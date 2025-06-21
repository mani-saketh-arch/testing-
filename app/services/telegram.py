"""
SafeIndy Assistant - Telegram Bot Service
Telegram bot integration for mobile emergency assistance
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import re

from telegram import (
    Update, 
    Bot, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove
)
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,
    filters,
    ContextTypes
)
from telegram.constants import ParseMode

from ..config import settings
from ..utils import timer, log_error, cache, sanitize_input, format_coordinates
from .llm import llm_service
from .rag import rag_service
from .external import location_service, weather_service, email_service
from .analytics import analytics_service

logger = logging.getLogger(__name__)


class TelegramService:
    """Telegram bot service for SafeIndy Assistant"""
    
    def __init__(self):
        self.bot = None
        self.application = None
        self.user_sessions = {}  # Store user session data
        self._initialize_bot()
    
    def _initialize_bot(self):
        """Initialize Telegram bot"""
        
        if not settings.TELEGRAM_BOT_TOKEN:
            logger.warning("⚠️ Telegram bot token not configured")
            return
        
        try:
            # Create bot instance
            self.bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
            
            # Create application
            self.application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
            
            # Register handlers
            self._register_handlers()
            
            logger.info("✅ Telegram bot initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Telegram bot: {e}")
            log_error(e, "Telegram bot initialization")
    
    def _register_handlers(self):
        """Register command and message handlers"""
        
        if not self.application:
            return
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.handle_start))
        self.application.add_handler(CommandHandler("help", self.handle_help))
        self.application.add_handler(CommandHandler("emergency", self.handle_emergency_command))
        self.application.add_handler(CommandHandler("location", self.handle_location_command))
        self.application.add_handler(CommandHandler("weather", self.handle_weather_command))
        self.application.add_handler(CommandHandler("services", self.handle_services_command))
        
        # Message handlers
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.add_handler(MessageHandler(filters.LOCATION, self.handle_location))
        self.application.add_handler(MessageHandler(filters.CONTACT, self.handle_contact))
        
        # Callback query handler for inline buttons
        self.application.add_handler(CallbackQueryHandler(self.handle_callback_query))
        
        # Error handler
        self.application.add_error_handler(self.handle_error)
    
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        
        user = update.effective_user
        chat_id = update.effective_chat.id
        
        try:
            # Initialize user session
            self.user_sessions[chat_id] = {
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "started_at": datetime.now(),
                "location": None
            }
            
            welcome_message = f"""🚨 **Welcome to SafeIndy Assistant!**

Hello {user.first_name}! I'm your AI-powered emergency response and civic assistance bot for Indianapolis.

**What I can help you with:**
🆘 Emergency assistance and guidance
🏛️ Indianapolis city services information
🌤️ Weather alerts and updates
📍 Location-based resources
🚔 Police and fire department contacts
🏥 Hospital and medical services

**Quick Commands:**
/emergency - Report an emergency
/location - Share your location
/weather - Get current weather
/services - City services info
/help - Show all commands

**To get started:** Just type your question or use the location button below to share your location for better assistance.

⚠️ **Important:** For immediate life-threatening emergencies, always call 911 directly!"""

            # Create location sharing keyboard
            keyboard = [
                [KeyboardButton("📍 Share My Location", request_location=True)],
                [KeyboardButton("🆘 Emergency"), KeyboardButton("🌤️ Weather")],
                [KeyboardButton("🏛️ City Services"), KeyboardButton("ℹ️ Help")]
            ]
            
            reply_markup = ReplyKeyboardMarkup(
                keyboard, 
                resize_keyboard=True, 
                one_time_keyboard=False
            )
            
            await update.message.reply_text(
                welcome_message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
            
            logger.info(f"New Telegram user started: {user.username or user.first_name} ({user.id})")
            
        except Exception as e:
            log_error(e, "Telegram start command", {"user_id": user.id})
            await update.message.reply_text("Sorry, I'm having trouble starting up. Please try again.")
    
    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        
        help_message = """🔧 **SafeIndy Assistant Commands**

**Emergency & Safety:**
/emergency - Report an emergency situation
📍 Share location - Get location-specific help

**Information Services:**
/weather - Current Indianapolis weather
/services - City services and contacts
/location - Location services and directions

**Quick Actions:**
• Type any question about Indianapolis
• Share your location for better assistance
• Use "emergency" in your message for urgent help

**Examples:**
"Is there a fire station near me?"
"What's the weather like?"
"I need to report a pothole"
"Emergency - car accident on I-65"

**Important Emergency Numbers:**
🚨 Emergency: 911
🚔 IMPD Non-Emergency: 317-327-3811
🏛️ Mayor's Action Center: 317-327-4622

⚠️ **Remember:** For immediate life-threatening emergencies, always call 911 first!"""

        await update.message.reply_text(help_message, parse_mode=ParseMode.MARKDOWN)
    
    async def handle_emergency_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /emergency command"""
        
        emergency_message = """🚨 **EMERGENCY ASSISTANCE**

If this is a **life-threatening emergency**, please **CALL 911 IMMEDIATELY**.

**For other urgent situations:**
1. Describe your emergency below
2. Share your location if safe to do so
3. I'll provide immediate guidance and contacts

**Emergency Services:**
🚨 Emergency: 911
🚔 Police Non-Emergency: 317-327-3811
🚑 Poison Control: 1-800-222-1222
🔥 Fire Department: 911

**What to include in your message:**
• What happened?
• Where are you? (address or location)
• Are you safe?
• Do you need immediate help?

Type your emergency situation now, and I'll help guide you to the right resources."""

        # Create emergency action buttons
        keyboard = [
            [InlineKeyboardButton("🚨 Call 911", url="tel:911")],
            [InlineKeyboardButton("🚔 IMPD Non-Emergency", url="tel:3173273811")],
            [InlineKeyboardButton("📍 Share Location", callback_data="share_location")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            emergency_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    async def handle_location_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /location command"""
        
        location_message = """📍 **Location Services**

Share your location to get:
• Nearest emergency services
• Location-specific weather alerts
• Nearby Indianapolis resources
• Accurate directions and guidance

**How to share location:**
1. Use the 📍 button below
2. Or type an address like "123 Main St"
3. Or describe your location

Your location data is only used to provide better assistance and is not stored permanently."""

        # Create location sharing button
        keyboard = [[KeyboardButton("📍 Share My Location", request_location=True)]]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
        
        await update.message.reply_text(
            location_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    async def handle_weather_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /weather command"""
        
        try:
            # Get current weather for Indianapolis
            weather_data = await weather_service.get_current_weather()
            
            if weather_data:
                weather_message = f"""🌤️ **Indianapolis Weather**

**Current Conditions:**
Temperature: {weather_data['temperature']:.1f}°F (feels like {weather_data['feels_like']:.1f}°F)
Condition: {weather_data['description']}
Humidity: {weather_data['humidity']}%
Wind Speed: {weather_data['wind_speed']:.1f} mph

**Location:** {weather_data['location']}
**Updated:** {datetime.now().strftime('%I:%M %p')}"""

                # Check for weather alerts
                alerts = await weather_service.get_weather_alerts()
                if alerts:
                    weather_message += "\n\n⚠️ **Weather Alerts:**\n"
                    for alert in alerts[:2]:  # Show first 2 alerts
                        weather_message += f"• {alert['title']}\n"
                
            else:
                weather_message = "Sorry, I couldn't retrieve current weather information. Please try again later."
            
            await update.message.reply_text(weather_message, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            log_error(e, "Telegram weather command")
            await update.message.reply_text("Sorry, I'm having trouble getting weather information.")
    
    async def handle_services_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /services command"""
        
        services_message = """🏛️ **Indianapolis City Services**

**Emergency Services:**
🚨 Emergency: 911
🚔 IMPD Non-Emergency: 317-327-3811
🚑 EMS: 911
🔥 Fire Department: 911

**City Services:**
📞 Indianapolis 311: 311
🏛️ Mayor's Action Center: 317-327-4622
🌐 RequestIndy: indy.gov/request

**Utilities:**
⚡ AES Indiana: 317-261-8111
💧 Citizens Energy: 317-924-3311
🗑️ Republic Services: 317-923-9100

**Health & Safety:**
🏥 Marion County Health: 317-221-2222
🦠 Poison Control: 1-800-222-1222

**Transportation:**
🚌 IndyGo: 317-635-3344
🚗 Traffic Issues: 317-327-4622

For specific questions, just ask me! Example: "How do I report a pothole?" or "What's the number for Citizens Energy?\""""

        # Create quick action buttons
        keyboard = [
            [InlineKeyboardButton("📞 Call 311", url="tel:311"),
             InlineKeyboardButton("🌐 RequestIndy", url="https://www.indy.gov/activity/request-indy")],
            [InlineKeyboardButton("🏛️ Mayor's Office", url="tel:3173274622")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            services_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages"""
        
        user = update.effective_user
        chat_id = update.effective_chat.id
        message_text = update.message.text
        
        try:
            # Sanitize input
            clean_message = sanitize_input(message_text, max_length=2000)
            
            # Check if this is a quick action button press
            if message_text in ["🆘 Emergency", "🌤️ Weather", "🏛️ City Services", "ℹ️ Help"]:
                await self._handle_quick_action(update, message_text)
                return
            
            # Prepare context for AI processing
            context_data = {
                "platform": "telegram",
                "user_id": user.id,
                "username": user.username,
                "chat_id": chat_id
            }
            
            # Add location context if available
            user_session = self.user_sessions.get(chat_id, {})
            if user_session.get("location"):
                context_data["user_location"] = user_session["location"]
            
            # Show typing indicator
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            
            # Process message with RAG service
            response_data = await rag_service.generate_response_with_context(
                user_query=clean_message,
                context=context_data
            )
            
            # Format response for Telegram
            formatted_response = self._format_response_for_telegram(response_data)
            
            # Send response
            await self._send_response(update, formatted_response, response_data)
            
            # Track interaction for analytics
            request_info = {
                "platform": "telegram",
                "user_agent": f"Telegram Bot API",
                "latitude": user_session.get("location", {}).get("latitude"),
                "longitude": user_session.get("location", {}).get("longitude")
            }
            
            await analytics_service.track_user_interaction(
                session_id=f"telegram_{chat_id}",
                user_query=clean_message,
                response_data=response_data,
                request_info=request_info
            )
            
            # Handle emergency if detected
            if response_data.get("emergency"):
                await self._handle_emergency_response(update, response_data, clean_message)
            
        except Exception as e:
            log_error(e, "Telegram message handling", {"user_id": user.id, "message": message_text[:100]})
            await update.message.reply_text(
                "Sorry, I'm having trouble processing your message. Please try again or use /help for assistance."
            )
    
    async def handle_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle location sharing"""
        
        user = update.effective_user
        chat_id = update.effective_chat.id
        location = update.message.location
        
        try:
            latitude = location.latitude
            longitude = location.longitude
            
            # Store location in user session
            if chat_id not in self.user_sessions:
                self.user_sessions[chat_id] = {}
            
            self.user_sessions[chat_id]["location"] = {
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": datetime.now()
            }
            
            # Get address from coordinates
            address_info = await location_service.reverse_geocode(latitude, longitude)
            
            # Get nearby emergency services
            nearby_services = await location_service.find_nearest_emergency_services(latitude, longitude)
            
            location_message = f"""📍 **Location Received**

**Your Location:**
{format_coordinates(latitude, longitude)}
{address_info.get('formatted_address', 'Address not available') if address_info else ''}

**Location saved for better assistance!**

**Nearest Emergency Services:**"""

            # Add nearby services
            if nearby_services.get("hospitals"):
                location_message += f"\n🏥 **Nearest Hospital:** {nearby_services['hospitals'][0]['name']}"
            
            if nearby_services.get("police"):
                location_message += f"\n🚔 **Nearest Police:** {nearby_services['police'][0]['name']}"
            
            if nearby_services.get("fire_stations"):
                location_message += f"\n🚒 **Nearest Fire Station:** {nearby_services['fire_stations'][0]['name']}"
            
            location_message += "\n\nNow I can provide more accurate, location-specific assistance!"
            
            # Create action buttons
            keyboard = [
                [InlineKeyboardButton("🆘 Report Emergency", callback_data="emergency_here")],
                [InlineKeyboardButton("🌤️ Local Weather", callback_data="weather_here")],
                [InlineKeyboardButton("🏥 Nearest Hospital", callback_data="nearest_hospital")]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                location_message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
            
            logger.info(f"Location received from user {user.id}: {latitude}, {longitude}")
            
        except Exception as e:
            log_error(e, "Telegram location handling", {"user_id": user.id})
            await update.message.reply_text("Thank you for sharing your location! I'll use it to provide better assistance.")
    
    async def handle_contact(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle contact sharing"""
        
        await update.message.reply_text(
            "Thank you for sharing your contact! This information helps me provide better emergency assistance if needed."
        )
    
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks"""
        
        query = update.callback_query
        await query.answer()
        
        callback_data = query.data
        
        try:
            if callback_data == "share_location":
                keyboard = [[KeyboardButton("📍 Share My Location", request_location=True)]]
                reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
                
                await query.edit_message_text(
                    "Please use the button below to share your location:",
                    reply_markup=reply_markup
                )
            
            elif callback_data == "emergency_here":
                await query.edit_message_text(
                    "🚨 **Emergency at your location**\n\nDescribe your emergency situation and I'll provide immediate guidance and contacts."
                )
            
            elif callback_data == "weather_here":
                user_session = self.user_sessions.get(query.message.chat_id, {})
                location = user_session.get("location")
                
                if location:
                    weather_data = await weather_service.get_current_weather(
                        location["latitude"], 
                        location["longitude"]
                    )
                    
                    if weather_data:
                        weather_text = f"""🌤️ **Weather at Your Location**

Temperature: {weather_data['temperature']:.1f}°F
Condition: {weather_data['description']}
Humidity: {weather_data['humidity']}%"""
                        
                        await query.edit_message_text(weather_text, parse_mode=ParseMode.MARKDOWN)
                    else:
                        await query.edit_message_text("Sorry, couldn't get weather for your location.")
                else:
                    await query.edit_message_text("Please share your location first to get local weather.")
            
            elif callback_data == "nearest_hospital":
                user_session = self.user_sessions.get(query.message.chat_id, {})
                location = user_session.get("location")
                
                if location:
                    services = await location_service.find_nearest_emergency_services(
                        location["latitude"], 
                        location["longitude"]
                    )
                    
                    if services.get("hospitals"):
                        hospital = services["hospitals"][0]
                        hospital_text = f"""🏥 **Nearest Hospital**

**{hospital['name']}**
Address: {hospital['address']}
Rating: {hospital.get('rating', 'N/A')}

For medical emergencies, call 911 for ambulance service."""
                        
                        await query.edit_message_text(hospital_text, parse_mode=ParseMode.MARKDOWN)
                    else:
                        await query.edit_message_text("No hospitals found nearby. For medical emergencies, call 911.")
                else:
                    await query.edit_message_text("Please share your location first to find nearby hospitals.")
            
        except Exception as e:
            log_error(e, "Telegram callback query", {"callback_data": callback_data})
            await query.edit_message_text("Sorry, I'm having trouble processing that request.")
    
    async def handle_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        
        logger.error(f"Telegram bot error: {context.error}")
        
        try:
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    "Sorry, something went wrong. Please try again or use /help for assistance."
                )
        except:
            pass  # Don't raise exceptions in error handler
    
    async def _handle_quick_action(self, update: Update, action: str):
        """Handle quick action button presses"""
        
        if action == "🆘 Emergency":
            await self.handle_emergency_command(update, None)
        elif action == "🌤️ Weather":
            await self.handle_weather_command(update, None)
        elif action == "🏛️ City Services":
            await self.handle_services_command(update, None)
        elif action == "ℹ️ Help":
            await self.handle_help(update, None)
    
    def _format_response_for_telegram(self, response_data: Dict[str, Any]) -> str:
        """Format AI response for Telegram"""
        
        if not response_data.get("success"):
            return "Sorry, I'm having trouble generating a response right now. Please try again."
        
        response_text = response_data.get("response", "")
        
        # Add source information if available
        sources = response_data.get("sources", [])
        if sources:
            response_text += "\n\n📚 **Sources:**"
            for i, source in enumerate(sources[:2], 1):  # Limit to 2 sources for Telegram
                if source.get("type") == "document":
                    response_text += f"\n{i}. {source.get('title', 'Document')}"
                elif source.get("type") == "web":
                    response_text += f"\n{i}. {source.get('title', 'Web Source')}"
        
        # Limit response length for Telegram (4096 character limit)
        if len(response_text) > 4000:
            response_text = response_text[:3950] + "...\n\n*Response truncated. Ask for more details if needed.*"
        
        return response_text
    
    async def _send_response(self, update: Update, response_text: str, response_data: Dict[str, Any]):
        """Send formatted response to user"""
        
        try:
            # Check if this is an emergency response
            if response_data.get("emergency"):
                # Add emergency header
                emergency_header = "🚨 **EMERGENCY DETECTED**\n\n"
                response_text = emergency_header + response_text
                
                # Add emergency action buttons
                keyboard = [
                    [InlineKeyboardButton("🚨 Call 911", url="tel:911")],
                    [InlineKeyboardButton("🚔 Non-Emergency", url="tel:3173273811")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(
                    response_text,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=reply_markup
                )
            else:
                await update.message.reply_text(
                    response_text,
                    parse_mode=ParseMode.MARKDOWN
                )
        
        except Exception as e:
            # Fallback to plain text if markdown fails
            try:
                await update.message.reply_text(response_text)
            except Exception as e2:
                log_error(e2, "Telegram response sending")
                await update.message.reply_text("Sorry, I'm having trouble sending my response.")
    
    async def _handle_emergency_response(self, update: Update, response_data: Dict[str, Any], original_message: str):
        """Handle emergency-specific response actions"""
        
        try:
            user = update.effective_user
            chat_id = update.effective_chat.id
            
            # Get location if available
            user_session = self.user_sessions.get(chat_id, {})
            location_info = user_session.get("location")
            
            # Prepare emergency alert data
            emergency_details = {
                "type": response_data.get("intent", "general_emergency"),
                "message": original_message,
                "confidence": response_data.get("confidence", 0.0),
                "keywords": response_data.get("emergency_keywords", []),
                "platform": "telegram",
                "user_id": user.id,
                "username": user.username,
                "chat_id": chat_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send emergency alert email
            await email_service.send_emergency_alert(
                emergency_details=emergency_details,
                location_info=location_info
            )
            
            # Send follow-up message to user
            follow_up_message = """🚨 **Emergency Alert Sent**

I've notified emergency services about your situation.

**What to do now:**
• If this is life-threatening: **Call 911 immediately**
• Stay calm and follow the guidance provided
• Keep your phone nearby for emergency contacts

**Emergency Numbers:**
🚨 Emergency: 911
🚔 Police: 317-327-3811
🏥 Poison Control: 1-800-222-1222"""

            await update.message.reply_text(follow_up_message, parse_mode=ParseMode.MARKDOWN)
            
            logger.info(f"Emergency alert processed for Telegram user {user.id}")
            
        except Exception as e:
            log_error(e, "Telegram emergency response handling")
    
    async def process_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming webhook data"""
        
        try:
            # Convert webhook data to Update object
            update = Update.de_json(webhook_data, self.bot)
            
            if update:
                # Process the update
                await self.application.process_update(update)
                
                return {
                    "success": True,
                    "message": "Webhook processed successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Invalid webhook data"
                }
        
        except Exception as e:
            log_error(e, "Telegram webhook processing", {"webhook_data": str(webhook_data)[:200]})
            return {
                "success": False,
                "error": str(e)
            }
    
    async def send_admin_notification(self, admin_chat_id: int, message: str):
        """Send notification to admin via Telegram"""
        
        try:
            if self.bot:
                await self.bot.send_message(
                    chat_id=admin_chat_id,
                    text=f"🔔 **SafeIndy Admin Alert**\n\n{message}",
                    parse_mode=ParseMode.MARKDOWN
                )
                return True
        except Exception as e:
            log_error(e, "Sending Telegram admin notification")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Telegram bot health"""
        
        health = {
            "bot_configured": bool(self.bot),
            "application_ready": bool(self.application),
            "webhook_configured": bool(settings.TELEGRAM_WEBHOOK_URL),
            "overall": "unknown"
        }
        
        if self.bot:
            try:
                # Test bot by getting info
                bot_info = await self.bot.get_me()
                health["bot_info"] = {
                    "username": bot_info.username,
                    "name": bot_info.first_name,
                    "id": bot_info.id
                }
                health["bot_status"] = "healthy"
            except Exception as e:
                health["bot_status"] = f"error: {str(e)[:50]}"
        else:
            health["bot_status"] = "not_configured"
        
        # Overall health
        if health["bot_configured"] and health["application_ready"]:
            health["overall"] = "healthy"
        elif health["bot_configured"]:
            health["overall"] = "degraded"
        else:
            health["overall"] = "not_configured"
        
        return health


# Global Telegram service instance
telegram_service = TelegramService()