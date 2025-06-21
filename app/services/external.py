"""
SafeIndy Assistant - External Services Integration
Google Maps, OpenWeather, Perplexity, and Email services
"""

import asyncio
import aiohttp
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from datetime import datetime
import googlemaps

from ..config import settings
from ..utils import timer, log_error, cache, format_coordinates

logger = logging.getLogger(__name__)


class ExternalServices:
    """Centralized external services integration"""
    
    def __init__(self):
        self.gmaps_client = None
        self.session = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize external service clients"""
        # Google Maps client
        try:
            self.gmaps_client = googlemaps.Client(key=settings.GOOGLE_MAPS_API_KEY)
            logger.info("✅ Google Maps client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Maps client: {e}")
            self.gmaps_client = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()


# Google Maps Services
class LocationService:
    """Google Maps integration for location services"""
    
    def __init__(self):
        self.gmaps_client = None
        try:
            self.gmaps_client = googlemaps.Client(key=settings.GOOGLE_MAPS_API_KEY)
            logger.info("✅ Google Maps client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Maps: {e}")
    
    @timer
    async def geocode_address(self, address: str) -> Optional[Dict[str, Any]]:
        """Convert address to coordinates"""
        if not self.gmaps_client:
            return None
        
        cache_key = f"geocode_{address.lower()}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Add Indianapolis context if not specified
            if "indianapolis" not in address.lower() and "indiana" not in address.lower():
                address += ", Indianapolis, IN"
            
            results = self.gmaps_client.geocode(address)
            
            if results:
                location = results[0]
                result = {
                    "latitude": location['geometry']['location']['lat'],
                    "longitude": location['geometry']['location']['lng'],
                    "formatted_address": location['formatted_address'],
                    "place_id": location['place_id'],
                    "address_components": location.get('address_components', [])
                }
                
                # Cache for 1 hour
                cache.set(cache_key, result, 3600)
                return result
        
        except Exception as e:
            log_error(e, f"Geocoding address: {address}")
        
        return None
    
    @timer
    async def reverse_geocode(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """Convert coordinates to address"""
        if not self.gmaps_client:
            return None
        
        cache_key = f"reverse_{latitude}_{longitude}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            results = self.gmaps_client.reverse_geocode((latitude, longitude))
            
            if results:
                location = results[0]
                result = {
                    "formatted_address": location['formatted_address'],
                    "place_id": location['place_id'],
                    "address_components": location.get('address_components', []),
                    "coordinates": format_coordinates(latitude, longitude)
                }
                
                # Cache for 24 hours
                cache.set(cache_key, result, 86400)
                return result
        
        except Exception as e:
            log_error(e, f"Reverse geocoding: {latitude}, {longitude}")
        
        return None
    
    @timer
    async def find_nearest_emergency_services(
        self, 
        latitude: float, 
        longitude: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find nearest emergency services"""
        if not self.gmaps_client:
            return {}
        
        cache_key = f"emergency_services_{latitude}_{longitude}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            location = (latitude, longitude)
            services = {}
            
            # Search for different types of emergency services
            service_types = {
                "hospitals": "hospital",
                "police": "police",
                "fire_stations": "fire_station",
                "pharmacies": "pharmacy"
            }
            
            for service_name, place_type in service_types.items():
                try:
                    places = self.gmaps_client.places_nearby(
                        location=location,
                        radius=10000,  # 10km radius
                        type=place_type
                    )
                    
                    service_list = []
                    for place in places.get('results', [])[:5]:  # Top 5 results
                        service_info = {
                            "name": place['name'],
                            "address": place.get('vicinity', 'Address not available'),
                            "rating": place.get('rating'),
                            "place_id": place['place_id'],
                            "location": place['geometry']['location']
                        }
                        service_list.append(service_info)
                    
                    services[service_name] = service_list
                
                except Exception as e:
                    logger.warning(f"Failed to fetch {service_name}: {e}")
                    services[service_name] = []
            
            # Cache for 6 hours
            cache.set(cache_key, services, 21600)
            return services
        
        except Exception as e:
            log_error(e, f"Finding emergency services near {latitude}, {longitude}")
            return {}


# Weather Service
class WeatherService:
    """OpenWeather API integration"""
    
    @timer
    async def get_current_weather(
        self, 
        latitude: float = None, 
        longitude: float = None
    ) -> Optional[Dict[str, Any]]:
        """Get current weather for Indianapolis or specified location"""
        
        # Use Indianapolis coordinates if not specified
        if latitude is None or longitude is None:
            latitude, longitude = settings.indianapolis_coordinates
        
        cache_key = f"weather_{latitude}_{longitude}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.openweathermap.org/data/2.5/weather"
                params = {
                    "lat": latitude,
                    "lon": longitude,
                    "appid": settings.OPENWEATHER_API_KEY,
                    "units": "imperial"  # Fahrenheit for US
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        weather_info = {
                            "temperature": data['main']['temp'],
                            "feels_like": data['main']['feels_like'],
                            "humidity": data['main']['humidity'],
                            "description": data['weather'][0]['description'].title(),
                            "condition": data['weather'][0]['main'],
                            "wind_speed": data.get('wind', {}).get('speed', 0),
                            "visibility": data.get('visibility', 0) / 1609.34,  # Convert to miles
                            "location": data['name'],
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Cache for 10 minutes
                        cache.set(cache_key, weather_info, 600)
                        return weather_info
        
        except Exception as e:
            log_error(e, f"Getting weather for {latitude}, {longitude}")
        
        return None
    
    @timer
    async def get_weather_alerts(
        self, 
        latitude: float = None, 
        longitude: float = None
    ) -> List[Dict[str, Any]]:
        """Get weather alerts for location"""
        
        # Use Indianapolis coordinates if not specified
        if latitude is None or longitude is None:
            latitude, longitude = settings.indianapolis_coordinates
        
        cache_key = f"weather_alerts_{latitude}_{longitude}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.openweathermap.org/data/2.5/onecall"
                params = {
                    "lat": latitude,
                    "lon": longitude,
                    "appid": settings.OPENWEATHER_API_KEY,
                    "exclude": "minutely,hourly,daily"  # Only get current and alerts
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        alerts = []
                        for alert in data.get('alerts', []):
                            alert_info = {
                                "title": alert['event'],
                                "description": alert['description'],
                                "severity": alert.get('severity', 'unknown'),
                                "start": datetime.fromtimestamp(alert['start']).isoformat(),
                                "end": datetime.fromtimestamp(alert['end']).isoformat(),
                                "sender": alert.get('sender_name', 'Weather Service')
                            }
                            alerts.append(alert_info)
                        
                        # Cache for 5 minutes
                        cache.set(cache_key, alerts, 300)
                        return alerts
        
        except Exception as e:
            log_error(e, f"Getting weather alerts for {latitude}, {longitude}")
        
        return []


# Perplexity Search Service
class SearchService:
    """Perplexity API for real-time Indianapolis data"""
    
    @timer
    async def search_indianapolis_info(
        self, 
        query: str, 
        focus: str = "indianapolis"
    ) -> Optional[Dict[str, Any]]:
        """Search for current Indianapolis information"""
        
        cache_key = f"search_{query.lower()}_{focus}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Enhance query with Indianapolis context
            enhanced_query = f"{query} Indianapolis Indiana current information"
            if focus == "emergency":
                enhanced_query += " emergency services police fire department"
            elif focus == "city_services":
                enhanced_query += " city government mayor's office 311"
            
            async with aiohttp.ClientSession() as session:
                url = "https://api.perplexity.ai/chat/completions"
                headers = {
                    "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "sonar-pro",
                    "messages": [
                        {
                            "role": "user",
                            "content": enhanced_query
                        }
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.2,
                    "search_domain_filter": ["indy.gov", "indianapolisrecorder.com", "indystar.com"],
                    "return_citations": True
                }
                
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('choices') and len(data['choices']) > 0:
                            result = {
                                "content": data['choices'][0]['message']['content'],
                                "citations": data.get('citations', []),
                                "timestamp": datetime.now().isoformat(),
                                "query": query
                            }
                            
                            # Cache for 15 minutes
                            cache.set(cache_key, result, 900)
                            return result
        
        except Exception as e:
            log_error(e, f"Perplexity search for: {query}")
        
        return None


# Email Service for Emergency Alerts
class EmailService:
    """SMTP email service for emergency notifications"""
    
    @timer
    async def send_emergency_alert(
        self,
        emergency_details: Dict[str, Any],
        location_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send emergency alert email"""
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = settings.SMTP_USERNAME
            msg['To'] = settings.EMERGENCY_EMAIL
            msg['Subject'] = f"🚨 SafeIndy Emergency Alert - {emergency_details.get('type', 'Emergency')}"
            
            # Build email body
            body = f"""
🚨 EMERGENCY ALERT from SafeIndy Assistant

Emergency Type: {emergency_details.get('type', 'Unknown')}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Confidence: {emergency_details.get('confidence', 'Unknown')}

USER MESSAGE:
{emergency_details.get('message', 'No message provided')}

LOCATION INFORMATION:
"""
            
            if location_info:
                body += f"""
Coordinates: {location_info.get('coordinates', 'Not available')}
Address: {location_info.get('formatted_address', 'Not available')}
Accuracy: {location_info.get('accuracy', 'Unknown')} meters
"""
            else:
                body += "Location not provided\n"
            
            body += f"""
RESPONSE DETAILS:
Platform: {emergency_details.get('platform', 'Web')}
Session ID: {emergency_details.get('session_id', 'Unknown')}
User Agent: {emergency_details.get('user_agent', 'Unknown')}

EMERGENCY KEYWORDS DETECTED:
{', '.join(emergency_details.get('keywords', []))}

---
This alert was generated automatically by SafeIndy Assistant.
Please verify the emergency by contacting the user or dispatching appropriate resources.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT)
            server.starttls()
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            text = msg.as_string()
            server.sendmail(settings.SMTP_USERNAME, settings.EMERGENCY_EMAIL, text)
            server.quit()
            
            logger.info(f"✅ Emergency alert email sent successfully")
            return True
            
        except Exception as e:
            log_error(e, "Sending emergency alert email", emergency_details)
            return False
    
    @timer
    async def send_admin_notification(
        self,
        subject: str,
        content: str,
        recipient: str = None
    ) -> bool:
        """Send notification email to admin"""
        
        if not recipient:
            recipient = settings.ADMIN_EMAIL
        
        try:
            msg = MIMEMultipart()
            msg['From'] = settings.SMTP_USERNAME
            msg['To'] = recipient
            msg['Subject'] = f"SafeIndy Admin: {subject}"
            
            body = f"""
SafeIndy Assistant Admin Notification

{content}

---
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System: SafeIndy Assistant v{settings.APP_VERSION}
Environment: {settings.ENVIRONMENT}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT)
            server.starttls()
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            text = msg.as_string()
            server.sendmail(settings.SMTP_USERNAME, recipient, text)
            server.quit()
            
            logger.info(f"✅ Admin notification sent to {recipient}")
            return True
            
        except Exception as e:
            log_error(e, "Sending admin notification", {"subject": subject, "recipient": recipient})
            return False


# Indianapolis Specific Services
class IndyDataService:
    """Indianapolis-specific data and services"""
    
    @timer
    async def get_city_services_info(self, service_type: str = None) -> Dict[str, Any]:
        """Get Indianapolis city services information"""
        
        cache_key = f"city_services_{service_type or 'all'}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Indianapolis city services data
        services = {
            "emergency": {
                "911": {
                    "name": "Emergency Services",
                    "number": "911",
                    "description": "Fire, Police, Medical Emergency"
                },
                "non_emergency": {
                    "name": "IMPD Non-Emergency",
                    "number": settings.INDY_NON_EMERGENCY,
                    "description": "Indianapolis Metropolitan Police Department"
                }
            },
            "city_services": {
                "311": {
                    "name": "Indianapolis 311",
                    "number": "311",
                    "description": "Non-emergency city services",
                    "website": "https://www.indy.gov/activity/request-indy"
                },
                "mayors_action_center": {
                    "name": "Mayor's Action Center",
                    "number": "317-327-4622",
                    "description": "City service requests and complaints",
                    "website": "https://www.indy.gov/activity/mayors-action-center"
                }
            },
            "utilities": {
                "citizens_energy": {
                    "name": "Citizens Energy Group",
                    "number": "317-924-3311",
                    "emergency": "317-924-3311",
                    "description": "Gas and water utilities"
                },
                "aes_indiana": {
                    "name": "AES Indiana",
                    "number": "317-261-8111",
                    "emergency": "317-261-8111",
                    "description": "Electric utility"
                }
            },
            "health": {
                "health_department": {
                    "name": "Marion County Health Department",
                    "number": "317-221-2222",
                    "description": "Public health services",
                    "website": "https://marionhealth.org"
                }
            },
            "transportation": {
                "indygo": {
                    "name": "IndyGo Public Transit",
                    "number": "317-635-3344",
                    "description": "Public transportation",
                    "website": "https://www.indygo.net"
                },
                "traffic_management": {
                    "name": "Traffic Management",
                    "number": "317-327-4622",
                    "description": "Traffic signals and road issues"
                }
            }
        }
        
        if service_type and service_type in services:
            result = {service_type: services[service_type]}
        else:
            result = services
        
        # Cache for 6 hours
        cache.set(cache_key, result, 21600)
        return result
    
    @timer
    async def get_neighborhood_info(self, location: str) -> Optional[Dict[str, Any]]:
        """Get Indianapolis neighborhood information"""
        
        cache_key = f"neighborhood_{location.lower()}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Indianapolis neighborhoods data
        neighborhoods = {
            "downtown": {
                "name": "Downtown Indianapolis",
                "description": "Central business district",
                "police_district": "Central District",
                "council_district": "At-large",
                "landmarks": ["Monument Circle", "Lucas Oil Stadium", "Bankers Life Fieldhouse"]
            },
            "broad_ripple": {
                "name": "Broad Ripple",
                "description": "Arts and entertainment district",
                "police_district": "North District",
                "council_district": "6",
                "landmarks": ["Broad Ripple Village", "Monon Trail"]
            },
            "fountain_square": {
                "name": "Fountain Square",
                "description": "Historic cultural district",
                "police_district": "Southeast District",
                "council_district": "14",
                "landmarks": ["Fountain Square Theatre", "Duckpin Bowling"]
            },
            # Add more neighborhoods as needed
        }
        
        location_key = location.lower().replace(" ", "_").replace("-", "_")
        
        if location_key in neighborhoods:
            result = neighborhoods[location_key]
            # Cache for 24 hours
            cache.set(cache_key, result, 86400)
            return result
        
        return None


# Service health checks
async def check_external_services_health() -> Dict[str, Any]:
    """Check health of all external services"""
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Test Google Maps
    try:
        location_service = LocationService()
        if location_service.gmaps_client:
            # Quick geocoding test
            result = await location_service.geocode_address("Indianapolis, IN")
            health_status["services"]["google_maps"] = "healthy" if result else "error"
        else:
            health_status["services"]["google_maps"] = "not_configured"
    except Exception as e:
        health_status["services"]["google_maps"] = f"error: {str(e)[:50]}"
    
    # Test OpenWeather
    try:
        weather_service = WeatherService()
        weather = await weather_service.get_current_weather()
        health_status["services"]["openweather"] = "healthy" if weather else "error"
    except Exception as e:
        health_status["services"]["openweather"] = f"error: {str(e)[:50]}"
    
    # Test Perplexity
    try:
        search_service = SearchService()
        result = await search_service.search_indianapolis_info("Indianapolis weather")
        health_status["services"]["perplexity"] = "healthy" if result else "error"
    except Exception as e:
        health_status["services"]["perplexity"] = f"error: {str(e)[:50]}"
    
    # Test Email (just check configuration)
    try:
        if all([settings.SMTP_SERVER, settings.SMTP_USERNAME, settings.SMTP_PASSWORD]):
            health_status["services"]["email"] = "configured"
        else:
            health_status["services"]["email"] = "not_configured"
    except Exception as e:
        health_status["services"]["email"] = f"error: {str(e)[:50]}"
    
    # Overall health
    healthy_services = sum(1 for status in health_status["services"].values() 
                          if status in ["healthy", "configured"])
    total_services = len(health_status["services"])
    
    if healthy_services == total_services:
        health_status["overall"] = "healthy"
    elif healthy_services > total_services // 2:
        health_status["overall"] = "degraded"
    else:
        health_status["overall"] = "error"
    
    return health_status


# Global service instances
location_service = LocationService()
weather_service = WeatherService()
search_service = SearchService()
email_service = EmailService()
indy_data_service = IndyDataService()