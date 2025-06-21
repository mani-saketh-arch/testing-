"""
SafeIndy Assistant - API Routes
API endpoints and Telegram webhook processing
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..utils import timer, log_error, sanitize_input
from ..services import (
    llm_service,
    rag_service,
    location_service,
    weather_service,
    indy_data_service,
    telegram_service,
    analytics_service,
    email_service
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Pydantic models for API requests
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    location_accuracy: Optional[float] = None
    platform: str = "api"


class EmergencyRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    contact_info: Optional[str] = None
    session_id: Optional[str] = None


class LocationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


# API endpoints
@router.post("/chat")
@timer
async def api_chat(
    request: Request,
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    user_agent: Optional[str] = Header(None)
):
    """API endpoint for chat functionality"""
    
    start_time = datetime.now()
    
    try:
        # Generate session ID if not provided
        session_id = chat_request.session_id or str(uuid4())
        
        # Sanitize input
        clean_message = sanitize_input(chat_request.message, max_length=2000)
        
        # Prepare context
        context_data = {
            "platform": chat_request.platform,
            "session_id": session_id,
            "timestamp": start_time.isoformat()
        }
        
        # Add location context if provided
        if chat_request.latitude and chat_request.longitude:
            context_data["user_location"] = (chat_request.latitude, chat_request.longitude)
            
            # Get address from coordinates
            try:
                address_info = await location_service.reverse_geocode(
                    chat_request.latitude, 
                    chat_request.longitude
                )
                if address_info:
                    context_data["user_address"] = address_info.get("formatted_address", "")
            except Exception as e:
                logger.warning(f"Failed to get address for API request: {e}")
        
        # Process with RAG service
        response_data = await rag_service.generate_response_with_context(
            user_query=clean_message,
            context=context_data
        )
        
        # Calculate response time
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        response_data["response_time_ms"] = response_time_ms
        
        # Track interaction for analytics
        request_info = {
            "platform": chat_request.platform,
            "user_agent": user_agent or "",
            "ip_address": request.client.host,
            "latitude": chat_request.latitude,
            "longitude": chat_request.longitude,
            "location_accuracy": chat_request.location_accuracy
        }
        
        await analytics_service.track_user_interaction(
            session_id=session_id,
            user_query=clean_message,
            response_data=response_data,
            request_info=request_info
        )
        
        # Handle emergency if detected
        if response_data.get("emergency"):
            try:
                emergency_details = {
                    "type": response_data.get("intent", "emergency"),
                    "message": clean_message,
                    "confidence": response_data.get("confidence", 0.0),
                    "keywords": response_data.get("emergency_keywords", []),
                    "platform": chat_request.platform,
                    "session_id": session_id,
                    "user_agent": user_agent or "",
                    "timestamp": datetime.now().isoformat()
                }
                
                location_info = None
                if chat_request.latitude and chat_request.longitude:
                    location_info = {
                        "latitude": chat_request.latitude,
                        "longitude": chat_request.longitude,
                        "accuracy": chat_request.location_accuracy,
                        "coordinates": f"{chat_request.latitude}, {chat_request.longitude}",
                        "formatted_address": context_data.get("user_address", "")
                    }
                
                # Send emergency alert asynchronously
                asyncio.create_task(
                    email_service.send_emergency_alert(emergency_details, location_info)
                )
                
            except Exception as e:
                logger.error(f"Failed to send emergency alert via API: {e}")
        
        # Prepare API response
        api_response = {
            "success": response_data.get("success", True),
            "response": response_data.get("response", ""),
            "session_id": session_id,
            "emergency": response_data.get("emergency", False),
            "intent": response_data.get("intent", "unknown"),
            "confidence": response_data.get("confidence", 0.0),
            "sources": response_data.get("sources", []),
            "response_time_ms": response_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add emergency-specific data
        if response_data.get("emergency"):
            api_response.update({
                "emergency_contacts": settings.emergency_contacts,
                "emergency_keywords": response_data.get("emergency_keywords", [])
            })
        
        return JSONResponse(content=api_response)
    
    except Exception as e:
        error_msg = f"API chat processing error: {e}"
        log_error(e, "API chat processing", {
            "session_id": chat_request.session_id,
            "message": clean_message[:100] if 'clean_message' in locals() else chat_request.message[:100]
        })
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": error_msg,
                "response": "I apologize, but I'm experiencing technical difficulties. If this is an emergency, please call 911 directly.",
                "emergency_contacts": settings.emergency_contacts,
                "timestamp": datetime.now().isoformat()
            }
        )


@router.post("/emergency")
@timer
async def api_emergency(
    request: Request,
    emergency_request: EmergencyRequest,
    db: Session = Depends(get_db),
    user_agent: Optional[str] = Header(None)
):
    """API endpoint for emergency reporting"""
    
    try:
        # Generate session ID if not provided
        session_id = emergency_request.session_id or str(uuid4())
        
        # Sanitize input
        clean_message = sanitize_input(emergency_request.message, max_length=2000)
        
        # Force emergency classification
        context_data = {
            "platform": "api",
            "session_id": session_id,
            "emergency_detected": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add location context
        if emergency_request.latitude and emergency_request.longitude:
            context_data["user_location"] = (emergency_request.latitude, emergency_request.longitude)
            
            # Get address
            try:
                address_info = await location_service.reverse_geocode(
                    emergency_request.latitude, 
                    emergency_request.longitude
                )
                if address_info:
                    context_data["user_address"] = address_info.get("formatted_address", "")
            except Exception as e:
                logger.warning(f"Failed to get emergency address: {e}")
        
        # Process emergency with LLM
        response_data = await llm_service.generate_response(
            user_message=clean_message,
            context=context_data,
            emergency_mode=True
        )
        
        # Ensure emergency flag is set
        response_data["emergency"] = True
        response_data["intent"] = "emergency"
        
        # Send emergency alert
        emergency_details = {
            "type": "api_emergency",
            "message": clean_message,
            "confidence": 1.0,  # API emergency calls are assumed urgent
            "contact_info": emergency_request.contact_info,
            "platform": "api",
            "session_id": session_id,
            "user_agent": user_agent or "",
            "timestamp": datetime.now().isoformat()
        }
        
        location_info = None
        if emergency_request.latitude and emergency_request.longitude:
            location_info = {
                "latitude": emergency_request.latitude,
                "longitude": emergency_request.longitude,
                "coordinates": f"{emergency_request.latitude}, {emergency_request.longitude}",
                "formatted_address": context_data.get("user_address", "")
            }
        
        # Send alert immediately
        alert_success = await email_service.send_emergency_alert(emergency_details, location_info)
        
        # Track emergency interaction
        request_info = {
            "platform": "api",
            "user_agent": user_agent or "",
            "ip_address": request.client.host,
            "latitude": emergency_request.latitude,
            "longitude": emergency_request.longitude
        }
        
        await analytics_service.track_user_interaction(
            session_id=session_id,
            user_query=clean_message,
            response_data=response_data,
            request_info=request_info
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "Emergency alert sent",
            "response": response_data.get("response", "Emergency services have been notified."),
            "alert_sent": alert_success,
            "emergency_contacts": settings.emergency_contacts,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        log_error(e, "API emergency processing", {
            "session_id": emergency_request.session_id,
            "message": emergency_request.message[:100]
        })
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Emergency processing failed. Please call 911 directly.",
                "emergency_contacts": settings.emergency_contacts,
                "timestamp": datetime.now().isoformat()
            }
        )


@router.post("/location/reverse-geocode")
async def api_reverse_geocode(location_request: LocationRequest):
    """API endpoint for reverse geocoding"""
    
    try:
        address_info = await location_service.reverse_geocode(
            location_request.latitude,
            location_request.longitude
        )
        
        if not address_info:
            return JSONResponse(
                status_code=404,
                content={"error": "Address not found for coordinates"}
            )
        
        return JSONResponse(content={
            "success": True,
            "address": address_info,
            "coordinates": {
                "latitude": location_request.latitude,
                "longitude": location_request.longitude
            }
        })
    
    except Exception as e:
        log_error(e, "API reverse geocoding", {
            "lat": location_request.latitude,
            "lon": location_request.longitude
        })
        
        return JSONResponse(
            status_code=500,
            content={"error": "Reverse geocoding failed"}
        )


@router.get("/location/nearby")
async def api_nearby_services(
    latitude: float,
    longitude: float,
    service_type: Optional[str] = None
):
    """API endpoint for finding nearby services"""
    
    try:
        # Validate coordinates
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinates")
        
        # Get nearby services
        nearby_services = await location_service.find_nearest_emergency_services(latitude, longitude)
        
        if service_type and service_type in nearby_services:
            services = {service_type: nearby_services[service_type]}
        else:
            services = nearby_services
        
        return JSONResponse(content={
            "success": True,
            "location": {"latitude": latitude, "longitude": longitude},
            "services": services,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        log_error(e, "API nearby services", {"lat": latitude, "lon": longitude})
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to find nearby services"}
        )


@router.get("/weather/current")
async def api_current_weather(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
):
    """API endpoint for current weather"""
    
    try:
        # Use Indianapolis coordinates if not specified
        if not latitude or not longitude:
            latitude, longitude = settings.indianapolis_coordinates
        
        weather_data = await weather_service.get_current_weather(latitude, longitude)
        
        if not weather_data:
            return JSONResponse(
                status_code=503,
                content={"error": "Weather service unavailable"}
            )
        
        alerts = await weather_service.get_weather_alerts(latitude, longitude)
        
        return JSONResponse(content={
            "success": True,
            "weather": weather_data,
            "alerts": alerts,
            "location": {"latitude": latitude, "longitude": longitude},
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        log_error(e, "API weather", {"lat": latitude, "lon": longitude})
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get weather data"}
        )


@router.get("/services/city")
async def api_city_services(service_type: Optional[str] = None):
    """API endpoint for Indianapolis city services"""
    
    try:
        services_data = await indy_data_service.get_city_services_info(service_type)
        
        return JSONResponse(content={
            "success": True,
            "services": services_data,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        log_error(e, "API city services", {"service_type": service_type})
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get city services"}
        )


# Telegram webhook endpoint
@router.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """Telegram bot webhook endpoint"""
    
    try:
        # Verify webhook is from Telegram (basic security)
        # In production, you might want to verify the secret token
        
        # Get webhook data
        webhook_data = await request.json()
        
        # Process with Telegram service
        result = await telegram_service.process_webhook(webhook_data)
        
        if result.get("success"):
            return JSONResponse(content={"ok": True})
        else:
            logger.error(f"Telegram webhook processing failed: {result.get('error')}")
            return JSONResponse(content={"ok": False}, status_code=500)
    
    except Exception as e:
        log_error(e, "Telegram webhook processing", {"webhook_data": str(await request.body())[:200]})
        return JSONResponse(content={"ok": False}, status_code=500)


@router.get("/telegram/webhook/info")
async def telegram_webhook_info():
    """Get Telegram webhook information"""
    
    try:
        if not telegram_service.bot:
            return JSONResponse(
                status_code=503,
                content={"error": "Telegram bot not configured"}
            )
        
        # Get webhook info from Telegram
        webhook_info = await telegram_service.bot.get_webhook_info()
        
        return JSONResponse(content={
            "success": True,
            "webhook_info": {
                "url": webhook_info.url,
                "has_custom_certificate": webhook_info.has_custom_certificate,
                "pending_update_count": webhook_info.pending_update_count,
                "last_error_date": webhook_info.last_error_date,
                "last_error_message": webhook_info.last_error_message,
                "max_connections": webhook_info.max_connections,
                "allowed_updates": webhook_info.allowed_updates
            }
        })
    
    except Exception as e:
        log_error(e, "Getting Telegram webhook info")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get webhook info"}
        )


@router.post("/telegram/webhook/set")
async def set_telegram_webhook():
    """Set Telegram webhook URL"""
    
    try:
        if not telegram_service.bot:
            return JSONResponse(
                status_code=503,
                content={"error": "Telegram bot not configured"}
            )
        
        if not settings.TELEGRAM_WEBHOOK_URL:
            return JSONResponse(
                status_code=400,
                content={"error": "Webhook URL not configured"}
            )
        
        # Set webhook
        webhook_url = f"{settings.TELEGRAM_WEBHOOK_URL}/api/telegram/webhook"
        success = await telegram_service.bot.set_webhook(
            url=webhook_url,
            allowed_updates=["message", "callback_query"]
        )
        
        if success:
            return JSONResponse(content={
                "success": True,
                "message": "Webhook set successfully",
                "webhook_url": webhook_url
            })
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to set webhook"}
            )
    
    except Exception as e:
        log_error(e, "Setting Telegram webhook")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to set webhook"}
        )


@router.delete("/telegram/webhook")
async def delete_telegram_webhook():
    """Delete Telegram webhook"""
    
    try:
        if not telegram_service.bot:
            return JSONResponse(
                status_code=503,
                content={"error": "Telegram bot not configured"}
            )
        
        # Delete webhook
        success = await telegram_service.bot.delete_webhook()
        
        if success:
            return JSONResponse(content={
                "success": True,
                "message": "Webhook deleted successfully"
            })
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to delete webhook"}
            )
    
    except Exception as e:
        log_error(e, "Deleting Telegram webhook")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to delete webhook"}
        )


# Health and status endpoints
@router.get("/health")
async def api_health_check():
    """API health check endpoint"""
    
    try:
        health_status = {
            "status": "healthy",
            "service": "SafeIndy Assistant API",
            "version": settings.APP_VERSION,
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "chat": "/api/chat",
                "emergency": "/api/emergency",
                "weather": "/api/weather/current",
                "location": "/api/location/nearby",
                "services": "/api/services/city",
                "telegram": "/api/telegram/webhook"
            }
        }
        
        return JSONResponse(content=health_status)
    
    except Exception as e:
        log_error(e, "API health check")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/status")
async def api_status():
    """Detailed API status with service checks"""
    
    try:
        status_data = {
            "api_version": settings.APP_VERSION,
            "timestamp": datetime.now().isoformat(),
            "environment": settings.ENVIRONMENT,
            "services": {}
        }
        
        # Check LLM service
        try:
            llm_health = await llm_service.health_check()
            status_data["services"]["llm"] = llm_health.get("overall", "unknown")
        except Exception as e:
            status_data["services"]["llm"] = f"error: {str(e)[:50]}"
        
        # Check RAG service
        try:
            rag_health = await rag_service.health_check()
            status_data["services"]["rag"] = rag_health.get("overall", "unknown")
        except Exception as e:
            status_data["services"]["rag"] = f"error: {str(e)[:50]}"
        
        # Check Telegram service
        try:
            telegram_health = await telegram_service.health_check()
            status_data["services"]["telegram"] = telegram_health.get("overall", "unknown")
        except Exception as e:
            status_data["services"]["telegram"] = f"error: {str(e)[:50]}"
        
        # Check location service
        try:
            test_location = await location_service.reverse_geocode(*settings.indianapolis_coordinates)
            status_data["services"]["location"] = "healthy" if test_location else "error"
        except Exception as e:
            status_data["services"]["location"] = f"error: {str(e)[:50]}"
        
        # Check weather service
        try:
            weather_data = await weather_service.get_current_weather()
            status_data["services"]["weather"] = "healthy" if weather_data else "error"
        except Exception as e:
            status_data["services"]["weather"] = f"error: {str(e)[:50]}"
        
        # Overall status
        service_statuses = list(status_data["services"].values())
        healthy_count = sum(1 for status in service_statuses if status == "healthy")
        total_count = len(service_statuses)
        
        if healthy_count == total_count:
            status_data["overall_status"] = "healthy"
        elif healthy_count > total_count // 2:
            status_data["overall_status"] = "degraded"
        else:
            status_data["overall_status"] = "error"
        
        status_data["service_health_ratio"] = f"{healthy_count}/{total_count}"
        
        return JSONResponse(content=status_data)
    
    except Exception as e:
        log_error(e, "API status check")
        return JSONResponse(
            status_code=500,
            content={
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# Analytics endpoints for API consumers
@router.get("/analytics/metrics")
async def api_analytics_metrics(
    time_period: str = "24h",
    metric_type: str = "overview"
):
    """Get analytics metrics via API"""
    
    try:
        if metric_type == "overview":
            data = await analytics_service.get_dashboard_overview(time_period)
        elif metric_type == "geographic":
            data = await analytics_service.get_geographic_data(time_period)
        elif metric_type == "emergency":
            data = await analytics_service.get_emergency_analytics(time_period)
        elif metric_type == "trends":
            days = int(time_period.replace("d", "")) if time_period.endswith("d") else 7
            data = await analytics_service.get_usage_trends(days)
        else:
            raise HTTPException(status_code=400, detail="Invalid metric type")
        
        # Remove sensitive data for public API
        if "error" not in data:
            # Remove detailed location data for privacy
            if metric_type == "geographic" and "location_points" in data:
                # Only return aggregated data
                data["location_points"] = []
                data["emergency_points"] = []
        
        return JSONResponse(content={
            "success": True,
            "metric_type": metric_type,
            "time_period": time_period,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, f"API analytics: {metric_type}/{time_period}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get analytics data"}
        )


# Emergency contacts endpoint
@router.get("/emergency/contacts")
async def api_emergency_contacts():
    """Get emergency contact information"""
    
    try:
        emergency_data = {
            "contacts": settings.emergency_contacts,
            "location": {
                "city": "Indianapolis",
                "state": "Indiana",
                "coordinates": settings.indianapolis_coordinates
            },
            "services": await indy_data_service.get_city_services_info("emergency"),
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=emergency_data)
    
    except Exception as e:
        log_error(e, "API emergency contacts")
        return JSONResponse(
            status_code=500,
            content={
                "contacts": settings.emergency_contacts,
                "error": "Failed to get complete emergency information"
            }
        )


# Rate limiting info
@router.get("/rate-limits")
async def api_rate_limits():
    """Get API rate limiting information"""
    
    return JSONResponse(content={
        "rate_limits": {
            "requests_per_window": settings.RATE_LIMIT_REQUESTS,
            "window_seconds": settings.RATE_LIMIT_WINDOW,
            "description": f"{settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW} seconds"
        },
        "headers": {
            "X-RateLimit-Limit": "Requests allowed per window",
            "X-RateLimit-Remaining": "Requests remaining in current window",
            "X-RateLimit-Reset": "Time when rate limit resets"
        },
        "timestamp": datetime.now().isoformat()
    })


# API documentation endpoint
@router.get("/docs-info")
async def api_documentation_info():
    """Get API documentation information"""
    
    endpoints_info = {
        "chat": {
            "method": "POST",
            "path": "/api/chat",
            "description": "Process chat messages with AI",
            "requires": ["message"],
            "optional": ["session_id", "latitude", "longitude", "location_accuracy"]
        },
        "emergency": {
            "method": "POST", 
            "path": "/api/emergency",
            "description": "Report emergency situations",
            "requires": ["message"],
            "optional": ["latitude", "longitude", "contact_info", "session_id"]
        },
        "weather": {
            "method": "GET",
            "path": "/api/weather/current",
            "description": "Get current weather information",
            "optional": ["latitude", "longitude"]
        },
        "location": {
            "method": "GET",
            "path": "/api/location/nearby",
            "description": "Find nearby emergency services",
            "requires": ["latitude", "longitude"],
            "optional": ["service_type"]
        },
        "services": {
            "method": "GET",
            "path": "/api/services/city",
            "description": "Get Indianapolis city services",
            "optional": ["service_type"]
        }
    }
    
    return JSONResponse(content={
        "api_version": settings.APP_VERSION,
        "base_url": "/api",
        "authentication": "None required for public endpoints",
        "rate_limiting": f"{settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW} seconds",
        "endpoints": endpoints_info,
        "emergency_note": "For life-threatening emergencies, always call 911 directly",
        "timestamp": datetime.now().isoformat()
    })


# Error handlers for API routes
@router.exception_handler(422)
async def api_validation_error(request: Request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "details": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


@router.exception_handler(429)
async def api_rate_limit_error(request: Request, exc):
    """Handle rate limiting errors"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": f"Too many requests. Limit: {settings.RATE_LIMIT_REQUESTS} per {settings.RATE_LIMIT_WINDOW} seconds",
            "timestamp": datetime.now().isoformat()
        }
    )
