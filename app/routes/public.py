"""
SafeIndy Assistant - Public Routes
Chat interface, emergency handling, and main pages
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db, log_user_interaction
from ..utils import timer, log_error, sanitize_input, format_timestamp
from ..services import (
    llm_service,
    rag_service, 
    location_service,
    weather_service,
    indy_data_service,
    email_service,
    analytics_service
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Templates
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Landing page with SafeIndy overview"""
    
    try:
        # Get basic city info
        emergency_contacts = settings.emergency_contacts
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
            "emergency_contacts": emergency_contacts,
            "indianapolis_coordinates": settings.indianapolis_coordinates
        })
    
    except Exception as e:
        log_error(e, "Landing page rendering")
        raise HTTPException(status_code=500, detail="Error loading page")


@router.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Main chat interface page"""
    
    try:
        # Generate session ID for this chat session
        session_id = str(uuid4())
        
        # Get current Indianapolis weather for context
        weather_data = await weather_service.get_current_weather()
        
        # Get emergency contacts
        emergency_contacts = settings.emergency_contacts
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "session_id": session_id,
            "app_name": settings.APP_NAME,
            "emergency_contacts": emergency_contacts,
            "weather_data": weather_data,
            "chat_mode": True,
            "google_maps_api_key": settings.GOOGLE_MAPS_API_KEY,
            "indianapolis_coordinates": settings.indianapolis_coordinates
        })
    
    except Exception as e:
        log_error(e, "Chat interface rendering")
        raise HTTPException(status_code=500, detail="Error loading chat interface")


@router.get("/about", response_class=HTMLResponse) 
async def about_page(request: Request):
    """About SafeIndy page"""
    
    try:
        return templates.TemplateResponse("about.html", {
            "request": request,
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
            "emergency_contacts": settings.emergency_contacts
        })
    
    except Exception as e:
        log_error(e, "About page rendering")
        raise HTTPException(status_code=500, detail="Error loading about page")


@router.post("/chat/message")
@timer
async def process_chat_message(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    location_accuracy: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """Process chat message and return AI response"""
    
    start_time = time.time()
    
    try:
        # Sanitize and validate input
        clean_message = sanitize_input(message, max_length=2000)
        if not clean_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Prepare request context
        request_info = {
            "platform": "web",
            "user_agent": request.headers.get("user-agent", ""),
            "ip_address": request.client.host,
            "latitude": latitude,
            "longitude": longitude,
            "location_accuracy": location_accuracy
        }
        
        # Build context for AI processing
        context_data = {
            "platform": "web",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add location context if provided
        if latitude and longitude:
            context_data["user_location"] = (latitude, longitude)
            
            # Get address from coordinates
            try:
                address_info = await location_service.reverse_geocode(latitude, longitude)
                if address_info:
                    context_data["user_address"] = address_info.get("formatted_address", "")
            except Exception as e:
                logger.warning(f"Failed to get address for coordinates: {e}")
        
        # 1. Classify intent and detect emergencies
        intent_result = await llm_service.classify_intent(clean_message)
        
        # 2. Generate response with RAG
        if intent_result.get("emergency", False):
            # Emergency: use fast model and include emergency context
            context_data["emergency_detected"] = True
            context_data["emergency_type"] = intent_result.get("emergency_type", [])
            
            response_data = await rag_service.generate_response_with_context(
                user_query=clean_message,
                context=context_data,
                max_context_length=1500
            )
            
            # Enhance with emergency-specific information
            response_data.update({
                "emergency": True,
                "intent": intent_result.get("intent", "emergency"),
                "confidence": intent_result.get("confidence", 0.8),
                "emergency_keywords": intent_result.get("keywords", []),
                "emergency_type": intent_result.get("emergency_type", [])
            })
        else:
            # Non-emergency: use full RAG processing
            response_data = await rag_service.generate_response_with_context(
                user_query=clean_message,
                context=context_data
            )
            
            # Add intent classification results
            response_data.update({
                "emergency": False,
                "intent": intent_result.get("intent", "information"),
                "confidence": intent_result.get("confidence", 0.5)
            })
        
        # 3. Add response timing
        response_time_ms = int((time.time() - start_time) * 1000)
        response_data["response_time_ms"] = response_time_ms
        
        # 4. Handle emergency actions
        if response_data.get("emergency"):
            try:
                # Send emergency alert email
                emergency_details = {
                    "type": response_data.get("intent", "emergency"),
                    "message": clean_message,
                    "confidence": response_data.get("confidence", 0.0),
                    "keywords": response_data.get("emergency_keywords", []),
                    "platform": "web",
                    "session_id": session_id,
                    "user_agent": request_info.get("user_agent", ""),
                    "timestamp": datetime.now().isoformat()
                }
                
                location_info = None
                if latitude and longitude:
                    location_info = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "accuracy": location_accuracy,
                        "coordinates": f"{latitude}, {longitude}",
                        "formatted_address": context_data.get("user_address", "")
                    }
                
                # Send alert asynchronously
                asyncio.create_task(
                    email_service.send_emergency_alert(emergency_details, location_info)
                )
                
                logger.info(f"Emergency alert initiated for session {session_id}")
                
            except Exception as e:
                logger.error(f"Failed to send emergency alert: {e}")
                # Don't fail the response if email fails
        
        # 5. Track interaction for analytics
        try:
            await analytics_service.track_user_interaction(
                session_id=session_id,
                user_query=clean_message,
                response_data=response_data,
                request_info=request_info
            )
        except Exception as e:
            logger.warning(f"Failed to track interaction: {e}")
            # Don't fail the response if analytics fails
        
        # 6. Prepare response
        chat_response = {
            "success": response_data.get("success", True),
            "response": response_data.get("response", "I apologize, but I'm having trouble generating a response."),
            "emergency": response_data.get("emergency", False),
            "intent": response_data.get("intent", "unknown"),
            "confidence": response_data.get("confidence", 0.0),
            "sources": response_data.get("sources", []),
            "response_time_ms": response_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add emergency-specific fields
        if response_data.get("emergency"):
            chat_response.update({
                "emergency_contacts": settings.emergency_contacts,
                "emergency_keywords": response_data.get("emergency_keywords", []),
                "emergency_type": response_data.get("emergency_type", [])
            })
        
        return JSONResponse(content=chat_response)
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Chat processing error: {e}"
        log_error(e, "Chat message processing", {
            "session_id": session_id,
            "message": clean_message[:100] if 'clean_message' in locals() else message[:100]
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


@router.get("/emergency")
async def emergency_info(request: Request):
    """Emergency information and quick access"""
    
    try:
        # Get emergency contacts and services
        emergency_contacts = settings.emergency_contacts
        
        # Get nearby emergency services for Indianapolis center
        lat, lon = settings.indianapolis_coordinates
        nearby_services = await location_service.find_nearest_emergency_services(lat, lon)
        
        emergency_data = {
            "emergency_contacts": emergency_contacts,
            "nearby_services": nearby_services,
            "indianapolis_coordinates": settings.indianapolis_coordinates
        }
        
        return JSONResponse(content=emergency_data)
    
    except Exception as e:
        log_error(e, "Emergency info endpoint")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to load emergency information",
                "emergency_contacts": settings.emergency_contacts
            }
        )


@router.get("/weather")
async def current_weather(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
):
    """Get current weather information"""
    
    try:
        # Use provided coordinates or default to Indianapolis
        if not latitude or not longitude:
            latitude, longitude = settings.indianapolis_coordinates
        
        # Get current weather
        weather_data = await weather_service.get_current_weather(latitude, longitude)
        
        if not weather_data:
            return JSONResponse(
                status_code=503,
                content={"error": "Weather service unavailable"}
            )
        
        # Get weather alerts
        alerts = await weather_service.get_weather_alerts(latitude, longitude)
        
        response_data = {
            "weather": weather_data,
            "alerts": alerts,
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        log_error(e, "Weather endpoint", {"lat": latitude, "lon": longitude})
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get weather information"}
        )


@router.get("/services")
async def city_services(service_type: Optional[str] = None):
    """Get Indianapolis city services information"""
    
    try:
        services_data = await indy_data_service.get_city_services_info(service_type)
        
        return JSONResponse(content={
            "services": services_data,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        log_error(e, "City services endpoint", {"service_type": service_type})
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get city services information"}
        )


@router.get("/location/nearby")
async def nearby_services(
    latitude: float,
    longitude: float,
    service_type: Optional[str] = None
):
    """Get nearby emergency services"""
    
    try:
        # Validate coordinates
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinates")
        
        # Get nearby services
        if service_type:
            # Specific service type (future enhancement)
            nearby_services = await location_service.find_nearest_emergency_services(latitude, longitude)
            services = nearby_services.get(service_type, [])
        else:
            # All emergency services
            services = await location_service.find_nearest_emergency_services(latitude, longitude)
        
        # Get address for the location
        address_info = await location_service.reverse_geocode(latitude, longitude)
        
        return JSONResponse(content={
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "address": address_info.get("formatted_address", "") if address_info else ""
            },
            "services": services,
            "timestamp": datetime.now().isoformat()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, "Nearby services endpoint", {"lat": latitude, "lon": longitude})
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to find nearby services"}
        )


@router.get("/health")
async def public_health_check():
    """Public health check endpoint"""
    
    try:
        health_status = {
            "status": "healthy",
            "service": "SafeIndy Assistant",
            "version": settings.APP_VERSION,
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Quick health checks for critical services
        try:
            # Test LLM service
            llm_health = await llm_service.health_check()
            health_status["components"]["ai"] = "healthy" if llm_health.get("overall") == "healthy" else "degraded"
        except:
            health_status["components"]["ai"] = "error"
        
        try:
            # Test weather service
            weather_data = await weather_service.get_current_weather()
            health_status["components"]["weather"] = "healthy" if weather_data else "error"
        except:
            health_status["components"]["weather"] = "error"
        
        try:
            # Test location service
            test_location = await location_service.reverse_geocode(*settings.indianapolis_coordinates)
            health_status["components"]["location"] = "healthy" if test_location else "error"
        except:
            health_status["components"]["location"] = "error"
        
        # Overall status
        if all(status == "healthy" for status in health_status["components"].values()):
            health_status["status"] = "healthy"
        elif any(status == "healthy" for status in health_status["components"].values()):
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "error"
        
        return JSONResponse(content=health_status)
    
    except Exception as e:
        log_error(e, "Public health check")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "service": "SafeIndy Assistant",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/status")
async def system_status():
    """System status for monitoring"""
    
    try:
        # Get basic metrics
        status_info = {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "timestamp": datetime.now().isoformat(),
            "uptime": "System operational",  # Could track actual uptime
            "location": "Indianapolis, IN",
            "coordinates": settings.indianapolis_coordinates,
            "emergency_contacts": settings.emergency_contacts
        }
        
        return JSONResponse(content=status_info)
    
    except Exception as e:
        log_error(e, "System status endpoint")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get system status"}
        )


# Error handlers for this router

@router.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors for public routes"""
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "error": "Page not found",
        "app_name": settings.APP_NAME
    }, status_code=404)


@router.exception_handler(500)
async def server_error_handler(request: Request, exc: HTTPException):
    """Handle 500 errors for public routes"""
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "message": "Please try again or contact emergency services if urgent",
            "emergency_contacts": settings.emergency_contacts,
            "timestamp": datetime.now().isoformat()
        }
    )
