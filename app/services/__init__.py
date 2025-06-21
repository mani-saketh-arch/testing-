"""
SafeIndy Assistant - Services Package
Business logic and external service integrations
"""

# Import all services for easy access
from .llm import llm_service
from .rag import rag_service
from .external import (
    location_service,
    weather_service,
    search_service,
    email_service,
    indy_data_service,
    check_external_services_health
)
from .documents import document_processor
from .analytics import analytics_service
from .telegram import telegram_service

__all__ = [
    # Core AI services
    "llm_service",
    "rag_service",
    
    # External services
    "location_service",
    "weather_service", 
    "search_service",
    "email_service",
    "indy_data_service",
    "check_external_services_health",
    
    # Document processing
    "document_processor",
    
    # Analytics
    "analytics_service",
    
    # Telegram bot
    "telegram_service"
]