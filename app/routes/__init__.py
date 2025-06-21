"""
SafeIndy Assistant - Routes Package
FastAPI route handlers and API endpoints
"""

# Import all route modules for easy registration
from . import public, admin, api, analytics

# Import routers for main app registration
from .public import router as public_router
from .admin import router as admin_router
from .api import router as api_router
from .analytics import router as analytics_router

__all__ = [
    "public",
    "admin", 
    "api",
    "analytics",
    "public_router",
    "admin_router",
    "api_router", 
    "analytics_router"
]
