"""
SafeIndy Assistant - Application Package
AI-powered emergency response and civic assistance for Indianapolis
"""

__version__ = "2.0.0"
__author__ = "SafeIndy Development Team"
__description__ = "AI-powered emergency response and civic assistance for Indianapolis"

# Import core components for easy access
from .config import settings
from .database import get_db, init_database
from .auth import create_access_token, verify_token, get_current_user

__all__ = [
    "settings",
    "get_db", 
    "init_database",
    "create_access_token",
    "verify_token", 
    "get_current_user"
]