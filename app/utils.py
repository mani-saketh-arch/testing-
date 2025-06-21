"""
SafeIndy Assistant - Shared Utility Functions
Common utility functions used across the application
"""

import os
import uuid
import json
import hashlib
import mimetypes
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import re
import asyncio
from functools import wraps
import time

from pandas import Timedelta

from .config import settings

logger = logging.getLogger(__name__)


# File handling utilities
def generate_unique_filename(original_filename: str) -> str:
    """Generate unique filename while preserving extension"""
    file_extension = Path(original_filename).suffix.lower()
    unique_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{timestamp}_{unique_id}{file_extension}"


def validate_file_type(filename: str) -> bool:
    """Validate if file type is allowed"""
    file_extension = Path(filename).suffix.lower().lstrip('.')
    return file_extension in settings.ALLOWED_FILE_TYPES


def validate_file_size(file_size: int) -> bool:
    """Validate if file size is within limits"""
    return file_size <= settings.MAX_FILE_SIZE


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get comprehensive file information"""
    if not os.path.exists(file_path):
        return {}
    
    stat = os.stat(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    
    return {
        "filename": os.path.basename(file_path),
        "size": stat.st_size,
        "mime_type": mime_type,
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "extension": Path(file_path).suffix.lower()
    }


def ensure_directory_exists(directory_path: str) -> bool:
    """Ensure directory exists, create if not"""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        return False


def safe_filename(filename: str) -> str:
    """Create safe filename by removing/replacing dangerous characters"""
    # Remove or replace dangerous characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing whitespace and dots
    safe_name = safe_name.strip('. ')
    # Ensure it's not empty
    if not safe_name:
        safe_name = "unnamed_file"
    
    return safe_name


# String and text utilities
def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\-.,!?():]', '', text)
    
    return text


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text (simple implementation)"""
    # Simple keyword extraction - remove common words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'under', 'over',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'must', 'shall', 'a', 'an', 'this', 'that', 'these', 'those'
    }
    
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count frequency and return most common
    word_count = {}
    for word in keywords:
        word_count[word] = word_count.get(word, 0) + 1
    
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:max_keywords]]


# Location utilities
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in miles"""
    import math
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in miles
    earth_radius = 3959
    return earth_radius * c


def is_in_indianapolis(latitude: float, longitude: float, radius_miles: float = 15) -> bool:
    """Check if coordinates are within Indianapolis area"""
    indy_lat, indy_lon = settings.indianapolis_coordinates
    distance = calculate_distance(latitude, longitude, indy_lat, indy_lon)
    return distance <= radius_miles


def format_coordinates(latitude: float, longitude: float) -> str:
    """Format coordinates for display"""
    lat_dir = "N" if latitude >= 0 else "S"
    lon_dir = "E" if longitude >= 0 else "W"
    
    return f"{abs(latitude):.6f}°{lat_dir}, {abs(longitude):.6f}°{lon_dir}"


# Data validation utilities
def validate_email(email: str) -> bool:
    """Validate email format"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None


def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Check if it's a valid US phone number (10 or 11 digits)
    return len(digits) in [10, 11] and (len(digits) == 10 or digits[0] == '1')


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input"""
    if not text:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', text)
    
    # Truncate to max length
    sanitized = sanitized[:max_length]
    
    # Clean up whitespace
    sanitized = ' '.join(sanitized.split())
    
    return sanitized


# JSON utilities
def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """Safely load JSON string"""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely dump object to JSON string"""
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return default


# Timing utilities
def timer(func):
    """Decorator to time function execution"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.3f} seconds")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.3f} seconds")
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def format_timestamp(timestamp: datetime, format_type: str = "full") -> str:
    """Format timestamp for display"""
    if not timestamp:
        return "N/A"
    
    # Ensure timezone awareness
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    # Convert to local timezone (Indianapolis is UTC-5/-4)
    local_timestamp = timestamp
    
    if format_type == "full":
        return local_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    elif format_type == "date":
        return local_timestamp.strftime("%Y-%m-%d")
    elif format_type == "time":
        return local_timestamp.strftime("%H:%M:%S")
    elif format_type == "relative":
        return format_relative_time(local_timestamp)
    else:
        return local_timestamp.isoformat()


def format_relative_time(timestamp: datetime) -> str:
    """Format timestamp as relative time (e.g., '2 hours ago')"""
    now = datetime.now(timezone.utc)
    
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    diff = now - timestamp
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds // 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        return timestamp.strftime("%Y-%m-%d")


# Emergency detection utilities
def detect_emergency_keywords(text: str) -> Dict[str, Any]:
    """Detect emergency keywords in text"""
    emergency_keywords = {
        "fire": ["fire", "smoke", "burning", "flames", "burn"],
        "medical": ["heart attack", "overdose", "unconscious", "bleeding", "choking", 
                   "can't breathe", "chest pain", "stroke", "seizure"],
        "violence": ["shooting", "stabbing", "assault", "robbery", "break in", 
                    "domestic violence", "threat", "weapon"],
        "accident": ["accident", "crash", "collision", "injured", "emergency"],
        "utility": ["gas leak", "power line", "water main", "explosion"],
        "general": ["help", "emergency", "urgent", "911", "call police", "ambulance"]
    }
    
    text_lower = text.lower()
    detected_categories = []
    matched_keywords = []
    
    for category, keywords in emergency_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected_categories.append(category)
                matched_keywords.append(keyword)
    
    # Calculate confidence based on matches
    confidence = min(len(set(detected_categories)) * 0.3 + len(matched_keywords) * 0.1, 1.0)
    
    return {
        "is_emergency": len(detected_categories) > 0,
        "confidence": confidence,
        "categories": list(set(detected_categories)),
        "keywords": matched_keywords
    }


# Data formatting utilities
def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def format_phone_number(phone: str) -> str:
    """Format phone number for display"""
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits[0] == '1':
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    else:
        return phone  # Return original if can't format


def format_address(address_components: Dict[str, str]) -> str:
    """Format address components into readable string"""
    parts = []
    
    # Street address
    if address_components.get("street_number") and address_components.get("route"):
        parts.append(f"{address_components['street_number']} {address_components['route']}")
    
    # City
    if address_components.get("locality"):
        parts.append(address_components["locality"])
    
    # State
    if address_components.get("administrative_area_level_1"):
        parts.append(address_components["administrative_area_level_1"])
    
    # ZIP code
    if address_components.get("postal_code"):
        parts.append(address_components["postal_code"])
    
    return ", ".join(parts)


# Caching utilities
class SimpleCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            value, expiry = self.cache[key]
            if datetime.now() < expiry:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        expiry = datetime.now() + Timedelta(seconds=ttl)
        self.cache[key] = (value, expiry)
    
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        # Clean expired entries first
        now = datetime.now()
        expired_keys = [k for k, (v, exp) in self.cache.items() if now >= exp]
        for key in expired_keys:
            del self.cache[key]
        
        return len(self.cache)


# Global cache instance
cache = SimpleCache(default_ttl=300)  # 5 minutes default


# Error handling utilities
def create_error_response(
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = 500
) -> Dict[str, Any]:
    """Create standardized error response"""
    response = {
        "error": error_type,
        "message": message,
        "status_code": status_code,
        "timestamp": datetime.now().isoformat()
    }
    
    if details:
        response["details"] = details
    
    return response


def log_error(
    error: Exception,
    context: str = "",
    additional_data: Optional[Dict[str, Any]] = None
) -> None:
    """Log error with context"""
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "timestamp": datetime.now().isoformat()
    }
    
    if additional_data:
        error_data.update(additional_data)
    
    logger.error(f"Error in {context}: {error}", extra=error_data)


# Health check utilities
def check_service_health() -> Dict[str, Any]:
    """Check health of various services"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "unknown",
            "file_system": "unknown",
            "cache": "unknown"
        }
    }
    
    # Check database
    try:
        from .database import get_db_context
        with get_db_context() as db:
            db.execute("SELECT 1")
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check file system
    try:
        test_file = os.path.join(settings.UPLOAD_DIRECTORY, ".health_check")
        with open(test_file, "w") as f:
            f.write("health_check")
        os.remove(test_file)
        health_status["services"]["file_system"] = "healthy"
    except Exception as e:
        health_status["services"]["file_system"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check cache
    try:
        cache.set("health_check", "ok", 10)
        result = cache.get("health_check")
        if result == "ok":
            health_status["services"]["cache"] = "healthy"
        else:
            health_status["services"]["cache"] = "error: cache not working"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["cache"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status