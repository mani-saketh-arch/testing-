"""
SafeIndy Assistant - Configuration Management
Centralized settings and environment variable handling
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Optional, Any
import os
from pathlib import Path
import json

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Application Settings
    APP_NAME: str = "SafeIndy Assistant"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True
    SECRET_KEY: str = Field(..., description="Secret key for sessions")
    ENVIRONMENT: str = "development"
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./data/safeindy.db"
    
    # JWT Authentication
    JWT_SECRET_KEY: str = Field(..., description="JWT secret key")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AWS Configuration (Claude Sonnet 4)
    AWS_ACCESS_KEY_ID: str = Field(..., description="AWS access key")
    AWS_SECRET_ACCESS_KEY: str = Field(..., description="AWS secret key")
    AWS_REGION: str = "us-east-1"
    AWS_BEDROCK_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    # Backup LLM Service
    GROQ_API_KEY: str = Field(..., description="Groq API key for backup LLM")
    
    # Vector Database
    QDRANT_URL: str = Field(..., description="Qdrant cluster URL")
    QDRANT_API_KEY: str = Field(..., description="Qdrant API key")
    QDRANT_COLLECTION_WEB: str = "web_knowledge"
    QDRANT_COLLECTION_DOCS: str = "admin_documents"
    
    # External API Services
    GOOGLE_MAPS_API_KEY: str = Field(..., description="Google Maps API key")
    OPENWEATHER_API_KEY: str = Field(..., description="OpenWeather API key")
    PERPLEXITY_API_KEY: str = Field(..., description="Perplexity API key")
    COHERE_API_KEY: str = Field(..., description="Cohere API key for embeddings")
    
    # Email Configuration
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = Field(..., description="SMTP username")
    SMTP_PASSWORD: str = Field(..., description="SMTP password")
    EMERGENCY_EMAIL: str = "emergency@indianapolis.gov"
    
    # Telegram Bot (Optional)
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_WEBHOOK_URL: Optional[str] = None
    
    # Indianapolis Configuration
    INDY_LATITUDE: float = 39.7684
    INDY_LONGITUDE: float = -86.1581
    INDY_EMERGENCY_NUMBER: str = "911"
    INDY_NON_EMERGENCY: str = "317-327-3811"
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 10485760  # 10MB
    ALLOWED_FILE_TYPES: Any = "pdf,doc,docx"  # Can be string or list
    UPLOAD_DIRECTORY: str = "./data/uploads"
    IMAGES_DIRECTORY: str = "./data/images"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 30
    RATE_LIMIT_WINDOW: int = 300  # 5 minutes
    
    # CORS Settings
    CORS_ORIGINS: Any = "http://localhost:3000,http://localhost:8000"  # Can be string or list
    
    # Admin Configuration
    ADMIN_EMAIL: str = "admin@indianapolis.gov"
    ADMIN_PASSWORD: str = Field(..., description="Default admin password")
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./data/logs/safeindy.log"
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list to consistent format"""
        if isinstance(v, str):
            # Parse comma-separated string
            return [origin.strip() for origin in v.split(",")]
        elif isinstance(v, list):
            # Already a list, just clean it up
            return [str(origin).strip() for origin in v]
        else:
            # Fallback to default
            return ["http://localhost:3000", "http://localhost:8000"]
    
    @field_validator("ALLOWED_FILE_TYPES", mode="before")
    @classmethod
    def parse_file_types(cls, v):
        """Parse file types from string or list to consistent format"""
        if isinstance(v, str):
            # Parse comma-separated string
            return [ft.strip().lower() for ft in v.split(",")]
        elif isinstance(v, list):
            # Already a list, just clean it up
            return [str(ft).strip().lower() for ft in v]
        else:
            # Fallback to default
            return ["pdf", "doc", "docx"]
    
    @property
    def database_path(self) -> Path:
        """Get database file path"""
        if self.DATABASE_URL.startswith("sqlite:///"):
            return Path(self.DATABASE_URL.replace("sqlite:///", ""))
        return Path("./data/safeindy.db")
    
    @property
    def emergency_contacts(self) -> dict:
        """Get emergency contact numbers"""
        return {
            "emergency": self.INDY_EMERGENCY_NUMBER,
            "non_emergency": self.INDY_NON_EMERGENCY,
            "email": self.EMERGENCY_EMAIL
        }
    
    @property
    def indianapolis_coordinates(self) -> tuple:
        """Get Indianapolis coordinates as tuple"""
        return (self.INDY_LATITUDE, self.INDY_LONGITUDE)
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list"""
        return self.CORS_ORIGINS
    
    @property
    def allowed_file_types_list(self) -> List[str]:
        """Get allowed file types as a list"""
        return self.ALLOWED_FILE_TYPES
    
    def validate_configuration(self) -> bool:
        """Validate that all required configurations are set"""
        required_fields = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY", 
            "GROQ_API_KEY",
            "QDRANT_URL",
            "QDRANT_API_KEY",
            "GOOGLE_MAPS_API_KEY",
            "OPENWEATHER_API_KEY",
            "PERPLEXITY_API_KEY",
            "COHERE_API_KEY",
            "SECRET_KEY",
            "JWT_SECRET_KEY"
        ]
        
        missing_fields = []
        for field in required_fields:
            value = getattr(self, field, None)
            if not value or value == f"your-{field.lower().replace('_', '-')}":
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")
        
        return True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Emergency response configuration
EMERGENCY_KEYWORDS = [
    "emergency", "help", "urgent", "fire", "police", "ambulance",
    "accident", "injured", "robbery", "assault", "shooting",
    "heart attack", "overdose", "unconscious", "bleeding",
    "gas leak", "break in", "domestic violence"
]

EMERGENCY_CONFIDENCE_THRESHOLD = 0.8

# Indianapolis specific data
INDIANAPOLIS_NEIGHBORHOODS = [
    "Downtown", "Broad Ripple", "Fountain Square", "Mass Ave",
    "Irvington", "Garfield Park", "Butler-Tarkington", "Old Northside",
    "Lockerbie Square", "Fall Creek Place", "Holy Cross", "Near Eastside"
]

CITY_SERVICES = {
    "311": "Non-emergency city services",
    "Mayor's Action Center": "City service requests",
    "IMPD": "Indianapolis Metropolitan Police",
    "IFD": "Indianapolis Fire Department",
    "DPW": "Department of Public Works",
    "Health Department": "Marion County Health Department"
}

# Model configuration
LLM_CONFIG = {
    "aws_claude": {
        "model_id": settings.AWS_BEDROCK_MODEL_ID,
        "max_tokens": 2048,
        "temperature": 0.3,
        "timeout": 30
    },
    "groq_backup": {
        "model": "llama-3.1-8b-instant",
        "max_tokens": 1024,
        "temperature": 0.2,
        "timeout": 10
    }
}

# RAG configuration
RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "similarity_threshold": 0.7,
    "max_results": 5,
    "hybrid_search_ratio": 0.7  # 70% semantic, 30% keyword
}