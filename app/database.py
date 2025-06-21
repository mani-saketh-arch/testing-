"""
SafeIndy Assistant - Database Setup and Models
SQLite database configuration with SQLAlchemy models
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from contextlib import contextmanager
import os
from datetime import datetime
from typing import Generator

from .config import settings

# Create database directory if it doesn't exist
os.makedirs(os.path.dirname(settings.database_path), exist_ok=True)

# Database setup
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=settings.DEBUG  # Log SQL queries in debug mode
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database Models
class Admin(Base):
    """Admin user model for authentication and management"""
    __tablename__ = "admins"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(String(50), default="admin")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    last_login = Column(DateTime)
    
    # Relationships
    documents = relationship("Document", back_populates="uploaded_by_user")


class Document(Base):
    """Document metadata for uploaded PDFs"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String(100))
    uploaded_by = Column(Integer, ForeignKey("admins.id"))
    uploaded_at = Column(DateTime, default=func.now())
    processed_at = Column(DateTime)
    status = Column(String(50), default="pending")  # pending, processing, completed, error
    page_count = Column(Integer)
    extraction_summary = Column(Text)
    vector_collection_id = Column(String(255))  # Qdrant collection reference
    
    # Relationships
    uploaded_by_user = relationship("Admin", back_populates="documents")
    images = relationship("DocumentImage", back_populates="document")


class DocumentImage(Base):
    """Images extracted from documents"""
    __tablename__ = "document_images"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    image_filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    page_number = Column(Integer)
    image_type = Column(String(50))  # photo, chart, diagram, text_image
    ocr_text = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="images")


class UserInteraction(Base):
    """User interaction tracking for analytics"""
    __tablename__ = "user_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    user_query = Column(Text, nullable=False)
    query_type = Column(String(100))  # emergency, information, city_service, etc.
    intent_confidence = Column(Float)
    latitude = Column(Float)
    longitude = Column(Float)
    location_accuracy = Column(Float)
    response_time_ms = Column(Integer)
    emergency_detected = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=func.now(), index=True)
    platform = Column(String(50))  # web, telegram
    user_agent = Column(Text)
    response_length = Column(Integer)
    sources_used = Column(Text)  # JSON string of sources
    satisfaction_rating = Column(Integer)  # 1-5 rating if provided


class AppSetting(Base):
    """Application settings and configuration"""
    __tablename__ = "app_settings"
    
    key = Column(String(255), primary_key=True)
    value = Column(Text)
    description = Column(Text)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    updated_by = Column(Integer, ForeignKey("admins.id"))


class SystemLog(Base):
    """System events and error logging"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    message = Column(Text, nullable=False)
    module = Column(String(100))
    function = Column(String(100))
    user_id = Column(Integer, ForeignKey("admins.id"))
    session_id = Column(String(255))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    timestamp = Column(DateTime, default=func.now(), index=True)
    additional_data = Column(Text)  # JSON string for extra context


# Database dependency
def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Database session context manager"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_database():
    """Initialize database and create tables"""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create default admin user if not exists
    with get_db_context() as db:
        admin = db.query(Admin).filter(Admin.email == settings.ADMIN_EMAIL).first()
        if not admin:
            from .auth import get_password_hash
            
            default_admin = Admin(
                email=settings.ADMIN_EMAIL,
                password_hash=get_password_hash(settings.ADMIN_PASSWORD),
                name="System Administrator",
                role="super_admin",
                is_active=True
            )
            db.add(default_admin)
            db.commit()
            
            print(f"✅ Created default admin user: {settings.ADMIN_EMAIL}")
        
        # Create default app settings
        default_settings = [
            ("emergency_threshold", "0.8", "Confidence threshold for emergency detection"),
            ("max_response_length", "2000", "Maximum response length in characters"),
            ("enable_location_tracking", "true", "Enable location tracking for analytics"),
            ("maintenance_mode", "false", "Enable maintenance mode"),
        ]
        
        for key, value, description in default_settings:
            setting = db.query(AppSetting).filter(AppSetting.key == key).first()
            if not setting:
                new_setting = AppSetting(
                    key=key,
                    value=value,
                    description=description
                )
                db.add(new_setting)
        
        db.commit()


# Database utility functions
def log_user_interaction(
    db: Session,
    session_id: str,
    user_query: str,
    query_type: str = None,
    intent_confidence: float = None,
    latitude: float = None,
    longitude: float = None,
    location_accuracy: float = None,
    response_time_ms: int = None,
    emergency_detected: bool = False,
    platform: str = "web",
    user_agent: str = None,
    response_length: int = None,
    sources_used: str = None
):
    """Log user interaction for analytics"""
    interaction = UserInteraction(
        session_id=session_id,
        user_query=user_query,
        query_type=query_type,
        intent_confidence=intent_confidence,
        latitude=latitude,
        longitude=longitude,
        location_accuracy=location_accuracy,
        response_time_ms=response_time_ms,
        emergency_detected=emergency_detected,
        platform=platform,
        user_agent=user_agent,
        response_length=response_length,
        sources_used=sources_used
    )
    
    db.add(interaction)
    db.commit()
    return interaction


def log_system_event(
    db: Session,
    level: str,
    message: str,
    module: str = None,
    function: str = None,
    user_id: int = None,
    session_id: str = None,
    ip_address: str = None,
    user_agent: str = None,
    additional_data: str = None
):
    """Log system event"""
    log_entry = SystemLog(
        level=level,
        message=message,
        module=module,
        function=function,
        user_id=user_id,
        session_id=session_id,
        ip_address=ip_address,
        user_agent=user_agent,
        additional_data=additional_data
    )
    
    db.add(log_entry)
    db.commit()
    return log_entry


def get_app_setting(db: Session, key: str, default: str = None):
    """Get application setting value"""
    setting = db.query(AppSetting).filter(AppSetting.key == key).first()
    return setting.value if setting else default


def set_app_setting(db: Session, key: str, value: str, description: str = None, user_id: int = None):
    """Set application setting value"""
    setting = db.query(AppSetting).filter(AppSetting.key == key).first()
    
    if setting:
        setting.value = value
        setting.updated_by = user_id
        setting.updated_at = func.now()
    else:
        setting = AppSetting(
            key=key,
            value=value,
            description=description,
            updated_by=user_id
        )
        db.add(setting)
    
    db.commit()
    return setting