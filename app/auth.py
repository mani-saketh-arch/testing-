"""
SafeIndy Assistant - JWT Authentication
Admin authentication and authorization system
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import logging

from .config import settings
from .database import get_db, Admin

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.JWT_SECRET_KEY, 
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None


def authenticate_admin(db: Session, email: str, password: str) -> Optional[Admin]:
    """Authenticate admin user"""
    admin = db.query(Admin).filter(
        Admin.email == email,
        Admin.is_active == True
    ).first()
    
    if not admin:
        logger.warning(f"Admin authentication failed: user not found - {email}")
        return None
    
    if not verify_password(password, admin.password_hash):
        logger.warning(f"Admin authentication failed: invalid password - {email}")
        return None
    
    # Update last login
    admin.last_login = datetime.utcnow()
    db.commit()
    
    logger.info(f"Admin authenticated successfully: {email}")
    return admin


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> Admin:
    """Get current authenticated admin user"""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = verify_token(credentials.credentials)
        if payload is None:
            raise credentials_exception
        
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    admin = db.query(Admin).filter(
        Admin.email == email,
        Admin.is_active == True
    ).first()
    
    if admin is None:
        raise credentials_exception
    
    return admin


async def get_optional_user(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[Admin]:
    """Get current user if authenticated, None otherwise"""
    
    try:
        # Try to get token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        payload = verify_token(token)
        
        if payload is None:
            return None
        
        email: str = payload.get("sub")
        if email is None:
            return None
        
        admin = db.query(Admin).filter(
            Admin.email == email,
            Admin.is_active == True
        ).first()
        
        return admin
        
    except Exception as e:
        logger.debug(f"Optional auth failed: {e}")
        return None


def require_role(required_role: str):
    """Decorator to require specific admin role"""
    def role_checker(current_user: Admin = Depends(get_current_user)):
        if current_user.role != required_role and current_user.role != "super_admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires {required_role} role"
            )
        return current_user
    return role_checker


def create_admin_token(admin: Admin) -> Dict[str, Any]:
    """Create token response for admin"""
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": admin.email, "role": admin.role},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "admin": {
            "id": admin.id,
            "email": admin.email,
            "name": admin.name,
            "role": admin.role,
            "last_login": admin.last_login.isoformat() if admin.last_login else None
        }
    }


# Rate limiting for authentication
class AuthRateLimiter:
    """Simple rate limiter for authentication attempts"""
    
    def __init__(self):
        self.attempts = {}  # IP -> (attempts, last_attempt_time)
        self.max_attempts = 5
        self.window_minutes = 15
    
    def is_allowed(self, ip_address: str) -> bool:
        """Check if authentication attempt is allowed"""
        now = datetime.utcnow()
        
        if ip_address in self.attempts:
            attempts, last_attempt = self.attempts[ip_address]
            
            # Reset if window has passed
            if (now - last_attempt).total_seconds() > (self.window_minutes * 60):
                self.attempts[ip_address] = (1, now)
                return True
            
            # Check if max attempts reached
            if attempts >= self.max_attempts:
                return False
            
            # Increment attempts
            self.attempts[ip_address] = (attempts + 1, now)
            return True
        else:
            # First attempt
            self.attempts[ip_address] = (1, now)
            return True
    
    def reset_attempts(self, ip_address: str):
        """Reset attempts for successful authentication"""
        if ip_address in self.attempts:
            del self.attempts[ip_address]


# Global rate limiter instance
auth_rate_limiter = AuthRateLimiter()


async def check_auth_rate_limit(request: Request):
    """Check authentication rate limit middleware"""
    client_ip = request.client.host
    
    if not auth_rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication attempts. Please try again later."
        )
    
    return client_ip


# Password validation
def validate_password(password: str) -> Dict[str, Any]:
    """Validate password strength"""
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one number")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# Session management for web interface
def create_session_token(admin_id: int, session_data: Dict[str, Any] = None) -> str:
    """Create session token for web interface"""
    data = {
        "admin_id": admin_id,
        "session_type": "web",
        "created_at": datetime.utcnow().isoformat()
    }
    
    if session_data:
        data.update(session_data)
    
    return create_access_token(data)


def verify_session_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify session token"""
    payload = verify_token(token)
    
    if payload and payload.get("session_type") == "web":
        return payload
    
    return None