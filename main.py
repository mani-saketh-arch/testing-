"""
SafeIndy Assistant - FastAPI Application Entry Point
AI-powered emergency response and civic assistance for Indianapolis
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager
import os

from app.config import settings
from app.database import init_database, get_db
from app.routes import public, admin, api, analytics


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 Starting SafeIndy Assistant...")
    
    # Initialize database
    try:
        await init_database()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise
    
    # Create data directories
    os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)
    os.makedirs(settings.IMAGES_DIRECTORY, exist_ok=True)
    os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
    
    logger.info(f"📍 Indianapolis coordinates: ({settings.INDY_LATITUDE}, {settings.INDY_LONGITUDE})")
    logger.info(f"🆘 Emergency number: {settings.INDY_EMERGENCY_NUMBER}")
    logger.info("💬 SafeIndy Assistant ready to serve Indianapolis residents!")
    
    yield
    
    # Shutdown
    logger.info("👋 SafeIndy Assistant shutting down...")


# Create FastAPI application
app = FastAPI(
    title="SafeIndy Assistant",
    description="AI-powered emergency response and civic assistance for Indianapolis",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(public.router, tags=["public"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(api.router, prefix="/api", tags=["api"])
app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": str(request.state.timestamp) if hasattr(request.state, 'timestamp') else None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "status_code": 500
        }
    )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header and request logging"""
    import time
    
    start_time = time.time()
    request.state.timestamp = start_time
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path} - {request.client.host}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log response
    logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "SafeIndy Assistant",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": str(time.time())
    }


@app.get("/")
async def root():
    """Root endpoint redirect"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/chat")


if __name__ == "__main__":
    import time
    
    print(f"\n🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"🌍 Environment: {settings.ENVIRONMENT}")
    print(f"🔧 Debug Mode: {settings.DEBUG}")
    print(f"📍 Indianapolis coordinates: ({settings.INDY_LATITUDE}, {settings.INDY_LONGITUDE})")
    print(f"🆘 Emergency number: {settings.INDY_EMERGENCY_NUMBER}")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )