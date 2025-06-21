"""
SafeIndy Assistant - Admin Routes
Admin panel, authentication, and document management
"""
# Import io for StreamingResponse
import io
import asyncio
import os
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from fastapi import (
    APIRouter, 
    Request, 
    Form, 
    File, 
    UploadFile,
    HTTPException, 
    Depends,
    status
)
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db, Admin, Document, log_system_event
from ..auth import (
    authenticate_admin,
    create_admin_token,
    get_current_user,
    check_auth_rate_limit,
    auth_rate_limiter,
    validate_password
)
from ..utils import timer, log_error, format_timestamp, format_file_size
from ..services import (
    document_processor,
    analytics_service,
    rag_service
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Templates
templates = Jinja2Templates(directory="templates")

# Security
security = HTTPBearer()


@router.get("/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Admin login page"""
    
    try:
        return templates.TemplateResponse("admin_login.html", {
            "request": request,
            "app_name": settings.APP_NAME
        })
    
    except Exception as e:
        log_error(e, "Admin login page rendering")
        raise HTTPException(status_code=500, detail="Error loading login page")


@router.post("/login")
@timer
async def admin_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Admin authentication endpoint"""
    
    client_ip = await check_auth_rate_limit(request)
    
    try:
        # Authenticate admin
        admin = authenticate_admin(db, email, password)
        
        if not admin:
            # Log failed attempt
            log_system_event(
                db=db,
                level="WARNING",
                message=f"Failed admin login attempt: {email}",
                module="admin_auth",
                function="admin_login",
                ip_address=client_ip,
                user_agent=request.headers.get("user-agent")
            )
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Reset rate limiting on successful login
        auth_rate_limiter.reset_attempts(client_ip)
        
        # Create token
        token_data = create_admin_token(admin)
        
        # Log successful login
        log_system_event(
            db=db,
            level="INFO",
            message=f"Admin login successful: {admin.email}",
            module="admin_auth",
            function="admin_login",
            user_id=admin.id,
            ip_address=client_ip,
            user_agent=request.headers.get("user-agent")
        )
        
        logger.info(f"Admin {admin.email} logged in successfully")
        
        return JSONResponse(content={
            "success": True,
            "message": "Login successful",
            "token_data": token_data,
            "redirect_url": "/admin/dashboard"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, "Admin login", {"email": email, "ip": client_ip})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login system error"
        )


@router.get("/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    current_user: Admin = Depends(get_current_user)
):
    """Admin dashboard page"""
    
    try:
        # Get dashboard overview data
        dashboard_data = await analytics_service.get_dashboard_overview("24h")
        
        # Get recent document uploads
        recent_documents = await document_processor.list_documents(
            uploaded_by=current_user.id,
            limit=10
        )
        
        # Get system health
        rag_health = await rag_service.health_check()
        
        return templates.TemplateResponse("admin_dashboard.html", {
            "request": request,
            "current_user": current_user,
            "dashboard_data": dashboard_data,
            "recent_documents": recent_documents.get("documents", []),
            "rag_health": rag_health,
            "app_name": settings.APP_NAME
        })
    
    except Exception as e:
        log_error(e, "Admin dashboard rendering", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Error loading dashboard")


@router.get("/documents", response_class=HTMLResponse) 
async def admin_documents_page(
    request: Request,
    current_user: Admin = Depends(get_current_user)
):
    """Document management page"""
    
    try:
        return templates.TemplateResponse("admin_dashboard.html", {
            "request": request,
            "current_user": current_user,
            "page": "documents",
            "app_name": settings.APP_NAME
        })
    
    except Exception as e:
        log_error(e, "Admin documents page rendering", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Error loading documents page")


@router.post("/documents/upload")
@timer
async def upload_document(
    file: UploadFile = File(...),
    current_user: Admin = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Validate file size
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / 1024 / 1024:.1f} MB"
            )
        
        # Process the uploaded file
        result = await document_processor.process_uploaded_file(
            file_content=file_content,
            filename=file.filename,
            uploaded_by=current_user.id,
            file_size=file_size
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Upload failed"))
        
        # Log successful upload
        log_system_event(
            db=db,
            level="INFO",
            message=f"Document uploaded: {file.filename}",
            module="admin_documents",
            function="upload_document",
            user_id=current_user.id,
            additional_data=f'{{"document_id": {result.get("document_id")}, "file_size": {file_size}}}'
        )
        
        logger.info(f"Document uploaded by admin {current_user.email}: {file.filename}")
        
        return JSONResponse(content={
            "success": True,
            "message": "Document uploaded and processing started",
            "document": {
                "id": result.get("document_id"),
                "filename": result.get("filename"),
                "original_name": result.get("original_name"),
                "file_size": result.get("file_size"),
                "file_size_formatted": format_file_size(result.get("file_size", 0))
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, "Document upload", {
            "user_id": current_user.id,
            "filename": file.filename if file else "unknown"
        })
        raise HTTPException(status_code=500, detail="Upload processing failed")


@router.get("/documents/list")
async def list_documents(
    page: int = 1,
    limit: int = 20,
    status_filter: Optional[str] = None,
    current_user: Admin = Depends(get_current_user)
):
    """List documents with pagination"""
    
    try:
        offset = (page - 1) * limit
        
        # Get documents list
        documents_result = await document_processor.list_documents(
            uploaded_by=current_user.id if current_user.role != "super_admin" else None,
            status=status_filter,
            limit=limit,
            offset=offset
        )
        
        if not documents_result.get("success"):
            raise HTTPException(status_code=500, detail="Failed to list documents")
        
        # Format documents for display
        formatted_documents = []
        for doc in documents_result.get("documents", []):
            formatted_doc = {
                **doc,
                "file_size_formatted": format_file_size(doc.get("file_size", 0)),
                "uploaded_at_formatted": format_timestamp(
                    datetime.fromisoformat(doc["uploaded_at"]) if doc.get("uploaded_at") else None,
                    "relative"
                )
            }
            formatted_documents.append(formatted_doc)
        
        return JSONResponse(content={
            "success": True,
            "documents": formatted_documents,
            "pagination": {
                "page": page,
                "limit": limit,
                "total_count": documents_result.get("total_count", 0),
                "has_more": documents_result.get("has_more", False)
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, "Listing documents", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.get("/documents/{document_id}")
async def get_document_details(
    document_id: int,
    current_user: Admin = Depends(get_current_user)
):
    """Get detailed document information"""
    
    try:
        # Get document details
        document_details = await document_processor.get_document_details(document_id)
        
        if not document_details:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user has access to this document
        if (current_user.role != "super_admin" and 
            document_details.get("uploaded_by") != current_user.id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Format for display
        formatted_details = {
            **document_details,
            "file_size_formatted": format_file_size(document_details.get("file_size", 0)),
            "uploaded_at_formatted": format_timestamp(
                datetime.fromisoformat(document_details["uploaded_at"]) if document_details.get("uploaded_at") else None,
                "full"
            ),
            "processed_at_formatted": format_timestamp(
                datetime.fromisoformat(document_details["processed_at"]) if document_details.get("processed_at") else None,
                "full"
            )
        }
        
        return JSONResponse(content={
            "success": True,
            "document": formatted_details
        })
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, f"Getting document details: {document_id}", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Failed to get document details")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: Admin = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a document"""
    
    try:
        # Get document details first for logging
        document_details = await document_processor.get_document_details(document_id)
        
        if not document_details:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user has access to delete this document
        if (current_user.role != "super_admin" and 
            document_details.get("uploaded_by") != current_user.id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete the document
        success = await document_processor.delete_document(document_id, current_user.id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document")
        
        # Log deletion
        log_system_event(
            db=db,
            level="INFO",
            message=f"Document deleted: {document_details.get('original_name')}",
            module="admin_documents",
            function="delete_document",
            user_id=current_user.id,
            additional_data=f'{{"document_id": {document_id}}}'
        )
        
        logger.info(f"Document {document_id} deleted by admin {current_user.email}")
        
        return JSONResponse(content={
            "success": True,
            "message": "Document deleted successfully"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, f"Deleting document: {document_id}", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.get("/documents/{document_id}/images/{image_id}")
async def get_document_image(
    document_id: int,
    image_id: int,
    current_user: Admin = Depends(get_current_user)
):
    """Get document image file"""
    
    try:
        # Check document access
        document_details = await document_processor.get_document_details(document_id)
        
        if not document_details:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if (current_user.role != "super_admin" and 
            document_details.get("uploaded_by") != current_user.id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get image file
        image_data = await document_processor.get_document_image(image_id)
        
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        image_content, content_type, filename = image_data
        
        return StreamingResponse(
            io.BytesIO(image_content),
            media_type=content_type,
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, f"Getting document image: {document_id}/{image_id}", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Failed to get image")


@router.get("/analytics/overview")
async def analytics_overview(
    time_period: str = "24h",
    current_user: Admin = Depends(get_current_user)
):
    """Get analytics overview"""
    
    try:
        # Get dashboard overview
        overview_data = await analytics_service.get_dashboard_overview(time_period)
        
        return JSONResponse(content={
            "success": True,
            "data": overview_data
        })
    
    except Exception as e:
        log_error(e, f"Analytics overview: {time_period}", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Failed to get analytics overview")


@router.get("/analytics/geographic")
async def analytics_geographic(
    time_period: str = "7d",
    current_user: Admin = Depends(get_current_user)
):
    """Get geographic analytics data"""
    
    try:
        # Get geographic data
        geo_data = await analytics_service.get_geographic_data(time_period)
        
        return JSONResponse(content={
            "success": True,
            "data": geo_data
        })
    
    except Exception as e:
        log_error(e, f"Geographic analytics: {time_period}", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Failed to get geographic data")


@router.get("/analytics/emergency")
async def analytics_emergency(
    time_period: str = "30d",
    current_user: Admin = Depends(get_current_user)
):
    """Get emergency analytics"""
    
    try:
        # Get emergency analytics
        emergency_data = await analytics_service.get_emergency_analytics(time_period)
        
        return JSONResponse(content={
            "success": True,
            "data": emergency_data
        })
    
    except Exception as e:
        log_error(e, f"Emergency analytics: {time_period}", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Failed to get emergency analytics")


@router.get("/analytics/documents")
async def analytics_documents(
    current_user: Admin = Depends(get_current_user)
):
    """Get document analytics"""
    
    try:
        # Get document analytics
        doc_data = await analytics_service.get_document_analytics()
        
        return JSONResponse(content={
            "success": True,
            "data": doc_data
        })
    
    except Exception as e:
        log_error(e, "Document analytics", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Failed to get document analytics")


@router.get("/system/health")
async def system_health_check(
    current_user: Admin = Depends(get_current_user)
):
    """Comprehensive system health check"""
    
    try:
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "services": {}
        }
        
        # Check RAG service
        try:
            rag_health = await rag_service.health_check()
            health_data["services"]["rag"] = rag_health
        except Exception as e:
            health_data["services"]["rag"] = {"overall": "error", "error": str(e)}
        
        # Check analytics service
        try:
            # Quick analytics test
            overview = await analytics_service.get_dashboard_overview("1h")
            health_data["services"]["analytics"] = "healthy" if overview else "error"
        except Exception as e:
            health_data["services"]["analytics"] = f"error: {str(e)[:50]}"
        
        # Check document processor
        try:
            # Quick document list test
            doc_test = await document_processor.list_documents(limit=1)
            health_data["services"]["documents"] = "healthy" if doc_test.get("success") else "error"
        except Exception as e:
            health_data["services"]["documents"] = f"error: {str(e)[:50]}"
        
        # Get collection stats
        try:
            collection_stats = await rag_service.get_collection_stats()
            health_data["collection_stats"] = collection_stats
        except Exception as e:
            health_data["collection_stats"] = {"error": str(e)}
        
        # Determine overall status
        service_statuses = [
            health_data["services"]["rag"].get("overall", "error"),
            health_data["services"]["analytics"],
            health_data["services"]["documents"]
        ]
        
        if all(status == "healthy" for status in service_statuses):
            health_data["overall_status"] = "healthy"
        elif any(status == "healthy" for status in service_statuses):
            health_data["overall_status"] = "degraded"
        else:
            health_data["overall_status"] = "error"
        
        return JSONResponse(content=health_data)
    
    except Exception as e:
        log_error(e, "System health check", {"user_id": current_user.id})
        return JSONResponse(
            status_code=500,
            content={
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.post("/system/settings")
async def update_system_settings(
    request: Request,
    current_user: Admin = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update system settings"""
    
    # Only super admins can update system settings
    if current_user.role != "super_admin":
        raise HTTPException(status_code=403, detail="Super admin access required")
    
    try:
        # Get form data
        form_data = await request.form()
        settings_data = dict(form_data)
        
        # Validate and update settings
        # This would update app settings in the database
        # For now, just log the action
        
        log_system_event(
            db=db,
            level="INFO",
            message="System settings updated",
            module="admin_system",
            function="update_system_settings",
            user_id=current_user.id,
            additional_data=str(settings_data)
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "System settings updated successfully"
        })
    
    except Exception as e:
        log_error(e, "Updating system settings", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Failed to update settings")


@router.post("/logout")
async def admin_logout(
    current_user: Admin = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Admin logout"""
    
    try:
        # Log logout
        log_system_event(
            db=db,
            level="INFO",
            message=f"Admin logout: {current_user.email}",
            module="admin_auth",
            function="admin_logout",
            user_id=current_user.id
        )
        
        logger.info(f"Admin {current_user.email} logged out")
        
        return JSONResponse(content={
            "success": True,
            "message": "Logged out successfully",
            "redirect_url": "/admin/login"
        })
    
    except Exception as e:
        log_error(e, "Admin logout", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Logout failed")


@router.get("/profile")
async def admin_profile(
    current_user: Admin = Depends(get_current_user)
):
    """Get admin profile information"""
    
    try:
        profile_data = {
            "id": current_user.id,
            "email": current_user.email,
            "name": current_user.name,
            "role": current_user.role,
            "is_active": current_user.is_active,
            "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
            "last_login": current_user.last_login.isoformat() if current_user.last_login else None,
            "last_login_formatted": format_timestamp(current_user.last_login, "relative") if current_user.last_login else "Never"
        }
        
        return JSONResponse(content={
            "success": True,
            "profile": profile_data
        })
    
    except Exception as e:
        log_error(e, "Getting admin profile", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Failed to get profile")


@router.post("/change-password")
async def change_admin_password(
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    current_user: Admin = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change admin password"""
    
    try:
        # Validate current password
        from ..auth import verify_password, get_password_hash
        
        if not verify_password(current_password, current_user.password_hash):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        
        # Validate new password
        if new_password != confirm_password:
            raise HTTPException(status_code=400, detail="New passwords do not match")
        
        password_validation = validate_password(new_password)
        if not password_validation["valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Password validation failed: {'; '.join(password_validation['errors'])}"
            )
        
        # Update password
        current_user.password_hash = get_password_hash(new_password)
        db.commit()
        
        # Log password change
        log_system_event(
            db=db,
            level="INFO",
            message=f"Password changed for admin: {current_user.email}",
            module="admin_auth",
            function="change_admin_password",
            user_id=current_user.id
        )
        
        logger.info(f"Password changed for admin {current_user.email}")
        
        return JSONResponse(content={
            "success": True,
            "message": "Password changed successfully"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, "Changing admin password", {"user_id": current_user.id})
        raise HTTPException(status_code=500, detail="Failed to change password")


