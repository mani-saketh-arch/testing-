"""
SafeIndy Assistant - Document Processing Service
PDF processing, text extraction, and image handling
"""

import os
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from datetime import datetime
import mimetypes

# PDF processing imports
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import io

from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db_context, Document, DocumentImage, log_system_event
from ..utils import (
    generate_unique_filename, 
    validate_file_type, 
    validate_file_size,
    get_file_info,
    ensure_directory_exists,
    safe_filename,
    timer,
    log_error
)
from .rag import rag_service

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Document processing and management service"""
    
    def __init__(self):
        # Ensure required directories exist
        ensure_directory_exists(settings.UPLOAD_DIRECTORY)
        ensure_directory_exists(settings.IMAGES_DIRECTORY)
        
        # Configure tesseract path if needed (for OCR)
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux
        
        logger.info("✅ Document processor initialized")
    
    @timer
    async def process_uploaded_file(
        self,
        file_content: bytes,
        filename: str,
        uploaded_by: int,
        file_size: int = None
    ) -> Dict[str, Any]:
        """Process an uploaded file through the complete pipeline"""
        
        try:
            # Validate file
            if not validate_file_type(filename):
                return {
                    "success": False,
                    "error": f"File type not allowed. Allowed types: {', '.join(settings.ALLOWED_FILE_TYPES)}"
                }
            
            if file_size and not validate_file_size(file_size):
                return {
                    "success": False,
                    "error": f"File too large. Maximum size: {settings.MAX_FILE_SIZE / 1024 / 1024:.1f} MB"
                }
            
            # Generate unique filename and save file
            unique_filename = generate_unique_filename(filename)
            file_path = os.path.join(settings.UPLOAD_DIRECTORY, unique_filename)
            
            # Save file to disk
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Get file info
            file_info = get_file_info(file_path)
            mime_type = file_info.get("mime_type", mimetypes.guess_type(filename)[0])
            
            # Create database record
            with get_db_context() as db:
                document = Document(
                    filename=unique_filename,
                    original_name=safe_filename(filename),
                    file_path=file_path,
                    file_size=file_info.get("size", file_size),
                    mime_type=mime_type,
                    uploaded_by=uploaded_by,
                    status="pending"
                )
                
                db.add(document)
                db.commit()
                db.refresh(document)
                
                document_id = document.id
                
                # Log the upload
                log_system_event(
                    db=db,
                    level="INFO",
                    message=f"Document uploaded: {filename}",
                    module="document_processor",
                    function="process_uploaded_file",
                    user_id=uploaded_by,
                    additional_data=f'{{"file_size": {file_info.get("size", 0)}, "mime_type": "{mime_type}"}}'
                )
            
            # Process the document asynchronously
            processing_result = await self._process_document_content(
                document_id=document_id,
                file_path=file_path,
                mime_type=mime_type
            )
            
            return {
                "success": True,
                "document_id": document_id,
                "filename": unique_filename,
                "original_name": filename,
                "file_size": file_info.get("size", file_size),
                "processing_result": processing_result
            }
        
        except Exception as e:
            error_msg = f"Failed to process uploaded file: {e}"
            log_error(e, "File upload processing", {"filename": filename})
            
            # Clean up file if it was created
            if 'file_path' in locals() and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            
            return {
                "success": False,
                "error": error_msg
            }
    
    @timer
    async def _process_document_content(
        self,
        document_id: int,
        file_path: str,
        mime_type: str
    ) -> Dict[str, Any]:
        """Process document content (text extraction, image extraction, RAG indexing)"""
        
        try:
            with get_db_context() as db:
                # Update document status
                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    return {"success": False, "error": "Document not found"}
                
                document.status = "processing"
                db.commit()
            
            processing_results = {
                "text_extraction": {"success": False},
                "image_extraction": {"success": False},
                "rag_indexing": {"success": False}
            }
            
            # 1. Extract text from PDF
            if mime_type == "application/pdf" or file_path.lower().endswith('.pdf'):
                text_result = await self._extract_text_from_pdf(file_path)
                processing_results["text_extraction"] = text_result
                
                # 2. Extract images from PDF
                image_result = await self._extract_images_from_pdf(document_id, file_path)
                processing_results["image_extraction"] = image_result
                
                # 3. Add to RAG system
                if text_result.get("success"):
                    rag_result = await self._add_to_rag_system(
                        document_id, 
                        file_path,
                        text_result.get("extracted_text", "")
                    )
                    processing_results["rag_indexing"] = rag_result
            
            # Update document with results
            with get_db_context() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    # Calculate page count
                    page_count = processing_results["text_extraction"].get("page_count", 0)
                    
                    # Create summary
                    summary_parts = []
                    if processing_results["text_extraction"].get("success"):
                        summary_parts.append(f"Text extracted: {processing_results['text_extraction'].get('char_count', 0)} characters")
                    if processing_results["image_extraction"].get("success"):
                        summary_parts.append(f"Images extracted: {processing_results['image_extraction'].get('image_count', 0)}")
                    if processing_results["rag_indexing"].get("success"):
                        summary_parts.append(f"RAG indexed: {processing_results['rag_indexing'].get('nodes_created', 0)} nodes")
                    
                    document.page_count = page_count
                    document.extraction_summary = "; ".join(summary_parts)
                    document.processed_at = datetime.utcnow()
                    document.status = "completed" if any(r.get("success") for r in processing_results.values()) else "error"
                    
                    # Set vector collection ID if RAG indexing succeeded
                    if processing_results["rag_indexing"].get("success"):
                        document.vector_collection_id = settings.QDRANT_COLLECTION_DOCS
                    
                    db.commit()
                    
                    # Log processing completion
                    log_system_event(
                        db=db,
                        level="INFO" if document.status == "completed" else "WARNING",
                        message=f"Document processing {document.status}: {document.original_name}",
                        module="document_processor",
                        function="_process_document_content",
                        additional_data=str(processing_results)
                    )
            
            return {
                "success": True,
                "processing_results": processing_results
            }
        
        except Exception as e:
            error_msg = f"Document content processing failed: {e}"
            log_error(e, "Document content processing", {"document_id": document_id})
            
            # Update document status to error
            try:
                with get_db_context() as db:
                    document = db.query(Document).filter(Document.id == document_id).first()
                    if document:
                        document.status = "error"
                        document.extraction_summary = f"Processing failed: {str(e)}"
                        db.commit()
            except:
                pass
            
            return {
                "success": False,
                "error": error_msg
            }
    
    @timer
    async def _extract_text_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF file"""
        
        try:
            extracted_text = ""
            page_count = 0
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
            
            # Clean up extracted text
            extracted_text = extracted_text.strip()
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "char_count": len(extracted_text),
                "page_count": page_count
            }
        
        except Exception as e:
            log_error(e, f"PDF text extraction: {file_path}")
            return {
                "success": False,
                "error": str(e),
                "page_count": 0
            }
    
    @timer
    async def _extract_images_from_pdf(self, document_id: int, file_path: str) -> Dict[str, Any]:
        """Extract images from PDF and perform OCR"""
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(file_path, dpi=150)  # Lower DPI for faster processing
            
            extracted_images = []
            total_ocr_text = ""
            
            for page_num, image in enumerate(images):
                try:
                    # Generate unique filename for image
                    image_filename = f"doc_{document_id}_page_{page_num + 1}_{uuid.uuid4().hex[:8]}.png"
                    image_path = os.path.join(settings.IMAGES_DIRECTORY, image_filename)
                    
                    # Save image
                    image.save(image_path, "PNG")
                    
                    # Perform OCR on the image
                    ocr_text = ""
                    try:
                        ocr_text = pytesseract.image_to_string(image, lang='eng')
                        total_ocr_text += f"\n--- Page {page_num + 1} OCR ---\n{ocr_text}\n"
                    except Exception as ocr_error:
                        logger.warning(f"OCR failed for page {page_num + 1}: {ocr_error}")
                        ocr_text = "OCR processing failed"
                    
                    # Determine image type (simple heuristic)
                    image_type = "page_image"
                    if len(ocr_text.strip()) > 100:
                        image_type = "text_image"
                    
                    # Save image metadata to database
                    with get_db_context() as db:
                        doc_image = DocumentImage(
                            document_id=document_id,
                            image_filename=image_filename,
                            image_path=image_path,
                            page_number=page_num + 1,
                            image_type=image_type,
                            ocr_text=ocr_text[:2000],  # Limit OCR text length
                            description=f"Page {page_num + 1} from PDF document"
                        )
                        
                        db.add(doc_image)
                        db.commit()
                    
                    extracted_images.append({
                        "filename": image_filename,
                        "page_number": page_num + 1,
                        "ocr_text_length": len(ocr_text),
                        "image_type": image_type
                    })
                
                except Exception as page_error:
                    logger.warning(f"Failed to process page {page_num + 1}: {page_error}")
                    continue
            
            return {
                "success": True,
                "image_count": len(extracted_images),
                "images": extracted_images,
                "total_ocr_text": total_ocr_text
            }
        
        except Exception as e:
            log_error(e, f"PDF image extraction: {file_path}")
            return {
                "success": False,
                "error": str(e),
                "image_count": 0
            }
    
    @timer
    async def _add_to_rag_system(
        self, 
        document_id: int, 
        file_path: str,
        extracted_text: str
    ) -> Dict[str, Any]:
        """Add document to RAG system for semantic search"""
        
        try:
            # Get document metadata
            with get_db_context() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    return {"success": False, "error": "Document not found"}
                
                metadata = {
                    "document_id": str(document_id),
                    "original_name": document.original_name,
                    "uploaded_by": document.uploaded_by,
                    "uploaded_at": document.uploaded_at.isoformat() if document.uploaded_at else None,
                    "file_size": document.file_size,
                    "mime_type": document.mime_type
                }
            
            # Add to RAG system
            rag_result = await rag_service.process_document(
                file_path=file_path,
                document_id=str(document_id),
                metadata=metadata
            )
            
            return rag_result
        
        except Exception as e:
            log_error(e, f"RAG system indexing: document {document_id}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @timer
    async def get_document_details(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get comprehensive document details"""
        
        try:
            with get_db_context() as db:
                # Get document
                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    return None
                
                # Get associated images
                images = db.query(DocumentImage).filter(
                    DocumentImage.document_id == document_id
                ).all()
                
                # Build response
                document_details = {
                    "id": document.id,
                    "filename": document.filename,
                    "original_name": document.original_name,
                    "file_size": document.file_size,
                    "mime_type": document.mime_type,
                    "uploaded_by": document.uploaded_by,
                    "uploaded_at": document.uploaded_at.isoformat() if document.uploaded_at else None,
                    "processed_at": document.processed_at.isoformat() if document.processed_at else None,
                    "status": document.status,
                    "page_count": document.page_count,
                    "extraction_summary": document.extraction_summary,
                    "vector_collection_id": document.vector_collection_id,
                    "images": []
                }
                
                # Add image details
                for image in images:
                    image_info = {
                        "id": image.id,
                        "filename": image.image_filename,
                        "page_number": image.page_number,
                        "image_type": image.image_type,
                        "ocr_text_length": len(image.ocr_text) if image.ocr_text else 0,
                        "description": image.description,
                        "created_at": image.created_at.isoformat() if image.created_at else None
                    }
                    document_details["images"].append(image_info)
                
                return document_details
        
        except Exception as e:
            log_error(e, f"Getting document details: {document_id}")
            return None
    
    @timer
    async def delete_document(self, document_id: int, user_id: int = None) -> bool:
        """Delete document and all associated files"""
        
        try:
            with get_db_context() as db:
                # Get document
                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    return False
                
                # Get associated images
                images = db.query(DocumentImage).filter(
                    DocumentImage.document_id == document_id
                ).all()
                
                # Delete physical files
                files_to_delete = [document.file_path]
                for image in images:
                    files_to_delete.append(image.image_path)
                
                for file_path in files_to_delete:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"Deleted file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete file {file_path}: {e}")
                
                # Remove from RAG system
                if document.vector_collection_id:
                    await rag_service.delete_document(str(document_id))
                
                # Delete database records
                # Delete images first (foreign key constraint)
                for image in images:
                    db.delete(image)
                
                # Delete document
                db.delete(document)
                db.commit()
                
                # Log deletion
                log_system_event(
                    db=db,
                    level="INFO",
                    message=f"Document deleted: {document.original_name}",
                    module="document_processor",
                    function="delete_document",
                    user_id=user_id,
                    additional_data=f'{{"document_id": {document_id}, "files_deleted": {len(files_to_delete)}}}'
                )
                
                logger.info(f"✅ Document {document_id} deleted successfully")
                return True
        
        except Exception as e:
            log_error(e, f"Deleting document: {document_id}")
            return False
    
    @timer
    async def list_documents(
        self, 
        uploaded_by: int = None,
        status: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List documents with filtering and pagination"""
        
        try:
            with get_db_context() as db:
                # Build query
                query = db.query(Document)
                
                if uploaded_by:
                    query = query.filter(Document.uploaded_by == uploaded_by)
                
                if status:
                    query = query.filter(Document.status == status)
                
                # Get total count
                total_count = query.count()
                
                # Apply pagination and ordering
                documents = query.order_by(Document.uploaded_at.desc()).offset(offset).limit(limit).all()
                
                # Build response
                document_list = []
                for doc in documents:
                    # Get image count
                    image_count = db.query(DocumentImage).filter(
                        DocumentImage.document_id == doc.id
                    ).count()
                    
                    doc_info = {
                        "id": doc.id,
                        "filename": doc.filename,
                        "original_name": doc.original_name,
                        "file_size": doc.file_size,
                        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                        "status": doc.status,
                        "page_count": doc.page_count,
                        "image_count": image_count,
                        "extraction_summary": doc.extraction_summary
                    }
                    document_list.append(doc_info)
                
                return {
                    "success": True,
                    "documents": document_list,
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < total_count
                }
        
        except Exception as e:
            log_error(e, "Listing documents")
            return {
                "success": False,
                "error": str(e),
                "documents": [],
                "total_count": 0
            }
    
    async def get_document_image(self, image_id: int) -> Optional[Tuple[bytes, str, str]]:
        """Get document image file content"""
        
        try:
            with get_db_context() as db:
                image = db.query(DocumentImage).filter(DocumentImage.id == image_id).first()
                if not image or not os.path.exists(image.image_path):
                    return None
                
                # Read image file
                with open(image.image_path, 'rb') as f:
                    image_content = f.read()
                
                # Determine content type
                content_type = "image/png"  # Default, since we save as PNG
                
                return image_content, content_type, image.image_filename
        
        except Exception as e:
            log_error(e, f"Getting document image: {image_id}")
            return None


# Global document processor instance
document_processor = DocumentProcessor()