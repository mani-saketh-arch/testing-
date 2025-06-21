"""
SafeIndy Assistant - RAG Service
LlamaIndex + Qdrant integration for document processing and retrieval
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from datetime import datetime

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.readers.file import PDFReader
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ..config import settings, RAG_CONFIG
from ..utils import timer, log_error, cache
from .llm import llm_service
from .external import search_service

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieval-Augmented Generation service using LlamaIndex and Qdrant"""
    
    def __init__(self):
        self.qdrant_client = None
        self.cohere_embeddings = None
        self.pdf_reader = None
        self.node_parser = None
        self.vector_stores = {}
        self.indices = {}
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize RAG components"""
        try:
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
            
            # Initialize Cohere embeddings
            self.cohere_embeddings = CohereEmbedding(
                api_key=settings.COHERE_API_KEY,
                model_name="embed-english-v3.0"
            )
            
            # Set global LlamaIndex settings
            Settings.embed_model = self.cohere_embeddings
            Settings.chunk_size = RAG_CONFIG["chunk_size"]
            Settings.chunk_overlap = RAG_CONFIG["chunk_overlap"]
            
            # Initialize PDF reader
            self.pdf_reader = PDFReader()
            
            # Initialize node parser
            self.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=RAG_CONFIG["chunk_size"],
                chunk_overlap=RAG_CONFIG["chunk_overlap"]
            )
            
            # Initialize vector stores and indices
            asyncio.create_task(self._setup_collections())
            
            logger.info("✅ RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG service: {e}")
            log_error(e, "RAG service initialization")
    
    async def _setup_collections(self):
        """Setup Qdrant collections for different data types"""
        try:
            collections = [
                settings.QDRANT_COLLECTION_WEB,
                settings.QDRANT_COLLECTION_DOCS
            ]
            
            for collection_name in collections:
                # Check if collection exists
                try:
                    collection_info = self.qdrant_client.get_collection(collection_name)
                    logger.info(f"✅ Collection {collection_name} already exists")
                except Exception:
                    # Create collection if it doesn't exist
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=1024,  # Cohere embedding dimension
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"✅ Created collection: {collection_name}")
                
                # Setup vector store for this collection
                vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=collection_name
                )
                self.vector_stores[collection_name] = vector_store
                
                # Create index for this collection
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                try:
                    # Try to load existing index
                    index = load_index_from_storage(storage_context)
                    logger.info(f"✅ Loaded existing index for {collection_name}")
                except:
                    # Create new index if none exists
                    index = VectorStoreIndex([], storage_context=storage_context)
                    logger.info(f"✅ Created new index for {collection_name}")
                
                self.indices[collection_name] = index
        
        except Exception as e:
            logger.error(f"❌ Failed to setup collections: {e}")
            log_error(e, "Collection setup")
    
    @timer
    async def process_document(
        self, 
        file_path: str, 
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a document and add it to the knowledge base"""
        
        try:
            # Read document
            documents = self.pdf_reader.load_data(file_path)
            
            if not documents:
                return {
                    "success": False,
                    "error": "No content extracted from document"
                }
            
            # Add metadata to documents
            doc_metadata = {
                "document_id": document_id,
                "file_path": file_path,
                "processed_at": datetime.now().isoformat(),
                "source_type": "admin_upload"
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            for doc in documents:
                doc.metadata.update(doc_metadata)
            
            # Parse documents into nodes
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Add nodes to the admin documents index
            admin_index = self.indices.get(settings.QDRANT_COLLECTION_DOCS)
            if admin_index:
                admin_index.insert_nodes(nodes)
                
                # Store processing results
                result = {
                    "success": True,
                    "document_id": document_id,
                    "nodes_created": len(nodes),
                    "total_content_length": sum(len(doc.text) for doc in documents),
                    "metadata": doc_metadata
                }
                
                logger.info(f"✅ Processed document {document_id}: {len(nodes)} nodes created")
                return result
            else:
                return {
                    "success": False,
                    "error": "Admin documents index not available"
                }
        
        except Exception as e:
            error_msg = f"Failed to process document {document_id}: {e}"
            log_error(e, "Document processing", {"document_id": document_id, "file_path": file_path})
            return {
                "success": False,
                "error": error_msg
            }
    
    @timer
    async def search_documents(
        self,
        query: str,
        collection: str = None,
        max_results: int = 5,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Search documents using hybrid retrieval"""
        
        if similarity_threshold is None:
            similarity_threshold = RAG_CONFIG["similarity_threshold"]
        
        # Determine which collection(s) to search
        collections_to_search = []
        if collection:
            collections_to_search = [collection]
        else:
            # Search admin docs first (priority), then web knowledge
            collections_to_search = [
                settings.QDRANT_COLLECTION_DOCS,
                settings.QDRANT_COLLECTION_WEB
            ]
        
        all_results = []
        
        for collection_name in collections_to_search:
            if collection_name not in self.indices:
                continue
            
            try:
                # Get index for this collection
                index = self.indices[collection_name]
                
                # Create retriever
                retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=max_results
                )
                
                # Perform search
                nodes = retriever.retrieve(query)
                
                # Process results
                for node in nodes:
                    if node.score >= similarity_threshold:
                        result = {
                            "content": node.text,
                            "score": node.score,
                            "metadata": node.metadata,
                            "collection": collection_name,
                            "node_id": node.node_id
                        }
                        all_results.append(result)
                
                # If we found results in admin docs, prioritize them
                if collection_name == settings.QDRANT_COLLECTION_DOCS and all_results:
                    break
            
            except Exception as e:
                logger.warning(f"Search failed for collection {collection_name}: {e}")
                continue
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:max_results]
    
    @timer
    async def generate_response_with_context(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None,
        max_context_length: int = 2000
    ) -> Dict[str, Any]:
        """Generate response using RAG with context from documents and external sources"""
        
        try:
            # 1. Search internal documents first (admin uploads have priority)
            document_results = await self.search_documents(
                user_query,
                max_results=RAG_CONFIG["max_results"]
            )
            
            # 2. Search external sources for real-time data
            external_results = None
            if "indianapolis" in user_query.lower() or "indy" in user_query.lower():
                external_results = await search_service.search_indianapolis_info(user_query)
            
            # 3. Build enhanced context
            enhanced_context = context.copy() if context else {}
            
            # Add document context
            if document_results:
                doc_context = []
                total_length = 0
                
                for result in document_results:
                    content = result["content"]
                    if total_length + len(content) <= max_context_length:
                        doc_context.append({
                            "content": content,
                            "source": result["metadata"].get("file_path", "Document"),
                            "score": result["score"]
                        })
                        total_length += len(content)
                    else:
                        break
                
                enhanced_context["document_context"] = doc_context
            
            # Add external search results
            if external_results:
                enhanced_context["search_results"] = external_results["content"][:500]
                enhanced_context["citations"] = external_results.get("citations", [])
            
            # 4. Generate response using LLM with enhanced context
            llm_response = await llm_service.generate_response(
                user_query,
                context=enhanced_context
            )
            
            # 5. Build comprehensive response
            response = {
                "success": llm_response.get("success", False),
                "response": llm_response.get("response", "I apologize, but I'm having trouble generating a response right now."),
                "model_used": llm_response.get("model_used", "unknown"),
                "response_time_ms": llm_response.get("response_time_ms", 0),
                "sources": []
            }
            
            # Add sources
            if document_results:
                for result in document_results[:3]:  # Top 3 document sources
                    source_info = {
                        "type": "document",
                        "title": result["metadata"].get("original_name", "Document"),
                        "relevance_score": result["score"],
                        "collection": result["collection"]
                    }
                    response["sources"].append(source_info)
            
            if external_results and external_results.get("citations"):
                for citation in external_results["citations"][:2]:  # Top 2 external sources
                    source_info = {
                        "type": "web",
                        "title": citation.get("title", "Web Source"),
                        "url": citation.get("url", ""),
                        "domain": citation.get("domain", "")
                    }
                    response["sources"].append(source_info)
            
            return response
        
        except Exception as e:
            error_msg = f"RAG response generation failed: {e}"
            log_error(e, "RAG response generation", {"query": user_query[:100]})
            
            return {
                "success": False,
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again or contact emergency services directly if this is urgent.",
                "error": error_msg,
                "sources": []
            }
    
    @timer
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the knowledge base"""
        
        try:
            # Find and delete nodes with matching document_id
            admin_index = self.indices.get(settings.QDRANT_COLLECTION_DOCS)
            if not admin_index:
                return False
            
            # Get all nodes and filter by document_id
            # Note: This is a simplified approach. In production, you might want
            # to maintain a mapping of document_id to node_ids for efficiency
            
            # For now, we'll use Qdrant's filter functionality
            filter_condition = {
                "must": [
                    {
                        "key": "document_id",
                        "match": {"value": document_id}
                    }
                ]
            }
            
            # Delete points matching the filter
            self.qdrant_client.delete(
                collection_name=settings.QDRANT_COLLECTION_DOCS,
                points_selector=filter_condition
            )
            
            logger.info(f"✅ Deleted document {document_id} from knowledge base")
            return True
        
        except Exception as e:
            log_error(e, f"Deleting document {document_id}")
            return False
    
    @timer
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base collections"""
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "collections": {}
        }
        
        for collection_name in [settings.QDRANT_COLLECTION_WEB, settings.QDRANT_COLLECTION_DOCS]:
            try:
                collection_info = self.qdrant_client.get_collection(collection_name)
                stats["collections"][collection_name] = {
                    "status": collection_info.status,
                    "points_count": collection_info.points_count,
                    "vectors_count": collection_info.vectors_count,
                    "indexed_vectors_count": collection_info.indexed_vectors_count
                }
            except Exception as e:
                stats["collections"][collection_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of RAG services"""
        
        health = {
            "qdrant": "unknown",
            "cohere_embeddings": "unknown",
            "collections": "unknown",
            "overall": "unknown"
        }
        
        # Test Qdrant connection
        try:
            collections = self.qdrant_client.get_collections()
            health["qdrant"] = "healthy"
        except Exception as e:
            health["qdrant"] = f"error: {str(e)[:50]}"
        
        # Test Cohere embeddings
        try:
            if self.cohere_embeddings:
                test_embedding = self.cohere_embeddings.get_text_embedding("test")
                health["cohere_embeddings"] = "healthy" if test_embedding else "error"
            else:
                health["cohere_embeddings"] = "not_configured"
        except Exception as e:
            health["cohere_embeddings"] = f"error: {str(e)[:50]}"
        
        # Test collections
        try:
            stats = await self.get_collection_stats()
            if all(col.get("status") == "green" for col in stats["collections"].values()):
                health["collections"] = "healthy"
            else:
                health["collections"] = "degraded"
        except Exception as e:
            health["collections"] = f"error: {str(e)[:50]}"
        
        # Overall health
        if all(status == "healthy" for status in [health["qdrant"], health["cohere_embeddings"], health["collections"]]):
            health["overall"] = "healthy"
        elif any(status == "healthy" for status in [health["qdrant"], health["cohere_embeddings"]]):
            health["overall"] = "degraded"
        else:
            health["overall"] = "error"
        
        return health


# Global RAG service instance
rag_service = RAGService()