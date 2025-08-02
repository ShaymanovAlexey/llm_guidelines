import asyncio
from typing import List, Dict, Any, Optional, Type
from abc import ABC, abstractmethod
from document_processor import DocumentProcessor
from vector_store import VectorStore


class BaseRAGSystem(ABC):
    """Base class for RAG systems defining common interface and functionality."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, vector_store_class: Type[VectorStore] = VectorStore, **vector_store_kwargs):
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = vector_store_class(**vector_store_kwargs)
    
    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents to the RAG system."""
        pass
    
    @abstractmethod
    async def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        pass
    
    def _generate_answer(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Generate an answer based on the question and retrieved context."""
        # Simple template-based answer generation
        # In a production system, you would use an LLM like GPT-3.5/4
        
        if not context:
            return "I don't have enough information to answer your question."
        
        # Combine relevant context
        relevant_text = "\n\n".join([doc['text'] for doc in context])
        
        # Simple answer template
        answer = f"Based on the available information:\n\n{relevant_text}\n\n"
        answer += f"This information was retrieved from: {', '.join(set([doc['source'] for doc in context]))}"
        
        return answer
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            vector_stats = await self.vector_store.get_collection_stats()
            return {
                'success': True,
                'total_documents': vector_stats['total_documents'],
                'collection_name': vector_stats['collection_name']
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def list_documents(self) -> Dict[str, Any]:
        """List all documents in the system."""
        try:
            documents = await self.vector_store.list_documents()
            return {
                'success': True,
                'documents': documents
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def clear_documents(self) -> Dict[str, Any]:
        """Clear all documents from the system."""
        try:
            await self.vector_store.delete_collection()
            return {
                'success': True,
                'message': 'All documents cleared successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Basic health check for the RAG system."""
        try:
            vector_stats = await self.vector_store.get_collection_stats()
            return {
                'status': 'healthy',
                'vector_store': {
                    'status': 'healthy',
                    'total_documents': vector_stats['total_documents'],
                    'collection_name': vector_stats['collection_name']
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            } 