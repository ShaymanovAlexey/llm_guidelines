import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
from datetime import datetime

from vector_store import VectorStore
from bm25_search import AsyncBM25Search, SearchResult as BM25SearchResult

# Import Langfuse integration
try:
    from langfuse_integration import LangfuseManager, create_rag_trace, score_rag_response
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    print("Warning: Langfuse integration not available")

@dataclass
class HybridSearchResult:
    """Represents a hybrid search result combining vector and BM25 scores."""
    doc_id: str
    title: str
    content: str
    vector_score: float
    bm25_score: float
    combined_score: float
    chunk_index: int
    metadata: Dict
    source: str  # 'vector', 'bm25', or 'hybrid'

class HybridSearchSystem:
    """
    Hybrid search system that combines vector search (ChromaDB) and BM25 search
    for improved document retrieval performance.
    
    This system provides:
    - Vector search for semantic similarity
    - BM25 search for keyword-based retrieval
    - Hybrid scoring for optimal results
    """
    
    def __init__(self, 
                 vector_store: VectorStore,
                 bm25_store: AsyncBM25Search,
                 vector_weight: float = 0.6,
                 bm25_weight: float = 0.4):
        """
        Initialize hybrid search system.
        
        Args:
            vector_store: VectorStore instance for vector search
            bm25_store: AsyncBM25Search instance for BM25 search
            vector_weight: Weight for vector search scores (0.0 to 1.0)
            bm25_weight: Weight for BM25 search scores (0.0 to 1.0)
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # Ensure weights sum to 1.0
        total_weight = vector_weight + bm25_weight
        if total_weight != 1.0:
            self.vector_weight = vector_weight / total_weight
            self.bm25_weight = bm25_weight / total_weight
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to both vector store and BM25 index.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'
        
        Returns:
            List of document IDs
        """
        doc_ids = []
        
        for doc in documents:
            doc_id = str(uuid.uuid4())
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            title = metadata.get('title', f'Document {doc_id[:8]}')
            
            # Add to vector store
            await self.vector_store.add_documents([{
                'id': doc_id,
                'text': content,
                'metadata': metadata
            }])
            
            # Add to BM25 index
            await self.bm25_store.add_document(
                doc_id=doc_id,
                title=title,
                content=content,
                chunk_index=metadata.get('chunk_index', 0),
                metadata=metadata
            )
            
            doc_ids.append(doc_id)
        
        return doc_ids
    
    async def search(self, 
                    query: str, 
                    top_k: int = 10,
                    search_type: str = 'hybrid') -> List[HybridSearchResult]:
        """
        Search for documents using the specified search type.
        
        Args:
            query: Search query
            top_k: Number of results to return
            search_type: 'vector', 'bm25', or 'hybrid'
        
        Returns:
            List of HybridSearchResult objects
        """
        if search_type == 'vector':
            return await self._vector_search(query, top_k)
        elif search_type == 'bm25':
            return await self._bm25_search(query, top_k)
        elif search_type == 'hybrid':
            return await self._hybrid_search(query, top_k)
        else:
            raise ValueError(f"Invalid search type: {search_type}. Use 'vector', 'bm25', or 'hybrid'")
    
    async def _vector_search(self, query: str, top_k: int) -> List[HybridSearchResult]:
        """Perform vector search."""
        results = await self.vector_store.similarity_search(query, top_k)
        
        hybrid_results = []
        for result in results:
            hybrid_results.append(HybridSearchResult(
                doc_id=result.get('id', str(uuid.uuid4())),
                title=result['metadata'].get('title', 'Unknown'),
                content=result['text'],
                vector_score=result['similarity_score'],
                bm25_score=0.0,
                combined_score=result['similarity_score'],
                chunk_index=result['metadata'].get('chunk_index', 0),
                metadata=result['metadata'],
                source='vector'
            ))
        
        return hybrid_results
    
    async def _bm25_search(self, query: str, top_k: int) -> List[HybridSearchResult]:
        """Perform BM25 search."""
        results = await self.bm25_store.search(query, top_k)
        
        hybrid_results = []
        for result in results:
            hybrid_results.append(HybridSearchResult(
                doc_id=result.doc_id,
                title=result.title,
                content=result.content,
                vector_score=0.0,
                bm25_score=result.score,
                combined_score=result.score,
                chunk_index=result.chunk_index,
                metadata=result.metadata,
                source='bm25'
            ))
        
        return hybrid_results
    
    async def _hybrid_search(self, query: str, top_k: int) -> List[HybridSearchResult]:
        """Perform hybrid search combining vector and BM25 results."""
        # Get results from both search methods
        vector_results = await self._vector_search(query, top_k * 2)
        bm25_results = await self._bm25_search(query, top_k * 2)
        
        # Create a map of doc_id to results for easy lookup
        doc_map = {}
        
        # Add vector results
        for result in vector_results:
            doc_map[result.doc_id] = result
        
        # Add or update with BM25 results
        for result in bm25_results:
            if result.doc_id in doc_map:
                # Update existing result with BM25 score
                doc_map[result.doc_id].bm25_score = result.bm25_score
                doc_map[result.doc_id].combined_score = (
                    self.vector_weight * doc_map[result.doc_id].vector_score +
                    self.bm25_weight * result.bm25_score
                )
                doc_map[result.doc_id].source = 'hybrid'
            else:
                # Add new result
                result.combined_score = result.bm25_score
                doc_map[result.doc_id] = result
        
        # Convert to list and sort by combined score
        hybrid_results = list(doc_map.values())
        hybrid_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return hybrid_results[:top_k]
    
    async def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get a specific document by ID."""
        # Try vector store first
        doc = await self.vector_store.get_document(doc_id)
        if doc:
            return doc
        
        # Try BM25 store
        doc = await self.bm25_store.get_document(doc_id)
        if doc:
            return {
                'id': doc.doc_id,
                'content': doc.content,
                'metadata': doc.metadata
            }
        
        return None
    
    async def get_statistics(self) -> Dict:
        """Get statistics about the hybrid search system."""
        vector_stats = await self.vector_store.get_collection_stats()
        bm25_stats = await self.bm25_store.get_statistics()
        
        return {
            'vector_store': vector_stats,
            'bm25_store': bm25_stats,
            'weights': {
                'vector_weight': self.vector_weight,
                'bm25_weight': self.bm25_weight
            }
        }
    
    async def clear_all(self):
        """Clear all documents from both stores."""
        await self.vector_store.clear_all()
        await self.bm25_store.clear_all()
    
    async def update_weights(self, vector_weight: float, bm25_weight: float):
        """Update the weights for hybrid search."""
        total_weight = vector_weight + bm25_weight
        if total_weight != 1.0:
            self.vector_weight = vector_weight / total_weight
            self.bm25_weight = bm25_weight / total_weight
        else:
            self.vector_weight = vector_weight
            self.bm25_weight = bm25_weight


class HybridRAGSystem:
    """
    Complete RAG system with hybrid search capabilities.
    Combines document retrieval with answer generation.
    """
    
    def __init__(self, 
                 hybrid_search: HybridSearchSystem,
                 generator,
                 max_context_length: int = 4000,
                 langfuse_manager=None):
        """
        Initialize hybrid RAG system.
        
        Args:
            hybrid_search: HybridSearchSystem instance
            generator: Answer generator (e.g., OllamaGenerator)
            max_context_length: Maximum context length for generation
            langfuse_manager: Optional LangfuseManager for observability
        """
        self.hybrid_search = hybrid_search
        self.generator = generator
        self.max_context_length = max_context_length
        self.langfuse_manager = langfuse_manager
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the hybrid search system."""
        return await self.hybrid_search.add_documents(documents)
    
    async def query(self, 
                   question: str, 
                   top_k: int = 5,
                   search_type: str = 'hybrid',
                   max_tokens: Optional[int] = None,
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the RAG system with hybrid search.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            search_type: 'vector', 'bm25', or 'hybrid'
            max_tokens: Maximum tokens for answer generation
            user_id: User identifier for tracing
            session_id: Session identifier for tracing
        
        Returns:
            Dictionary with answer, context, and metadata
        """
        # Create Langfuse trace if available
        trace = None
        
        if self.langfuse_manager and self.langfuse_manager.is_enabled():
            trace = create_rag_trace(
                self.langfuse_manager,
                question,
                user_id=user_id,
                session_id=session_id
            )
        
        try:
            # Search for relevant documents with tracing
            if trace and self.langfuse_manager.is_enabled():
                with self.langfuse_manager.client.start_as_current_span(
                    name="document_search",
                    input={
                        "query": question,
                        "top_k": top_k,
                        "search_type": search_type
                    },
                    metadata={"operation": "hybrid_search"}
                ) as search_span:
                    search_results = await self.hybrid_search.search(question, top_k, search_type)
                    # Update span with results - use Langfuse client methods directly
                    try:
                        self.langfuse_manager.client.update_current_span(
                            output={
                                "results_count": len(search_results),
                                "search_type": search_type
                            },
                            metadata={"search_success": True}
                        )
                    except Exception as e:
                        print(f"⚠️  Failed to update search span: {e}")
            else:
                search_results = await self.hybrid_search.search(question, top_k, search_type)
            
            if not search_results:
                if trace:
                    trace.end(
                        output={
                            'answer': 'No relevant documents found.',
                            'context': [],
                            'search_type': search_type,
                            'scores': {}
                        },
                        status_message="No results found"
                    )
                
                return {
                    'answer': 'No relevant documents found.',
                    'context': [],
                    'search_type': search_type,
                    'scores': {}
                }
            
            # Prepare context for generation
            context_docs = []
            total_length = 0
            
            for result in search_results:
                if total_length + len(result.content) > self.max_context_length:
                    break
                
                context_docs.append({
                    'content': result.content,
                    'title': result.title,
                    'score': result.combined_score,
                    'source': result.source
                })
                total_length += len(result.content)
            
            # Generate answer with tracing
            if trace and self.langfuse_manager.is_enabled():
                with self.langfuse_manager.client.start_as_current_generation(
                    name=f"ollama-{getattr(self.generator, 'model_name', 'unknown')}",
                    model=getattr(self.generator, 'model_name', 'unknown'),
                    input=f"Question: {question}\n\nContext: {context_docs}",
                    model_parameters={"max_tokens": max_tokens},
                    metadata={
                        "context_length": total_length,
                        "context_docs_count": len(context_docs),
                        "search_type": search_type
                    }
                ) as generation_span:
                    # Generate answer
                    if hasattr(self.generator, 'generate_answer'):
                        answer = await self.generator.generate_answer(question, context_docs, max_tokens)
                    else:
                        # Fallback to simple template
                        answer = self._generate_template_answer(question, context_docs)
                    
                    # Update generation with completion - use Langfuse client methods directly
                    try:
                        self.langfuse_manager.client.update_current_generation(
                            output=answer
                        )
                    except Exception as e:
                        print(f"⚠️  Failed to update generation: {e}")
                    
                    # Score the response
                    relevance_score = min(1.0, len(answer) / 100)  # Simple heuristic
                    helpfulness_score = 0.8  # Could be improved with actual evaluation
                    accuracy_score = 0.9  # Could be improved with actual evaluation
                    
                    # Add scores to the generation using Langfuse client methods directly
                    # Note: Scoring functionality temporarily disabled due to API changes
                    # TODO: Implement proper scoring when Langfuse API is updated
                    try:
                        # For now, just log the scores
                        print(f"📊 Generated scores - Relevance: {relevance_score:.2f}, Helpfulness: {helpfulness_score:.2f}, Accuracy: {accuracy_score:.2f}")
                    except Exception as e:
                        print(f"⚠️  Failed to log scores: {e}")
            else:
                # Generate answer without tracing
                if hasattr(self.generator, 'generate_answer'):
                    answer = await self.generator.generate_answer(question, context_docs, max_tokens)
                else:
                    # Fallback to simple template
                    answer = self._generate_template_answer(question, context_docs)
            
            result = {
                'answer': answer,
                'context': context_docs,
                'search_type': search_type,
                'scores': {
                    'vector_scores': [r.vector_score for r in search_results],
                    'bm25_scores': [r.bm25_score for r in search_results],
                    'combined_scores': [r.combined_score for r in search_results]
                }
            }
            
            # End trace successfully
            if trace:
                trace.end(
                    output=result,
                    metadata={
                        "search_type": search_type,
                        "context_length": total_length,
                        "answer_length": len(answer)
                    }
                )
            
            return result
            
        except Exception as e:
            # End trace with error
            if trace:
                trace.end(status_message=str(e))
            
            # Re-raise the exception
            raise
    
    def _generate_template_answer(self, question: str, context_docs: List[Dict]) -> str:
        """Generate a simple template-based answer."""
        if not context_docs:
            return "I don't have enough information to answer this question."
        
        context_text = "\n\n".join([f"Source: {doc['title']}\n{doc['content']}" 
                                   for doc in context_docs])
        
        return f"""Based on the available information:

{context_text}

This information should help answer your question: {question}"""
    
    async def get_statistics(self) -> Dict:
        """Get statistics about the RAG system."""
        try:
            search_stats = await self.hybrid_search.get_statistics()
            
            stats = {
                'search_system': search_stats,
                'max_context_length': self.max_context_length,
                'generator_type': type(self.generator).__name__
            }
            
            if hasattr(self.generator, 'get_stats'):
                stats['generator'] = await self.generator.get_stats()
            
            return stats
        except Exception as e:
            print(f"⚠️  Error getting statistics: {e}")
            return {
                'error': str(e),
                'max_context_length': self.max_context_length,
                'generator_type': type(self.generator).__name__
            } 