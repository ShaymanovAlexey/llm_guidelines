import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
from datetime import datetime

from vector_store import VectorStore
from bm25_search import AsyncBM25Search, SearchResult as BM25SearchResult

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
                'content': content,
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
        Search documents using the specified search method.
        
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
            raise ValueError(f"Invalid search_type: {search_type}")
    
    async def _vector_search(self, query: str, top_k: int) -> List[HybridSearchResult]:
        """Perform vector search only."""
        results = await self.vector_store.similarity_search(query, k=top_k)
        
        hybrid_results = []
        for result in results:
            hybrid_results.append(HybridSearchResult(
                doc_id=result['id'],
                title=result['metadata'].get('title', f'Document {result["id"][:8]}'),
                content=result['content'],
                vector_score=result['similarity_score'],
                bm25_score=0.0,
                combined_score=result['similarity_score'],
                chunk_index=result['metadata'].get('chunk_index', 0),
                metadata=result['metadata'],
                source='vector'
            ))
        
        return hybrid_results
    
    async def _bm25_search(self, query: str, top_k: int) -> List[HybridSearchResult]:
        """Perform BM25 search only."""
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
        vector_results = await self.vector_store.similarity_search(query, k=top_k * 2)
        bm25_results = await self.bm25_store.search(query, top_k * 2)
        
        # Create lookup dictionaries
        vector_lookup = {result['id']: result for result in vector_results}
        bm25_lookup = {result.doc_id: result for result in bm25_results}
        
        # Combine results
        all_doc_ids = set(vector_lookup.keys()) | set(bm25_lookup.keys())
        combined_results = []
        
        for doc_id in all_doc_ids:
            vector_result = vector_lookup.get(doc_id)
            bm25_result = bm25_lookup.get(doc_id)
            
            # Get scores (normalize to 0-1 range)
            vector_score = vector_result['similarity_score'] if vector_result else 0.0
            bm25_score = bm25_result.score if bm25_result else 0.0
            
            # Normalize BM25 score (assuming typical range 0-10)
            normalized_bm25_score = min(bm25_score / 10.0, 1.0)
            
            # Calculate combined score
            combined_score = (self.vector_weight * vector_score + 
                            self.bm25_weight * normalized_bm25_score)
            
            # Get document information
            if vector_result:
                title = vector_result['metadata'].get('title', f'Document {doc_id[:8]}')
                content = vector_result['content']
                chunk_index = vector_result['metadata'].get('chunk_index', 0)
                metadata = vector_result['metadata']
            elif bm25_result:
                title = bm25_result.title
                content = bm25_result.content
                chunk_index = bm25_result.chunk_index
                metadata = bm25_result.metadata
            else:
                continue
            
            combined_results.append(HybridSearchResult(
                doc_id=doc_id,
                title=title,
                content=content,
                vector_score=vector_score,
                bm25_score=bm25_score,
                combined_score=combined_score,
                chunk_index=chunk_index,
                metadata=metadata,
                source='hybrid'
            ))
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        return combined_results[:top_k]
    
    async def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a document by ID from both stores."""
        # Try vector store first
        vector_doc = await self.vector_store.get_document(doc_id)
        if vector_doc:
            return vector_doc
        
        # Try BM25 store
        bm25_doc = await self.bm25_store.get_document(doc_id)
        if bm25_doc:
            return bm25_doc
        
        return None
    
    async def get_statistics(self) -> Dict:
        """Get statistics from both search systems."""
        vector_stats = await self.vector_store.get_statistics()
        bm25_stats = await self.bm25_store.get_statistics()
        
        return {
            'vector_store': vector_stats,
            'bm25_store': bm25_stats,
            'hybrid_config': {
                'vector_weight': self.vector_weight,
                'bm25_weight': self.bm25_weight
            }
        }
    
    async def clear_all(self):
        """Clear all documents from both stores."""
        await self.vector_store.clear_collection()
        await self.bm25_store.clear_index()
    
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
                 max_context_length: int = 4000):
        """
        Initialize hybrid RAG system.
        
        Args:
            hybrid_search: HybridSearchSystem instance
            generator: Answer generator (e.g., OllamaGenerator)
            max_context_length: Maximum context length for generation
        """
        self.hybrid_search = hybrid_search
        self.generator = generator
        self.max_context_length = max_context_length
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the hybrid search system."""
        return await self.hybrid_search.add_documents(documents)
    
    async def query(self, 
                   question: str, 
                   top_k: int = 5,
                   search_type: str = 'hybrid',
                   max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Query the RAG system with hybrid search.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            search_type: 'vector', 'bm25', or 'hybrid'
            max_tokens: Maximum tokens for answer generation
        
        Returns:
            Dictionary with answer, context, and metadata
        """
        # Search for relevant documents
        search_results = await self.hybrid_search.search(question, top_k, search_type)
        
        if not search_results:
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
        
        # Generate answer
        if hasattr(self.generator, 'generate_answer'):
            answer = await self.generator.generate_answer(question, context_docs, max_tokens)
        else:
            # Fallback to simple template
            answer = self._generate_template_answer(question, context_docs)
        
        return {
            'answer': answer,
            'context': context_docs,
            'search_type': search_type,
            'scores': {
                'vector_scores': [r.vector_score for r in search_results],
                'bm25_scores': [r.bm25_score for r in search_results],
                'combined_scores': [r.combined_score for r in search_results]
            }
        }
    
    def _generate_template_answer(self, question: str, context_docs: List[Dict]) -> str:
        """Generate a simple template-based answer."""
        if not context_docs:
            return "I don't have enough information to answer this question."
        
        context_text = "\n\n".join([f"Source: {doc['title']}\n{doc['content']}" 
                                   for doc in context_docs])
        
        return f"""Based on the available information:

{context_text}

Question: {question}

Answer: The information above provides relevant context for your question. Please review the sources for detailed information."""
    
    async def get_statistics(self) -> Dict:
        """Get system statistics."""
        search_stats = await self.hybrid_search.get_statistics()
        return {
            'search_system': search_stats,
            'generator': type(self.generator).__name__,
            'max_context_length': self.max_context_length
        } 