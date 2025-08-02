import os
import asyncio
from typing import List, Dict, Any, Optional, Type
from concurrent.futures import ThreadPoolExecutor
from base_rag_system import BaseRAGSystem
from ollama_generator import OllamaGenerator
from vector_store import VectorStore


class AdvancedRAGSystem(BaseRAGSystem):
    """Advanced RAG system with concurrent processing and Ollama integration."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, max_workers: int = 4, 
                 ollama_model: str = "llama2", ollama_url: str = "http://localhost:11434",
                 vector_store_class: Type[VectorStore] = VectorStore, **vector_store_kwargs):
        super().__init__(chunk_size, chunk_overlap, vector_store_class=vector_store_class, **vector_store_kwargs)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        
        # Initialize Ollama generator
        self.ollama_generator = OllamaGenerator(model_name=ollama_model, base_url=ollama_url)
        self.use_ollama = True
    
    async def add_documents_concurrent(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents with concurrent processing."""
        try:
            # Process documents into chunks concurrently
            chunk_tasks = []
            for doc in documents:
                if doc['type'] == 'file':
                    task = asyncio.get_event_loop().run_in_executor(
                        self.executor, 
                        self.document_processor.process_file, 
                        doc['content'], 
                        doc['filename']
                    )
                else:
                    task = asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.document_processor.process_text,
                        doc['content'],
                        doc['source']
                    )
                chunk_tasks.append(task)
            
            # Wait for all chunking tasks to complete
            chunk_results = await asyncio.gather(*chunk_tasks)
            
            # Combine all chunks
            all_chunks = []
            for chunks in chunk_results:
                all_chunks.extend(chunks)
            
            # Add to vector store
            await self.vector_store.add_documents(all_chunks)
            
            return {
                'success': True,
                'message': f'Successfully processed {len(all_chunks)} chunks from {len(documents)} documents (concurrent)',
                'chunks_processed': len(all_chunks),
                'documents_processed': len(documents)
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing documents: {str(e)}',
                'error': str(e)
            }
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents to the RAG system."""
        try:
            # Process documents into chunks
            all_chunks = []
            for doc in documents:
                if doc['type'] == 'pdf':
                    # Handle PDF files
                    chunks = self.document_processor.process_pdf(doc['content'], doc['filename'])
                elif doc['type'] == 'file':
                    # Handle text-based files
                    chunks = self.document_processor.process_file(doc['content'], doc['filename'])
                else:
                    # Handle pasted text
                    chunks = self.document_processor.process_text(doc['content'], doc['source'])
                all_chunks.extend(chunks)
            
            # Add to vector store
            await self.vector_store.add_documents(all_chunks)
            
            return {
                'success': True,
                'message': f'Successfully processed {len(all_chunks)} chunks from {len(documents)} documents',
                'chunks_processed': len(all_chunks)
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing documents: {str(e)}',
                'error': str(e)
            }
    
    async def batch_query(self, questions: List[str], k: int = 3) -> List[Dict[str, Any]]:
        """Process multiple queries concurrently."""
        tasks = [self.query(question, k) for question in questions]
        results = await asyncio.gather(*tasks)
        return results
    
    async def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        try:
            # Retrieve relevant documents
            retrieved_docs = await self.vector_store.similarity_search(question, k)
            
            # Create context from retrieved documents
            context = []
            for doc in retrieved_docs:
                context.append({
                    'text': doc['text'],
                    'source': doc['metadata'].get('topic') or doc['metadata'].get('source', 'Unknown'),
                    'similarity_score': doc['similarity_score']
                })
            
            print("self.use_ollama", self.use_ollama)
            
            # Generate answer using Ollama or fallback to template
            if self.use_ollama:
                try:
                    # If no context found, return a special response for UI handling
                    if not context:
                        return {
                            'success': True,
                            'question': question,
                            'answer': 'NO_CONTEXT_FOUND',
                            'context': [],
                            'sources': [],
                            'needs_user_input': True
                        }
                    
                    answer = await self.ollama_generator.generate_answer(question, context)
                except Exception as e:
                    # Fallback to template-based approach if Ollama fails
                    answer = self._generate_answer(question, context)
            else:
                answer = self._generate_answer(question, context)
            
            return {
                'success': True,
                'question': question,
                'answer': answer,
                'context': context,
                'sources': list(set([doc['source'] for doc in context])) if context else []
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing query: {str(e)}',
                'error': str(e)
            }

    async def query_without_context(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Query the RAG system without using context (general answer from model)."""
        try:
            if self.use_ollama:
                try:
                    # Generate answer without context
                    general_prompt = f"""Based on my general knowledge, here's what I can tell you about: {question}

                    Please note that this is general information and may not be specific to your particular context or the most up-to-date information available."""
                    
                    loop = asyncio.get_event_loop()
                    answer = await loop.run_in_executor(None, self.ollama_generator._call_ollama, general_prompt)
                    
                    return {
                        'success': True,
                        'question': question,
                        'answer': answer.strip(),
                        'context': [],
                        'sources': []
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'message': f'Error generating general answer: {str(e)}',
                        'error': str(e)
                    }
            else:
                return {
                    'success': False,
                    'message': 'Ollama is not enabled for general answers',
                    'error': 'Ollama disabled'
                }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing query without context: {str(e)}',
                'error': str(e)
            }
    
    async def get_embedding_models(self) -> List[str]:
        """Get available embedding models."""
        return self.vector_store.get_available_embedding_models()

    async def switch_embedding_model(self, model_name: str) -> Dict[str, Any]:
        """Switch to a different embedding model."""
        return self.vector_store.switch_embedding_model(model_name)

    async def get_current_embedding_model(self) -> str:
        return self.vector_store.get_current_embedding_model()

    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            vector_stats = await self.vector_store.get_collection_stats()
            return {
                'success': True,
                'total_documents': vector_stats['total_documents'],
                'collection_name': vector_stats['collection_name'],
                'max_workers': self.max_workers,
                'embedding_model': self.vector_store.get_current_embedding_model(),
                'current_ollama_model': self.ollama_generator.model_name
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for the RAG system."""
        try:
            # Get base health check
            base_health = await super().health_check()
            
            # Check Ollama if enabled
            ollama_status = None
            if self.use_ollama:
                ollama_status = await self.ollama_generator.health_check()
            
            # Update base health with additional info
            base_health.update({
                'ollama': ollama_status,
                'max_workers': self.max_workers,
                'use_ollama': self.use_ollama
            })
            
            return base_health
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def get_ollama_models(self) -> List[str]:
        """Get available Ollama models."""
        return self.ollama_generator.get_available_models()
    
    async def switch_ollama_model(self, model_name: str) -> dict:
        """Switch to a different Ollama model."""
        try:
            success = self.ollama_generator.switch_model(model_name)
            if success:
                return {
                    "success": True,
                    "message": f"Switched to Ollama model '{model_name}'",
                    "current_model": self.ollama_generator.model_name
                }
            else:
                return {
                    "success": False,
                    "message": f"Model '{model_name}' not found or could not be switched.",
                    "current_model": self.ollama_generator.model_name
                }
        except Exception as e:
            return {
                "success": False,
                "message": str(e),
                "current_model": self.ollama_generator.model_name
            }
    
    async def set_ollama_system_prompt(self, prompt: str) -> Dict[str, Any]:
        """Set a custom system prompt for Ollama."""
        try:
            self.ollama_generator.set_system_prompt(prompt)
            return {
                'success': True,
                'message': 'System prompt updated successfully',
                'prompt': prompt
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def toggle_ollama(self, enable: bool | None = None) -> Dict[str, Any]:
        """Toggle Ollama usage on/off."""
        if enable is None:
            # Toggle current state
            self.use_ollama = not self.use_ollama
        else:
            self.use_ollama = enable
        
        return {
            'success': True,
            'message': f'Ollama {"enabled" if self.use_ollama else "disabled"}',
            'use_ollama': self.use_ollama
        }
    
    def get_current_ollama_model(self) -> str:
        return self.ollama_generator.model_name
    
    async def remove_vector_duplicates(self) -> dict:
        if hasattr(self.vector_store, 'remove_duplicates'):
            return await self.vector_store.remove_duplicates()
        return {"success": False, "message": "Duplicate removal not supported by this vector store."}
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

    def get_persist_directory(self) -> str:
        return self.vector_store.persist_directory

    async def switch_persist_directory(self, persist_directory: str) -> dict:
        # Save current config
        model_name = self.vector_store.get_current_embedding_model()
        threshold = getattr(self.vector_store, 'duplicate_threshold', 0.9)
        collection_name = self.vector_store.collection_name
        # Re-instantiate vector store with new directory
        from fuzzy_vector_store import FuzzyVectorStore
        self.vector_store = FuzzyVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model_name=model_name,
            duplicate_threshold=threshold
        )
        return {"success": True, "persist_directory": persist_directory} 