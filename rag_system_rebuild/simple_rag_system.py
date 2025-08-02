from typing import List, Dict, Any
from base_rag_system import BaseRAGSystem


class SimpleRAGSystem(BaseRAGSystem):
    """Simple RAG system with basic functionality."""
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents to the RAG system."""
        try:
            # Process documents into chunks
            all_chunks = []
            for doc in documents:
                if doc['type'] == 'file':
                    chunks = self.document_processor.process_file(doc['content'], doc['filename'])
                else:
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
    
    async def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        try:
            # Retrieve relevant documents
            retrieved_docs = await self.vector_store.similarity_search(question, k)
            
            if not retrieved_docs:
                return {
                    'success': False,
                    'message': 'No relevant documents found',
                    'answer': 'I could not find any relevant information to answer your question.',
                    'context': []
                }
            
            # Create context from retrieved documents
            context = []
            for doc in retrieved_docs:
                context.append({
                    'text': doc['text'],
                    'source': doc['metadata'].get('source', 'Unknown'),
                    'similarity_score': doc['similarity_score']
                })
            
            # Generate answer using template-based approach
            answer = self._generate_answer(question, context)
            
            return {
                'success': True,
                'question': question,
                'answer': answer,
                'context': context,
                'sources': list(set([doc['source'] for doc in context]))
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing query: {str(e)}',
                'error': str(e)
            } 