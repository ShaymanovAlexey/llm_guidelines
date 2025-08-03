import os
import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """Handles vector storage and similarity search using ChromaDB."""
    
    # List of available embedding models
    AVAILABLE_EMBEDDING_MODELS = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'paraphrase-MiniLM-L6-v2',
        'multi-qa-MiniLM-L6-cos-v1',
        'distiluse-base-multilingual-cased-v2'
    ]

    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./chroma_db", embedding_model_name: str = 'all-MiniLM-L6-v2', duplicate_threshold: float = 0.9):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.duplicate_threshold = duplicate_threshold
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Try to get existing collection first, create if it doesn't exist
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    @classmethod
    def get_available_embedding_models(cls) -> List[str]:
        return cls.AVAILABLE_EMBEDDING_MODELS

    def get_current_embedding_model(self) -> str:
        return self.embedding_model_name

    def switch_embedding_model(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self.AVAILABLE_EMBEDDING_MODELS:
            return {"success": False, "message": f"Model '{model_name}' is not available."}
        try:
            self.embedding_model_name = model_name
            self.embedding_model = SentenceTransformer(model_name)
            return {"success": True, "message": f"Switched to embedding model '{model_name}'", "current_model": model_name}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def _generate_summary(self, content: str, max_length: int = 150) -> str:
        """Generate a summary of the content."""
        if not content:
            return ""
        
        # Simple summary generation: take first few sentences
        import re
        sentences = re.split(r'[.!?]+', content.strip())
        summary_parts = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_length + len(sentence) <= max_length:
                summary_parts.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        summary = '. '.join(summary_parts)
        if summary and not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    async def add_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200, topic: str = None) -> None:
        """Add documents to the vector store, splitting into chunks and skipping fuzzy duplicates. Optionally add a topic to metadata."""
        # Split documents into chunks first, with topic
        chunked_documents = self.chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap, topic=topic)
        # Get all existing documents and their embeddings
        existing_docs = await self.list_documents()
        existing_texts = [doc['text'] for doc in existing_docs]
        if existing_texts:
            loop = asyncio.get_event_loop()
            existing_embeddings = await loop.run_in_executor(None, self.embedding_model.encode, existing_texts)
            existing_embeddings = np.array(existing_embeddings)
        else:
            existing_embeddings = np.array([])

        # Prepare new docs and their embeddings
        texts = [doc['text'] for doc in chunked_documents]
        metadatas = [doc['metadata'] for doc in chunked_documents]
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self.embedding_model.encode, texts)
        embeddings = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)  # type: ignore

        # Fuzzy duplicate detection
        unique_indices = []
        for i, emb in enumerate(embeddings):
            is_duplicate = False
            if existing_embeddings.size > 0:
                similarities = cosine_similarity([emb], existing_embeddings)[0]
                if np.max(similarities) > self.duplicate_threshold:
                    is_duplicate = True
            if not is_duplicate:
                unique_indices.append(i)

        # Only add unique docs
        unique_texts = [texts[i] for i in unique_indices]
        unique_metadatas = [metadatas[i] for i in unique_indices]
        unique_embeddings = [embeddings[i] for i in unique_indices]
        timestamp = int(time.time() * 1000)  # milliseconds
        ids = [f"doc_{timestamp}_{i}_{str(uuid.uuid4())[:8]}" for i in range(len(unique_texts))]

        if unique_texts:
            await loop.run_in_executor(None, lambda: self.collection.add(
                embeddings=unique_embeddings,
                documents=unique_texts,
                metadatas=unique_metadatas,
                ids=ids
            ))
    
    async def similarity_search(self, query: str, k: int = 3, topic: str = None, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """Search for similar documents, prioritizing topic similarity first, then document similarity."""
        loop = asyncio.get_event_loop()
        # Generate query embedding asynchronously
        query_embedding = await loop.run_in_executor(None, self.embedding_model.encode, [query])
        query_embedding = query_embedding.tolist()
        # Get all unique topics in the collection
        all_docs = await self.list_documents()
        topic_to_docs = {}
        for doc in all_docs:
            meta_topic = doc['metadata'].get('topic') if doc['metadata'] else None
            if meta_topic:
                topic_to_docs.setdefault(meta_topic, []).append(doc)
        # If there are topics, compute topic similarity
        topic_scores = []
        if topic_to_docs:
            topic_embeddings = await loop.run_in_executor(None, self.embedding_model.encode, list(topic_to_docs.keys()))
            topic_similarities = cosine_similarity(query_embedding, topic_embeddings)[0]
            for i, t in enumerate(topic_to_docs.keys()):
                topic_scores.append((t, topic_similarities[i]))
            # Sort topics by similarity
            topic_scores.sort(key=lambda x: x[1], reverse=True)
        # Gather documents from most similar topics
        results_out = []
        used_doc_ids = set()
        for t, _ in topic_scores:
            docs = topic_to_docs[t]
            # Compute document similarity within this topic
            doc_texts = [d['text'] for d in docs]
            doc_embeddings = await loop.run_in_executor(None, self.embedding_model.encode, doc_texts)
            doc_similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            doc_entries = [
                {
                    'text': docs[i]['text'],
                    'metadata': docs[i]['metadata'],
                    'similarity_score': float(doc_similarities[i]),
                    'id': docs[i].get('id')
                }
                for i in range(len(docs))
            ]
            doc_entries.sort(key=lambda x: x['similarity_score'], reverse=True)
            for entry in doc_entries:
                if entry['id'] not in used_doc_ids:
                    results_out.append(entry)
                    used_doc_ids.add(entry['id'])
                if len(results_out) >= k:
                    break
            if len(results_out) >= k:
                break
        # If not enough, search all documents (regardless of topic)
        if len(results_out) < k:
            remaining_docs = [d for d in all_docs if d.get('id') not in used_doc_ids]
            if remaining_docs:
                doc_texts = [d['text'] for d in remaining_docs]
                doc_embeddings = await loop.run_in_executor(None, self.embedding_model.encode, doc_texts)
                doc_similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
                doc_entries = [
                    {
                        'text': remaining_docs[i]['text'],
                        'metadata': remaining_docs[i]['metadata'],
                        'similarity_score': float(doc_similarities[i]),
                        'id': remaining_docs[i].get('id')
                    }
                    for i in range(len(remaining_docs))
                ]
                doc_entries.sort(key=lambda x: x['similarity_score'], reverse=True)
                for entry in doc_entries:
                    if entry['id'] not in used_doc_ids:
                        results_out.append(entry)
                        used_doc_ids.add(entry['id'])
                    if len(results_out) >= k:
                        break
        # Filter by minimum similarity threshold
        filtered_results = [entry for entry in results_out if entry['similarity_score'] >= min_similarity]
        
        # Remove 'id' from output
        for entry in filtered_results:
            entry.pop('id', None)
        return filtered_results[:k]
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, self.collection.count)
        return {
            'total_documents': count,
            'collection_name': self.collection_name
        }
    
    async def delete_collection(self) -> None:
        """Delete the entire collection."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.client.delete_collection, self.collection_name)
        self.collection = await loop.run_in_executor(None, lambda: self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        ))
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the collection."""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self.collection.get)
        
        documents = []
        if results['documents'] and results['metadatas'] and results['ids']:
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                documents.append({
                    'id': results['ids'][i],
                    'text': doc,
                    'metadata': metadata
                })
        
        return documents
    
    def chunk_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200, topic: str = None) -> List[Dict[str, Any]]:
        """Split each document into chunks by paragraphs, then by sentences if paragraph exceeds chunk_size."""
        chunked_docs = []
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {}).copy() if doc.get('metadata') else {}
            
            # Add timestamp if not present
            if 'timestamp' not in metadata:
                metadata['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                metadata['created_at'] = metadata['timestamp']
            
            # Generate summary if not present
            if 'summary' not in metadata:
                metadata['summary'] = self._generate_summary(text)
            
            # Always set topic: prefer argument, then doc metadata, else 'Unknown'
            meta_topic = topic or metadata.get('topic') or 'UnknownSource'
            metadata['topic'] = meta_topic
            # Preprocess text (remove extra whitespace and special chars except punctuation)
            import re
            cleaned_text = re.sub(r'\s+', ' ', text)
            cleaned_text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', cleaned_text)
            cleaned_text = cleaned_text.strip()
            # Split by paragraphs (double newlines preferred, fallback to single newline)
            if '\n\n' in text:
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            else:
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            chunk_id = 0
            for paragraph in paragraphs:
                if len(paragraph) <= chunk_size:
                    # Paragraph is small enough, keep as one chunk
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_id': chunk_id,
                        'chunk_size': len(paragraph),
                        'total_chunks': len(paragraphs)
                    })
                    chunked_docs.append({'text': paragraph, 'metadata': chunk_metadata})
                    chunk_id += 1
                else:
                    # Paragraph is too long, split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    current_chunk = ""
                    overlap_sentences = []
                    for sentence in sentences:
                        # Add overlap sentences to current chunk
                        if overlap_sentences:
                            current_chunk += " ".join(overlap_sentences) + " "
                        if len(current_chunk) + len(sentence) <= chunk_size:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk.strip():
                                chunk_metadata = metadata.copy()
                                chunk_metadata.update({
                                    'chunk_id': chunk_id,
                                    'chunk_size': len(current_chunk.strip()),
                                    'total_chunks': len(paragraphs)
                                })
                                chunked_docs.append({'text': current_chunk.strip(), 'metadata': chunk_metadata})
                                chunk_id += 1
                            # Prepare overlap for next chunk
                            if chunk_overlap > 0:
                                # Keep last sentence(s) for overlap
                                overlap_sentences = [sentence]
                                current_chunk = ""
                            else:
                                overlap_sentences = []
                                current_chunk = sentence + " "
                    # Add remaining chunk
                    if current_chunk.strip():
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            'chunk_id': chunk_id,
                            'chunk_size': len(current_chunk.strip()),
                            'total_chunks': len(paragraphs)
                        })
                        chunked_docs.append({'text': current_chunk.strip(), 'metadata': chunk_metadata})
                        chunk_id += 1
        return chunked_docs 