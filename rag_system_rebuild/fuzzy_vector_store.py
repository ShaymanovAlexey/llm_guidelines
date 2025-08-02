from vector_store import VectorStore
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio
import time
import uuid

class FuzzyVectorStore(VectorStore):
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./chroma_db", embedding_model_name: str = 'all-MiniLM-L6-v2', duplicate_threshold: float = 0.9):
        super().__init__(collection_name, persist_directory, embedding_model_name)
        self.duplicate_threshold = duplicate_threshold

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store, skipping fuzzy duplicates."""
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
        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
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

    async def remove_duplicates(self) -> dict:
        """Remove fuzzy duplicate documents from the collection, keeping only the first occurrence."""
        all_docs = await self.list_documents()
        if not all_docs:
            return {"success": True, "removed": 0, "message": "No documents to process."}
        texts = [doc['text'] for doc in all_docs]
        ids = [doc['id'] for doc in all_docs]
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self.embedding_model.encode, texts)
        embeddings = np.array(embeddings)
        keep_indices = []
        removed_indices = set()
        for i, emb in enumerate(embeddings):
            if i in removed_indices:
                continue
            keep_indices.append(i)
            sims = cosine_similarity([emb], embeddings)[0]
            for j, sim in enumerate(sims):
                if j != i and sim > self.duplicate_threshold:
                    removed_indices.add(j)
        # IDs to remove
        remove_ids = [ids[i] for i in range(len(ids)) if i not in keep_indices]
        # Remove from collection
        if remove_ids:
            await loop.run_in_executor(None, lambda: self.collection.delete(ids=remove_ids))
        return {"success": True, "removed": len(remove_ids), "message": f"Removed {len(remove_ids)} duplicates."} 