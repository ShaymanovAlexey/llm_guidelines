#!/usr/bin/env python3
"""
Find where the news_with_summaries collection is stored.
"""

import asyncio
import sys
import os
import chromadb
from chromadb.config import Settings

# Add the rag_system_rebuild path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag_system_rebuild')))

from vector_store import VectorStore
from config import VECTOR_STORE_CONFIG

async def find_collection():
    """Find where the news_with_summaries collection is stored."""
    
    print("Finding news_with_summaries Collection")
    print("=" * 50)
    
    # Method 1: Direct ChromaDB client
    print("\n1. Checking with ChromaDB client:")
    try:
        client = chromadb.PersistentClient(
            path='../rag_system_rebuild/chroma_db',
            settings=Settings(anonymized_telemetry=False)
        )
        
        collections = client.list_collections()
        print(f"Found {len(collections)} collections:")
        for col in collections:
            print(f"  - {col.name} ({col.count()} documents)")
            
        # Try to get the specific collection
        try:
            collection = client.get_collection("news_with_summaries")
            print(f"\n✅ Found news_with_summaries collection with {collection.count()} documents")
        except Exception as e:
            print(f"\n❌ Could not get news_with_summaries collection: {e}")
            
    except Exception as e:
        print(f"Error with ChromaDB client: {e}")
    
    # Method 2: Our VectorStore wrapper
    print("\n2. Checking with our VectorStore wrapper:")
    try:
        store = VectorStore(
            collection_name=VECTOR_STORE_CONFIG['collection_name'],
            persist_directory=VECTOR_STORE_CONFIG['persist_directory']
        )
        
        stats = await store.get_collection_stats()
        print(f"✅ VectorStore can access collection: {stats['collection_name']}")
        print(f"   Total documents: {stats['total_documents']}")
        
        # List some documents
        docs = await store.list_documents()
        print(f"   Retrieved {len(docs)} documents")
        
        if docs:
            metadata = docs[0].get('metadata', {})
            print(f"   Sample document: {metadata.get('title', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Error with VectorStore: {e}")
    
    # Method 3: Check different persist directories
    print("\n3. Checking different persist directories:")
    directories = [
        '../rag_system_rebuild/chroma_db',
        '../rag_system_rebuild/chroma_db/rag_documents',
        './chroma_db',
        './chroma_db/rag_documents'
    ]
    
    for directory in directories:
        try:
            client = chromadb.PersistentClient(
                path=directory,
                settings=Settings(anonymized_telemetry=False)
            )
            collections = client.list_collections()
            print(f"  {directory}: {len(collections)} collections")
            for col in collections:
                print(f"    - {col.name} ({col.count()} documents)")
        except Exception as e:
            print(f"  {directory}: Error - {e}")

if __name__ == "__main__":
    asyncio.run(find_collection()) 