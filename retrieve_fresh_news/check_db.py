#!/usr/bin/env python3
"""
Check if the news_with_summaries collection exists and has documents.
"""

import asyncio
import sys
import os

# Add the rag_system_rebuild path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag_system_rebuild')))

from vector_store import VectorStore
from config import VECTOR_STORE_CONFIG

async def check_database():
    """Check the database status."""
    
    print("Checking Database Status")
    print("=" * 50)
    
    try:
        # Try to create/access the collection
        store = VectorStore(
            collection_name=VECTOR_STORE_CONFIG['collection_name'],
            persist_directory=VECTOR_STORE_CONFIG['persist_directory']
        )
        
        # Get stats
        stats = await store.get_collection_stats()
        print(f"Collection: {stats['collection_name']}")
        print(f"Total documents: {stats['total_documents']}")
        
        # List documents
        docs = await store.list_documents()
        print(f"Retrieved {len(docs)} documents")
        
        if docs:
            print("\nFirst 3 documents:")
            for i, doc in enumerate(docs[:3], 1):
                metadata = doc.get('metadata', {})
                print(f"{i}. Title: {metadata.get('title', 'N/A')}")
                print(f"   Summary: {metadata.get('summary', 'N/A')[:100]}...")
                print()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_database()) 