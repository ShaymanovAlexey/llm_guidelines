#!/usr/bin/env python3
"""
Debug script for vector store issues
"""

import asyncio
from vector_store import VectorStore

async def debug_vector_store():
    print("=== Debugging Vector Store ===\n")
    
    # Create a fresh vector store
    vector_store = VectorStore(collection_name="debug_test")
    
    print("1. Initial Collection Stats:")
    stats = await vector_store.get_collection_stats()
    print(f"   📊 {stats}")
    
    print("\n2. Adding Test Document:")
    test_doc = [{
        'id': 'test_doc_1',
        'text': 'This is a test document about artificial intelligence and machine learning.',
        'metadata': {'source': 'test', 'topic': 'AI'}
    }]
    
    try:
        await vector_store.add_documents(test_doc)
        print("   ✅ Document added successfully")
    except Exception as e:
        print(f"   ❌ Error adding document: {e}")
        return
    
    print("\n3. Collection Stats After Addition:")
    stats = await vector_store.get_collection_stats()
    print(f"   📊 {stats}")
    
    print("\n4. Listing All Documents:")
    try:
        all_docs = await vector_store.list_documents()
        print(f"   📝 Found {len(all_docs)} documents:")
        for i, doc in enumerate(all_docs):
            print(f"      Doc {i+1}: {doc}")
    except Exception as e:
        print(f"   ❌ Error listing documents: {e}")
    
    print("\n5. Testing Similarity Search:")
    try:
        results = await vector_store.similarity_search("artificial intelligence", k=3)
        print(f"   🔍 Search results: {len(results)} found")
        for i, result in enumerate(results):
            print(f"      Result {i+1}: {result}")
    except Exception as e:
        print(f"   ❌ Error in similarity search: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n6. Testing Direct ChromaDB Query:")
    try:
        # Try to get documents directly from ChromaDB
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, vector_store.collection.count)
        print(f"   🗄️  ChromaDB collection count: {count}")
        
        if count > 0:
            # Try to get all documents from ChromaDB
            all_chroma_docs = await loop.run_in_executor(None, lambda: vector_store.collection.get())
            print(f"   📋 ChromaDB documents: {all_chroma_docs}")
        else:
            print("   ⚠️  ChromaDB collection is empty")
            
    except Exception as e:
        print(f"   ❌ Error querying ChromaDB directly: {e}")
    
    print("\n7. Testing Embedding Generation:")
    try:
        test_text = "This is a test text for embedding generation."
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, vector_store.embedding_model.encode, [test_text])
        print(f"   🔢 Embedding generated successfully")
        print(f"   📐 Embedding shape: {embedding.shape}")
        print(f"   🔢 Embedding dimension: {embedding.shape[1]}")
    except Exception as e:
        print(f"   ❌ Error generating embedding: {e}")

if __name__ == "__main__":
    asyncio.run(debug_vector_store()) 