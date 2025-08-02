import asyncio
import sqlite3
import json
from bm25_search import AsyncBM25Search

async def debug_bm25_detailed():
    """Detailed debug of BM25 search to find the issue."""
    print("=== Detailed BM25 Debug ===")
    
    # Initialize BM25 search
    bm25 = AsyncBM25Search("debug_detailed.db")
    
    # Add a simple document
    await bm25.add_document(
        doc_id="test_doc",
        title="Test Document",
        content="artificial intelligence computer science",
        metadata={"source": "test"}
    )
    
    # Check the database directly
    print("\n=== Database Contents ===")
    with sqlite3.connect("debug_detailed.db") as conn:
        cursor = conn.cursor()
        
        # Check documents table
        print("Documents table:")
        cursor.execute('SELECT * FROM documents')
        for row in cursor.fetchall():
            print(f"  {row}")
        
        # Check terms table
        print("\nTerms table:")
        cursor.execute('SELECT * FROM terms')
        for row in cursor.fetchall():
            print(f"  {row}")
        
        # Check doc_frequency table
        print("\nDoc frequency table:")
        cursor.execute('SELECT * FROM doc_frequency')
        for row in cursor.fetchall():
            print(f"  {row}")
        
        # Check statistics table
        print("\nStatistics table:")
        cursor.execute('SELECT * FROM statistics')
        for row in cursor.fetchall():
            print(f"  {row}")
    
    # Check in-memory structures
    print("\n=== In-Memory Structures ===")
    print(f"Documents: {len(bm25.bm25.documents)}")
    print(f"Doc lengths: {len(bm25.bm25.doc_lengths)}")
    print(f"Term freq: {len(bm25.bm25.term_freq)}")
    print(f"Doc freq: {len(bm25.bm25.doc_freq)}")
    print(f"Total docs: {bm25.bm25.total_docs}")
    print(f"Avg doc length: {bm25.bm25.avgdl}")
    
    # Print some sample data
    if bm25.bm25.documents:
        print("\nSample document:")
        doc_id = list(bm25.bm25.documents.keys())[0]
        doc = bm25.bm25.documents[doc_id]
        print(f"  ID: {doc_id}")
        print(f"  Title: {doc['title']}")
        print(f"  Content: {doc['content']}")
        print(f"  Length: {doc['length']}")
    
    if bm25.bm25.term_freq:
        print("\nSample term frequencies:")
        for term, docs in list(bm25.bm25.term_freq.items())[:5]:
            print(f"  '{term}': {docs}")
    
    if bm25.bm25.doc_freq:
        print("\nSample document frequencies:")
        for term, freq in list(bm25.bm25.doc_freq.items())[:5]:
            print(f"  '{term}': {freq}")
    
    # Test search with debug info
    print("\n=== Search Debug ===")
    query = "artificial intelligence"
    print(f"Query: '{query}'")
    
    # Check tokenization
    tokens = bm25.bm25._tokenize(query)
    print(f"Tokens: {tokens}")
    
    # Check if tokens exist in term_freq
    for token in tokens:
        if token in bm25.bm25.term_freq:
            print(f"  Token '{token}' found in term_freq: {bm25.bm25.term_freq[token]}")
        else:
            print(f"  Token '{token}' NOT found in term_freq")
    
    # Check if tokens exist in doc_freq
    for token in tokens:
        if token in bm25.bm25.doc_freq:
            print(f"  Token '{token}' found in doc_freq: {bm25.bm25.doc_freq[token]}")
        else:
            print(f"  Token '{token}' NOT found in doc_freq")
    
    # Test search
    results = await bm25.search(query, top_k=3)
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"  - {result.title} (Score: {result.score:.3f})")
    
    # Clean up
    await bm25.clear_index()

async def test_fresh_bm25():
    """Test with a completely fresh BM25 instance."""
    print("\n=== Fresh BM25 Test ===")
    
    # Create a new instance
    bm25 = AsyncBM25Search("fresh_test.db")
    
    # Add document
    await bm25.add_document(
        doc_id="fresh_doc",
        title="Fresh Document",
        content="artificial intelligence is a field of computer science",
        metadata={"source": "fresh"}
    )
    
    # Test search immediately
    results = await bm25.search("artificial", top_k=3)
    print(f"Search for 'artificial': {len(results)} results")
    for result in results:
        print(f"  - {result.title} (Score: {result.score:.3f})")
    
    results = await bm25.search("intelligence", top_k=3)
    print(f"Search for 'intelligence': {len(results)} results")
    for result in results:
        print(f"  - {result.title} (Score: {result.score:.3f})")
    
    results = await bm25.search("artificial intelligence", top_k=3)
    print(f"Search for 'artificial intelligence': {len(results)} results")
    for result in results:
        print(f"  - {result.title} (Score: {result.score:.3f})")
    
    # Clean up
    await bm25.clear_index()

async def main():
    """Run detailed debug tests."""
    print("Detailed BM25 Debug Tests")
    print("=" * 40)
    
    await debug_bm25_detailed()
    await test_fresh_bm25()
    
    print("\nDetailed debug tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 