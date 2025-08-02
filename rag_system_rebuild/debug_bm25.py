import asyncio
import re
from bm25_search import AsyncBM25Search

async def debug_bm25_tokenization():
    """Debug BM25 tokenization to understand why some queries don't work."""
    print("=== BM25 Tokenization Debug ===")
    
    # Initialize BM25 search
    bm25 = AsyncBM25Search("debug_bm25.db")
    
    # Test documents
    test_docs = [
        {
            "title": "AI Document",
            "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.",
            "metadata": {"source": "test"}
        },
        {
            "title": "ML Document", 
            "content": "Machine Learning is a subset of artificial intelligence that provides systems the ability to learn.",
            "metadata": {"source": "test"}
        }
    ]
    
    # Add documents
    for i, doc in enumerate(test_docs):
        await bm25.add_document(
            doc_id=f"debug_doc_{i}",
            title=doc["title"],
            content=doc["content"],
            metadata=doc["metadata"]
        )
    
    # Test tokenization function directly
    def tokenize(text):
        """Copy of the tokenization function from BM25Search."""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        tokens = [token for token in tokens if len(token) > 2]
        return tokens
    
    # Test queries
    test_queries = [
        "artificial intelligence",
        "machine learning", 
        "AI",
        "computer science",
        "intelligent machines"
    ]
    
    print("\n=== Tokenization Analysis ===")
    for query in test_queries:
        tokens = tokenize(query)
        print(f"Query: '{query}' -> Tokens: {tokens}")
        
        # Check if tokens exist in documents
        results = await bm25.search(query, top_k=3)
        print(f"  Results: {len(results)}")
        if results:
            for result in results:
                print(f"    - {result.title} (Score: {result.score:.3f})")
    
    # Check what terms are actually in the index
    print("\n=== Index Analysis ===")
    stats = await bm25.get_statistics()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Unique terms: {stats['unique_terms']}")
    print(f"Total terms: {stats['total_terms']}")
    
    # Get a sample document to see what terms were indexed
    doc = await bm25.get_document("debug_doc_0")
    if doc:
        print(f"\nSample document content: {doc['content']}")
        doc_tokens = tokenize(doc['content'])
        print(f"Document tokens: {doc_tokens}")
    
    # Clean up
    await bm25.clear_index()

async def test_simple_queries():
    """Test with very simple queries to see what works."""
    print("\n=== Simple Query Test ===")
    
    bm25 = AsyncBM25Search("simple_test.db")
    
    # Simple document
    await bm25.add_document(
        doc_id="simple_doc",
        title="Simple AI Doc",
        content="artificial intelligence is a field of computer science",
        metadata={"source": "simple"}
    )
    
    # Test simple queries
    simple_queries = [
        "artificial",
        "intelligence", 
        "computer",
        "science",
        "field",
        "artificial intelligence",
        "computer science"
    ]
    
    for query in simple_queries:
        results = await bm25.search(query, top_k=3)
        print(f"Query: '{query}' -> {len(results)} results")
        if results:
            for result in results:
                print(f"  - {result.title} (Score: {result.score:.3f})")
    
    await bm25.clear_index()

async def main():
    """Run debug tests."""
    print("BM25 Debug Tests")
    print("=" * 30)
    
    await debug_bm25_tokenization()
    await test_simple_queries()
    
    print("\nDebug tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 