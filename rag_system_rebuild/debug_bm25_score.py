import asyncio
import math
from bm25_search import AsyncBM25Search

async def debug_bm25_score():
    """Debug BM25 score calculation step by step."""
    print("=== BM25 Score Calculation Debug ===")
    
    # Initialize BM25 search
    bm25 = AsyncBM25Search("debug_score.db")
    
    # Add a simple document
    await bm25.add_document(
        doc_id="test_doc",
        title="Test Document",
        content="artificial intelligence computer science",
        metadata={"source": "test"}
    )
    
    # Test query
    query = "artificial intelligence"
    print(f"Query: '{query}'")
    
    # Get tokens
    tokens = bm25.bm25._tokenize(query)
    print(f"Tokens: {tokens}")
    
    # Debug each step for the first token
    if tokens:
        term = tokens[0]  # 'artificial'
        doc_id = "test_doc"
        
        print(f"\nDebugging term: '{term}'")
        
        # Check if term exists in term_freq
        if term in bm25.bm25.term_freq:
            print(f"  Term found in term_freq: {bm25.bm25.term_freq[term]}")
            
            # Check if doc_id exists for this term
            if doc_id in bm25.bm25.term_freq[term]:
                tf = bm25.bm25.term_freq[term][doc_id]
                print(f"  Term frequency (tf): {tf}")
                
                # Get document length
                doc_len = bm25.bm25.doc_lengths[doc_id]
                print(f"  Document length: {doc_len}")
                
                # Calculate IDF
                idf = bm25.bm25._calculate_idf(term)
                print(f"  IDF: {idf}")
                
                # Calculate BM25 components
                k1 = bm25.bm25.k1
                b = bm25.bm25.b
                avgdl = bm25.bm25.avgdl
                
                print(f"  BM25 parameters: k1={k1}, b={b}, avgdl={avgdl}")
                
                # Calculate numerator and denominator
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                
                print(f"  Numerator: {tf} * ({k1} + 1) = {numerator}")
                print(f"  Denominator: {tf} + {k1} * (1 - {b} + {b} * ({doc_len} / {avgdl})) = {denominator}")
                
                # Calculate final score for this term
                term_score = idf * (numerator / denominator)
                print(f"  Term score: {idf} * ({numerator} / {denominator}) = {term_score}")
                
                # Calculate full BM25 score
                full_score = bm25.bm25._calculate_bm25_score(doc_id, tokens)
                print(f"  Full BM25 score: {full_score}")
                
            else:
                print(f"  Document {doc_id} not found for term '{term}'")
        else:
            print(f"  Term '{term}' not found in term_freq")
    
    # Test the search function
    print(f"\n=== Testing Search Function ===")
    results = await bm25.search(query, top_k=3)
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"  - {result.title} (Score: {result.score:.6f})")
    
    # Clean up
    await bm25.clear_index()

async def test_multiple_docs():
    """Test with multiple documents to see if IDF calculation works better."""
    print("\n=== Multiple Documents Test ===")
    
    bm25 = AsyncBM25Search("debug_multiple.db")
    
    # Add multiple documents
    docs = [
        ("doc1", "AI Document", "artificial intelligence is a field of computer science"),
        ("doc2", "ML Document", "machine learning is a subset of artificial intelligence"),
        ("doc3", "CS Document", "computer science includes many subfields"),
    ]
    
    for doc_id, title, content in docs:
        await bm25.add_document(doc_id, title, content, metadata={"source": "test"})
        print(f"Added: {title}")
    
    # Test search
    query = "artificial intelligence"
    print(f"\nQuery: '{query}'")
    
    results = await bm25.search(query, top_k=3)
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"  - {result.title} (Score: {result.score:.6f})")
    
    # Clean up
    await bm25.clear_index()

async def main():
    """Run BM25 score debug tests."""
    print("BM25 Score Calculation Debug")
    print("=" * 40)
    
    await debug_bm25_score()
    await test_multiple_docs()
    
    print("\nBM25 score debug completed!")

if __name__ == "__main__":
    asyncio.run(main()) 