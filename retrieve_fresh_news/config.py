"""
Configuration file for news extraction and summary generation.
"""

# Summary Generator Configuration
SUMMARY_CONFIG = {
    # Choose your preferred summary generator:
    # 'simple' - Fast, rule-based (no dependencies)
    # 'ollama' - LLM-based (requires Ollama server)
    # 'huggingface' - Transformer-based (requires transformers)
    'generator_type': 'ollama',
    
    # Summary length in characters
    'max_length': 150,
    
    # Ollama configuration (if using 'ollama' generator)
    'ollama': {
        'model_name': 'llama3.2:latest',
        'base_url': 'http://localhost:11434'
    },
    
    # Hugging Face configuration (if using 'huggingface' generator)
    'huggingface': {
        'model_name': 'facebook/bart-large-cnn'
    }
}

# Vector Store Configuration
VECTOR_STORE_CONFIG = {
    'persist_directory': '..rag_system_rebuild/rag_storage/vector_embeddings',
    'collection_name': 'news_with_summaries_visible',  # Visible collection
    'chunk_size': 1000,
    'chunk_overlap': 200
}

# BM25 Database Configuration
BM25_CONFIG = {
    'database_path': '../rag_system_rebuild/rag_storage/bm25_database/bm25_news.db',
    'collection_name': 'news_documents'
}

# News Sources Configuration
NEWS_SOURCES = {
    'ainvest': {
        'name': 'AI Investment News',
        'base_url': 'https://www.ainvest.com',
        'news_url': 'https://www.ainvest.com/news/articles-latest/',
        'topic': 'AI Investment News'
    },
    'bitcoin': {
        'name': 'Bitcoin News',
        'base_url': 'https://news.bitcoin.com',
        'news_url': 'https://news.bitcoin.com/latest-news',
        'topic': 'Bitcoin News'
    }
}

def get_summary_kwargs():
    """Get keyword arguments for the selected summary generator."""
    config = SUMMARY_CONFIG
    generator_type = config['generator_type']
    
    if generator_type == 'ollama':
        return config['ollama']
    elif generator_type == 'huggingface':
        return config['huggingface']
    else:
        return {}

def print_available_generators():
    """Print available summary generators and their descriptions."""
    from summary_generator import SummaryGeneratorFactory
    
    print("Available Summary Generators:")
    print("=" * 50)
    
    generators = SummaryGeneratorFactory.get_available_generators()
    for gen_type, description in generators.items():
        current = " (CURRENT)" if gen_type == SUMMARY_CONFIG['generator_type'] else ""
        print(f"• {gen_type}{current}")
        print(f"  {description}")
        print()
    
    print(f"Current configuration: {SUMMARY_CONFIG['generator_type']}")
    print(f"Summary length: {SUMMARY_CONFIG['max_length']} characters") 