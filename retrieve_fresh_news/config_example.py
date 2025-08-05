"""
Example script demonstrating how to use the new dataclass-based configuration.
"""

from config import AppConfig, config


def example_basic_usage():
    """Example of basic configuration usage."""
    print("=== Basic Configuration Usage ===")
    
    # Access configuration values
    print(f"Summary generator type: {config.summary.generator_type}")
    print(f"Summary max length: {config.summary.max_length}")
    print(f"Ollama model: {config.summary.ollama.model_name}")
    print(f"Vector store path: {config.vector_store.persist_directory}")
    
    # Get keyword arguments for components
    summary_kwargs = config.get_summary_kwargs()
    print(f"Summary kwargs: {summary_kwargs}")


def example_custom_config():
    """Example of creating custom configuration."""
    print("\n=== Custom Configuration ===")
    
    # Create a custom configuration
    custom_config = AppConfig(
        summary=config.summary.__class__(
            generator_type="huggingface",
            max_length=200,
            huggingface=config.summary.huggingface.__class__(
                model_name="facebook/bart-large-xsum"
            )
        ),
        vector_store=config.vector_store.__class__(
            persist_directory="custom_storage/vectors",
            chunk_size=1500,
            chunk_overlap=300
        )
    )
    
    print(f"Custom generator type: {custom_config.summary.generator_type}")
    print(f"Custom max length: {custom_config.summary.max_length}")
    print(f"Custom vector store path: {custom_config.vector_store.persist_directory}")
    print(f"Custom chunk size: {custom_config.vector_store.chunk_size}")


def example_news_sources():
    """Example of working with news sources configuration."""
    print("\n=== News Sources Configuration ===")
    
    # Access news sources
    for source_name, source_config in config.news_sources.__dict__.items():
        if hasattr(source_config, 'name'):
            print(f"Source: {source_name}")
            print(f"  Name: {source_config.name}")
            print(f"  Base URL: {source_config.base_url}")
            print(f"  News URL: {source_config.news_url}")
            print(f"  Topic: {source_config.topic}")
            print()


def example_backward_compatibility():
    """Example showing backward compatibility with old dictionary format."""
    print("\n=== Backward Compatibility ===")
    
    # Import the old-style configs
    from config import SUMMARY_CONFIG, VECTOR_STORE_CONFIG, BM25_CONFIG, NEWS_SOURCES
    
    print("Old-style SUMMARY_CONFIG:")
    print(f"  Generator type: {SUMMARY_CONFIG['generator_type']}")
    print(f"  Max length: {SUMMARY_CONFIG['max_length']}")
    
    print("\nOld-style VECTOR_STORE_CONFIG:")
    print(f"  Persist directory: {VECTOR_STORE_CONFIG['persist_directory']}")
    print(f"  Collection name: {VECTOR_STORE_CONFIG['collection_name']}")
    
    print("\nOld-style NEWS_SOURCES:")
    for source_name, source_config in NEWS_SOURCES.items():
        print(f"  {source_name}: {source_config['name']}")


def example_validation():
    """Example of configuration validation."""
    print("\n=== Configuration Validation ===")
    
    # The configuration should be valid by default
    print("Default configuration is valid")
    
    # You could add validation methods to the dataclass if needed
    # For example, checking if paths exist, validating URLs, etc.


if __name__ == "__main__":
    example_basic_usage()
    example_custom_config()
    example_news_sources()
    example_backward_compatibility()
    example_validation()
    
    print("\n=== Configuration Summary ===")
    config.print_available_generators() 