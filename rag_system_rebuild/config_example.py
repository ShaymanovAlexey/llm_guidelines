"""
Example script demonstrating how to use the new dataclass-based configuration for RAG System.
"""

from config import RAGSystemConfig, config, get_config, get_development_config, get_production_config, get_test_config


def example_basic_usage():
    """Example of basic configuration usage."""
    print("=== Basic Configuration Usage ===")
    
    # Access configuration values
    print(f"Ollama model: {config.ollama.model_name}")
    print(f"Ollama URL: {config.ollama.base_url}")
    print(f"Vector store path: {config.vector_store.persist_directory}")
    print(f"BM25 database path: {config.bm25.database_path}")
    print(f"Max workers: {config.processing.max_workers}")
    print(f"Default K: {config.search.default_k}")
    print(f"Web interface: {config.web_interface.host}:{config.web_interface.port}")
    
    # Get keyword arguments for components
    ollama_kwargs = config.get_ollama_kwargs()
    vector_kwargs = config.get_vector_store_kwargs()
    bm25_kwargs = config.get_bm25_kwargs()
    
    print(f"\nOllama kwargs: {ollama_kwargs}")
    print(f"Vector store kwargs: {vector_kwargs}")
    print(f"BM25 kwargs: {bm25_kwargs}")


def example_custom_config():
    """Example of creating custom configuration."""
    print("\n=== Custom Configuration ===")
    
    # Import the nested dataclass types
    from config import OllamaConfig, VectorStoreConfig, ProcessingConfig, SearchConfig
    
    # Create a custom configuration
    custom_config = RAGSystemConfig(
        ollama=OllamaConfig(
            model_name="llama3.2:latest",
            base_url="http://localhost:11434",
            system_prompt="You are a specialized AI assistant for technical documentation."
        ),
        vector_store=VectorStoreConfig(
            persist_directory="custom_storage/vectors",
            embedding_model="all-mpnet-base-v2",
            chunk_size=1500,
            chunk_overlap=300
        ),
        processing=ProcessingConfig(
            max_workers=8,
            batch_size=20,
            enable_concurrent_processing=True
        ),
        search=SearchConfig(
            default_k=5,
            max_k=15,
            enable_hybrid_search=True,
            bm25_weight=0.4,
            vector_weight=0.6
        )
    )
    
    print(f"Custom Ollama model: {custom_config.ollama.model_name}")
    print(f"Custom vector store path: {custom_config.vector_store.persist_directory}")
    print(f"Custom embedding model: {custom_config.vector_store.embedding_model}")
    print(f"Custom max workers: {custom_config.processing.max_workers}")
    print(f"Custom default K: {custom_config.search.default_k}")
    print(f"Custom weights - BM25: {custom_config.search.bm25_weight}, Vector: {custom_config.search.vector_weight}")


def example_environment_configs():
    """Example of environment-specific configurations."""
    print("\n=== Environment-Specific Configurations ===")
    
    # Development configuration
    dev_config = get_development_config()
    print("Development config:")
    print(f"  Log level: {dev_config.logging.level}")
    print(f"  Debug mode: {dev_config.web_interface.debug}")
    print(f"  Auto reload: {dev_config.web_interface.reload}")
    print(f"  Metrics enabled: {dev_config.enable_metrics}")
    
    # Production configuration
    prod_config = get_production_config()
    print("\nProduction config:")
    print(f"  Log level: {prod_config.logging.level}")
    print(f"  Debug mode: {prod_config.web_interface.debug}")
    print(f"  Auto reload: {prod_config.web_interface.reload}")
    print(f"  Metrics enabled: {prod_config.enable_metrics}")
    print(f"  Max workers: {prod_config.processing.max_workers}")
    
    # Test configuration
    test_config = get_test_config()
    print("\nTest config:")
    print(f"  Log level: {test_config.logging.level}")
    print(f"  Vector store path: {test_config.vector_store.persist_directory}")
    print(f"  BM25 database path: {test_config.bm25.database_path}")
    print(f"  Web interface port: {test_config.web_interface.port}")
    print(f"  Caching enabled: {test_config.enable_caching}")


def example_config_validation():
    """Example of configuration validation."""
    print("\n=== Configuration Validation ===")
    
    # Validate default configuration
    errors = config.validate()
    if errors:
        print("Validation errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Default configuration is valid")
    
    # Create an invalid configuration for demonstration
    from config import ProcessingConfig, SearchConfig
    
    invalid_config = RAGSystemConfig(
        processing=ProcessingConfig(max_workers=0),
        search=SearchConfig(
            default_k=0,
            bm25_weight=0.8,
            vector_weight=0.1  # Doesn't sum to 1.0
        )
    )
    
    errors = invalid_config.validate()
    if errors:
        print("\nInvalid configuration errors:")
        for error in errors:
            print(f"  - {error}")


def example_advanced_usage():
    """Example of advanced configuration usage."""
    print("\n=== Advanced Configuration Usage ===")
    
    # Using the configuration factory
    try:
        dev_config = get_config("development")
        prod_config = get_config("production")
        test_config = get_config("test")
        
        print("Successfully created environment-specific configs")
        print(f"  Development port: {dev_config.web_interface.port}")
        print(f"  Production port: {prod_config.web_interface.port}")
        print(f"  Test port: {test_config.web_interface.port}")
        
        # Try invalid environment
        invalid_config = get_config("invalid")
    except ValueError as e:
        print(f"Expected error for invalid environment: {e}")
    
    # Print full configuration
    print("\nFull configuration details:")
    config.print_config()


def example_integration_with_rag_system():
    """Example of how to integrate configuration with RAG system."""
    print("\n=== RAG System Integration Example ===")
    
    # This shows how you would use the configuration with the actual RAG system
    print("Example of initializing AdvancedRAGSystem with config:")
    print("```python")
    print("from advanced_rag_system import AdvancedRAGSystem")
    print("from config import config")
    print()
    print("# Initialize with configuration")
    print("rag_system = AdvancedRAGSystem(")
    print(f"    max_workers={config.processing.max_workers},")
    print(f"    ollama_model='{config.ollama.model_name}',")
    print(f"    ollama_url='{config.ollama.base_url}',")
    print(f"    chunk_size={config.vector_store.chunk_size},")
    print(f"    chunk_overlap={config.vector_store.chunk_overlap}")
    print(")")
    print("```")
    
    print("\nExample of initializing FastAPI with config:")
    print("```python")
    print("import uvicorn")
    print("from config import config")
    print()
    print("# Start server with configuration")
    print("uvicorn.run(")
    print("    'main:app',")
    print(f"    host='{config.web_interface.host}',")
    print(f"    port={config.web_interface.port},")
    print(f"    debug={config.web_interface.debug},")
    print(f"    reload={config.web_interface.reload}")
    print(")")
    print("```")


if __name__ == "__main__":
    example_basic_usage()
    example_custom_config()
    example_environment_configs()
    example_config_validation()
    example_advanced_usage()
    example_integration_with_rag_system() 