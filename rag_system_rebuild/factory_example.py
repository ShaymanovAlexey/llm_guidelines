"""
Example script demonstrating how to use the RAG System Factory and Model Factory.
"""

import asyncio
from rag_factory import RAGSystemFactory, RAGSystemConfig, RAGSystemType, VectorStoreType, get_config
from model_factory import ModelFactory, ModelConfig, ModelType, ModelProvider, ModelRegistry, get_model_config


def example_rag_system_factory():
    """Example of using the RAG System Factory."""
    print("=== RAG System Factory Examples ===")
    
    # Get available systems
    available_systems = RAGSystemFactory.get_available_systems()
    print("Available RAG Systems:")
    for system_type, description in available_systems.items():
        print(f"  • {system_type}: {description}")
    
    # Get available vector stores
    available_stores = RAGSystemFactory.get_available_vector_stores()
    print("\nAvailable Vector Stores:")
    for store_type, description in available_stores.items():
        print(f"  • {store_type}: {description}")
    
    # Create different RAG system configurations
    print("\n--- Creating Different RAG Systems ---")
    
    # Simple RAG system
    simple_config = RAGSystemConfig(
        system_type=RAGSystemType.SIMPLE,
        vector_store_type=VectorStoreType.STANDARD,
        use_ollama=False
    )
    print(f"Simple RAG Config: {simple_config.system_type.value}")
    
    # Advanced RAG system
    advanced_config = RAGSystemConfig(
        system_type=RAGSystemType.ADVANCED,
        vector_store_type=VectorStoreType.FUZZY,
        use_ollama=True,
        ollama_model="llama2:7b"
    )
    print(f"Advanced RAG Config: {advanced_config.system_type.value}")
    
    # Hybrid RAG system
    hybrid_config = RAGSystemConfig(
        system_type=RAGSystemType.HYBRID,
        vector_store_type=VectorStoreType.FUZZY,
        use_ollama=True,
        enable_hybrid_search=True,
        bm25_weight=0.4,
        vector_weight=0.6
    )
    print(f"Hybrid RAG Config: {hybrid_config.system_type.value}")
    
    # Validate configurations
    print("\n--- Configuration Validation ---")
    for config_name, config in [("Simple", simple_config), ("Advanced", advanced_config), ("Hybrid", hybrid_config)]:
        errors = config.validate()
        if errors:
            print(f"{config_name} config errors: {errors}")
        else:
            print(f"{config_name} config is valid")


def example_model_factory():
    """Example of using the Model Factory."""
    print("\n=== Model Factory Examples ===")
    
    # Get available models
    available_models = ModelFactory.get_available_models()
    print("Available Model Types:")
    for model_type, description in available_models.items():
        print(f"  • {model_type}: {description}")
    
    # Get available providers
    available_providers = ModelFactory.get_available_providers()
    print("\nAvailable Model Providers:")
    for provider, description in available_providers.items():
        print(f"  • {provider}: {description}")
    
    # Create different model configurations
    print("\n--- Creating Different Model Configurations ---")
    
    # Ollama model
    ollama_config = ModelConfig(
        model_type=ModelType.OLLAMA,
        provider=ModelProvider.OLLAMA,
        model_name="llama2:7b",
        temperature=0.7
    )
    print(f"Ollama Model Config: {ollama_config.model_name}")
    
    # Template model
    template_config = ModelConfig(
        model_type=ModelType.TEMPLATE,
        provider=ModelProvider.LOCAL,
        model_name="template"
    )
    print(f"Template Model Config: {template_config.model_name}")
    
    # Validate configurations
    print("\n--- Model Configuration Validation ---")
    for config_name, config in [("Ollama", ollama_config), ("Template", template_config)]:
        errors = config.validate()
        if errors:
            print(f"{config_name} config errors: {errors}")
        else:
            print(f"{config_name} config is valid")


def example_predefined_configs():
    """Example of using predefined configurations."""
    print("\n=== Predefined Configurations ===")
    
    # RAG system configurations
    print("RAG System Configurations:")
    try:
        simple_rag = get_config("simple", "development")
        print(f"  • Simple RAG (dev): {simple_rag.system_type.value}")
        
        advanced_rag = get_config("advanced", "production")
        print(f"  • Advanced RAG (prod): {advanced_rag.system_type.value}, Workers: {advanced_rag.max_workers}")
        
        hybrid_rag = get_config("hybrid", "test")
        print(f"  • Hybrid RAG (test): {hybrid_rag.system_type.value}")
        
    except ValueError as e:
        print(f"  Error: {e}")
    
    # Model configurations
    print("\nModel Configurations:")
    try:
        fast_model = get_model_config("fast")
        print(f"  • Fast Model: {fast_model.model_name}, Temp: {fast_model.temperature}")
        
        accurate_model = get_model_config("accurate")
        print(f"  • Accurate Model: {accurate_model.model_name}, Temp: {accurate_model.temperature}")
        
        creative_model = get_model_config("creative")
        print(f"  • Creative Model: {creative_model.model_name}, Temp: {creative_model.temperature}")
        
    except ValueError as e:
        print(f"  Error: {e}")


def example_model_registry():
    """Example of using the Model Registry."""
    print("\n=== Model Registry Examples ===")
    
    # List registered models
    registered_models = ModelRegistry.list_models()
    print("Registered Models:")
    for name, model_name in registered_models.items():
        print(f"  • {name}: {model_name}")
    
    # Register a custom model
    custom_config = ModelConfig(
        model_type=ModelType.OLLAMA,
        provider=ModelProvider.OLLAMA,
        model_name="llama2:13b",
        temperature=0.3,
        max_tokens=1500
    )
    ModelRegistry.register_model("custom", custom_config)
    print(f"\nRegistered custom model: {custom_config.model_name}")
    
    # Get model from registry
    retrieved_config = ModelRegistry.get_model("custom")
    if retrieved_config:
        print(f"Retrieved custom model: {retrieved_config.model_name}, Temp: {retrieved_config.temperature}")


async def example_rag_system_creation():
    """Example of creating and using RAG systems."""
    print("\n=== RAG System Creation Examples ===")
    
    # Create a simple RAG system
    try:
        simple_config = RAGSystemConfig(
            system_type=RAGSystemType.SIMPLE,
            vector_store_type=VectorStoreType.STANDARD,
            use_ollama=False,
            persist_directory="test_storage/vectors"
        )
        
        print("Creating simple RAG system...")
        # Note: We'll comment out actual creation to avoid errors in example
        # simple_rag = RAGSystemFactory.create_rag_system(simple_config)
        # print(f"Created {type(simple_rag).__name__}")
        
        print("Simple RAG system configuration created successfully")
        
    except Exception as e:
        print(f"Error creating simple RAG system: {e}")
    
    # Create model generator
    try:
        model_config = ModelConfig(
            model_type=ModelType.TEMPLATE,
            provider=ModelProvider.LOCAL,
            model_name="template"
        )
        
        print("Creating template model generator...")
        model_generator = ModelFactory.create_model(model_config)
        print(f"Created {type(model_generator).__name__}")
        
        # Test health check
        health = await model_generator.health_check()
        print(f"Model health: {health['status']}")
        
    except Exception as e:
        print(f"Error creating model generator: {e}")


def example_integration():
    """Example of integrating RAG system and model factories."""
    print("\n=== Integration Examples ===")
    
    # Create a complete RAG system with custom model
    print("Creating integrated RAG system configuration:")
    
    # RAG system config
    rag_config = RAGSystemConfig(
        system_type=RAGSystemType.ADVANCED,
        vector_store_type=VectorStoreType.FUZZY,
        use_ollama=True,
        ollama_model="llama2:7b",
        max_workers=4,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Model config
    model_config = ModelConfig(
        model_type=ModelType.OLLAMA,
        provider=ModelProvider.OLLAMA,
        model_name="llama2:7b",
        temperature=0.7,
        max_tokens=1000,
        system_prompt="You are a helpful AI assistant for document Q&A."
    )
    
    print(f"RAG System: {rag_config.system_type.value}")
    print(f"Vector Store: {rag_config.vector_store_type.value}")
    print(f"Model: {model_config.model_name}")
    print(f"Temperature: {model_config.temperature}")
    print(f"Max Workers: {rag_config.max_workers}")
    
    # Validate both configurations
    rag_errors = rag_config.validate()
    model_errors = model_config.validate()
    
    if not rag_errors and not model_errors:
        print("Both configurations are valid!")
    else:
        if rag_errors:
            print(f"RAG config errors: {rag_errors}")
        if model_errors:
            print(f"Model config errors: {model_errors}")


def example_usage_patterns():
    """Example of common usage patterns."""
    print("\n=== Common Usage Patterns ===")
    
    print("1. Quick setup with defaults:")
    print("   config = get_config('advanced', 'development')")
    print("   rag_system = RAGSystemFactory.create_rag_system(config)")
    
    print("\n2. Custom model with RAG system:")
    print("   model_config = get_model_config('fast')")
    print("   model = ModelFactory.create_model(model_config)")
    
    print("\n3. Environment-specific setup:")
    print("   # Development")
    print("   dev_config = get_config('simple', 'development')")
    print("   # Production")
    print("   prod_config = get_config('hybrid', 'production')")
    
    print("\n4. Model registry usage:")
    print("   ModelRegistry.register_model('my_model', custom_config)")
    print("   model = ModelRegistry.create_model('my_model')")
    
    print("\n5. Dictionary-based configuration:")
    print("   config_dict = {'system_type': 'advanced', 'use_ollama': True}")
    print("   rag_system = RAGSystemFactory.create_from_dict(config_dict)")


async def main():
    """Main function to run all examples."""
    print("RAG System Factory and Model Factory Examples")
    print("=" * 60)
    
    example_rag_system_factory()
    example_model_factory()
    example_predefined_configs()
    example_model_registry()
    await example_rag_system_creation()
    example_integration()
    example_usage_patterns()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main()) 