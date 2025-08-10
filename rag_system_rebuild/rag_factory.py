"""
RAG System Factory for creating different RAG implementations.
"""

from typing import Dict, Any, Type, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from base_rag_system import BaseRAGSystem
from simple_rag_system import SimpleRAGSystem
from advanced_rag_system import AdvancedRAGSystem
from hybrid_search_system import HybridRAGSystem, HybridSearchSystem
from vector_store import VectorStore
from fuzzy_vector_store import FuzzyVectorStore
from bm25_search import AsyncBM25Search
from ollama_generator import OllamaGenerator


class RAGSystemType(Enum):
    """Enumeration of available RAG system types."""
    SIMPLE = "simple"
    ADVANCED = "advanced"
    HYBRID = "hybrid"


class VectorStoreType(Enum):
    """Enumeration of available vector store types."""
    STANDARD = "standard"
    FUZZY = "fuzzy"


@dataclass
class RAGSystemConfig:
    """Configuration for RAG system instantiation."""
    system_type: RAGSystemType = RAGSystemType.ADVANCED
    vector_store_type: VectorStoreType = VectorStoreType.STANDARD
    
    # Processing configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_workers: int = 4
    
    # Vector store configuration
    persist_directory: str = "rag_storage/vector_embeddings"
    collection_name: str = "documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    duplicate_threshold: float = 0.9
    
    # BM25 configuration
    bm25_database_path: str = "rag_storage/bm25_database/bm25_documents.db"
    bm25_collection_name: str = "documents"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # Ollama configuration
    ollama_model: str = "llama2"
    ollama_url: str = "http://localhost:11434"
    ollama_system_prompt: str = "You are a helpful AI assistant that provides accurate and concise answers based on the given context."
    use_ollama: bool = True
    
    # Search configuration
    default_k: int = 3
    max_k: int = 10
    similarity_threshold: float = 0.7
    enable_hybrid_search: bool = True
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    
    # Hybrid search configuration
    max_context_length: int = 4000
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate system type
        if not isinstance(self.system_type, RAGSystemType):
            errors.append(f"Invalid system_type: {self.system_type}")
        
        # Validate vector store type
        if not isinstance(self.vector_store_type, VectorStoreType):
            errors.append(f"Invalid vector_store_type: {self.vector_store_type}")
        
        # Validate numeric ranges
        if self.chunk_size < 100:
            errors.append("chunk_size must be at least 100")
        
        if self.chunk_overlap < 0:
            errors.append("chunk_overlap must be non-negative")
        
        if self.max_workers < 1:
            errors.append("max_workers must be at least 1")
        
        if self.default_k < 1:
            errors.append("default_k must be at least 1")
        
        if not 0 <= self.bm25_weight <= 1:
            errors.append("bm25_weight must be between 0 and 1")
        
        if not 0 <= self.vector_weight <= 1:
            errors.append("vector_weight must be between 0 and 1")
        
        if abs(self.bm25_weight + self.vector_weight - 1.0) > 0.01:
            errors.append("bm25_weight + vector_weight should equal 1.0")
        
        if not 0 <= self.duplicate_threshold <= 1:
            errors.append("duplicate_threshold must be between 0 and 1")
        
        # Validate URLs
        if not self.ollama_url.startswith(('http://', 'https://')):
            errors.append("ollama_url must be a valid HTTP/HTTPS URL")
        
        return errors


class RAGSystemFactory:
    """Factory for creating RAG system instances."""
    
    _vector_store_classes = {
        VectorStoreType.STANDARD: VectorStore,
        VectorStoreType.FUZZY: FuzzyVectorStore
    }
    
    _rag_system_classes = {
        RAGSystemType.SIMPLE: SimpleRAGSystem,
        RAGSystemType.ADVANCED: AdvancedRAGSystem,
        RAGSystemType.HYBRID: HybridRAGSystem
    }
    
    @classmethod
    def get_available_systems(cls) -> Dict[str, str]:
        """Get available RAG system types and their descriptions."""
        return {
            RAGSystemType.SIMPLE.value: "Simple RAG system with template-based answers",
            RAGSystemType.ADVANCED.value: "Advanced RAG system with concurrent processing and Ollama",
            RAGSystemType.HYBRID.value: "Hybrid RAG system combining vector and BM25 search"
        }
    
    @classmethod
    def get_available_vector_stores(cls) -> Dict[str, str]:
        """Get available vector store types and their descriptions."""
        return {
            VectorStoreType.STANDARD.value: "Standard vector store with ChromaDB",
            VectorStoreType.FUZZY.value: "Fuzzy vector store with duplicate detection"
        }
    
    @classmethod
    def create_vector_store(cls, config: RAGSystemConfig) -> VectorStore:
        """Create vector store instance based on configuration."""
        vector_store_class = cls._vector_store_classes[config.vector_store_type]
        
        vector_store_kwargs = {
            'persist_directory': config.persist_directory,
            'collection_name': config.collection_name,
            'embedding_model': config.embedding_model
        }
        
        # Add fuzzy-specific parameters
        if config.vector_store_type == VectorStoreType.FUZZY:
            vector_store_kwargs['duplicate_threshold'] = config.duplicate_threshold
        
        return vector_store_class(**vector_store_kwargs)
    
    @classmethod
    def create_bm25_store(cls, config: RAGSystemConfig) -> AsyncBM25Search:
        """Create BM25 store instance based on configuration."""
        return AsyncBM25Search(
            database_path=config.bm25_database_path,
            collection_name=config.bm25_collection_name,
            k1=config.bm25_k1,
            b=config.bm25_b
        )
    
    @classmethod
    def create_ollama_generator(cls, config: RAGSystemConfig) -> Optional[OllamaGenerator]:
        """Create Ollama generator instance based on configuration."""
        if not config.use_ollama:
            return None
        
        return OllamaGenerator(
            model_name=config.ollama_model,
            base_url=config.ollama_url,
            system_prompt=config.ollama_system_prompt
        )
    
    @classmethod
    def create_rag_system(cls, config: RAGSystemConfig) -> BaseRAGSystem:
        """Create RAG system instance based on configuration."""
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
        
        # Create vector store
        vector_store = cls.create_vector_store(config)
        
        if config.system_type == RAGSystemType.SIMPLE:
            return cls._create_simple_rag_system(config, vector_store)
        
        elif config.system_type == RAGSystemType.ADVANCED:
            return cls._create_advanced_rag_system(config, vector_store)
        
        elif config.system_type == RAGSystemType.HYBRID:
            return cls._create_hybrid_rag_system(config, vector_store)
        
        else:
            raise ValueError(f"Unsupported RAG system type: {config.system_type}")
    
    @classmethod
    def _create_simple_rag_system(cls, config: RAGSystemConfig, vector_store: VectorStore) -> SimpleRAGSystem:
        """Create simple RAG system."""
        return SimpleRAGSystem(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            vector_store_class=type(vector_store),
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model
        )
    
    @classmethod
    def _create_advanced_rag_system(cls, config: RAGSystemConfig, vector_store: VectorStore) -> AdvancedRAGSystem:
        """Create advanced RAG system."""
        return AdvancedRAGSystem(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_workers=config.max_workers,
            ollama_model=config.ollama_model,
            ollama_url=config.ollama_url,
            vector_store_class=type(vector_store),
            duplicate_threshold=config.duplicate_threshold
        )
    
    @classmethod
    def _create_hybrid_rag_system(cls, config: RAGSystemConfig, vector_store: VectorStore) -> HybridRAGSystem:
        """Create hybrid RAG system."""
        # Create BM25 store
        bm25_store = cls.create_bm25_store(config)
        
        # Create hybrid search system
        hybrid_search = HybridSearchSystem(
            vector_store=vector_store,
            bm25_store=bm25_store,
            vector_weight=config.vector_weight,
            bm25_weight=config.bm25_weight
        )
        
        # Create Ollama generator
        generator = cls.create_ollama_generator(config)
        
        return HybridRAGSystem(
            hybrid_search=hybrid_search,
            generator=generator,
            max_context_length=config.max_context_length
        )
    
    @classmethod
    def create_from_dict(cls, config_dict: Dict[str, Any]) -> BaseRAGSystem:
        """Create RAG system from dictionary configuration."""
        # Convert string values to enums
        if 'system_type' in config_dict and isinstance(config_dict['system_type'], str):
            config_dict['system_type'] = RAGSystemType(config_dict['system_type'])
        
        if 'vector_store_type' in config_dict and isinstance(config_dict['vector_store_type'], str):
            config_dict['vector_store_type'] = VectorStoreType(config_dict['vector_store_type'])
        
        # Create configuration object
        config = RAGSystemConfig(**config_dict)
        
        # Create and return RAG system
        return cls.create_rag_system(config)


# Predefined configurations
def get_simple_config() -> RAGSystemConfig:
    """Get configuration for simple RAG system."""
    return RAGSystemConfig(
        system_type=RAGSystemType.SIMPLE,
        vector_store_type=VectorStoreType.STANDARD,
        use_ollama=False
    )


def get_advanced_config() -> RAGSystemConfig:
    """Get configuration for advanced RAG system."""
    return RAGSystemConfig(
        system_type=RAGSystemType.ADVANCED,
        vector_store_type=VectorStoreType.FUZZY,
        use_ollama=True
    )


def get_hybrid_config() -> RAGSystemConfig:
    """Get configuration for hybrid RAG system."""
    return RAGSystemConfig(
        system_type=RAGSystemType.HYBRID,
        vector_store_type=VectorStoreType.FUZZY,
        use_ollama=True,
        enable_hybrid_search=True,
        bm25_weight=0.3,
        vector_weight=0.7
    )


def get_development_config() -> RAGSystemConfig:
    """Get configuration optimized for development."""
    config = get_advanced_config()
    config.max_workers = 2
    config.chunk_size = 800
    config.chunk_overlap = 100
    return config


def get_production_config() -> RAGSystemConfig:
    """Get configuration optimized for production."""
    config = get_hybrid_config()
    config.max_workers = 8
    config.chunk_size = 1200
    config.chunk_overlap = 200
    config.default_k = 5
    return config


def get_test_config() -> RAGSystemConfig:
    """Get configuration optimized for testing."""
    config = get_simple_config()
    config.persist_directory = "test_storage/vector_embeddings"
    config.bm25_database_path = "test_storage/bm25_database/test.db"
    config.use_ollama = False
    return config


# Configuration factory
def get_config(system_type: str = "advanced", environment: str = "development") -> RAGSystemConfig:
    """Get configuration for specified system type and environment."""
    # System type configurations
    system_configs = {
        "simple": get_simple_config,
        "advanced": get_advanced_config,
        "hybrid": get_hybrid_config
    }
    
    # Environment modifiers
    env_modifiers = {
        "development": get_development_config,
        "production": get_production_config,
        "test": get_test_config
    }
    
    if system_type not in system_configs:
        raise ValueError(f"Unknown system type: {system_type}. Available: {list(system_configs.keys())}")
    
    if environment not in env_modifiers:
        raise ValueError(f"Unknown environment: {environment}. Available: {list(env_modifiers.keys())}")
    
    # Get base configuration for system type
    config = system_configs[system_type]()
    
    # Apply environment-specific modifications
    env_config = env_modifiers[environment]()
    
    # Merge configurations (environment takes precedence)
    for field in config.__dataclass_fields__:
        env_value = getattr(env_config, field)
        if env_value is not None:
            setattr(config, field, env_value)
    
    return config 