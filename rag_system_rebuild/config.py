"""
Configuration file for RAG System Rebuild.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type, List
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM integration."""
    model_name: str = "llama2"
    base_url: str = "http://localhost:11434"
    system_prompt: str = "You are a helpful AI assistant that provides accurate and concise answers based on the given context."
    use_ollama: bool = True


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    persist_directory: str = "rag_storage/vector_embeddings"
    collection_name: str = "documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    duplicate_threshold: float = 0.9


@dataclass
class BM25Config:
    """Configuration for BM25 search."""
    database_path: str = "rag_storage/bm25_database/bm25_documents.db"
    collection_name: str = "documents"
    k1: float = 1.5
    b: float = 0.75


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    max_workers: int = 4
    batch_size: int = 10
    enable_concurrent_processing: bool = True


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    default_k: int = 3
    max_k: int = 10
    similarity_threshold: float = 0.7
    enable_hybrid_search: bool = True
    bm25_weight: float = 0.3
    vector_weight: float = 0.7


@dataclass
class WebInterfaceConfig:
    """Configuration for web interface."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = True
    title: str = "RAG System Rebuild"
    version: str = "2.0.0"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    enable_console: bool = True


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    enable_health_checks: bool = True
    check_interval: int = 30  # seconds
    timeout: int = 10  # seconds


@dataclass
class LangfuseConfig:
    """Configuration for Langfuse observability."""
    enabled: bool = True
    public_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", ""))
    host: str = field(default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    project_name: str = "rag-system-rebuild"
    enable_tracing: bool = True
    enable_scoring: bool = True
    enable_metrics: bool = True
    trace_user_id: str = "default-user"
    trace_session_id: str = "default-session"


@dataclass
class RAGSystemConfig:
    """Main RAG system configuration."""
    # Core components
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    
    # Interface and monitoring
    web_interface: WebInterfaceConfig = field(default_factory=WebInterfaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    
    # System features
    enable_fallback: bool = True
    enable_metrics: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    
    def get_vector_store_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for vector store initialization."""
        return {
            'persist_directory': self.vector_store.persist_directory,
            'collection_name': self.vector_store.collection_name,
            'embedding_model': self.vector_store.embedding_model,
            'duplicate_threshold': self.vector_store.duplicate_threshold
        }
    
    def get_bm25_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for BM25 initialization."""
        return {
            'database_path': self.bm25.database_path,
            'collection_name': self.bm25.collection_name,
            'k1': self.bm25.k1,
            'b': self.bm25.b
        }
    
    def get_ollama_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for Ollama initialization."""
        return {
            'model_name': self.ollama.model_name,
            'base_url': self.ollama.base_url,
            'system_prompt': self.ollama.system_prompt
        }
    
    def get_processing_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for processing configuration."""
        return {
            'max_workers': self.processing.max_workers,
            'batch_size': self.processing.batch_size,
            'enable_concurrent_processing': self.processing.enable_concurrent_processing
        }
    
    def get_search_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for search configuration."""
        return {
            'default_k': self.search.default_k,
            'max_k': self.search.max_k,
            'similarity_threshold': self.search.similarity_threshold,
            'enable_hybrid_search': self.search.enable_hybrid_search,
            'bm25_weight': self.search.bm25_weight,
            'vector_weight': self.search.vector_weight
        }
    
    def get_langfuse_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for Langfuse initialization."""
        return {
            'public_key': self.langfuse.public_key,
            'secret_key': self.langfuse.secret_key,
            'host': self.langfuse.host,
            'project_name': self.langfuse.project_name
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate paths
        if not Path(self.vector_store.persist_directory).parent.exists():
            errors.append(f"Vector store persist directory parent does not exist: {self.vector_store.persist_directory}")
        
        if not Path(self.bm25.database_path).parent.exists():
            errors.append(f"BM25 database directory does not exist: {self.bm25.database_path}")
        
        # Validate numeric ranges
        if self.processing.max_workers < 1:
            errors.append("max_workers must be at least 1")
        
        if self.search.default_k < 1:
            errors.append("default_k must be at least 1")
        
        if not 0 <= self.search.bm25_weight <= 1:
            errors.append("bm25_weight must be between 0 and 1")
        
        if not 0 <= self.search.vector_weight <= 1:
            errors.append("vector_weight must be between 0 and 1")
        
        if abs(self.search.bm25_weight + self.search.vector_weight - 1.0) > 0.01:
            errors.append("bm25_weight + vector_weight should equal 1.0")
        
        if not 0 <= self.vector_store.duplicate_threshold <= 1:
            errors.append("duplicate_threshold must be between 0 and 1")
        
        # Validate URLs
        if not self.ollama.base_url.startswith(('http://', 'https://')):
            errors.append("Ollama base_url must be a valid HTTP/HTTPS URL")
        
        # Validate Langfuse configuration
        if self.langfuse.enabled:
            if not self.langfuse.public_key or not self.langfuse.secret_key:
                errors.append("Langfuse is enabled but credentials are missing")
            if not self.langfuse.host.startswith(('http://', 'https://')):
                errors.append("Langfuse host must be a valid HTTP/HTTPS URL")
        
        return errors
    
    def print_config(self):
        """Print current configuration."""
        print("RAG System Configuration:")
        print("=" * 50)
        print(f"Ollama Model: {self.ollama.model_name}")
        print(f"Ollama URL: {self.ollama.base_url}")
        print(f"Vector Store: {self.vector_store.persist_directory}")
        print(f"BM25 Database: {self.bm25.database_path}")
        print(f"Max Workers: {self.processing.max_workers}")
        print(f"Default K: {self.search.default_k}")
        print(f"Hybrid Search: {self.search.enable_hybrid_search}")
        print(f"BM25 Weight: {self.search.bm25_weight}")
        print(f"Vector Weight: {self.search.vector_weight}")
        print(f"Web Interface: {self.web_interface.host}:{self.web_interface.port}")
        print(f"Log Level: {self.logging.level}")
        print(f"Langfuse Enabled: {self.langfuse.enabled}")
        if self.langfuse.enabled:
            print(f"Langfuse Host: {self.langfuse.host}")
            print(f"Langfuse Project: {self.langfuse.project_name}")
        print("=" * 50)


def get_development_config() -> RAGSystemConfig:
    """Get development configuration."""
    config = RAGSystemConfig()
    config.logging.level = "DEBUG"
    config.web_interface.debug = True
    config.web_interface.reload = True
    return config


def get_production_config() -> RAGSystemConfig:
    """Get production configuration."""
    config = RAGSystemConfig()
    config.logging.level = "WARNING"
    config.web_interface.debug = False
    config.web_interface.reload = False
    config.processing.max_workers = 8
    config.enable_metrics = True
    return config


def get_test_config() -> RAGSystemConfig:
    """Get test configuration."""
    config = RAGSystemConfig()
    config.vector_store.persist_directory = "rag_storage/test_vector_embeddings"
    config.bm25.database_path = "rag_storage/test_bm25_database/test_bm25_documents.db"
    config.logging.level = "DEBUG"
    config.web_interface.debug = True
    config.langfuse.enabled = False  # Disable Langfuse for tests
    return config


def get_config(environment: str = "development") -> RAGSystemConfig:
    """Get configuration based on environment."""
    if environment == "production":
        return get_production_config()
    elif environment == "test":
        return get_test_config()
    else:
        return get_development_config() 