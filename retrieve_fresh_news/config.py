"""
Configuration file for news extraction and summary generation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM integration."""
    model_name: str = "llama3.2:latest"
    base_url: str = "http://localhost:11434"


@dataclass
class HuggingFaceConfig:
    """Configuration for Hugging Face transformers."""
    model_name: str = "facebook/bart-large-cnn"


@dataclass
class SummaryConfig:
    """Configuration for summary generation."""
    generator_type: str = "ollama"  # 'simple', 'ollama', 'huggingface'
    max_length: int = 150
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    persist_directory: str = "../rag_system_rebuild/rag_storage/vector_embeddings"
    collection_name: str = "news_with_summaries_visible"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class BM25Config:
    """Configuration for BM25 database."""
    database_path: str = "../rag_system_rebuild/rag_storage/bm25_database/bm25_news.db"
    collection_name: str = "news_documents"


@dataclass
class NewsSource:
    """Configuration for a single news source."""
    name: str
    base_url: str
    news_url: str
    topic: str


@dataclass
class NewsSourcesConfig:
    """Configuration for all news sources."""
    ainvest: NewsSource = field(default_factory=lambda: NewsSource(
        name="AI Investment News",
        base_url="https://www.ainvest.com",
        news_url="https://www.ainvest.com/news/articles-latest/",
        topic="AI Investment News"
    ))
    bitcoin: NewsSource = field(default_factory=lambda: NewsSource(
        name="Bitcoin News",
        base_url="https://news.bitcoin.com",
        news_url="https://news.bitcoin.com/latest-news",
        topic="Bitcoin News"
    ))


@dataclass
class AppConfig:
    """Main application configuration."""
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    news_sources: NewsSourcesConfig = field(default_factory=NewsSourcesConfig)
    
    def get_summary_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for the selected summary generator."""
        generator_type = self.summary.generator_type
        
        if generator_type == 'ollama':
            return {
                'model_name': self.summary.ollama.model_name,
                'base_url': self.summary.ollama.base_url
            }
        elif generator_type == 'huggingface':
            return {
                'model_name': self.summary.huggingface.model_name
            }
        else:
            return {}
    
    def print_available_generators(self):
        """Print available summary generators and their descriptions."""
        from summary_generator import SummaryGeneratorFactory
        
        print("Available Summary Generators:")
        print("=" * 50)
        
        generators = SummaryGeneratorFactory.get_available_generators()
        for gen_type, description in generators.items():
            current = " (CURRENT)" if gen_type == self.summary.generator_type else ""
            print(f"• {gen_type}{current}")
            print(f"  {description}")
            print()
        
        print(f"Current configuration: {self.summary.generator_type}")
        print(f"Summary length: {self.summary.max_length} characters")


# Create default configuration instance
config = AppConfig()

# Backward compatibility - keep the old dictionary-based configs
SUMMARY_CONFIG = {
    'generator_type': config.summary.generator_type,
    'max_length': config.summary.max_length,
    'ollama': {
        'model_name': config.summary.ollama.model_name,
        'base_url': config.summary.ollama.base_url
    },
    'huggingface': {
        'model_name': config.summary.huggingface.model_name
    }
}

VECTOR_STORE_CONFIG = {
    'persist_directory': config.vector_store.persist_directory,
    'collection_name': config.vector_store.collection_name,
    'chunk_size': config.vector_store.chunk_size,
    'chunk_overlap': config.vector_store.chunk_overlap
}

BM25_CONFIG = {
    'database_path': config.bm25.database_path,
    'collection_name': config.bm25.collection_name
}

NEWS_SOURCES = {
    'ainvest': {
        'name': config.news_sources.ainvest.name,
        'base_url': config.news_sources.ainvest.base_url,
        'news_url': config.news_sources.ainvest.news_url,
        'topic': config.news_sources.ainvest.topic
    },
    'bitcoin': {
        'name': config.news_sources.bitcoin.name,
        'base_url': config.news_sources.bitcoin.base_url,
        'news_url': config.news_sources.bitcoin.news_url,
        'topic': config.news_sources.bitcoin.topic
    }
}


def get_summary_kwargs():
    """Get keyword arguments for the selected summary generator."""
    return config.get_summary_kwargs()


def print_available_generators():
    """Print available summary generators and their descriptions."""
    config.print_available_generators() 