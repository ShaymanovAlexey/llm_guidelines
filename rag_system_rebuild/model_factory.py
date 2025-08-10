"""
Model Factory for managing different LLM models and generators.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from ollama_generator import OllamaGenerator


class ModelType(Enum):
    """Enumeration of available model types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    ANTHROPIC = "anthropic"
    TEMPLATE = "template"


class ModelProvider(Enum):
    """Enumeration of available model providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """Configuration for LLM model instantiation."""
    model_type: ModelType = ModelType.OLLAMA
    provider: ModelProvider = ModelProvider.OLLAMA
    
    # Model identification
    model_name: str = "llama2"
    model_version: Optional[str] = None
    
    # API configuration
    api_key: Optional[str] = None
    base_url: str = "http://localhost:11434"
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # System prompt
    system_prompt: str = "You are a helpful AI assistant that provides accurate and concise answers based on the given context."
    
    # Model-specific parameters
    context_length: Optional[int] = None
    embedding_model: Optional[str] = None
    
    # Connection settings
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Cost and rate limiting
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate model type
        if not isinstance(self.model_type, ModelType):
            errors.append(f"Invalid model_type: {self.model_type}")
        
        # Validate provider
        if not isinstance(self.provider, ModelProvider):
            errors.append(f"Invalid provider: {self.provider}")
        
        # Validate numeric ranges
        if not 0 <= self.temperature <= 2:
            errors.append("temperature must be between 0 and 2")
        
        if not 0 <= self.top_p <= 1:
            errors.append("top_p must be between 0 and 1")
        
        if self.max_tokens is not None and self.max_tokens < 1:
            errors.append("max_tokens must be at least 1")
        
        if self.timeout < 1:
            errors.append("timeout must be at least 1 second")
        
        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")
        
        # Validate URLs
        if not self.base_url.startswith(('http://', 'https://')):
            errors.append("base_url must be a valid HTTP/HTTPS URL")
        
        # Provider-specific validation
        if self.provider == ModelProvider.OPENAI and not self.api_key:
            errors.append("OpenAI provider requires api_key")
        
        if self.provider == ModelProvider.ANTHROPIC and not self.api_key:
            errors.append("Anthropic provider requires api_key")
        
        return errors


class BaseModelGenerator(ABC):
    """Abstract base class for model generators."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def generate_with_context(self, question: str, context: str, **kwargs) -> str:
        """Generate answer with context."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check model health."""
        pass


class OllamaModelGenerator(BaseModelGenerator):
    """Ollama model generator implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.generator = OllamaGenerator(
            model_name=config.model_name,
            base_url=config.base_url,
            system_prompt=config.system_prompt
        )
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama."""
        try:
            response = await self.generator.generate(prompt, **kwargs)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def generate_with_context(self, question: str, context: str, **kwargs) -> str:
        """Generate answer with context using Ollama."""
        try:
            full_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            response = await self.generator.generate(full_prompt, **kwargs)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama model health."""
        try:
            # Try to generate a simple response
            response = await self.generate("Hello", max_tokens=10)
            return {
                'status': 'healthy',
                'provider': 'ollama',
                'model': self.config.model_name,
                'response_time': 'fast'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'provider': 'ollama',
                'model': self.config.model_name,
                'error': str(e)
            }


class TemplateModelGenerator(BaseModelGenerator):
    """Template-based model generator for fallback."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate template-based response."""
        return f"Based on the information provided: {prompt[:100]}..."
    
    async def generate_with_context(self, question: str, context: str, **kwargs) -> str:
        """Generate template-based answer with context."""
        context_summary = context[:200] + "..." if len(context) > 200 else context
        return f"Based on the available context: {context_summary}\n\nQuestion: {question}\n\nAnswer: This is a template-based response. For more detailed answers, please use an LLM model."
    
    async def health_check(self) -> Dict[str, Any]:
        """Template generator is always healthy."""
        return {
            'status': 'healthy',
            'provider': 'template',
            'model': 'template',
            'response_time': 'instant'
        }


class ModelFactory:
    """Factory for creating model generators."""
    
    _generator_classes = {
        ModelType.OLLAMA: OllamaModelGenerator,
        ModelType.TEMPLATE: TemplateModelGenerator
    }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get available model types and their descriptions."""
        return {
            ModelType.OLLAMA.value: "Ollama local LLM models",
            ModelType.OPENAI.value: "OpenAI GPT models (requires API key)",
            ModelType.ANTHROPIC.value: "Anthropic Claude models (requires API key)",
            ModelType.HUGGINGFACE.value: "Hugging Face transformer models",
            ModelType.TEMPLATE.value: "Template-based fallback generator"
        }
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, str]:
        """Get available model providers and their descriptions."""
        return {
            ModelProvider.OLLAMA.value: "Ollama local server",
            ModelProvider.OPENAI.value: "OpenAI API",
            ModelProvider.ANTHROPIC.value: "Anthropic API",
            ModelProvider.HUGGINGFACE.value: "Hugging Face models",
            ModelProvider.LOCAL.value: "Local model files"
        }
    
    @classmethod
    def create_model(cls, config: ModelConfig) -> BaseModelGenerator:
        """Create model generator instance based on configuration."""
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Model configuration validation failed: {errors}")
        
        # Get generator class
        generator_class = cls._generator_classes.get(config.model_type)
        if not generator_class:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        # Create and return generator
        return generator_class(config)
    
    @classmethod
    def create_from_dict(cls, config_dict: Dict[str, Any]) -> BaseModelGenerator:
        """Create model generator from dictionary configuration."""
        # Convert string values to enums
        if 'model_type' in config_dict and isinstance(config_dict['model_type'], str):
            config_dict['model_type'] = ModelType(config_dict['model_type'])
        
        if 'provider' in config_dict and isinstance(config_dict['provider'], str):
            config_dict['provider'] = ModelProvider(config_dict['provider'])
        
        # Create configuration object
        config = ModelConfig(**config_dict)
        
        # Create and return model generator
        return cls.create_model(config)


# Predefined model configurations
def get_ollama_config(model_name: str = "llama2") -> ModelConfig:
    """Get configuration for Ollama model."""
    return ModelConfig(
        model_type=ModelType.OLLAMA,
        provider=ModelProvider.OLLAMA,
        model_name=model_name,
        base_url="http://localhost:11434",
        system_prompt="You are a helpful AI assistant that provides accurate and concise answers based on the given context."
    )


def get_openai_config(model_name: str = "gpt-3.5-turbo") -> ModelConfig:
    """Get configuration for OpenAI model."""
    return ModelConfig(
        model_type=ModelType.OPENAI,
        provider=ModelProvider.OPENAI,
        model_name=model_name,
        base_url="https://api.openai.com/v1",
        temperature=0.7,
        max_tokens=1000,
        system_prompt="You are a helpful AI assistant that provides accurate and concise answers based on the given context."
    )


def get_anthropic_config(model_name: str = "claude-3-sonnet-20240229") -> ModelConfig:
    """Get configuration for Anthropic model."""
    return ModelConfig(
        model_type=ModelType.ANTHROPIC,
        provider=ModelProvider.ANTHROPIC,
        model_name=model_name,
        base_url="https://api.anthropic.com",
        temperature=0.7,
        max_tokens=1000,
        system_prompt="You are a helpful AI assistant that provides accurate and concise answers based on the given context."
    )


def get_template_config() -> ModelConfig:
    """Get configuration for template-based model."""
    return ModelConfig(
        model_type=ModelType.TEMPLATE,
        provider=ModelProvider.LOCAL,
        model_name="template",
        system_prompt="Template-based response generator"
    )


def get_fast_config() -> ModelConfig:
    """Get configuration for fast local model."""
    return ModelConfig(
        model_type=ModelType.OLLAMA,
        provider=ModelProvider.OLLAMA,
        model_name="llama2:7b",
        temperature=0.5,
        max_tokens=500,
        timeout=15
    )


def get_accurate_config() -> ModelConfig:
    """Get configuration for accurate model."""
    return ModelConfig(
        model_type=ModelType.OLLAMA,
        provider=ModelProvider.OLLAMA,
        model_name="llama2:13b",
        temperature=0.3,
        max_tokens=1000,
        timeout=60
    )


def get_creative_config() -> ModelConfig:
    """Get configuration for creative model."""
    return ModelConfig(
        model_type=ModelType.OLLAMA,
        provider=ModelProvider.OLLAMA,
        model_name="llama2:7b",
        temperature=0.9,
        max_tokens=800,
        top_p=0.95
    )


# Model configuration factory
def get_model_config(model_type: str = "ollama", model_name: str = "llama2") -> ModelConfig:
    """Get model configuration for specified type and name."""
    # Model type configurations
    model_configs = {
        "ollama": get_ollama_config,
        "openai": get_openai_config,
        "anthropic": get_anthropic_config,
        "template": get_template_config
    }
    
    # Special configurations
    special_configs = {
        "fast": get_fast_config,
        "accurate": get_accurate_config,
        "creative": get_creative_config
    }
    
    if model_type in model_configs:
        config = model_configs[model_type]()
        if model_name != config.model_name:
            config.model_name = model_name
        return config
    
    elif model_type in special_configs:
        return special_configs[model_type]()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_configs.keys()) + list(special_configs.keys())}")


# Model registry for easy access
class ModelRegistry:
    """Registry for managing model configurations."""
    
    _models: Dict[str, ModelConfig] = {}
    
    @classmethod
    def register_model(cls, name: str, config: ModelConfig):
        """Register a model configuration."""
        cls._models[name] = config
    
    @classmethod
    def get_model(cls, name: str) -> Optional[ModelConfig]:
        """Get registered model configuration."""
        return cls._models.get(name)
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """List all registered models."""
        return {name: config.model_name for name, config in cls._models.items()}
    
    @classmethod
    def create_model(cls, name: str) -> Optional[BaseModelGenerator]:
        """Create model generator from registered configuration."""
        config = cls.get_model(name)
        if config:
            return ModelFactory.create_model(config)
        return None


# Register common models
ModelRegistry.register_model("default", get_ollama_config())
ModelRegistry.register_model("fast", get_fast_config())
ModelRegistry.register_model("accurate", get_accurate_config())
ModelRegistry.register_model("creative", get_creative_config())
ModelRegistry.register_model("template", get_template_config()) 