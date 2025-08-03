import re
import asyncio
import os
import sys
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

# Add the rag_system_rebuild path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag_system_rebuild')))

class SummaryGenerator(ABC):
    """Abstract base class for summary generators."""
    
    @abstractmethod
    async def generate_summary(self, content: str, max_length: int = 150) -> str:
        """Generate a summary from the given content."""
        pass

class SimpleSummaryGenerator(SummaryGenerator):
    """Simple rule-based summary generator using first sentences."""
    
    async def generate_summary(self, content: str, max_length: int = 150) -> str:
        """Generate a summary by taking the first few sentences."""
        if not content:
            return ""
        
        # Split by sentence boundaries
        sentences = re.split(r'[.!?]\s+', content.strip())
        summary_parts = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add space for the period and space that will be added
            sentence_length = len(sentence) + 2
            
            if current_length + sentence_length <= max_length:
                summary_parts.append(sentence)
                current_length += sentence_length
            else:
                # If this is the first sentence and it's too long, truncate it
                if i == 0 and len(sentence) > max_length:
                    truncated = sentence[:max_length-3] + "..."
                    summary_parts.append(truncated)
                break
        
        summary = '. '.join(summary_parts)
        if summary and not summary.endswith('.'):
            summary += '.'
        
        return summary

class OllamaSummaryGenerator(SummaryGenerator):
    """Summary generator using Ollama LLM."""
    
    def __init__(self, model_name: str = "llama3.2:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._client = None
    
    async def _get_client(self):
        """Get or create the Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError("Ollama Python client not installed. Run: pip install ollama")
        return self._client
    
    async def generate_summary(self, content: str, max_length: int = 150) -> str:
        """Generate a summary using Ollama LLM."""
        if not content:
            return ""
        
        try:
            client = await self._get_client()
            
            # Create a prompt for summarization
            prompt = f"""Please provide a concise summary of the following text in {max_length} characters or less:

{content[:2000]}  # Limit input to avoid token limits

Summary:"""
            
            # Generate summary
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            
            summary = response['message']['content'].strip()
            
            # Ensure summary doesn't exceed max_length
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return summary
            
        except Exception as e:
            print(f"Error generating summary with Ollama: {e}")
            # Fallback to simple summary
            simple_gen = SimpleSummaryGenerator()
            return await simple_gen.generate_summary(content, max_length)

class HuggingFaceSummaryGenerator(SummaryGenerator):
    """Summary generator using Hugging Face transformers."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
    
    async def _load_model(self):
        """Load the Hugging Face model and tokenizer."""
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqGeneration.from_pretrained(self.model_name)
                
                if torch.cuda.is_available():
                    self._model = self._model.to('cuda')
                    
            except ImportError:
                raise ImportError("Transformers not installed. Run: pip install transformers torch")
    
    async def generate_summary(self, content: str, max_length: int = 150) -> str:
        """Generate a summary using Hugging Face model."""
        if not content:
            return ""
        
        try:
            await self._load_model()
            
            # Tokenize input
            inputs = self._tokenizer(content[:1024], return_tensors="pt", max_length=1024, truncation=True)
            
            # Generate summary
            summary_ids = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._model.generate(
                    inputs["input_ids"],
                    max_length=max_length//4,  # Approximate token count
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            )
            
            # Decode summary
            summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return summary.strip()
            
        except Exception as e:
            print(f"Error generating summary with Hugging Face: {e}")
            # Fallback to simple summary
            simple_gen = SimpleSummaryGenerator()
            return await simple_gen.generate_summary(content, max_length)

class SummaryGeneratorFactory:
    """Factory for creating summary generators."""
    
    GENERATORS = {
        'simple': SimpleSummaryGenerator,
        'ollama': OllamaSummaryGenerator,
        'huggingface': HuggingFaceSummaryGenerator
    }
    
    @classmethod
    def create(cls, generator_type: str, **kwargs) -> SummaryGenerator:
        """Create a summary generator of the specified type."""
        if generator_type not in cls.GENERATORS:
            raise ValueError(f"Unknown generator type: {generator_type}. Available: {list(cls.GENERATORS.keys())}")
        
        return cls.GENERATORS[generator_type](**kwargs)
    
    @classmethod
    def get_available_generators(cls) -> Dict[str, str]:
        """Get available generator types and their descriptions."""
        return {
            'simple': 'Rule-based summary using first sentences (fast, no dependencies)',
            'ollama': 'LLM-based summary using Ollama (requires Ollama server)',
            'huggingface': 'Transformer-based summary using Hugging Face models (requires transformers)'
        }

# Default generator instance
_default_generator = SimpleSummaryGenerator()

async def generate_summary(content: str, max_length: int = 150, generator_type: str = 'simple', **kwargs) -> str:
    """Convenience function to generate summaries."""
    if generator_type == 'simple':
        return await _default_generator.generate_summary(content, max_length)
    else:
        generator = SummaryGeneratorFactory.create(generator_type, **kwargs)
        return await generator.generate_summary(content, max_length) 