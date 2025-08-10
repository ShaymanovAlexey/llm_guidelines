#!/usr/bin/env python3
"""
Simple Langfuse Example with Ollama
Demonstrates basic usage of Langfuse for LLM observability and tracing using local Ollama models.
"""

import asyncio
import time
from typing import Dict, Any
from langfuse import Langfuse
from langfuse.model import CreateTrace, CreateSpan, CreateGeneration
import ollama
from config import (
    LANGFUSE_PUBLIC_KEY, 
    LANGFUSE_SECRET_KEY, 
    LANGFUSE_HOST,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    PROJECT_NAME
)

class LangfuseOllamaExample:
    def __init__(self):
        """Initialize Langfuse client and Ollama client."""
        # Initialize Langfuse client
        self.langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        
        # Initialize Ollama client
        ollama.set_host(OLLAMA_HOST)
        
        print(f"✅ Langfuse client initialized for project: {PROJECT_NAME}")
        print(f"✅ Ollama client initialized with host: {OLLAMA_HOST}")
        print(f"✅ Using model: {OLLAMA_MODEL}")
    
    def call_ollama(self, prompt: str, model: str = None) -> str:
        """Make a call to Ollama and return the response."""
        try:
            model_to_use = model or OLLAMA_MODEL
            response = ollama.chat(model=model_to_use, messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            return response['message']['content']
        except Exception as e:
            print(f"⚠️  Ollama call failed: {e}")
            return f"This is a simulated response for demonstration purposes. (Ollama error: {e})"
    
    def create_simple_trace(self, user_input: str) -> str:
        """
        Create a simple trace with a single generation using Ollama.
        This demonstrates the most basic usage of Langfuse.
        """
        print(f"\n🔍 Creating simple trace for: '{user_input}'")
        
        # Create a trace
        trace = self.langfuse.trace(
            name="simple-ollama-call",
            user_id="demo-user",
            metadata={"example_type": "simple_trace", "llm_provider": "ollama"}
        )
        
        try:
            # Simulate LLM call with timing
            start_time = time.time()
            
            # Create a generation span
            generation = trace.generation(
                name=f"ollama-{OLLAMA_MODEL}",
                model=OLLAMA_MODEL,
                model_parameters={"temperature": 0.7, "max_tokens": 150},
                prompt=user_input,
                completion="",  # Will be updated after the call
                metadata={"provider": "ollama", "local": True}
            )
            
            # Make the actual Ollama call
            response = self.call_ollama(user_input)
            
            # Update the generation with the actual response
            generation.update(completion=response)
            
            # End the generation
            generation.end()
            
            # End the trace
            trace.end()
            
            elapsed_time = time.time() - start_time
            
            print(f"✅ Simple trace completed in {elapsed_time:.2f}s")
            print(f"📝 Response: {response[:100]}...")
            return "Simple trace completed successfully"
            
        except Exception as e:
            print(f"❌ Error in simple trace: {e}")
            trace.end(status_message=str(e))
            return f"Error: {e}"
    
    def create_complex_trace(self, user_query: str) -> str:
        """
        Create a more complex trace with multiple spans and generations using Ollama.
        This demonstrates advanced tracing capabilities.
        """
        print(f"\n🔍 Creating complex trace for: '{user_query}'")
        
        # Create a trace
        trace = self.langfuse.trace(
            name="complex-rag-pipeline-ollama",
            user_id="demo-user",
            metadata={"example_type": "complex_trace", "pipeline": "rag", "llm_provider": "ollama"}
        )
        
        try:
            # Span 1: Document retrieval
            retrieval_span = trace.span(
                name="document-retrieval",
                input={"query": user_query, "search_type": "vector_search"},
                metadata={"component": "retriever"}
            )
            
            # Simulate retrieval time
            time.sleep(0.5)
            retrieved_docs = [
                {"title": "Document 1", "content": "Relevant content about the query"},
                {"title": "Document 2", "content": "Additional relevant information"}
            ]
            
            retrieval_span.end(
                output={"retrieved_documents": retrieved_docs, "count": len(retrieved_docs)},
                metadata={"retrieval_score": 0.85}
            )
            
            # Span 2: Context preparation
            context_span = trace.span(
                name="context-preparation",
                input={"documents": retrieved_docs, "query": user_query},
                metadata={"component": "context_builder"}
            )
            
            time.sleep(0.3)
            context = f"Context: {retrieved_docs[0]['content']} {retrieved_docs[1]['content']}"
            
            context_span.end(
                output={"prepared_context": context, "context_length": len(context)},
                metadata={"context_quality": "high"}
            )
            
            # Span 3: LLM generation with Ollama
            generation_span = trace.generation(
                name=f"ollama-{OLLAMA_MODEL}",
                model=OLLAMA_MODEL,
                model_parameters={"temperature": 0.3, "max_tokens": 200},
                prompt=f"Based on this context: {context}\n\nAnswer this question: {user_query}",
                completion="",  # Will be updated after the call
                metadata={"provider": "ollama", "local": True, "pipeline": "rag"}
            )
            
            # Make the actual Ollama call
            response = self.call_ollama(f"Based on this context: {context}\n\nAnswer this question: {user_query}")
            
            # Update the generation with the actual response
            generation_span.update(completion=response)
            
            generation_span.end()
            
            # End the trace
            trace.end(
                output={"final_response": response, "pipeline_success": True},
                metadata={"total_documents": len(retrieved_docs), "pipeline_type": "rag"}
            )
            
            print("✅ Complex trace completed successfully")
            print(f"📝 RAG Response: {response[:100]}...")
            return "Complex trace with multiple spans completed"
            
        except Exception as e:
            print(f"❌ Error in complex trace: {e}")
            trace.end(status_message=str(e))
            return f"Error: {e}"
    
    def create_score_trace(self, user_input: str) -> str:
        """
        Create a trace with scoring and evaluation using Ollama.
        This demonstrates how to track quality metrics.
        """
        print(f"\n🔍 Creating scored trace for: '{user_input}'")
        
        # Create a trace
        trace = self.langfuse.trace(
            name="scored-ollama-response",
            user_id="demo-user",
            metadata={"example_type": "scored_trace", "llm_provider": "ollama"}
        )
        
        try:
            # Create generation
            generation = trace.generation(
                name=f"ollama-{OLLAMA_MODEL}",
                model=OLLAMA_MODEL,
                model_parameters={"temperature": 0.5, "max_tokens": 100},
                prompt=user_input,
                completion="",  # Will be updated after the call
                metadata={"provider": "ollama", "local": True}
            )
            
            # Make the actual Ollama call
            response = self.call_ollama(user_input)
            
            # Update the generation with the actual response
            generation.update(completion=response)
            
            # Add scores
            generation.score(
                name="relevance",
                value=0.85,
                comment="Response is relevant to the query"
            )
            
            generation.score(
                name="helpfulness",
                value=0.78,
                comment="Response provides helpful information"
            )
            
            generation.score(
                name="accuracy",
                value=0.92,
                comment="Response appears accurate based on context"
            )
            
            generation.end()
            trace.end()
            
            print("✅ Scored trace completed successfully")
            print(f"📝 Response: {response[:100]}...")
            return "Trace with scoring completed"
            
        except Exception as e:
            print(f"❌ Error in scored trace: {e}")
            trace.end(status_message=str(e))
            return f"Error: {e}"
    
    def create_model_comparison_trace(self, user_input: str) -> str:
        """
        Create a trace comparing different Ollama models.
        This demonstrates how to track model performance differences.
        """
        print(f"\n🔍 Creating model comparison trace for: '{user_input}'")
        
        # Create a trace
        trace = self.langfuse.trace(
            name="ollama-model-comparison",
            user_id="demo-user",
            metadata={"example_type": "model_comparison", "llm_provider": "ollama"}
        )
        
        try:
            # List of models to compare
            models_to_test = ["llama2", "mistral", "codellama"]
            responses = {}
            
            for model in models_to_test:
                try:
                    # Create generation for each model
                    generation = trace.generation(
                        name=f"ollama-{model}",
                        model=model,
                        model_parameters={"temperature": 0.3, "max_tokens": 150},
                        prompt=user_input,
                        completion="",  # Will be updated after the call
                        metadata={"provider": "ollama", "local": True, "comparison": True}
                    )
                    
                    # Make the actual Ollama call
                    response = self.call_ollama(user_input, model)
                    responses[model] = response
                    
                    # Update the generation with the actual response
                    generation.update(completion=response)
                    
                    # Add a simple score based on response length
                    response_length_score = min(1.0, len(response) / 100)  # Normalize to 0-1
                    generation.score(
                        name="response_length",
                        value=response_length_score,
                        comment=f"Response length: {len(response)} characters"
                    )
                    
                    generation.end()
                    
                except Exception as e:
                    print(f"⚠️  Model {model} failed: {e}")
                    responses[model] = f"Error: {e}"
            
            # End the trace
            trace.end(
                output={"model_responses": responses, "models_tested": list(responses.keys())},
                metadata={"comparison_success": True, "total_models": len(responses)}
            )
            
            print("✅ Model comparison trace completed successfully")
            for model, response in responses.items():
                print(f"📝 {model}: {response[:80]}...")
            return "Model comparison trace completed"
            
        except Exception as e:
            print(f"❌ Error in model comparison trace: {e}")
            trace.end(status_message=str(e))
            return f"Error: {e}"
    
    def run_all_examples(self):
        """Run all example traces to demonstrate different capabilities."""
        print("🚀 Starting Langfuse + Ollama Examples")
        print("=" * 60)
        
        # Example 1: Simple trace
        self.create_simple_trace("What is machine learning?")
        
        # Example 2: Complex trace
        self.create_complex_trace("Explain the benefits of renewable energy")
        
        # Example 3: Scored trace
        self.create_score_trace("How does photosynthesis work?")
        
        # Example 4: Model comparison
        self.create_model_comparison_trace("Write a short poem about coding")
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("📊 Check your Langfuse dashboard to see the traces")
        print("🤖 All LLM calls were made using local Ollama models")
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'langfuse'):
            self.langfuse.flush()  # Ensure all data is sent
            print("🧹 Langfuse client cleaned up")

def main():
    """Main function to run the examples."""
    # Check if credentials are available
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        print("❌ Langfuse credentials not found!")
        print("Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your .env file")
        print("You can get these from: https://cloud.langfuse.com")
        return
    
    # Check if Ollama is accessible
    try:
        import ollama
        ollama.list()  # Test connection
        print("✅ Ollama connection successful")
    except Exception as e:
        print("⚠️  Ollama connection failed. Make sure Ollama is running:")
        print("   ollama serve")
        print(f"   Error: {e}")
        return
    
    # Create and run examples
    example = LangfuseOllamaExample()
    
    try:
        example.run_all_examples()
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        example.cleanup()

if __name__ == "__main__":
    main() 