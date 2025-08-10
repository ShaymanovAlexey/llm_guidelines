import asyncio
import json
from typing import List, Dict, Any, Optional
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage, SystemMessage


class OllamaGenerator:
    """Ollama-based answer generator for the RAG system."""
    
    def __init__(self, model_name: str = "llama3.2:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.ollama = OllamaLLM(
            model=model_name,
            base_url=base_url
        )
        
        # Default system prompt for RAG
        self.system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
        Always use the context information to answer questions. If the context doesn't contain enough information 
        to answer the question, say so clearly. Be concise but comprehensive in your answers.
        
        Guidelines:
        - Use only the information provided in the context
        - If you're not sure about something, say so
        - Be helpful and accurate
        - Cite sources when possible
        """
    
    async def generate_answer(self, question: str, context: List[Dict[str, Any]], max_tokens: Optional[int] = None) -> str:
        """Generate an answer using Ollama based on the question and context.
        Args:
            question: The user question.
            context: List of context dicts.
            max_tokens: Maximum number of tokens to generate (None for unlimited).
        """
        try:
            if not context:
                # No context available, ask user if they want to proceed without context
                prompt = f"""I don't have enough specific information in my knowledge base to answer your question: "{question}"

Would you like me to provide a general answer based on my training data without using specific context? 

If yes, I can attempt to answer your question using my general knowledge, but please note that this may not be as accurate or up-to-date as information from your specific documents.

Please respond with 'yes' if you want me to proceed, or 'no' if you'd prefer to rephrase your question or add more relevant documents to the knowledge base."""

                # Generate the question using Ollama
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self._call_ollama, prompt, max_tokens)
                
                if "yes" in response.lower():
                    # User agreed, provide answer without context
                    general_prompt = f"""Based on my general knowledge, here's what I can tell you about: {question}

Please note that this is general information and may not be specific to your particular context or the most up-to-date information available."""
                    
                    general_response = await loop.run_in_executor(None, self._call_ollama, general_prompt, max_tokens)
                    return general_response.strip()
                else:
                    return "I understand. Please rephrase your question or add more relevant documents to the knowledge base for a more accurate answer."
            
            # Prepare context text
            context_text = self._prepare_context(context)
            
            # Create the prompt
            prompt = f"""Context Information:\n{context_text}\n\nQuestion: {question}\n\nPlease answer the question based on the context information provided above. If the context doesn't contain enough information to answer the question, please say so clearly."""

            # Generate answer using Ollama
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._call_ollama, prompt, max_tokens)
            
            return response.strip()
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _call_ollama(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Call Ollama model synchronously."""
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            # Pass num_predict to OllamaLLM.invoke if max_tokens is set
            if max_tokens is not None:
                response = self.ollama.invoke(messages, num_predict=max_tokens)
            else:
                response = self.ollama.invoke(messages)
            return response
        
        except Exception as e:
            # Fallback to direct API call if langchain fails
            return self._fallback_api_call(prompt, max_tokens)
    
    def _fallback_api_call(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Fallback method using direct Ollama API."""
        try:
            import requests
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            if max_tokens is not None:
                payload["num_predict"] = max_tokens
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                return f"API Error: {response.status_code}"
        
        except Exception as e:
            return f"Fallback API Error: {str(e)}"
    
    def _prepare_context(self, context: List[Dict[str, Any]]) -> str:
        """Prepare context text for the prompt."""
        context_parts = []
        
        for i, ctx in enumerate(context, 1):
            source = ctx.get('metadata', {}).get('topic') or ctx.get('source') or 'UnknownALL'
            similarity = ctx.get('similarity_score', 0)
            text = ctx.get('text', '')
            
            context_parts.append(f"Source {i} ({source}, similarity: {similarity:.2f}):\n{text}\n")
        
        return "\n".join(context_parts)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if Ollama is available and the model is loaded."""
        try:
            import requests
            
            # Check if Ollama server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                if self.model_name in model_names:
                    return {
                        "status": "healthy",
                        "model": self.model_name,
                        "available_models": model_names,
                        "base_url": self.base_url
                    }
                else:
                    return {
                        "status": "model_not_found",
                        "model": self.model_name,
                        "available_models": model_names,
                        "base_url": self.base_url
                    }
            else:
                return {
                    "status": "server_error",
                    "error": f"HTTP {response.status_code}",
                    "base_url": self.base_url
                }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "base_url": self.base_url
            }
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set a custom system prompt."""
        self.system_prompt = prompt
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            import requests
            
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model.get("name", "") for model in models]
            return []
        
        except Exception:
            return []
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different Ollama model."""
        try:
            available_models = self.get_available_models()
            if model_name in available_models:
                self.model_name = model_name
                self.ollama = OllamaLLM(
                    model=model_name,
                    base_url=self.base_url
                )
                return True
            return False
        
        except Exception:
            return False 