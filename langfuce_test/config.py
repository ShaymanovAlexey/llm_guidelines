import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Langfuse configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

# Example project configuration
PROJECT_NAME = "langfuse-ollama-demo"
PROJECT_DESCRIPTION = "Simple Langfuse example with Ollama for local LLM inference" 