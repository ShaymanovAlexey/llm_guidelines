# RAG System Rebuild

A high-performance Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, and ChromaDB, featuring full async/await support and concurrent processing capabilities. This system uses a clean inheritance architecture for better maintainability and extensibility.

## Features

- **Inheritance Architecture**: Clean base class with simple and advanced implementations
- **Full Async Support**: All operations use async/await for better performance
- **Concurrent Processing**: Multi-threaded document processing and batch queries
- **Local LLM Integration**: Ollama integration for high-quality answer generation
- **Document Ingestion**: Upload files or paste text with automatic chunking
- **Vector Embeddings**: Using sentence-transformers for semantic search
- **Vector Storage**: ChromaDB for efficient similarity search
- **Semantic Search**: Retrieve relevant context for questions
- **Batch Operations**: Process multiple queries concurrently
- **Health Monitoring**: System health checks and statistics
- **Model Management**: Switch between different Ollama models
- **Custom Prompts**: Configure system prompts for different use cases
- **Modern Web Interface**: Beautiful, responsive UI with real-time interactions
- **REST API**: Complete API endpoints for programmatic access

## Architecture

The system uses a clean inheritance structure:

```
BaseRAGSystem (Abstract Base Class)
├── SimpleRAGSystem (Basic functionality)
└── AdvancedRAGSystem (Concurrent processing + Ollama)
```

### Core Components

- `base_rag_system.py`: Abstract base class defining the RAG interface
- `simple_rag_system.py`: Basic RAG implementation with template-based answers
- `advanced_rag_system.py`: Advanced implementation with concurrent processing and Ollama
- `document_processor.py`: Document ingestion and chunking
- `vector_store.py`: Async vector storage and retrieval
- `ollama_generator.py`: Ollama-based answer generation
- `main.py`: FastAPI application with web interface and API endpoints
- `templates/`: HTML templates for the web interface

## Setup

### Prerequisites

1. **Install Ollama** (for local LLM generation):
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Or download from https://ollama.ai/download
   ```

2. **Pull a model** (e.g., llama2):
   ```bash
   ollama pull llama2
   ```

3. **Start Ollama server**:
   ```bash
   ollama serve
   ```

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file (optional):
```bash
OPENAI_API_KEY=your_api_key_here
```

3. Run the application:
```bash
python main.py
```

4. Open your browser and go to `http://localhost:8000`

## Usage

### Choosing a RAG System

In `main.py`, you can choose between different RAG implementations:

```python
# Simple RAG system (template-based answers)
rag_system = SimpleRAGSystem()

# Advanced RAG system (concurrent processing + Ollama)
rag_system = AdvancedRAGSystem(
    max_workers=4,
    ollama_model="llama2",
    ollama_url="http://localhost:11434"
)
```

### Basic Usage

1. **Upload Documents**: Use the web interface to upload text files or paste text directly
2. **Ask Questions**: Type your questions and get answers based on the uploaded documents
3. **View Results**: See the retrieved context and generated answers

## API Endpoints

- `GET /`: Main web interface
- `POST /upload`: Upload documents
- `POST /query`: Query the RAG system
- `POST /batch-query`: Process multiple queries concurrently
- `GET /documents`: List uploaded documents
- `GET /stats`: Get system statistics
- `GET /health`: Health check endpoint
- `DELETE /documents`: Clear all documents

### Ollama Endpoints (AdvancedRAGSystem only)

- `GET /ollama/models`: Get available Ollama models
- `POST /ollama/switch-model`: Switch to a different model
- `POST /ollama/system-prompt`: Set custom system prompt
- `POST /ollama/toggle`: Enable/disable Ollama

## Configuration

The system uses:
- **Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Vector Database**: ChromaDB
- **LLM**: Ollama (default: llama2) - AdvancedRAGSystem only
- **Chunk Size**: 1000 characters with 200 character overlap
- **Retrieval**: Top 3 most similar chunks (configurable)
- **Concurrency**: Up to 4 worker threads for document processing (AdvancedRAGSystem)
- **Async Operations**: All I/O operations are non-blocking

### Ollama Configuration (AdvancedRAGSystem)

- **Default Model**: `llama2`
- **Server URL**: `http://localhost:11434`
- **Fallback**: Template-based answers if Ollama is unavailable
- **Model Switching**: Dynamic model switching via API
- **Custom Prompts**: Configurable system prompts

## Performance Features

### SimpleRAGSystem
- Basic document processing and querying
- Template-based answer generation
- Suitable for simple use cases

### AdvancedRAGSystem
- **Concurrent Document Processing**: Multiple documents processed simultaneously
- **Batch Query Processing**: Handle multiple queries in parallel
- **Async Vector Operations**: Non-blocking embedding and search operations
- **Thread Pool Management**: Efficient resource utilization
- **Health Monitoring**: Real-time system status tracking
- **Local LLM Processing**: Fast local inference with Ollama
- **Fallback Mechanisms**: Graceful degradation when services are unavailable

## Testing

```bash
# Test simple RAG system
python test_simple_rag.py

# Test advanced RAG system
python test_advanced_rag.py
```

## Inheritance Benefits

1. **Code Reuse**: Common functionality in the base class
2. **Extensibility**: Easy to add new RAG implementations
3. **Maintainability**: Clear separation of concerns
4. **Flexibility**: Choose the right implementation for your needs
5. **Testing**: Easy to test different implementations independently

## Extending the System

To create a new RAG implementation:

1. Inherit from `BaseRAGSystem`
2. Implement the required abstract methods (`add_documents`, `query`)
3. Override optional methods as needed
4. Add your custom functionality

Example:
```python
class CustomRAGSystem(BaseRAGSystem):
    def __init__(self, custom_param):
        super().__init__()
        self.custom_param = custom_param
    
    async def add_documents(self, documents):
        # Your custom implementation
        pass
    
    async def query(self, question, k=3):
        # Your custom implementation
        pass
``` 