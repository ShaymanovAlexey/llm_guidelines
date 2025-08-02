import os
from typing import List, Dict, Any
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from simple_rag_system import SimpleRAGSystem
from advanced_rag_system import AdvancedRAGSystem
from fuzzy_vector_store import FuzzyVectorStore

# Initialize FastAPI app
app = FastAPI(title="RAG System Rebuild", version="2.0.0")

# Initialize RAG system (you can choose between simple and advanced)
# rag_system = SimpleRAGSystem()  # Simple RAG system
rag_system = AdvancedRAGSystem(
    max_workers=4,
    ollama_model="llama2",  # Default Ollama model
    ollama_url="http://localhost:11434",  # Default Ollama URL
    vector_store_class=FuzzyVectorStore,
    duplicate_threshold=0.9
)  # Advanced RAG system with concurrent processing and Ollama

# Templates
templates = Jinja2Templates(directory="templates")

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    k: int = 3

class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    context: List[Dict[str, Any]]
    sources: List[str]

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main web interface."""
    stats = await rag_system.get_stats()
    return templates.TemplateResponse("index.html", {"request": request, "stats": stats})

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File([]),
    text_content: str = Form("")
):
    """Upload documents to the RAG system."""
    documents = []
    
    # Process uploaded files
    for file in files:
        if file.filename:
            content = await file.read()
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            if file_extension == '.pdf':
                # Handle PDF files
                documents.append({
                    'type': 'pdf',
                    'content': content,  # Keep as bytes for PDF processing
                    'filename': file.filename
                })
            else:
                # Handle text-based files
                try:
                    content_str = content.decode('utf-8')
                    documents.append({
                        'type': 'file',
                        'content': content_str,
                        'filename': file.filename
                    })
                except UnicodeDecodeError:
                    try:
                        content_str = content.decode('latin-1')
                        documents.append({
                            'type': 'file',
                            'content': content_str,
                            'filename': file.filename
                        })
                    except UnicodeDecodeError:
                        raise HTTPException(status_code=400, detail=f"Could not decode file {file.filename}")
    
    # Process pasted text
    if text_content.strip():
        documents.append({
            'type': 'text',
            'content': text_content,
            'source': 'pasted_text'
        })
    
    if not documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    result = await rag_system.add_documents(documents)
    return result

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    result = await rag_system.query(request.question, request.k)
    return result

@app.post("/query-without-context")
async def query_without_context(request: QueryRequest):
    """Query the RAG system without using context (general answer from model)."""
    result = await rag_system.query_without_context(request.question, request.k)
    return result

@app.get("/documents")
async def list_documents():
    """List all documents in the system."""
    return await rag_system.list_documents()

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    stats = await rag_system.get_stats()
    # Add embedding model if not present
    if 'embedding_model' not in stats and hasattr(rag_system, 'get_current_embedding_model'):
        stats['embedding_model'] = await rag_system.get_current_embedding_model()
    # Add current Ollama model if available
    if hasattr(rag_system, 'get_current_ollama_model'):
        stats['current_ollama_model'] = rag_system.get_current_ollama_model()
    return stats

@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the system."""
    return await rag_system.clear_documents()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return await rag_system.health_check()

@app.post("/batch-query")
async def batch_query(questions: List[str], k: int = 3):
    """Process multiple queries concurrently."""
    if hasattr(rag_system, 'batch_query'):
        return await rag_system.batch_query(questions, k)
    else:
        # Fallback to sequential processing
        results = []
        for question in questions:
            result = await rag_system.query(question, k)
            results.append(result)
        return results

# Ollama-specific endpoints (only available in AdvancedRAGSystem)
@app.get("/ollama/models")
async def get_ollama_models():
    """Get available Ollama models."""
    if hasattr(rag_system, 'get_ollama_models'):
        models = await rag_system.get_ollama_models()
        return {"models": models}
    else:
        return {"models": [], "message": "Ollama not available in this RAG system"}

@app.post("/ollama/switch-model")
async def switch_ollama_model(data: Dict[str, Any] = Body(...)):
    """Switch to a different Ollama model."""
    model_name = data.get('model_name')
    if not model_name:
        return {"success": False, "message": "No model_name provided"}
    if hasattr(rag_system, 'switch_ollama_model'):
        return await rag_system.switch_ollama_model(model_name)
    else:
        return {"success": False, "message": "Ollama not available in this RAG system"}

@app.post("/ollama/system-prompt")
async def set_ollama_system_prompt(prompt: str):
    """Set a custom system prompt for Ollama."""
    if hasattr(rag_system, 'set_ollama_system_prompt'):
        return await rag_system.set_ollama_system_prompt(prompt)
    else:
        return {"success": False, "message": "Ollama not available in this RAG system"}

@app.post("/ollama/toggle")
async def toggle_ollama(enable: bool | None = None):
    """Toggle Ollama usage on/off."""
    if hasattr(rag_system, 'toggle_ollama'):
        return rag_system.toggle_ollama(enable)
    else:
        return {"success": False, "message": "Ollama not available in this RAG system"}

@app.get("/embedding/models")
async def get_embedding_models():
    """Get available embedding models."""
    if hasattr(rag_system, 'get_embedding_models'):
        models = await rag_system.get_embedding_models()
        current = await rag_system.get_current_embedding_model()
        return {"models": models, "current_model": current}
    else:
        return {"models": [], "message": "Embedding model selection not available"}

@app.post("/embedding/switch-model")
async def switch_embedding_model(data: Dict[str, Any] = Body(...)):
    """Switch to a different embedding model."""
    model_name = data.get('model_name')
    if not model_name:
        return {"success": False, "message": "No model_name provided"}
    if hasattr(rag_system, 'switch_embedding_model'):
        return await rag_system.switch_embedding_model(model_name)
    else:
        return {"success": False, "message": "Embedding model selection not available"}

@app.post("/documents/remove-duplicates")
async def remove_duplicates():
    """Remove fuzzy duplicate documents from the vector store."""
    if hasattr(rag_system, 'remove_vector_duplicates'):
        return await rag_system.remove_vector_duplicates()
    return {"success": False, "message": "Duplicate removal not supported."}

@app.get("/vector/persist-directory")
async def get_persist_directory():
    if hasattr(rag_system, 'get_persist_directory'):
        return {"persist_directory": rag_system.get_persist_directory()}
    return {"persist_directory": None, "message": "Not supported."}

@app.post("/vector/switch-persist-directory")
async def switch_persist_directory(data: Dict[str, Any] = Body(...)):
    persist_directory = data.get('persist_directory')
    if not persist_directory:
        return {"success": False, "message": "No persist_directory provided."}
    if hasattr(rag_system, 'switch_persist_directory'):
        return await rag_system.switch_persist_directory(persist_directory)
    return {"success": False, "message": "Not supported."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 