import os
from typing import List, Dict, Any
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from vector_store import VectorStore
from bm25_search import AsyncBM25Search
from hybrid_search_system import HybridSearchSystem, HybridRAGSystem
from ollama_generator import OllamaGenerator
from config import get_config
from langfuse_integration import LangfuseManager

# Load configuration
config = get_config()

# Initialize FastAPI app
app = FastAPI(title="Hybrid RAG System", version="3.0.0")

# Initialize Langfuse manager
langfuse_manager = LangfuseManager(config.langfuse)

# Initialize components
vector_store = VectorStore("hybrid_rag")
bm25_store = AsyncBM25Search("hybrid_rag_bm25.db")

# Create hybrid search system
hybrid_search = HybridSearchSystem(
    vector_store=vector_store,
    bm25_store=bm25_store,
    vector_weight=0.6,
    bm25_weight=0.4
)

# Initialize generator
try:
    generator = OllamaGenerator(model_name="llama3.2:latest")
    print("Ollama generator initialized successfully")
except Exception as e:
    print(f"Ollama not available, using template generator: {e}")
    generator = None

# Create hybrid RAG system with Langfuse integration
rag_system = HybridRAGSystem(
    hybrid_search=hybrid_search,
    generator=generator,
    max_context_length=4000,
    langfuse_manager=langfuse_manager
)

# Templates
templates = Jinja2Templates(directory="templates")

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    search_type: str = "hybrid"  # 'vector', 'bm25', or 'hybrid'
    max_tokens: int = None

class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    context: List[Dict[str, Any]]
    search_type: str
    scores: Dict[str, List[float]]

class DocumentUpload(BaseModel):
    content: str
    title: str = None
    metadata: Dict[str, Any] = None

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main web interface."""
    stats = await rag_system.get_statistics()
    return templates.TemplateResponse("index.html", {"request": request, "stats": stats})

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File([]),
    text_content: str = Form("")
):
    """Upload documents to the hybrid RAG system."""
    documents = []
    
    # Process uploaded files
    for file in files:
        if file.filename:
            content = await file.read()
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            if file_extension == '.pdf':
                # Handle PDF files (simplified - you might want to add PDF processing)
                documents.append({
                    'content': content.decode('utf-8', errors='ignore'),
                    'metadata': {
                        'title': file.filename,
                        'source': 'pdf_upload',
                        'filename': file.filename
                    }
                })
            else:
                # Handle text-based files
                try:
                    content_str = content.decode('utf-8')
                    documents.append({
                        'content': content_str,
                        'metadata': {
                            'title': file.filename,
                            'source': 'file_upload',
                            'filename': file.filename
                        }
                    })
                except UnicodeDecodeError:
                    try:
                        content_str = content.decode('latin-1')
                        documents.append({
                            'content': content_str,
                            'metadata': {
                                'title': file.filename,
                                'source': 'file_upload',
                                'filename': file.filename
                            }
                        })
                    except UnicodeDecodeError:
                        raise HTTPException(status_code=400, detail=f"Could not decode file {file.filename}")
    
    # Process pasted text
    if text_content.strip():
        documents.append({
            'content': text_content,
            'metadata': {
                'title': 'Pasted Text',
                'source': 'pasted_text'
            }
        })
    
    if not documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    try:
        doc_ids = await rag_system.add_documents(documents)
        return {
            "success": True,
            "message": f"Successfully uploaded {len(doc_ids)} documents",
            "document_ids": doc_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}")

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Query the hybrid RAG system."""
    try:
        result = await rag_system.query(
            question=request.question,
            top_k=request.top_k,
            search_type=request.search_type,
            max_tokens=request.max_tokens
        )
        
        return QueryResponse(
            success=True,
            question=request.question,
            answer=result['answer'],
            context=result['context'],
            search_type=result['search_type'],
            scores=result['scores']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying RAG system: {str(e)}")

@app.post("/search")
async def search_documents(
    query: str,
    top_k: int = 10,
    search_type: str = "hybrid"
):
    """Search documents without generating answers."""
    try:
        results = await hybrid_search.search(query, top_k, search_type)
        
        return {
            "success": True,
            "query": query,
            "search_type": search_type,
            "results": [
                {
                    "doc_id": result.doc_id,
                    "title": result.title,
                    "content": result.content,
                    "vector_score": result.vector_score,
                    "bm25_score": result.bm25_score,
                    "combined_score": result.combined_score,
                    "source": result.source
                }
                for result in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all documents in the system."""
    try:
        stats = await hybrid_search.get_statistics()
        return {
            "success": True,
            "vector_documents": stats['vector_store'].get('total_documents', 0),
            "bm25_documents": stats['bm25_store'].get('total_documents', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = await rag_system.get_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the system."""
    try:
        await hybrid_search.clear_all()
        return {
            "success": True,
            "message": "All documents cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check vector store
        vector_stats = await vector_store.get_collection_stats()
        
        # Check BM25 store
        bm25_stats = await bm25_store.get_statistics()
        
        # Check generator
        generator_status = "available" if generator else "unavailable"
        
        return {
            "status": "healthy",
            "vector_store": "available",
            "bm25_store": "available",
            "generator": generator_status,
            "vector_documents": vector_stats.get('total_documents', 0),
            "bm25_documents": bm25_stats.get('total_documents', 0)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/batch-query")
async def batch_query(questions: List[str], top_k: int = 5, search_type: str = "hybrid"):
    """Process multiple queries in batch."""
    try:
        results = []
        for question in questions:
            result = await rag_system.query(question, top_k, search_type)
            results.append({
                "question": question,
                "answer": result['answer'],
                "context_count": len(result['context']),
                "search_type": result['search_type']
            })
        
        return {
            "success": True,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch queries: {str(e)}")

@app.post("/weights/update")
async def update_search_weights(vector_weight: float, bm25_weight: float):
    """Update the weights for hybrid search."""
    try:
        await hybrid_search.update_weights(vector_weight, bm25_weight)
        return {
            "success": True,
            "message": f"Weights updated: vector={vector_weight}, bm25={bm25_weight}",
            "new_weights": {
                "vector_weight": hybrid_search.vector_weight,
                "bm25_weight": hybrid_search.bm25_weight
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating weights: {str(e)}")

@app.get("/search/types")
async def get_search_types():
    """Get available search types."""
    return {
        "success": True,
        "search_types": [
            {
                "name": "vector",
                "description": "Semantic search using vector embeddings"
            },
            {
                "name": "bm25",
                "description": "Keyword-based search using BM25 algorithm"
            },
            {
                "name": "hybrid",
                "description": "Combined vector and BM25 search"
            }
        ]
    }

@app.post("/documents/add")
async def add_single_document(document: DocumentUpload):
    """Add a single document to the system."""
    try:
        doc = {
            'content': document.content,
            'metadata': document.metadata or {}
        }
        
        if document.title:
            doc['metadata']['title'] = document.title
        
        doc_ids = await rag_system.add_documents([doc])
        
        return {
            "success": True,
            "message": "Document added successfully",
            "document_id": doc_ids[0] if doc_ids else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 