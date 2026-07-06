"""
FastAPI server for MedHub Lite Q&A system.
Provides REST API endpoints for document management and Q&A.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from agents import AgentOrchestrator
import tempfile
import os

from qa_system import MedicalQASystem
import config

# Initialize FastAPI app
app = FastAPI(
    title="MedHub Lite API",
    description="Medical Document Q&A System for Claims Processing",
    version="1.0.0"
)

# Initialize Q&A system
qa_system = MedicalQASystem()

# Initialize multi-agent orchestrator
orchestrator = AgentOrchestrator(qa_system, qa_system.vector_store)

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QuestionResponse(BaseModel):
    answer: str
    sources: List[dict]
    num_sources: int
    confidence: str


class DocumentSummaryResponse(BaseModel):
    doc_id: str
    num_chunks: int
    medical_profile: dict
    metadata: dict


class AgentRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "MedHub Lite API",
        "version": "1.0.0",
        "description": "Medical Document Q&A System",
        "endpoints": {
            "POST /upload": "Upload a medical document",
            "POST /ask": "Ask a question about documents",
            "GET /documents/{doc_id}": "Get document summary",
            "GET /timeline": "Get medical timeline",
            "GET /stats": "Get system statistics",
            "DELETE /documents": "Clear all documents"
        }
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a medical document (PDF or TXT).
    
    Args:
        file: The document file
        
    Returns:
        Document metadata
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Process document
        document = qa_system.add_document(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "document": {
                "doc_id": document.metadata['doc_id'],
                "filename": document.metadata['filename'],
                "doc_type": document.metadata.get('doc_type', 'unknown'),
                "num_pages": document.metadata.get('num_pages', 0),
                "num_chunks": len(document.chunks)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the medical documents.
    
    Args:
        request: Question request with question text and optional top_k
        
    Returns:
        Answer with sources and citations
    """
    try:
        result = qa_system.ask_question(
            question=request.question,
            top_k=request.top_k
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


@app.post("/agent")
def agent_query(request: AgentRequest):
    """Multi-agent endpoint: routes to QA, extraction, or summarization agent."""
    try:
        return orchestrator.run(request.query, request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{doc_id}", response_model=DocumentSummaryResponse)
async def get_document_summary(doc_id: str):
    """
    Get a summary of a specific document.
    
    Args:
        doc_id: The document ID
        
    Returns:
        Document summary with medical profile
    """
    try:
        summary = qa_system.get_document_summary(doc_id)
        
        if 'error' in summary:
            raise HTTPException(status_code=404, detail=summary['error'])
        
        return summary
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document summary: {str(e)}")


@app.get("/timeline")
async def get_timeline(doc_id: Optional[str] = None):
    """
    Get medical timeline events.
    
    Args:
        doc_id: Optional document ID to filter by
        
    Returns:
        List of timeline events
    """
    try:
        events = qa_system.get_timeline(doc_id=doc_id)
        return {
            "events": events,
            "count": len(events)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting timeline: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    Get system statistics.
    
    Returns:
        System statistics
    """
    try:
        stats = qa_system.get_system_stats()
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.delete("/documents")
async def clear_documents():
    """
    Clear all documents from the system.
    
    Returns:
        Success message
    """
    try:
        qa_system.clear_all_documents()
        return {"status": "success", "message": "All documents cleared"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")


# Run the server
if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 Starting MedHub Lite API server...")
    print(f"📍 API will be available at: http://{config.API_HOST}:{config.API_PORT}")
    print(f"📖 API documentation at: http://{config.API_HOST}:{config.API_PORT}/docs\n")
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )
