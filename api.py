"""
FastAPI server for MedHub Lite Q&A system.
Provides REST API endpoints for document management and Q&A.
"""
from fastapi import Depends, FastAPI, File, Header, UploadFile, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from agents import AgentOrchestrator
import tempfile
import os
import hmac
import logging
import time
import uuid
from pathlib import Path

from qa_system import MedicalQASystem
import config
from observability import configure_logging, runtime_metrics

configure_logging()
logger = logging.getLogger("medhub.api")

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


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    if config.API_AUTH_KEY:
        authorized = bool(x_api_key) and hmac.compare_digest(x_api_key, config.API_AUTH_KEY)
    elif config.TENANT_API_KEYS:
        authorized = bool(x_api_key) and any(
            hmac.compare_digest(x_api_key, tenant_key)
            for tenant_key in config.TENANT_API_KEYS.values()
        )
    else:
        authorized = True  # Local development mode only; Terraform always injects a key.
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_tenant_id(
    x_tenant_id: str = Header(default=config.DEFAULT_TENANT_ID, alias="X-Tenant-ID")
) -> str:
    try:
        return MedicalQASystem.normalize_tenant_id(x_tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def authorize_tenant(
    x_tenant_id: str = Header(default=config.DEFAULT_TENANT_ID, alias="X-Tenant-ID"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    tenant_id = get_tenant_id(x_tenant_id)
    if config.TENANT_API_KEYS:
        expected = config.TENANT_API_KEYS.get(tenant_id)
        if not expected or not x_api_key or not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(status_code=401, detail="Invalid tenant credentials")
    else:
        require_api_key(x_api_key)
    return tenant_id


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled_request_error", extra={"request_id": request_id})
        raise
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    runtime_metrics.record(response.status_code, latency_ms)
    response.headers["X-Request-ID"] = request_id
    logger.info("request_complete", extra={
        "request_id": request_id, "method": request.method,
        "route": request.url.path, "status_code": response.status_code,
        "latency_ms": latency_ms,
    })
    return response

# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=config.MAX_QUERY_LENGTH)
    top_k: int = Field(default=5, ge=1, le=config.MAX_TOP_K)


class QuestionResponse(BaseModel):
    answer: str
    sources: List[dict]
    num_sources: int
    confidence: str
    answerable: bool
    retrieved_chunks: int
    cited_source_numbers: List[int] = Field(default_factory=list)
    top_similarity: Optional[float] = None
    claim_evidence: List[dict] = Field(default_factory=list)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    latency_ms: Dict[str, float] = Field(default_factory=dict)
    verification: Dict = Field(default_factory=dict)


class DocumentSummaryResponse(BaseModel):
    doc_id: str
    num_chunks: int
    medical_profile: dict
    metadata: dict


class AgentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=config.MAX_QUERY_LENGTH)
    top_k: int = Field(default=5, ge=1, le=config.MAX_TOP_K)


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
            "POST /agent": "Route a grounded Q&A, extraction, or summary task",
            "GET /documents/{doc_id}": "Get document summary",
            "GET /timeline": "Get medical timeline",
            "GET /stats": "Get system statistics",
            "DELETE /documents": "Clear all documents"
        }
    }


@app.get("/health/live")
async def health_live():
    return {"status": "ok", "version": config.APP_VERSION}


@app.get("/health/ready")
async def health_ready():
    try:
        stats = await run_in_threadpool(qa_system.get_system_stats)
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        return {"status": "ready", "version": config.APP_VERSION, "vector_store": stats}
    except Exception:
        raise HTTPException(status_code=503, detail="Service is not ready")


@app.get("/metrics")
async def service_metrics(_auth=Depends(require_api_key)):
    return runtime_metrics.snapshot()


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...), tenant_id: str = Depends(authorize_tenant),
):
    """
    Upload a medical document (PDF or TXT).
    
    Args:
        file: The document file
        
    Returns:
        Document metadata
    """
    try:
        safe_name = Path(file.filename or "").name
        extension = Path(safe_name).suffix.lower()
        if extension not in config.ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Only PDF and TXT files are supported")
        content = await file.read(config.MAX_UPLOAD_BYTES + 1)
        if len(content) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Embedding, parsing, and LLM SDK calls are blocking operations.
            document = await run_in_threadpool(
                qa_system.add_document, tmp_path, safe_name, tenant_id
            )
        finally:
            if os.path.exists(tmp_path):
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
    
    except HTTPException:
        raise
    except Exception:
        logger.exception("document_upload_failed")
        raise HTTPException(status_code=500, detail="Document processing failed")


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest, tenant_id: str = Depends(authorize_tenant),
):
    """
    Ask a question about the medical documents.
    
    Args:
        request: Question request with question text and optional top_k
        
    Returns:
        Answer with sources and citations
    """
    try:
        result = await run_in_threadpool(
            qa_system.ask_question, request.question, request.top_k, tenant_id
        )
        return result
    
    except Exception:
        logger.exception("question_answering_failed")
        raise HTTPException(status_code=500, detail="Question answering failed")


@app.post("/agent")
async def agent_query(
    request: AgentRequest, tenant_id: str = Depends(authorize_tenant),
):
    """Multi-agent endpoint: routes to QA, extraction, or summarization agent."""
    try:
        return await run_in_threadpool(orchestrator.run, request.query, request.top_k, tenant_id)
    except Exception:
        logger.exception("agent_query_failed")
        raise HTTPException(status_code=500, detail="Agent query failed")


@app.get("/documents/{doc_id}", response_model=DocumentSummaryResponse)
async def get_document_summary(
    doc_id: str, tenant_id: str = Depends(authorize_tenant),
):
    """
    Get a summary of a specific document.
    
    Args:
        doc_id: The document ID
        
    Returns:
        Document summary with medical profile
    """
    try:
        summary = await run_in_threadpool(qa_system.get_document_summary, doc_id, tenant_id)
        
        if 'error' in summary:
            raise HTTPException(status_code=404, detail=summary['error'])
        
        return summary
    
    except HTTPException:
        raise
    except Exception:
        logger.exception("document_summary_failed")
        raise HTTPException(status_code=500, detail="Document summary failed")


@app.get("/timeline")
async def get_timeline(
    doc_id: Optional[str] = None, tenant_id: str = Depends(authorize_tenant),
):
    """
    Get medical timeline events.
    
    Args:
        doc_id: Optional document ID to filter by
        
    Returns:
        List of timeline events
    """
    try:
        events = await run_in_threadpool(qa_system.get_timeline, doc_id, tenant_id)
        return {
            "events": events,
            "count": len(events)
        }
    
    except Exception:
        logger.exception("timeline_failed")
        raise HTTPException(status_code=500, detail="Timeline generation failed")


@app.get("/stats")
async def get_stats(_auth=Depends(require_api_key)):
    """
    Get system statistics.
    
    Returns:
        System statistics
    """
    try:
        stats = await run_in_threadpool(qa_system.get_system_stats)
        return stats
    
    except Exception:
        logger.exception("stats_failed")
        raise HTTPException(status_code=500, detail="Statistics unavailable")


@app.delete("/documents")
async def clear_documents(
    tenant_id: str = Depends(authorize_tenant)
):
    """
    Clear all documents from the system.
    
    Returns:
        Success message
    """
    try:
        deleted = await run_in_threadpool(qa_system.clear_all_documents, tenant_id)
        return {"status": "success", "deleted_chunks": deleted}
    
    except Exception:
        logger.exception("tenant_delete_failed")
        raise HTTPException(status_code=500, detail="Document deletion failed")


# Run the server
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting MedHub Lite API server...")
    print(f"📍 API will be available at: http://{config.API_HOST}:{config.API_PORT}")
    print(f"📖 API documentation at: http://{config.API_HOST}:{config.API_PORT}/docs\n")
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )
