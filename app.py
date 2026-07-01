"""
Personal Knowledge Base AI Q&A — Production-Grade FastAPI Application

Full pipeline:
  Upload → ParserRouter → HierarchyChunker → Dual Index (ChromaDB + BM25)
  Ask   → RetrievalAgent (dynamic tools) → CitationGenerator → Evaluator

Supports: PDF (digital/scanned), DOCX, DOC, TXT, MD, images, code files.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# --- Early env load ---
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Imports (after env load — config reads env vars) ---
from config import MAX_UPLOAD_SIZE_BYTES
from services.document_service import DocumentService
from services.qa_service import QAService
from retrieval.tools import set_document_service

# --- FastAPI app ---
app = FastAPI(
    title="Personal Knowledge Base AI Q&A",
    description="Upload documents, ask questions, get AI-powered answers with page-level citations.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directories ---
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Services ---
document_service = DocumentService()
set_document_service(document_service)  # Inject into retrieval tools
qa_service = QAService(document_service=document_service)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    doc_ids: Optional[list[str]] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict]
    chunks: list[dict]
    evaluation: Optional[dict] = None


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the web UI."""
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a document to the knowledge base.

    Supported formats: PDF, DOCX, DOC, TXT, MD, CSV, JSON, HTML,
    code files, PNG, JPG, TIFF.

    The system auto-detects file type and applies the best parser:
    - Digital PDF (< 50 pages): Marker (local ML, preserves headings)
    - Digital PDF (50+ pages): LlamaParse (cloud, optimized for long docs)
    - Scanned/image PDF: DashScope qwen-vl-ocr (multimodal extraction)
    - DOCX: Marker
    - Images: DashScope qwen-vl-ocr
    - Text/code: direct extraction
    """
    logger.info(f"Upload: filename={file.filename}, content_type={file.content_type}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Read file content
    content = await file.read()
    logger.info(f"File size: {len(content)} bytes")

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB.",
        )

    # Save to temp file
    file_id = str(uuid.uuid4())
    safe_name = f"{file_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_name

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        # Full pipeline: parse → chunk → dual index
        result = document_service.ingest(
            file_path=str(file_path),
            filename=file.filename,
        )

        # Cache source markdown for later evaluation (from the parser)
        # Note: The chunker doesn't store full source markdown.
        # For production, we'd store it alongside the doc_id.
        # Here we keep a lightweight reference.

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Document '{file.filename}' processed successfully.",
                **result,
            },
        )

    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process document: {str(e)}"
        )
    finally:
        if file_path.exists():
            file_path.unlink()


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """
    Ask a question about your uploaded documents.

    The system uses an agent to dynamically select retrieval tools,
    then generates an answer with mandatory page and section citations.
    Each claim includes a reference like:
    根据《文档名》第X页，第Y章节："原文引用"
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = qa_service.answer(
            question=req.question.strip(),
            doc_ids=req.doc_ids,
        )
        return AskResponse(**result)
    except Exception as e:
        logger.exception(f"Error answering: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to answer: {str(e)}"
        )


@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents with metadata."""
    try:
        docs = document_service.list_documents()
        return {"documents": docs}
    except Exception as e:
        logger.exception(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list documents: {str(e)}"
        )


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its chunks from both indexes."""
    try:
        ok = document_service.delete(doc_id)
        if ok:
            return {"status": "success", "message": f"Document '{doc_id}' deleted."}
        else:
            raise HTTPException(status_code=404, detail="Document not found.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete: {str(e)}"
        )


@app.get("/api/documents/{doc_id}/toc")
async def get_document_toc(doc_id: str):
    """Get the table of contents for a parsed document (if available)."""
    # TOC data is embedded in chunk metadata — reconstruct from chunks
    try:
        chunks = document_service.get_all_chunks(doc_id=doc_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found.")

        # Extract unique section entries from chunk metadata
        sections = {}
        for doc in chunks:
            meta = doc.metadata
            sid = meta.get("section_id", "")
            if sid and sid not in sections:
                sections[sid] = {
                    "section_id": sid,
                    "heading_hierarchy": meta.get("heading_hierarchy", ""),
                    "page_num": meta.get("page_num"),
                }
        return {"toc": sorted(sections.values(), key=lambda x: x.get("page_num") or 0)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting TOC: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_id}/page/{page_num}")
async def get_document_page(doc_id: str, page_num: int):
    """Get all chunks from a specific page of a document."""
    try:
        chunks = document_service.get_chunks_by_page(doc_id, page_num)
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for page {page_num}.",
            )
        return {
            "doc_id": doc_id,
            "page_num": page_num,
            "chunks": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in chunks
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting page: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check with system status."""
    return {
        "status": "ok",
        "deepseek_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
        "dashscope_configured": bool(os.getenv("DASHSCOPE_API_KEY")),
        "llamaparse_configured": bool(os.getenv("LLAMA_CLOUD_API_KEY")),
        "vector_chunks": document_service.vector_store.count(),
        "bm25_chunks": document_service.bm25_index.count(),
    }


# --- Startup ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
