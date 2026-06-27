"""
Personal Knowledge Base AI Q&A — FastAPI Application

Upload documents, ask questions, get AI-powered answers based on your content.
Built with LangChain LCEL for RAG, ChromaDB for vector storage, and
HuggingFace sentence-transformers for embeddings.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from document_processor import load_document
from qa_service import QAService

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("DEEPSEEK_API_KEY"):
    print("=" * 60)
    print("WARNING: DEEPSEEK_API_KEY not set!")
    print("Create a .env file with your DeepSeek API key:")
    print("  DEEPSEEK_API_KEY=sk-...")
    print("=" * 60)

# Initialize FastAPI app
app = FastAPI(
    title="Personal Knowledge Base AI Q&A",
    description="Upload your documents and ask questions about their content",
    version="1.0.0",
)

# Add CORS middleware (allows browser to make requests from any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize services — QAService owns the vector store internally
qa_service = QAService()


# --- Request/Response Models ---

class AskRequest(BaseModel):
    question: str
    doc_ids: Optional[list[str]] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks: list[dict]


class DocInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int


# --- API Endpoints ---

@app.get("/")
async def root():
    """Serve the web UI."""
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a document to the knowledge base.
    Supported formats: PDF, DOCX, TXT, MD, CSV, JSON, HTML, code files.
    """
    logger.info(f"Upload request received: filename={file.filename}, content_type={file.content_type}")

    if not file.filename:
        logger.warning("Upload rejected: no filename")
        raise HTTPException(status_code=400, detail="No file provided.")

    # Validate file size (50MB limit)
    content = await file.read()
    logger.info(f"File read: size={len(content)} bytes")
    if len(content) > 50 * 1024 * 1024:
        logger.warning(f"Upload rejected: file too large ({len(content)} bytes)")
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")

    # Save to temp file
    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename
    logger.info(f"Saving temp file to: {file_path}")

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        # Extract text, chunk, and wrap in LangChain Documents
        logger.info(f"Processing document: {file.filename}")
        documents = load_document(str(file_path), file.filename)
        logger.info(f"Document processed: {len(documents)} chunks")

        # Store in vector database
        doc_id = qa_service.add_document(documents)
        logger.info(f"Document stored: doc_id={doc_id}, filename={file.filename}")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Document '{file.filename}' uploaded successfully.",
                "doc_id": doc_id,
                "filename": file.filename,
                "chunks": len(documents),
            },
        )

    except ValueError as e:
        logger.error(f"ValueError processing {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        # Clean up temp file
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Temp file cleaned up: {file_path}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question and get an answer based on your uploaded documents.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = qa_service.answer(
            question=request.question.strip(),
            doc_ids=request.doc_ids,
        )
        return AskResponse(**result)
    except Exception as e:
        logger.exception(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")


@app.get("/api/documents")
async def list_documents():
    """
    List all uploaded documents in the knowledge base.
    """
    try:
        docs = qa_service.list_documents()
        return {"documents": docs}
    except Exception as e:
        logger.exception(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document and all its chunks from the knowledge base.
    """
    try:
        success = qa_service.delete_document(doc_id)
        if success:
            return {"status": "success", "message": f"Document '{doc_id}' deleted."}
        else:
            raise HTTPException(status_code=404, detail="Document not found.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "api_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
    }


# --- Startup ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
