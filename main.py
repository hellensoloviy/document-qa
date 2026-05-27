"""
Document Q&A API
================
A RAG-powered FastAPI service for querying documents and summarizing meetings.

Endpoints:
    GET  /          — health check
    POST /upload    — upload a PDF to the knowledge base
    POST /ask       — ask a question about uploaded documents  
    POST /summarize — summarize meeting notes into structured format

Run with: uvicorn main:app --reload
Docs at:  http://localhost:8000/docs
"""


import os
import shutil
import re

from rag_core import process_document, answer_question, summarize_meeting, extract_tasks
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Document Q&A API",
    description="RAG-powered document question answering and meeting summarization",
    version="1.0.0"
)

# Serve frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/ui")
def serve_frontend():
    return FileResponse("frontend/index.html")

# This allows the API to be called from a browser or frontend
# Allow all origins for local development and demo purposes.
# In production, replace "*" with your actual frontend domain:
# allow_origins=["https://yourdomain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response models ────────────────────────────────────────────────
# Pydantic models define what shape the request body should be
# FastAPI validates this automatically

class QuestionRequest(BaseModel):
    question: str

class SummarizeRequest(BaseModel):
    text: str

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — confirms the API is running and database status"""
    db_exists = os.path.exists("./chroma_db")
    return {
        "status": "running",
        "message": "Document Q&A API is ready",
        "database": "loaded" if db_exists else "empty — please upload a document first"
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document to be stored in the vector database.
    Accepts: PDF file
    Returns: number of pages and chunks processed
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Check file is not empty
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )
    # Reset file position after reading
    await file.seek(0)

    # Save uploaded file temporarily
    temp_path = f"./temp_{file.filename}"
    
    try:
        # Write the uploaded file to disk
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process it
        result = process_document(temp_path)
        
        return {
            "filename": file.filename,
            "pages_loaded": result["pages_loaded"],
            "chunks_created": result["chunks_created"],
            "message": "Document processed and stored successfully"
        }
    
    except Exception as e:
        # If anything goes wrong, return a clean error
        # Never expose raw error details to the client in production
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )
    
    finally:
        # Always clean up the temp file, even if there was an error
        # This is the 'finally' block — it runs no matter what
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded documents.
    Accepts: { "question": "your question here" }
    Returns: answer based on document content
    """
    # Sanitize first, then validate
    clean_question = sanitize_text(request.question)
    
    if not clean_question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        result = answer_question(clean_question)
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(e)}"
        )


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    """
    Summarize meeting notes or any text into structured format.
    Accepts: { "text": "your meeting notes here" }
    Returns: structured summary with decisions and action items
    """
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    # Sanitize input — remove invisible control characters
    clean_text = sanitize_text(request.text)
    
    if len(clean_text) < 50:
        raise HTTPException(
            status_code=400,
            detail="Text is too short to summarize meaningfully"
        )
    
    try:
        result = summarize_meeting(clean_text)
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to summarize: {str(e)}"
        )

@app.post("/extract-tasks")
async def extract_tasks_endpoint(request: SummarizeRequest):
    """
    Extract actionable tasks from meeting notes.
    Accepts: { "text": "your meeting notes here" }
    Returns: structured list of tasks with owners and priorities
    """
    clean_text = sanitize_text(request.text)
    
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(clean_text) < 50:
        raise HTTPException(status_code=400, detail="Text too short to extract tasks from")
    
    try:
        result = extract_tasks(clean_text)
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract tasks: {str(e)}"
        )


# ── Helper fuctions ──────────────────────────────────────────────────────────────

def sanitize_text(text: str) -> str:
    """Remove invisible control characters from user input"""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text).strip()