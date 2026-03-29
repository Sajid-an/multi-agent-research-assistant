from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from graph import research_graph
from agents.pdf_agent import add_pdf, vector_store
import shutil
import os
import json

router = APIRouter()

# ── Request/Response models ──────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str

class ResearchResponse(BaseModel):
    query: str
    report: str
    sources: list[str]
    agents_used: list[str]
    query_type: str

# ── Research endpoint ────────────────────────────────────────────

@router.post("/research/", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    print(f"\n[API] New research request: {request.query}")

    initial_state = {
        "query": request.query,
        "messages": [],
        "web_results": [],
        "arxiv_results": [],
        "pdf_results": [],
        "fact_check_results": [],
        "final_report": "",
        "sources": [],
        "next_agent": "",
        "planned_agents": [],
        "query_type": "",
    }

    try:
        result = research_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

    return ResearchResponse(
        query=request.query,
        report=result["final_report"],
        sources=result.get("sources", []),
        agents_used=result.get("planned_agents", []),
        query_type=result.get("query_type", "general")
    )

# ── PDF upload endpoint ──────────────────────────────────────────

@router.post("/upload/")
async def upload_pdf(file: UploadFile = File(description="Upload a PDF to include in research")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        add_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {str(e)}")

    return {
        "message": f"{file.filename} uploaded and indexed successfully",
        "file": file.filename,
        "path": file_path
    }

# ── List uploaded files ──────────────────────────────────────────

@router.get("/files/")
async def list_files():
    os.makedirs("uploads", exist_ok=True)
    files = [f for f in os.listdir("uploads") if f.endswith(".pdf")]
    return {
        "uploaded_files": files,
        "total": len(files),
        "index_ready": vector_store is not None
    }

# ── Clear uploaded files ─────────────────────────────────────────

@router.delete("/clear/")
async def clear_files():
    global vector_store
    from agents import pdf_agent
    pdf_agent.vector_store = None

    if os.path.exists("uploads"):
        shutil.rmtree("uploads")
        os.makedirs("uploads", exist_ok=True)

    return {"message": "All uploaded files cleared and index reset"}
