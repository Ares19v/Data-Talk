"""
api/app.py
──────────
FastAPI backend for the Voice-RAG pipeline.

Endpoints:
  POST /query        — text query → RAG answer
  POST /voice-query  — audio file upload → STT → RAG answer
  GET  /health       — health check + index stats
  GET  /             — redirect to /docs
"""
from __future__ import annotations

import os
import sys
import logging

# Make root importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional

from pipeline.harness import RAGRequest, RAGResponse, get_harness

app = FastAPI(
    title="Voice-RAG API",
    description="Voice-enabled RAG on ai4bharat/MSMARCO-XI | HH Goa 2026",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "data-talk-backend"
    }


@app.get("/stats")
def stats():
    harness = get_harness()
    total = harness.store.total_vectors()
    strategies = harness.store.get_strategies()
    return {
        "status": "ok",
        "total_vectors": total,
        "strategies": strategies,
        "index_loaded": total > 0,
    }


from pydantic import BaseModel, Field

class TextQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 5
    include_sources: bool = True


@app.post("/query", response_model=RAGResponse)
def text_query(request: TextQueryRequest):
    """
    Submit a text question and get a grounded answer with latency stats.
    """
    harness = get_harness()
    rag_request = RAGRequest(
        query=request.query,
        top_k=request.top_k,
        include_sources=request.include_sources,
    )
    response = harness.run(rag_request)
    return response


@app.post("/voice-query", response_model=RAGResponse)
async def voice_query(
    audio: UploadFile = File(..., description="WAV/MP3 audio file (max 30s, 16kHz recommended)"),
    language_hint: Optional[str] = Form(None, description="ISO language code hint e.g. 'hi-IN'"),
    top_k: int = Form(5),
    include_sources: bool = Form(True),
):
    """
    Upload an audio file → Sarvam STT transcription → RAG answer.
    Full voice pipeline end-to-end.
    """
    if audio.content_type not in {
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mpeg", "audio/mp3",
        "audio/ogg", "audio/webm",
        "audio/flac", "audio/aac",
    }:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type: {audio.content_type}. Use WAV or MP3.",
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="Audio file too large. Max 10MB.")

    harness = get_harness()
    response = harness.run_with_stt(
        audio_bytes=audio_bytes,
        language_hint=language_hint,
        top_k=top_k,
        include_sources=include_sources,
    )
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
