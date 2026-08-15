"""
ui/app.py
─────────
Gradio voice UI for the Voice-RAG pipeline.

Features:
- Microphone recording input
- Audio file upload fallback
- Text query fallback (direct text input)
- Displays answer + source passages + per-stage latency
- Connects to the FastAPI backend (or runs inline if no URL set)

Run standalone:
    python ui/app.py

Or with backend URL:
    API_URL=http://localhost:8000 python ui/app.py
"""
from __future__ import annotations

import os
import sys
import json
import logging
import tempfile

# Make root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

import gradio as gr

API_URL = os.getenv("API_URL", "")  # If empty, run inline


# ── API call helpers ──────────────────────────────────────────────────────────

def _query_api_text(query: str, top_k: int = 5) -> dict:
    """Call the FastAPI /query endpoint."""
    import httpx
    resp = httpx.post(
        f"{API_URL}/query",
        json={"query": query, "top_k": top_k, "include_sources": True},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def _query_api_voice(audio_path: str, top_k: int = 5) -> dict:
    """Call the FastAPI /voice-query endpoint."""
    import httpx
    with open(audio_path, "rb") as f:
        resp = httpx.post(
            f"{API_URL}/voice-query",
            files={"audio": ("audio.wav", f, "audio/wav")},
            data={"top_k": str(top_k), "include_sources": "true"},
            timeout=60.0,
        )
    resp.raise_for_status()
    return resp.json()


def _query_inline_text(query: str, top_k: int = 5) -> dict:
    """Call the harness directly (no API server needed)."""
    from pipeline.harness import get_harness, RAGRequest
    harness = get_harness()
    req = RAGRequest(query=query, top_k=top_k, include_sources=True)
    resp = harness.run(req)
    return resp.model_dump()


def _query_inline_voice(audio_path: str, top_k: int = 5) -> dict:
    """Call STT + harness directly."""
    from pipeline.harness import get_harness
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    harness = get_harness()
    resp = harness.run_with_stt(audio_bytes=audio_bytes, top_k=top_k, include_sources=True)
    return resp.model_dump()


# ── Response formatting ───────────────────────────────────────────────────────

def _format_response(data: dict) -> tuple[str, str, str]:
    """Returns (answer_text, sources_text, latency_text)."""
    answer = data.get("answer", "No answer returned.")
    success = data.get("success", True)
    guardrail_reason = data.get("guardrail_reason", "")

    if guardrail_reason:
        answer = f"⚠️ {answer}"

    # Sources
    sources = data.get("sources", [])
    if sources:
        src_lines = []
        for i, s in enumerate(sources[:5], 1):
            strategy = s.get("strategy", "?")
            score = s.get("rrf_score", 0)
            text = s.get("text", "")[:300]
            selected = "✅ " if s.get("is_selected") else ""
            src_lines.append(f"**[{i}] {selected}{strategy}** (RRF: {score:.3f})\n> {text}")
        sources_text = "\n\n".join(src_lines)
    else:
        sources_text = "_No sources retrieved._"

    # Latency
    lat = data.get("latency", {})
    lat_lines = []
    if lat.get("stt_ms"):
        lat_lines.append(f"🎤 STT: **{lat['stt_ms']:.0f}ms**")
    lat_lines.extend([
        f"🔢 Encode: **{lat.get('encode_ms', 0):.1f}ms**",
        f"🔍 FAISS: **{lat.get('retrieval_ms', 0):.1f}ms**",
        f"🔀 RRF: **{lat.get('rrf_ms', 0):.1f}ms**",
        f"🤖 LLM: **{lat.get('llm_ms', 0):.0f}ms**",
        f"🛡️ Guardrails: **{lat.get('guardrail_ms', 0):.1f}ms**",
        f"⏱️ **Total: {lat.get('total_pipeline_ms', 0):.0f}ms**",
    ])
    latency_text = "\n".join(lat_lines)

    return answer, sources_text, latency_text


# ── Gradio handlers ───────────────────────────────────────────────────────────

def handle_text_query(query: str, top_k: int):
    if not query.strip():
        return "Please enter a question.", "", ""
    try:
        if API_URL:
            data = _query_api_text(query.strip(), int(top_k))
        else:
            data = _query_inline_text(query.strip(), int(top_k))
        return _format_response(data)
    except Exception as e:
        logger.error(f"Text query error: {e}", exc_info=True)
        return f"❌ Error: {str(e)}", "", ""


def handle_voice_query(audio_path, top_k: int):
    if audio_path is None:
        return "Please record or upload audio.", "", ""
    try:
        if API_URL:
            data = _query_api_voice(audio_path, int(top_k))
        else:
            data = _query_inline_voice(audio_path, int(top_k))
        transcribed = data.get("query", "")
        answer, sources, latency = _format_response(data)
        if transcribed:
            answer = f"**📝 Transcribed:** _{transcribed}_\n\n{answer}"
        return answer, sources, latency
    except Exception as e:
        logger.error(f"Voice query error: {e}", exc_info=True)
        return f"❌ Error: {str(e)}", "", ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Voice-RAG | HH Goa 2026",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
        css=".answer-box { font-size: 1.1em; }",
    ) as demo:
        gr.Markdown(
            """
            # 🎙️ Voice-RAG — MS MARCO Indic QA
            **HH Goa 2026 Task 2** | Multi-strategy RAG on `ai4bharat/MSMARCO-XI`

            > Speak or type a question. The pipeline retrieves relevant context from a vector DB
            > and generates a grounded answer using an LLM.
            """
        )

        with gr.Tabs():
            # ── Voice tab ───────────────────────────────────────────────
            with gr.Tab("🎤 Voice Input"):
                gr.Markdown("Record or upload audio (WAV/MP3, max 30s, speak clearly).")
                with gr.Row():
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Record or Upload Audio",
                    )
                    top_k_voice = gr.Slider(1, 10, value=5, step=1, label="Context Chunks (top-k)")

                voice_btn = gr.Button("🔍 Transcribe & Search", variant="primary")
                voice_answer = gr.Markdown(label="Answer", elem_classes=["answer-box"])

                with gr.Accordion("📚 Source Passages", open=False):
                    voice_sources = gr.Markdown()
                with gr.Accordion("⏱️ Latency Breakdown", open=False):
                    voice_latency = gr.Markdown()

                voice_btn.click(
                    fn=handle_voice_query,
                    inputs=[audio_input, top_k_voice],
                    outputs=[voice_answer, voice_sources, voice_latency],
                )

            # ── Text tab ────────────────────────────────────────────────
            with gr.Tab("💬 Text Input"):
                gr.Markdown("Type your question directly (for testing without microphone).")
                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder="e.g. How does photosynthesis work?",
                        label="Question",
                        lines=2,
                    )
                    top_k_text = gr.Slider(1, 10, value=5, step=1, label="Context Chunks (top-k)")

                text_btn = gr.Button("🔍 Search", variant="primary")
                text_answer = gr.Markdown(label="Answer", elem_classes=["answer-box"])

                with gr.Accordion("📚 Source Passages", open=False):
                    text_sources = gr.Markdown()
                with gr.Accordion("⏱️ Latency Breakdown", open=False):
                    text_latency = gr.Markdown()

                text_btn.click(
                    fn=handle_text_query,
                    inputs=[text_input, top_k_text],
                    outputs=[text_answer, text_sources, text_latency],
                )
                text_input.submit(
                    fn=handle_text_query,
                    inputs=[text_input, top_k_text],
                    outputs=[text_answer, text_sources, text_latency],
                )

        gr.Markdown(
            """
            ---
            **Pipeline:** Sarvam STT → FAISS (IVFFlat) → RRF Fusion → Groq LLM
            | **4 chunking strategies:** Fixed-size · Semantic-sentence · Passage-aware · Hierarchical
            | **Guardrails:** Off-topic · Profanity · Grounding · Confidence
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False,
    )
