# Voice-RAG Build Tasks

## Phase 1: Scaffold & Config
- [x] Create project directory structure
- [x] .env.example
- [x] requirements.txt
- [x] README.md

## Phase 2: Ingest & Chunking
- [x] ingest/loader.py — HuggingFace streaming loader
- [x] ingest/chunkers.py — 4 chunking strategies

## Phase 3: Indexer
- [x] indexer/embedder.py — sentence-transformer encoder
- [x] indexer/faiss_store.py — FAISS IVF + Flat build/query
- [x] scripts/build_index.py — offline index builder

## Phase 4: Pipeline Core
- [x] stt/sarvam.py — Sarvam STT wrapper
- [x] pipeline/guardrails.py — input/output guardrails
- [x] pipeline/retriever.py — multi-strategy RRF retrieval
- [x] pipeline/generator.py — LLM answer generation
- [x] pipeline/harness.py — orchestration + retries

## Phase 5: Analytics
- [x] analytics/latency.py — P50/P70/P100 benchmark

## Phase 6: API + UI
- [x] api/app.py — FastAPI endpoints
- [x] ui/app.py — Gradio voice UI
- [x] app.py — HuggingFace Spaces entry point

## Phase 7: Tests
- [x] tests/bench.py — latency benchmark
- [x] tests/test_guardrails.py — guardrail unit tests
- [x] tests/smoke_test.py — full pipeline smoke test

## Phase 8: Verify
- [x] pip install requirements ✅
- [x] All 10 modules import cleanly ✅
- [x] 12/12 guardrail unit tests PASSED ✅
