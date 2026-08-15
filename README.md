# Voice-Enabled RAG System
## HH Goa 2026 — Task 2

A production-grade voice-enabled Retrieval-Augmented Generation pipeline on the **`ai4bharat/MSMARCO-XI`** dataset (MS MARCO translated into 22 Indic languages).

![Data Talk Preview](./preview.png)

**Pipeline:** `Voice Input → Sarvam STT → Multi-Strategy Chunking/FAISS → RRF Fusion → LLM → Guardrailed Answer`

---

## Architecture

```
Microphone (React/Vite UI)
     │
     ▼
Sarvam STT (saaras:v3)          ~300–800ms (network)
     │
     ▼
Input Guardrails                <5ms
  • Profanity filter
  • Off-topic cosine distance check
  • Length validation
     │
     ▼
Query Encoder (all-MiniLM-L6-v2) ~5ms
     │
     ▼
FAISS Multi-Strategy Search     ~8–15ms
  ┌─ Strategy 1: Fixed-size chunks (256 tok, 64 overlap)
  ├─ Strategy 2: Semantic sentence chunks (≤192 tok)
  ├─ Strategy 3: Passage-aware (dataset's own passages)
  └─ Strategy 4: Hierarchical (parent + child sentences)
     │
     ▼
RRF Fusion                      <1ms
     │
     ▼
Groq LLM (llama-3.1-8b-instant) ~200–400ms
     │
     ▼
Output Guardrails               <5ms
  • Grounding check (ROUGE-1 recall ≥ 0.10)
  • Confidence gate (FAISS distance threshold)
  • Hallucination heuristic
     │
     ▼
Structured Response (Pydantic)
```

## Latency Targets

| Path | Target |
|---|---|
| Retrieval only (encode + FAISS + RRF) | **< 200ms** ✅ |
| Full pipeline with LLM | ~400–600ms |
| Full with STT | ~700–1400ms |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your Sarvam AI and Groq keys
```

### 3. Build the vector index (one-time)

```bash
python scripts/build_index.py
```

This streams the MSMARCO-XI dataset, applies all 4 chunking strategies, embeds everything with `all-MiniLM-L6-v2`, and saves FAISS indices.

### 4. Run the API Server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
Endpoints: `POST /query`, `POST /voice-query`, `GET /health`

### 5. Run the React Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## Benchmark

```bash
# Run P50/P70/P90/P100 latency benchmark (100 queries)
python tests/bench.py --n 100

# Run unit tests
pytest tests/test_guardrails.py -v

# Full smoke test (requires built index)
python tests/smoke_test.py
```

---

## Chunking Strategies

| Strategy | Description |
|---|---|
| **Fixed-size + overlap** | 256-token windows, 64-token overlap |
| **Semantic sentence** | NLTK sentence split, merge up to 192 tokens |
| **Passage-aware** | Uses `passages.English_passages` from the dataset directly |
| **Hierarchical** | Parent = full answer, children = individual sentences with `parent_id` |

All four strategies build separate FAISS partitions. At query time, results are merged using **Reciprocal Rank Fusion (RRF)**.

---

## Guardrails

**Input:**
- Profanity/unsafe content (`better_profanity`)
- Off-topic detection (cosine distance to domain centroid)
- Query length validation

**Output:**
- ROUGE-1 recall grounding check (answer must cite retrieved context)
- FAISS confidence gate (reject if top retrieved doc is too dissimilar)
- Hallucination heuristic (detect "I don't know" when context exists)

---

## Deployment (HuggingFace Spaces)

1. Create a new HF Space (Gradio SDK)
2. Push this repo
3. Add secrets: `SARVAM_API_KEY`, `GROQ_API_KEY`, `FAISS_INDEX_PATH`, `METADATA_PATH`
4. The pre-built index can be stored in the Space's persistent storage or downloaded from HF Hub

---

## Tech Stack

| Component | Library |
|---|---|
| STT | `sarvamai` (Saaras v3) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector DB | `faiss-cpu` (IVFFlat + FlatIP) |
| Chunking | Custom + `nltk` |
| LLM | `groq` (llama-3.1-8b-instant) |
| Guardrails | Custom + `better-profanity`, `rouge-score` |
| API | `fastapi` + `uvicorn` |
| UI | `gradio` |
| Retries | `tenacity` |
| Dataset | `datasets` (HuggingFace streaming) |
