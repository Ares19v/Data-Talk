"""
evaluate_pipeline.py
────────────────────
Evaluates the RAG pipeline's Retrieval and Generation accuracy using
the MSMARCO-XI validation split.

Metrics computed:
- Retrieval: Hit Rate @ 5, Mean Reciprocal Rank (MRR) @ 5
- Generation: ROUGE-L F1 against ground-truth answers
- Latency: Average end-to-end pipeline latency
"""

import sys
import time
import os
import logging
from statistics import mean
from tqdm import tqdm

from rouge_score import rouge_scorer
from datasets import load_dataset

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.harness import get_harness, RAGRequest
from ingest.loader import stream_dataset
from dotenv import load_dotenv

load_dotenv()

# Disable noisy logs
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

def is_hit(retrieved_text: str, gt_passages: list[str]) -> bool:
    """
    Check if retrieved chunk matches any ground truth passage.
    """
    ret_lower = retrieved_text.lower().strip()
    for gt in gt_passages:
        gt_lower = gt.lower().strip()
        if gt_lower in ret_lower or ret_lower in gt_lower:
            return True
        if len(gt_lower) > 50 and gt_lower[:50] in ret_lower:
            return True
    return False

def run_evaluation(num_samples: int = 50, top_k: int = 5):
    print(f"\n🚀 Starting RAG Evaluation (N={num_samples}, top_k={top_k})")
    print("Loading RAG harness (FAISS index + LLM)...")
    harness = get_harness()
    harness.input_guardrails.off_topic_threshold = 1.0 # Disable off-topic for eval
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    print("Streaming validation dataset...")
    ds = stream_dataset(train_sample_size=50, include_validation=True)

    results = []
    
    hits = 0
    mrr_sum = 0.0
    rouge_l_sum = 0.0
    latency_sum = 0.0
    failures = 0

    count = 0
    for record in tqdm(ds, total=num_samples, desc="Evaluating"):
        query = record.get("Eng_Query") or record.get("query")
        gt_answer = record.get("Eng_Answer") or record.get("Answer")
        passages_obj = record.get("passages") or {}
        eng_passages = passages_obj.get("English_passages", [])
        is_selected = passages_obj.get("is_selected", [])
        
        gt_passages = []
        for i, p in enumerate(eng_passages):
            if i < len(is_selected) and int(is_selected[i]) == 1 and p.strip():
                gt_passages.append(p)

        if not query or not gt_answer or not gt_passages:
            continue

        req = RAGRequest(query=query, top_k=top_k, include_sources=True)
        resp = harness.run(req)

        if not resp.success:
            failures += 1
            print(f"\n[FAIL] Query: '{query}' -> {resp.guardrail_reason}")
            continue

        hit_rank = 0
        for i, src in enumerate(resp.sources):
            if is_hit(src.text, gt_passages):
                hit_rank = i + 1
                break
        
        if hit_rank > 0:
            hits += 1
            mrr_sum += 1.0 / hit_rank

        r_scores = scorer.score(gt_answer, resp.answer)
        rouge_l_f1 = r_scores["rougeL"].fmeasure
        rouge_l_sum += rouge_l_f1

        latency_sum += resp.latency.total_pipeline_ms

        count += 1
        if count >= num_samples:
            break

    print("DEBUG: Finished loop, calculating metrics...")

    if count == 0:
        print("No valid records evaluated.")
        return

    hit_rate = hits / count
    mrr = mrr_sum / count
    avg_rouge_l = rouge_l_sum / count
    avg_latency = latency_sum / count

    print("\n" + "="*50)
    print(" 📊 EVALUATION RESULTS ")
    print("="*50)
    print(f"Total Evaluated : {count}")
    print(f"Guardrail Blocks: {failures}")
    print("-" * 50)
    print("RETRIEVAL (Top-{k})".format(k=top_k))
    print(f"Hit Rate        : {hit_rate:.2%}")
    print(f"MRR             : {mrr:.4f}")
    print("-" * 50)
    print("GENERATION")
    print(f"ROUGE-L F1      : {avg_rouge_l:.4f}")
    print("-" * 50)
    print("PERFORMANCE")
    print(f"Avg Latency     : {avg_latency:.0f} ms/query")
    print("="*50 + "\n")
    
    os._exit(0)

if __name__ == "__main__":
    run_evaluation(num_samples=5, top_k=5)
