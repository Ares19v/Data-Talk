"""
scripts/build_index.py
──────────────────────
One-time offline script to:
1. Stream MSMARCO-XI from HuggingFace
2. Apply all 4 chunking strategies
3. Embed all chunks
4. Build FAISS indices (per-strategy + combined)
5. Save to disk

Run:
    python scripts/build_index.py

Env vars (from .env):
    TRAIN_SAMPLE_SIZE=5000
    FAISS_INDEX_PATH=./data/faiss_index.bin
    METADATA_PATH=./data/metadata.pkl
"""
import os
import sys
import time
import logging

# Make root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_index")

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from ingest.loader import stream_dataset
from ingest.chunkers import MultiStrategyChunker
from indexer.faiss_store import FAISSStore

console = Console()

def main():
    index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index.bin")
    metadata_path = os.getenv("METADATA_PATH", "./data/metadata.pkl")
    train_sample = int(os.getenv("TRAIN_SAMPLE_SIZE", "5000"))

    console.print(f"[bold green]Voice-RAG Index Builder[/bold green]")
    console.print(f"  Train sample size : {train_sample}")
    console.print(f"  Index path        : {index_path}")
    console.print(f"  Metadata path     : {metadata_path}")
    console.print()

    chunker = MultiStrategyChunker()
    store = FAISSStore()

    all_chunks = []
    record_count = 0
    t0 = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Streaming & chunking dataset...", total=None)

        for record in stream_dataset(train_sample_size=train_sample):
            chunks = chunker.chunk(record)
            # assign chunk_ids
            for i, chunk in enumerate(chunks):
                if not chunk.chunk_id:
                    chunk.chunk_id = f"{record_count}_{chunk.strategy}_{i}"
            all_chunks.extend(chunks)
            record_count += 1

            if record_count % 1000 == 0:
                progress.update(
                    task,
                    description=f"Records: {record_count:,} | Chunks: {len(all_chunks):,}",
                )

    chunk_time = time.perf_counter() - t0
    console.print(f"[green]✓[/green] Chunked {record_count:,} records → {len(all_chunks):,} chunks in {chunk_time:.1f}s")

    # Deduplicate by text content
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        key = c.text.strip().lower()[:200]
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)
    console.print(f"[green]✓[/green] After dedup: {len(unique_chunks):,} unique chunks")

    # Build FAISS indices
    console.print("Building FAISS indices...")
    t1 = time.perf_counter()
    store.build_from_chunks(unique_chunks, batch_size=512)
    index_time = time.perf_counter() - t1
    console.print(f"[green]✓[/green] Indexed {store.total_vectors():,} vectors in {index_time:.1f}s")
    console.print(f"  Strategies: {store.get_strategies()}")

    # Save
    store.save(index_path, metadata_path)
    console.print(f"[bold green]✓ Done! Index saved to {metadata_path}[/bold green]")
    console.print(f"  Total time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
