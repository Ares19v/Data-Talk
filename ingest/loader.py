"""
ingest/loader.py
─────────────────
Streams the ai4bharat/MSMARCO-XI dataset from HuggingFace and yields
raw records for chunking. Supports validation split (always loaded)
and an optional sample of the train split.
"""
from __future__ import annotations

import os
import logging
from typing import Iterator

from datasets import load_dataset, IterableDataset

logger = logging.getLogger(__name__)


def _iter_split(split_name: str, max_records: int | None = None) -> Iterator[dict]:
    """
    Stream a single split of MSMARCO-XI.
    Each record has keys: query, Eng_Query, Answer, Eng_Answer,
    passages {English_passages, Translated_passages, is_selected},
    query_type, query_id, source_lang, target_lang, meta.
    """
    logger.info(f"Streaming split={split_name} max_records={max_records}")
    ds: IterableDataset = load_dataset(
        "ai4bharat/MSMARCO-XI",
        split=split_name,
        streaming=True,
        trust_remote_code=True,
    )
    count = 0
    try:
        for record in ds:
            if max_records is not None and count >= max_records:
                break
            yield record
            count += 1
    except Exception as e:
        logger.warning(f"Stopped streaming {split_name} early due to dataset error: {e}")
    logger.info(f"Finished streaming {count} records from {split_name}")


def stream_dataset(
    train_sample_size: int | None = None,
    include_validation: bool = True,
) -> Iterator[dict]:
    """
    Yield records from validation (up to train_sample_size/2) + train (up to train_sample_size).
    """
    train_n = int(os.getenv("TRAIN_SAMPLE_SIZE", "5000")) if train_sample_size is None else train_sample_size

    if include_validation:
        yield from _iter_split("validation", max_records=train_n)

    if train_n and train_n > 0:
        yield from _iter_split("train", max_records=train_n)
