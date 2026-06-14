"""Perhitungan metrik evaluasi retrieval."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .retrieval import RetrievedItem, dedupe_by_context_hash


def evaluate_one_query(
    items: Sequence[RetrievedItem],
    gold_hash: str,
    k: int,
) -> Dict[str, float]:
    """Evaluasi satu query berdasarkan satu gold context utama.

    Metrik utama:
    - Success@K: 1 jika gold context ditemukan dalam Top-K, 0 jika tidak.
    - MRR@K: reciprocal rank dari posisi pertama gold context pada Top-K.
    - nDCG@K: kualitas ranking dengan gain biner. Karena satu gold context,
      IDCG = 1 jika gold tersedia, sedangkan DCG = 1/log2(rank+1).
    """
    top = dedupe_by_context_hash(items)[:k]
    rel = [1 if item.text_hash == gold_hash else 0 for item in top]

    success_at_k = 1 if any(rel) else 0
    precision_at_k = sum(rel) / k
    recall_at_k = float(success_at_k)

    first_rank = 0
    for idx, val in enumerate(rel, start=1):
        if val == 1:
            first_rank = idx
            break

    mrr_at_k = 1.0 / first_rank if first_rank else 0.0
    ndcg_at_k = (1.0 / math.log2(first_rank + 1)) if first_rank else 0.0

    return {
        "success_at_k": float(success_at_k),
        "hit_at_k": float(success_at_k),  # alias agar output lama tetap terbaca
        "mrr_at_k": float(mrr_at_k),
        "mrr": float(mrr_at_k),  # alias lama
        "ndcg_at_k": float(ndcg_at_k),
        "precision_at_k": float(precision_at_k),
        "precision": float(precision_at_k),
        "recall_at_k": float(recall_at_k),
        "recall": float(recall_at_k),
        "first_relevant_rank": int(first_rank),
        "returned_unique_contexts": int(len(top)),
    }


def make_summary(metric_df: pd.DataFrame) -> pd.DataFrame:
    """Membuat ringkasan metrik per konfigurasi chunking, metode, dan K."""
    return (
        metric_df.groupby(
            [
                "chunk_config_id",
                "chunk_size",
                "chunk_overlap",
                "chunk_category",
                "method",
                "k",
            ],
            as_index=False,
        )
        .agg(
            n_queries=("ID", "count"),
            success_at_k=("success_at_k", "mean"),
            mrr_at_k=("mrr_at_k", "mean"),
            ndcg_at_k=("ndcg_at_k", "mean"),
            precision_at_k=("precision_at_k", "mean"),
            recall_at_k=("recall_at_k", "mean"),
            avg_first_relevant_rank=(
                "first_relevant_rank",
                lambda s: np.mean([x for x in s if x > 0]) if (s > 0).any() else 0,
            ),
        )
        .sort_values(
            ["k", "success_at_k", "mrr_at_k", "ndcg_at_k"],
            ascending=[True, False, False, False],
        )
    )


def choose_best_configuration(summary_df: pd.DataFrame, preferred_k: int = 10) -> pd.DataFrame:
    """Menentukan konfigurasi terbaik berdasarkan Success@K, MRR@K, dan nDCG@K."""
    candidates = summary_df[summary_df["k"] == preferred_k].copy()
    if candidates.empty:
        candidates = summary_df.copy()

    # Skor komposit hanya untuk ranking konfigurasi; metrik utama tetap dilaporkan terpisah.
    candidates["composite_score"] = (
        candidates["success_at_k"] * 0.40
        + candidates["mrr_at_k"] * 0.30
        + candidates["ndcg_at_k"] * 0.30
    )

    return candidates.sort_values(
        ["composite_score", "success_at_k", "mrr_at_k", "ndcg_at_k"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
