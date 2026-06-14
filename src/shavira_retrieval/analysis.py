"""Analisis pendukung: per sumber korpus, query gagal, dan matriks skenario."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .utils import simple_token_count


def make_scenario_matrix(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Membuat matriks 9 skenario utama untuk Tabel 3.5/hasil eksperimen."""
    cols = [
        "chunk_config_id",
        "chunk_size",
        "chunk_overlap",
        "chunk_category",
        "method",
        "k",
        "success_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "precision_at_k",
        "recall_at_k",
    ]
    return summary_df[cols].copy()


def make_summary_by_source(metric_df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan performa berdasarkan sumber korpus gold context."""
    group_cols = [
        "gold_source_file",
        "chunk_config_id",
        "chunk_size",
        "chunk_overlap",
        "method",
        "k",
    ]
    available = [c for c in group_cols if c in metric_df.columns]
    return (
        metric_df.groupby(available, as_index=False)
        .agg(
            n_queries=("ID", "count"),
            success_at_k=("success_at_k", "mean"),
            mrr_at_k=("mrr_at_k", "mean"),
            ndcg_at_k=("ndcg_at_k", "mean"),
        )
        .sort_values(
            ["gold_source_file", "k", "success_at_k", "mrr_at_k", "ndcg_at_k"],
            ascending=[True, True, False, False, False],
        )
    )


def infer_failure_cause(row: pd.Series) -> str:
    """Memberi kategori awal penyebab query gagal secara heuristik.

    Kategori ini bukan vonis final. Fungsinya untuk membantu analisis manual Bab IV.
    """
    query = str(row.get("Question", ""))
    top1 = str(row.get("top1_preview", ""))
    top_hashes = str(row.get("top_hashes", ""))
    returned = int(row.get("returned_unique_contexts", 0) or 0)
    token_count = simple_token_count(query)

    if returned == 0 or not top_hashes.strip():
        return "no_retrieval_result"
    if token_count <= 3:
        return "query_too_general"
    if len(top1) < 60:
        return "chunk_too_short_or_sparse_context"
    if len(top1) > 220:
        return "possible_noise_from_long_chunk_or_term_mismatch"
    return "gold_context_not_found_in_top_k_possible_term_mismatch"


def make_failed_queries(metric_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    """Mengambil query yang gagal menemukan gold context pada K maksimum."""
    if metric_df.empty or detail_df.empty:
        return pd.DataFrame()

    max_k = int(metric_df["k"].max())
    metric_max = metric_df[metric_df["k"] == max_k].copy()
    failed = metric_max[metric_max["success_at_k"] == 0].copy()

    join_cols = [
        "ID",
        "chunk_config_id",
        "chunk_size",
        "chunk_overlap",
        "method",
        "Question",
        "gold_hash",
        "gold_source_file",
        "gold_title",
        "gold_url",
        "top_hashes",
        "top_node_ids",
        "top_scores",
        "top_titles",
        "top_urls",
        "top1_preview",
    ]
    detail_cols = [c for c in join_cols if c in detail_df.columns]

    merged = failed.merge(
        detail_df[detail_cols],
        on=[
            "ID",
            "chunk_config_id",
            "chunk_size",
            "chunk_overlap",
            "method",
        ],
        how="left",
        suffixes=("", "_detail"),
    )

    for col in ["Question", "gold_hash", "gold_source_file", "gold_title", "gold_url"]:
        detail_col = f"{col}_detail"
        if detail_col in merged.columns:
            merged[col] = merged[col].fillna(merged[detail_col])
            merged = merged.drop(columns=[detail_col])

    merged["failure_cause_initial"] = merged.apply(infer_failure_cause, axis=1)
    merged["analysis_note"] = (
        "Kategori penyebab bersifat awal/heuristik. Validasi manual diperlukan "
        "untuk membedakan perbedaan istilah, query terlalu umum, chunk terlalu "
        "pendek/panjang, gold context terpecah, atau metadata/text_hash tidak sesuai."
    )

    order_cols = [
        "ID",
        "Question",
        "gold_source_file",
        "chunk_config_id",
        "chunk_size",
        "chunk_overlap",
        "method",
        "k",
        "success_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "returned_unique_contexts",
        "failure_cause_initial",
        "top_titles",
        "top_urls",
        "top1_preview",
        "gold_title",
        "gold_url",
        "analysis_note",
    ]
    return merged[[c for c in order_cols if c in merged.columns]].sort_values(
        ["chunk_config_id", "method", "ID"]
    )


def select_generation_sample(metric_df: pd.DataFrame, detail_df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Memilih sampel evaluasi generation dari konfigurasi terbaik.

    Prioritas: query yang berhasil pada Top-K, karena generation memang dievaluasi
    setelah retrieval terbaik menyediakan konteks.
    """
    if metric_df.empty or detail_df.empty:
        return pd.DataFrame()

    best_rows = metric_df[metric_df["is_best_configuration"] == 1].copy()
    if best_rows.empty:
        return pd.DataFrame()

    k = int(best_rows["k"].max())
    best_rows = best_rows[best_rows["k"] == k].copy()

    merged = best_rows.merge(
        detail_df,
        on=["ID", "chunk_config_id", "chunk_size", "chunk_overlap", "method"],
        how="left",
        suffixes=("", "_detail"),
    )

    merged = merged.sort_values(
        ["success_at_k", "mrr_at_k", "ndcg_at_k", "ID"],
        ascending=[False, False, False, True],
    )

    return merged.head(sample_size).copy()
