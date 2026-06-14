"""Penyimpanan konfigurasi dan output eksperimen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .constants import CHUNK_CONFIGS, PRIMARY_METRICS, RETRIEVAL_METHODS, SUPPORTING_METRICS
from .utils import clean_dataframe_for_excel


def save_experiment_config(
    args: argparse.Namespace,
    output_dir: Path,
    index_cache_paths: Dict[str, str],
) -> Path:
    """Menyimpan konfigurasi eksperimen agar hasil mudah dilacak."""
    config = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "jsonl_files": args.jsonl_files,
        "validation_file": args.validation_file,
        "mode": "single_config" if args.single_config else "grid_3x3",
        "chunk_configs": [
            {
                "chunk_config_id": c["chunk_config_id"],
                "chunk_size": c["chunk_size"],
                "chunk_overlap": c["chunk_overlap"],
                "chunk_category": c["chunk_category"],
            }
            for c in ([
                {
                    "chunk_config_id": "C_SINGLE",
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": args.chunk_overlap,
                    "chunk_category": "single_manual",
                }
            ] if args.single_config else CHUNK_CONFIGS)
        ],
        "eval_k": args.eval_k,
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "model_name": args.model_name,
        "embed_max_length": args.embed_max_length,
        "embed_batch_size": args.embed_batch_size,
        "bm25_mode": args.bm25_mode,
        "deduplicate": not args.no_deduplicate,
        "evaluation_level": "record/context hash",
        "primary_metrics": PRIMARY_METRICS,
        "supporting_metrics": SUPPORTING_METRICS,
        "retrieval_methods": RETRIEVAL_METHODS,
        "index_cache_dir": args.index_cache_dir,
        "index_cache_paths": index_cache_paths,
        "force_rebuild_index": args.force_rebuild_index,
        "generation_eval_enabled": bool(args.run_generation_eval),
        "ollama_model": args.ollama_model if args.run_generation_eval else None,
        "generation_sample_size": args.generation_sample_size if args.run_generation_eval else None,
        "generation_top_k": args.generation_top_k if args.run_generation_eval else None,
    }
    config_path = output_dir / "experiment_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config_path


def save_result_files(
    output_dir: Path,
    summary_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    scenario_matrix_df: pd.DataFrame,
    summary_by_source_df: pd.DataFrame,
    failed_queries_df: pd.DataFrame,
    best_configuration_df: pd.DataFrame,
    generation_df: Optional[pd.DataFrame] = None,
    generation_summary_df: Optional[pd.DataFrame] = None,
):
    """Menyimpan hasil eksperimen ke CSV dan Excel."""
    paths = {
        "summary": output_dir / "summary_metrics.csv",
        "metric": output_dir / "per_query_metrics.csv",
        "detail": output_dir / "retrieval_details_top10.csv",
        "scenario_matrix": output_dir / "scenario_matrix.csv",
        "summary_by_source": output_dir / "summary_by_source.csv",
        "failed_queries": output_dir / "failed_queries.csv",
        "best_configuration": output_dir / "best_configuration.csv",
        "xlsx": output_dir / "summary_and_details.xlsx",
    }

    summary_df.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    metric_df.to_csv(paths["metric"], index=False, encoding="utf-8-sig")
    detail_df.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    scenario_matrix_df.to_csv(paths["scenario_matrix"], index=False, encoding="utf-8-sig")
    summary_by_source_df.to_csv(paths["summary_by_source"], index=False, encoding="utf-8-sig")
    failed_queries_df.to_csv(paths["failed_queries"], index=False, encoding="utf-8-sig")
    best_configuration_df.to_csv(paths["best_configuration"], index=False, encoding="utf-8-sig")

    if generation_df is not None and not generation_df.empty:
        paths["generation_eval"] = output_dir / "generation_eval_limited.csv"
        generation_df.to_csv(paths["generation_eval"], index=False, encoding="utf-8-sig")

    if generation_summary_df is not None and not generation_summary_df.empty:
        paths["generation_summary"] = output_dir / "generation_eval_summary.csv"
        generation_summary_df.to_csv(paths["generation_summary"], index=False, encoding="utf-8-sig")

    # Excel bersifat tambahan. CSV tetap menjadi output utama karena lebih stabil
    # untuk file besar dan metadata dokumen yang panjang. Jika Excel gagal dibuat,
    # program tidak dihentikan; pesan error ditulis ke excel_export_warning.txt.
    try:
        with pd.ExcelWriter(paths["xlsx"], engine="openpyxl") as writer:
            sheets = {
                "summary_metrics": summary_df,
                "scenario_matrix": scenario_matrix_df,
                "summary_by_source": summary_by_source_df,
                "best_configuration": best_configuration_df,
                "failed_queries": failed_queries_df,
                "per_query_metrics": metric_df,
                "retrieval_details_top10": detail_df,
            }
            if generation_summary_df is not None and not generation_summary_df.empty:
                sheets["generation_eval_summary"] = generation_summary_df
            if generation_df is not None and not generation_df.empty:
                sheets["generation_eval_limited"] = generation_df

            for sheet_name, df in sheets.items():
                clean_dataframe_for_excel(df).to_excel(writer, sheet_name=sheet_name[:31], index=False)
    except Exception as exc:
        warning_path = output_dir / "excel_export_warning.txt"
        warning_path.write_text(
            "CSV berhasil disimpan, tetapi ekspor Excel gagal. "
            f"Gunakan file CSV sebagai output utama. Detail error: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        paths.pop("xlsx", None)
        paths["excel_warning"] = warning_path

    return paths
