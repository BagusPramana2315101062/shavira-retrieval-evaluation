"""Orkestrasi utama eksperimen retrieval SHAVIRA."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from .analysis import (
    make_failed_queries,
    make_scenario_matrix,
    make_summary_by_source,
    select_generation_sample,
)
from .constants import CHUNK_CONFIGS, REQUIRED_VALIDATION_COLUMNS
from .data_loader import build_gold_source_map, build_nodes, load_documents
from .generation_eval import make_generation_summary, run_generation_evaluation
from .indexing import build_retrievers, make_faiss_cache_path
from .metrics import choose_best_configuration, evaluate_one_query, make_summary
from .outputs import save_experiment_config, save_result_files
from .retrieval import dedupe_by_context_hash, retrieve_items, rrf_fusion
from .utils import format_preview, make_hash


def get_chunk_configs(args) -> List[Dict[str, Any]]:
    """Mengambil konfigurasi chunking sesuai mode eksperimen."""
    if args.single_config:
        return [
            {
                "chunk_config_id": "C_SINGLE",
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "chunk_category": "single_manual",
            }
        ]
    return CHUNK_CONFIGS


def make_top_contexts(items, max_items: int = 10) -> str:
    """Menyimpan Top-K teks untuk evaluasi generation terbatas."""
    selected = dedupe_by_context_hash(items)[:max_items]
    return "\n\n---CONTEXT_SEPARATOR---\n\n".join([x.text for x in selected])


def run_experiment(args) -> None:
    """Menjalankan eksperimen dari load data sampai output hasil."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_paths = [data_dir / name for name in args.jsonl_files]
    validation_path = data_dir / args.validation_file
    missing = [str(p) for p in jsonl_paths + [validation_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("File berikut belum ditemukan:\n" + "\n".join(missing))

    print("Memuat korpus JSONL")
    documents, stats = load_documents(jsonl_paths, deduplicate=not args.no_deduplicate)
    gold_source_map = build_gold_source_map(documents)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    val_df = pd.read_excel(validation_path)
    missing_cols = REQUIRED_VALIDATION_COLUMNS - set(val_df.columns)
    if missing_cols:
        raise ValueError(f"Kolom dataset validasi tidak lengkap. Kolom hilang: {sorted(missing_cols)}")

    if args.limit:
        val_df = val_df.head(args.limit).copy()

    val_df["gold_hash"] = val_df["Context"].astype(str).map(make_hash)
    val_df["gold_source_file"] = val_df["gold_hash"].map(
        lambda h: gold_source_map.get(str(h), {}).get("gold_source_file", "UNMATCHED_CONTEXT")
    )
    val_df["gold_title"] = val_df["gold_hash"].map(
        lambda h: gold_source_map.get(str(h), {}).get("gold_title", "")
    )
    val_df["gold_url"] = val_df["gold_hash"].map(
        lambda h: gold_source_map.get(str(h), {}).get("gold_url", "")
    )

    max_eval_k = max(args.eval_k)
    candidate_k = max(args.candidate_k, max_eval_k, args.generation_top_k)

    rows_detail: List[Dict[str, Any]] = []
    rows_metric: List[Dict[str, Any]] = []
    index_cache_paths: Dict[str, str] = {}
    started = time.time()

    for chunk_cfg in get_chunk_configs(args):
        chunk_config_id = chunk_cfg["chunk_config_id"]
        chunk_size = int(chunk_cfg["chunk_size"])
        chunk_overlap = int(chunk_cfg["chunk_overlap"])
        chunk_category = str(chunk_cfg.get("chunk_category", ""))

        print("\n" + "=" * 80)
        print(
            f"Chunking {chunk_config_id}: chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}, kategori={chunk_category}"
        )
        nodes = build_nodes(documents, chunk_size, chunk_overlap)
        print(f"Total nodes/chunks: {len(nodes)}")

        index_cache_path = make_faiss_cache_path(
            base_cache_dir=args.index_cache_dir,
            model_name=args.model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_max_length=args.embed_max_length,
            deduplicate=not args.no_deduplicate,
        )
        index_cache_paths[chunk_config_id] = index_cache_path

        bm25_retriever, vector_retriever = build_retrievers(
            nodes=nodes,
            model_name=args.model_name,
            dense_top_k=candidate_k,
            bm25_top_k=candidate_k,
            embed_max_length=args.embed_max_length,
            embed_batch_size=args.embed_batch_size,
            bm25_mode=args.bm25_mode,
            index_cache_dir=index_cache_path,
            force_rebuild_index=args.force_rebuild_index,
        )

        iterator = tqdm(
            val_df.iterrows(),
            total=len(val_df),
            desc=f"Evaluasi query {chunk_config_id}",
        )

        for _, row in iterator:
            qid = row["ID"]
            query = str(row["Question"])
            answer = str(row["Answer"])
            gold_hash = str(row["gold_hash"])
            gold_source_file = str(row["gold_source_file"])
            gold_title = str(row["gold_title"])
            gold_url = str(row["gold_url"])

            bm25_items = retrieve_items(bm25_retriever, query)
            vector_items = retrieve_items(vector_retriever, query)
            hybrid_items = rrf_fusion(
                [bm25_items, vector_items],
                rrf_k=args.rrf_k,
                top_k=candidate_k,
            )

            method_results = {
                "BM25": bm25_items,
                "BGE_M3_FAISS": vector_items,
                "HYBRID_RRF": hybrid_items,
            }

            for method, items in method_results.items():
                deduped = dedupe_by_context_hash(items)

                # Metrik dihitung untuk setiap K.
                for k in args.eval_k:
                    metrics = evaluate_one_query(deduped, gold_hash, k)
                    rows_metric.append(
                        {
                            "ID": qid,
                            "Question": query,
                            "Answer": answer,
                            "gold_hash": gold_hash,
                            "gold_source_file": gold_source_file,
                            "gold_title": gold_title,
                            "gold_url": gold_url,
                            "chunk_config_id": chunk_config_id,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "chunk_category": chunk_category,
                            "method": method,
                            "k": k,
                            **metrics,
                        }
                    )

                top10 = deduped[:10]
                rows_detail.append(
                    {
                        "ID": qid,
                        "Question": query,
                        "Answer": answer,
                        "gold_hash": gold_hash,
                        "gold_source_file": gold_source_file,
                        "gold_title": gold_title,
                        "gold_url": gold_url,
                        "chunk_config_id": chunk_config_id,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "chunk_category": chunk_category,
                        "method": method,
                        "top_hashes": " | ".join([x.text_hash for x in top10]),
                        "top_node_ids": " | ".join([x.node_id for x in top10]),
                        "top_scores": " | ".join(
                            [f"{x.score:.6f}" if not math.isnan(x.score) else "nan" for x in top10]
                        ),
                        "top_titles": " | ".join(
                            [str(x.metadata.get("title", ""))[:80] for x in top10]
                        ),
                        "top_urls": " | ".join(
                            [str(x.metadata.get("url", ""))[:120] for x in top10]
                        ),
                        "top1_preview": format_preview(top10[0].text) if top10 else "",
                        "top1_is_relevant": int(bool(top10 and top10[0].text_hash == gold_hash)),
                        "top_contexts": make_top_contexts(deduped, max_items=max(args.generation_top_k, 10)),
                    }
                )

    metric_df = pd.DataFrame(rows_metric)
    detail_df = pd.DataFrame(rows_detail)

    summary_df = make_summary(metric_df)
    best_configuration_df = choose_best_configuration(summary_df, preferred_k=max_eval_k)

    # Tandai baris konfigurasi terbaik di per-query metrics untuk evaluasi generation.
    metric_df["is_best_configuration"] = 0
    if not best_configuration_df.empty:
        best = best_configuration_df.iloc[0]
        mask = (
            (metric_df["chunk_config_id"] == best["chunk_config_id"])
            & (metric_df["method"] == best["method"])
            & (metric_df["k"] == int(best["k"]))
        )
        metric_df.loc[mask, "is_best_configuration"] = 1

    scenario_matrix_df = make_scenario_matrix(summary_df)
    summary_by_source_df = make_summary_by_source(metric_df)
    failed_queries_df = make_failed_queries(metric_df, detail_df)

    generation_df = pd.DataFrame()
    generation_summary_df = pd.DataFrame()
    if args.run_generation_eval:
        print("\nMenjalankan evaluasi generation terbatas dengan Ollama")
        sample_df = select_generation_sample(metric_df, detail_df, args.generation_sample_size)
        generation_df = run_generation_evaluation(sample_df, args)
        generation_summary_df = make_generation_summary(generation_df)

    config_path = save_experiment_config(args, output_dir, index_cache_paths)
    paths = save_result_files(
        output_dir=output_dir,
        summary_df=summary_df,
        metric_df=metric_df,
        detail_df=detail_df,
        scenario_matrix_df=scenario_matrix_df,
        summary_by_source_df=summary_by_source_df,
        failed_queries_df=failed_queries_df,
        best_configuration_df=best_configuration_df,
        generation_df=generation_df,
        generation_summary_df=generation_summary_df,
    )

    elapsed = time.time() - started
    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== BEST CONFIGURATION ===")
    print(best_configuration_df.head(5).to_string(index=False))

    print("\nFile output:")
    for key, path in paths.items():
        print(f"- {key}: {path}")
    print(f"- config: {config_path}")
    print(f"Durasi total eksperimen: {elapsed:.2f} detik")
