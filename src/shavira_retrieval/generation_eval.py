"""Evaluasi generation terbatas menggunakan Ollama melalui LlamaIndex."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import pandas as pd


def _extract_json(text: str) -> Dict[str, Any]:
    """Mengambil JSON dari respons LLM secara toleran."""
    text = str(text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def build_generation_prompt(question: str, contexts: List[str]) -> str:
    """Prompt generator jawaban berbasis konteks."""
    joined_context = "\n\n".join(
        [f"[KONTEKS {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    )
    return f"""Anda adalah asisten akademik SHAVIRA. Jawab pertanyaan pengguna hanya berdasarkan konteks yang diberikan. Jika konteks tidak memuat informasi yang cukup, nyatakan bahwa informasi tidak tersedia dalam konteks.

{joined_context}

PERTANYAAN:
{question}

JAWABAN BERBASIS KONTEKS:
"""


def build_judge_prompt(question: str, contexts: List[str], ground_truth: str, generated_answer: str) -> str:
    """Prompt penilaian jawaban untuk evaluasi generation terbatas."""
    joined_context = "\n\n".join(
        [f"[KONTEKS {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    )
    return f"""Nilai jawaban yang dihasilkan sistem berdasarkan konteks dan ground truth.

Kriteria skor 1-5:
1 = sangat buruk/tidak sesuai
2 = kurang sesuai
3 = cukup sesuai
4 = sesuai
5 = sangat sesuai

Berikan JSON saja dengan format:
{{
  "context_alignment": <1-5>,
  "ground_truth_alignment": <1-5>,
  "completeness": <1-5>,
  "outside_context_information": <0 atau 1>,
  "short_reason": "<alasan singkat>"
}}

KONTEKS:
{joined_context}

PERTANYAAN:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

JAWABAN SISTEM:
{generated_answer}
"""


def split_contexts_from_detail(row: pd.Series, max_contexts: int) -> List[str]:
    """Mengambil konteks dari output detail.

    Saat ini detail menyimpan preview Top-1. Untuk generation yang lebih kuat,
    `top_contexts` diisi oleh experiment.py dengan gabungan Top-K text.
    """
    if "top_contexts" in row and pd.notna(row["top_contexts"]):
        raw = str(row["top_contexts"])
        parts = [p.strip() for p in raw.split("\n\n---CONTEXT_SEPARATOR---\n\n") if p.strip()]
        return parts[:max_contexts]

    preview = str(row.get("top1_preview", "") or "").strip()
    return [preview] if preview else []


def run_generation_evaluation(sample_df: pd.DataFrame, args) -> pd.DataFrame:
    """Menjalankan evaluasi generation terbatas dengan Ollama.

    LLM dipakai sebagai generator dan judge sederhana. Output tetap harus
    dibaca sebagai evaluasi pendukung, bukan metrik utama retrieval.
    """
    if sample_df.empty:
        return pd.DataFrame()

    try:
        from llama_index.llms.ollama import Ollama
    except Exception as exc:
        raise RuntimeError(
            "Paket Ollama LlamaIndex belum tersedia. Install:\n"
            "pip install llama-index-llms-ollama"
        ) from exc

    llm = Ollama(
        model=args.ollama_model,
        base_url=args.ollama_base_url,
        request_timeout=float(args.ollama_timeout),
        temperature=0.0,
    )

    rows = []
    for _, row in sample_df.iterrows():
        question = str(row.get("Question", ""))
        ground_truth = str(row.get("Answer", ""))
        contexts = split_contexts_from_detail(row, max_contexts=int(args.generation_top_k))

        generation_prompt = build_generation_prompt(question, contexts)
        generated = str(llm.complete(generation_prompt)).strip()

        judge_prompt = build_judge_prompt(question, contexts, ground_truth, generated)
        judge_raw = str(llm.complete(judge_prompt)).strip()
        judge = _extract_json(judge_raw)

        rows.append(
            {
                "ID": row.get("ID", ""),
                "chunk_config_id": row.get("chunk_config_id", ""),
                "chunk_size": row.get("chunk_size", ""),
                "chunk_overlap": row.get("chunk_overlap", ""),
                "method": row.get("method", ""),
                "Question": question,
                "ground_truth_answer": ground_truth,
                "generated_answer": generated,
                "context_alignment": judge.get("context_alignment", ""),
                "ground_truth_alignment": judge.get("ground_truth_alignment", ""),
                "completeness": judge.get("completeness", ""),
                "outside_context_information": judge.get("outside_context_information", ""),
                "judge_reason": judge.get("short_reason", ""),
                "judge_raw": judge_raw,
                "n_contexts": len(contexts),
                "ollama_model": args.ollama_model,
            }
        )

    return pd.DataFrame(rows)


def make_generation_summary(generation_df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan skor evaluasi generation terbatas."""
    if generation_df.empty:
        return pd.DataFrame()

    numeric_cols = [
        "context_alignment",
        "ground_truth_alignment",
        "completeness",
        "outside_context_information",
    ]
    df = generation_df.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.groupby(["chunk_config_id", "chunk_size", "chunk_overlap", "method", "ollama_model"], as_index=False)
        .agg(
            n_samples=("ID", "count"),
            avg_context_alignment=("context_alignment", "mean"),
            avg_ground_truth_alignment=("ground_truth_alignment", "mean"),
            avg_completeness=("completeness", "mean"),
            outside_context_rate=("outside_context_information", "mean"),
        )
    )
