"""Fungsi utilitas umum."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


def make_hash(text: str) -> str:
    """Membuat hash stabil untuk teks context/record asal."""
    normalized = " ".join(str(text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    """Membuat nama aman untuk folder cache."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Membaca file JSONL sebagai iterator dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON tidak valid pada {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Record JSONL pada {path}:{line_no} bukan object JSON.")
            yield obj


def install_hint() -> str:
    """Pesan bantuan ketika dependency LlamaIndex belum lengkap."""
    return (
        "Dependency belum lengkap. Jalankan:\n"
        "pip install -r requirements.txt\n\n"
        "Jika memakai CUDA untuk embedding, pastikan PyTorch CUDA sudah sesuai versi driver.\n"
        "Jika memakai Ollama untuk evaluasi generation, jalankan juga:\n"
        "ollama serve\n"
        "ollama pull llama3.1:8b"
    )


def format_preview(text: str, max_chars: int = 250) -> str:
    """Memotong teks agar output CSV tetap mudah dibaca."""
    text = " ".join(str(text).split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def clean_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan nilai dataframe agar aman ditulis ke Excel/openpyxl.

    Excel/openpyxl menolak karakter kontrol tertentu dan membatasi panjang isi
    cell. Fungsi ini menjaga tipe numerik tetap numerik, membersihkan string,
    dan memotong teks yang terlalu panjang.
    """
    cleaned = df.copy()
    illegal = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")

    def clean_value(value):
        if value is None:
            return value
        try:
            if pd.isna(value):
                return value
        except Exception:
            pass

        if isinstance(value, (int, float, bool)):
            return value

        text = illegal.sub("", str(value))
        return text[:32000] if len(text) > 32000 else text

    for col in cleaned.columns:
        cleaned[col] = cleaned[col].map(clean_value)
    return cleaned

def simple_token_count(text: str) -> int:
    """Tokenisasi sederhana untuk analisis heuristik query gagal."""
    return len(re.findall(r"\w+", str(text).lower()))
