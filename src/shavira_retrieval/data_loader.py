"""Load data JSONL dan pembentukan node/chunk."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

from .utils import install_hint, make_hash, read_jsonl


def load_documents(jsonl_paths: Sequence[Path], deduplicate: bool = True):
    """Muat record JSONL menjadi LlamaIndex Document.

    Setiap record diberi metadata `source_file` dan `text_hash`. Metadata ini
    dipakai untuk analisis per sumber korpus dan pencocokan gold context.
    """
    try:
        from llama_index.core import Document
    except Exception as exc:
        raise RuntimeError(install_hint()) from exc

    documents = []
    seen_hashes = set()
    total = 0
    skipped_empty = 0
    skipped_duplicate = 0

    for jsonl_path in jsonl_paths:
        for record_no, obj in enumerate(read_jsonl(jsonl_path), start=1):
            total += 1
            text = str(obj.get("text", "") or "").strip()
            if not text:
                skipped_empty += 1
                continue

            h = make_hash(text)
            if deduplicate and h in seen_hashes:
                skipped_duplicate += 1
                continue
            seen_hashes.add(h)

            raw_meta = obj.get("metadata", {}) or {}
            metadata = {
                "source_file": jsonl_path.name,
                "source_record_no": record_no,
                "text_hash": h,
                "url": raw_meta.get("url", ""),
                "title": raw_meta.get("title", ""),
                "page": raw_meta.get("page", ""),
                "page_label": raw_meta.get("page_label", ""),
                "page_id": raw_meta.get("page_id", ""),
                "content_type": raw_meta.get("content_type", ""),
                "target_site": raw_meta.get("target_site", ""),
                "last_updated": raw_meta.get("last_updated", ""),
            }
            excluded_keys = list(metadata.keys())

            documents.append(
                Document(
                    text=text,
                    metadata=metadata,
                    excluded_embed_metadata_keys=excluded_keys,
                    excluded_llm_metadata_keys=excluded_keys,
                )
            )

    stats = {
        "total_records": total,
        "documents_used": len(documents),
        "skipped_empty": skipped_empty,
        "skipped_duplicate": skipped_duplicate,
    }
    return documents, stats


def build_gold_source_map(documents) -> Dict[str, Dict[str, str]]:
    """Membuat peta hash context ke metadata sumber korpus."""
    output: Dict[str, Dict[str, str]] = {}
    for doc in documents:
        metadata = dict(doc.metadata or {})
        text_hash = str(metadata.get("text_hash", ""))
        if text_hash:
            output[text_hash] = {
                "gold_source_file": str(metadata.get("source_file", "")),
                "gold_title": str(metadata.get("title", "")),
                "gold_url": str(metadata.get("url", "")),
                "gold_content_type": str(metadata.get("content_type", "")),
            }
    return output


def build_nodes(documents, chunk_size: int, chunk_overlap: int):
    """Chunking dokumen dengan SentenceSplitter LlamaIndex."""
    try:
        from llama_index.core.node_parser import SentenceSplitter
    except Exception as exc:
        raise RuntimeError(install_hint()) from exc

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

    # Node ID stabil agar cache dan output dapat ditelusuri.
    running_per_hash: Dict[str, int] = {}
    for node in nodes:
        h = node.metadata.get("text_hash", "nohash")
        idx = running_per_hash.get(h, 0)
        running_per_hash[h] = idx + 1
        node.id_ = f"{h}_chunk_{idx:04d}"

    return nodes
