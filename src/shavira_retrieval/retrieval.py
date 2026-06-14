"""Struktur hasil retrieval, deduplikasi, dan Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass
class RetrievedItem:
    """Representasi hasil retrieval yang tidak bergantung langsung pada LlamaIndex."""

    node_id: str
    text_hash: str
    score: float
    text: str
    metadata: Dict[str, Any]


def node_to_item(node_with_score) -> RetrievedItem:
    """Mengubah NodeWithScore LlamaIndex menjadi RetrievedItem."""
    node = node_with_score.node
    metadata = dict(node.metadata or {})
    node_id = getattr(node, "node_id", None) or getattr(node, "id_", "")
    text = getattr(node, "text", None) or node.get_content()
    score = node_with_score.score
    if score is None:
        score = float("nan")

    return RetrievedItem(
        node_id=str(node_id),
        text_hash=str(metadata.get("text_hash", "")),
        score=float(score),
        text=str(text),
        metadata=metadata,
    )


def retrieve_items(retriever, query: str) -> List[RetrievedItem]:
    """Menjalankan query pada retriever."""
    return [node_to_item(nws) for nws in retriever.retrieve(query)]


def dedupe_by_context_hash(items: Sequence[RetrievedItem]) -> List[RetrievedItem]:
    """Deduplikasi hasil berdasarkan record/context asal, bukan chunk."""
    seen = set()
    deduped = []
    for item in items:
        key = item.text_hash or item.node_id
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def rrf_fusion(
    ranked_lists: Sequence[Sequence[RetrievedItem]],
    rrf_k: int = 10,
    top_k: int = 10,
) -> List[RetrievedItem]:
    """Reciprocal Rank Fusion manual berbasis text_hash/node_id."""
    fused_scores: Dict[str, float] = {}
    best_item: Dict[str, RetrievedItem] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = item.text_hash or item.node_id
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))
            if key not in best_item:
                best_item[key] = item

    sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
    output = []
    for key in sorted_keys[:top_k]:
        item = best_item[key]
        output.append(
            RetrievedItem(
                node_id=item.node_id,
                text_hash=item.text_hash,
                score=fused_scores[key],
                text=item.text,
                metadata=item.metadata,
            )
        )
    return output
