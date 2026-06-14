"""Pembuatan embedding model, indeks FAISS, dan retriever BM25."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .utils import install_hint, safe_name


def make_embed_model(
    model_name: str,
    embed_max_length: Optional[int] = None,
    embed_batch_size: int = 8,
):
    """Membuat embedding model BGE-M3.

    GPU CUDA dipakai otomatis jika `torch.cuda.is_available()` bernilai True.
    """
    try:
        import torch
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except Exception as exc:
        raise RuntimeError(install_hint()) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding device: {device}")

    kwargs = {
        "model_name": model_name,
        "device": device,
        "embed_batch_size": embed_batch_size,
    }
    if embed_max_length:
        kwargs["max_length"] = embed_max_length

    try:
        return HuggingFaceEmbedding(**kwargs)
    except TypeError:
        # Fallback untuk versi LlamaIndex tertentu.
        return HuggingFaceEmbedding(model_name=model_name)


def build_bm25_kwargs(nodes, bm25_top_k: int, bm25_mode: str = "default") -> Dict[str, Any]:
    """Menyiapkan argumen BM25 untuk LlamaIndex.

    Mode default mencoba menghindari stemming/stopword bahasa Inggris karena
    korpus SHAVIRA berbahasa Indonesia dan memuat istilah administratif.
    """
    mode = str(bm25_mode or "default").strip().lower()
    bm25_kwargs: Dict[str, Any] = {
        "nodes": nodes,
        "similarity_top_k": bm25_top_k,
        "skip_stemming": True,
    }

    if mode in {"default", "none", "no_stopwords", "no-stopwords"}:
        bm25_kwargs["language"] = []
        print("BM25 mode: default lexical retrieval, no stemming, no custom stopwords")
        return bm25_kwargs

    if mode in {"english", "en"}:
        bm25_kwargs["language"] = "english"
        bm25_kwargs["skip_stemming"] = False
        try:
            import Stemmer

            bm25_kwargs["stemmer"] = Stemmer.Stemmer("english")
        except Exception:
            pass
        print("BM25 mode: english stopwords/stemming")
        return bm25_kwargs

    bm25_kwargs["language"] = []
    print(f"BM25 mode '{bm25_mode}' tidak dikenali. Menggunakan default no stopwords.")
    return bm25_kwargs


def make_faiss_cache_path(
    base_cache_dir: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_max_length: Optional[int],
    deduplicate: bool,
) -> str:
    """Membuat path cache FAISS berdasarkan konfigurasi eksperimen."""
    cache_name = (
        f"faiss_"
        f"model_{safe_name(model_name)}_"
        f"chunk_{chunk_size}_"
        f"overlap_{chunk_overlap}_"
        f"maxlen_{embed_max_length}_"
        f"faiss_IndexFlatIP_"
        f"dedup_{int(deduplicate)}"
    )
    return str(Path(base_cache_dir) / cache_name)


def build_retrievers(
    nodes,
    model_name: str,
    dense_top_k: int,
    bm25_top_k: int,
    embed_max_length: Optional[int] = None,
    embed_batch_size: int = 8,
    bm25_mode: str = "default",
    index_cache_dir: Optional[str] = None,
    force_rebuild_index: bool = False,
):
    """Bangun retriever BM25 dan FAISS-BGE-M3."""
    try:
        import faiss
        from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.vector_stores.faiss import FaissVectorStore
    except Exception as exc:
        raise RuntimeError(install_hint()) from exc

    Settings.llm = None
    embed_model = make_embed_model(
        model_name=model_name,
        embed_max_length=embed_max_length,
        embed_batch_size=embed_batch_size,
    )
    Settings.embed_model = embed_model

    cache_path = Path(index_cache_dir) if index_cache_dir else None
    vector_index = None

    if cache_path and cache_path.exists() and not force_rebuild_index:
        try:
            print(f"Memuat FAISS index dari cache: {cache_path}")
            vector_store = FaissVectorStore.from_persist_dir(str(cache_path))
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(cache_path),
            )
            vector_index = load_index_from_storage(storage_context)
            print("FAISS index berhasil dimuat dari cache. Embedding ulang dilewati.")
        except Exception as exc:
            print(f"Gagal memuat FAISS index dari cache: {exc}")
            print("FAISS index akan dibangun ulang.")

    if vector_index is None:
        test_emb = embed_model.get_text_embedding("tes dimensi embedding")
        dim = len(test_emb)
        faiss_index = faiss.IndexFlatIP(dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print(f"Membangun FAISS IndexFlatIP dengan dimensi embedding: {dim}")
        try:
            vector_index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                show_progress=True,
            )
        except TypeError:
            vector_index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                show_progress=True,
            )

        if cache_path:
            cache_path.mkdir(parents=True, exist_ok=True)
            vector_index.storage_context.persist(persist_dir=str(cache_path))
            print(f"FAISS index disimpan ke cache: {cache_path}")

    vector_retriever = vector_index.as_retriever(similarity_top_k=dense_top_k)

    print("Membangun BM25 retriever")
    bm25_kwargs = build_bm25_kwargs(
        nodes=nodes,
        bm25_top_k=bm25_top_k,
        bm25_mode=bm25_mode,
    )
    try:
        bm25_retriever = BM25Retriever.from_defaults(**bm25_kwargs)
    except Exception as exc:
        print(f"BM25 default no-stopwords gagal: {exc}")
        print("Mencoba fallback BM25Retriever tanpa parameter language/skip_stemming.")
        bm25_kwargs.pop("language", None)
        bm25_kwargs.pop("skip_stemming", None)
        bm25_retriever = BM25Retriever.from_defaults(**bm25_kwargs)

    return bm25_retriever, vector_retriever
