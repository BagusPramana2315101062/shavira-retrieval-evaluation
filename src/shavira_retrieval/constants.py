"""Konstanta default untuk eksperimen retrieval SHAVIRA."""

DEFAULT_JSONL_FILES = [
    "jdih_undiksha_ac_id.jsonl",
    "upttik_undiksha_ac_id.jsonl",
    "undiksha_ac_id_pmb.jsonl",
    "undiksha_ac_id_tentang_undiksha.jsonl",
]

DEFAULT_VALIDATION_FILE = "Dataset_Validasi_Retrieval_SHAVIRA.xlsx"

# Sesuai kerangka revisi pembimbing: chunk pendek, sedang/baseline, panjang.
CHUNK_CONFIGS = [
    {
        "chunk_config_id": "C1",
        "chunk_size": 256,
        "chunk_overlap": 32,
        "chunk_category": "pendek",
    },
    {
        "chunk_config_id": "C2",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "chunk_category": "sedang_baseline",
    },
    {
        "chunk_config_id": "C3",
        "chunk_size": 768,
        "chunk_overlap": 96,
        "chunk_category": "panjang",
    },
]

DEFAULT_EVAL_K = [5, 10]
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_CANDIDATE_K = 50
DEFAULT_RRF_K = 10

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_EMBED_MAX_LENGTH = 512
DEFAULT_EMBED_BATCH_SIZE = 8
DEFAULT_BM25_MODE = "default"
DEFAULT_INDEX_CACHE_DIR = "storage/faiss_index_cache"

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 300.0
DEFAULT_GENERATION_SAMPLE_SIZE = 50
DEFAULT_GENERATION_TOP_K = 5

REQUIRED_VALIDATION_COLUMNS = {"ID", "Context", "Question", "Answer"}
RETRIEVAL_METHODS = ["BM25", "BGE_M3_FAISS", "HYBRID_RRF"]
PRIMARY_METRICS = ["success_at_k", "mrr_at_k", "ndcg_at_k"]
SUPPORTING_METRICS = ["precision_at_k", "recall_at_k"]
