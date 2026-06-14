# SHAVIRA LlamaIndex Retrieval Evaluation

Project ini digunakan untuk mengevaluasi performa **BM25**, **BGE-M3 + FAISS**, dan **Hybrid Retrieval berbasis Reciprocal Rank Fusion (RRF)** pada knowledge search SHAVIRA. Versi ini sudah disesuaikan dengan rancangan revisi proposal: eksperimen dilakukan pada **3 konfigurasi chunking × 3 metode retrieval** serta dievaluasi menggunakan **Success@K, MRR@K, dan nDCG@K**.

## Tujuan project

Project ini mendukung penelitian berjudul:

> Evaluasi Komparatif BM25, FAISS, dan Hybrid Retrieval pada Berbagai Konfigurasi Chunking terhadap Relevansi Knowledge Search SHAVIRA.

Fokus utama project adalah tahap **retrieval**. Tahap **generation** hanya disediakan sebagai evaluasi lanjutan terbatas menggunakan Ollama setelah konfigurasi retrieval terbaik diperoleh.

## Desain eksperimen

### Konfigurasi chunking

| ID | Chunk size | Chunk overlap | Kategori |
|---|---:|---:|---|
| C1 | 256 | 32 | Pendek |
| C2 | 512 | 64 | Sedang/baseline |
| C3 | 768 | 96 | Panjang |

### Metode retrieval

| Metode | Deskripsi |
|---|---|
| BM25 | Lexical retrieval berbasis kecocokan istilah. |
| BGE_M3_FAISS | Dense retrieval menggunakan embedding BAAI/bge-m3 dan FAISS IndexFlatIP. |
| HYBRID_RRF | Fusion hasil BM25 dan FAISS menggunakan Reciprocal Rank Fusion. |

Total skenario utama: **9 skenario**.

### Metrik evaluasi

Metrik utama:

- `success_at_k`
- `mrr_at_k`
- `ndcg_at_k`

Metrik pendukung:

- `precision_at_k`
- `recall_at_k`

Nilai K yang digunakan: **K = 5** dan **K = 10**.

## Struktur folder

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
├── results_summary/
│   ├── experiment_config.json
│   ├── summary_metrics.csv
│   ├── scenario_matrix.csv
│   ├── summary_by_source.csv
│   ├── best_configuration.csv
│   ├── generation_eval_summary.csv
│   └── generation_eval_limited.csv
└── src/
    ├── check_dataset.py
    ├── evaluate_retrieval.py
    └── shavira_retrieval/
        ├── analysis.py
        ├── cli.py
        ├── constants.py
        ├── data_loader.py
        ├── experiment.py
        ├── generation_eval.py
        ├── indexing.py
        ├── metrics.py
        ├── outputs.py
        ├── retrieval.py
        └── utils.py
```

Folder berikut sengaja **tidak diunggah** ke GitHub:

- `data/raw/` untuk dataset dan korpus mentah.
- `storage/` untuk cache FAISS index.
- `results/` dan `results_full/` untuk output eksperimen lokal.
- `.venv/` untuk virtual environment.

## Persiapan data

Buat folder:

```bash
mkdir -p data/raw
```

Letakkan file berikut di `data/raw/`:

```text
data/raw/jdih_undiksha_ac_id.jsonl
data/raw/upttik_undiksha_ac_id.jsonl
data/raw/undiksha_ac_id_pmb.jsonl
data/raw/undiksha_ac_id_tentang_undiksha.jsonl
data/raw/Dataset_Validasi_Retrieval_SHAVIRA.xlsx
```

## Instalasi

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Untuk CUDA, pasang PyTorch sesuai driver/GPU. Contoh untuk CUDA 11.8:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Cek CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## Validasi dataset

```powershell
python src/check_dataset.py --data-dir data/raw
```

## Menjalankan eksperimen retrieval

### Uji cepat

```powershell
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results --limit 10 --embed-batch-size 8
```

### Full retrieval evaluation

```powershell
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results_full --embed-batch-size 8
```

### Membangun cache FAISS satu per satu

```powershell
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results/cache_C1 --single-config --chunk-size 256 --chunk-overlap 32 --limit 1 --embed-batch-size 8
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results/cache_C2 --single-config --chunk-size 512 --chunk-overlap 64 --limit 1 --embed-batch-size 8
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results/cache_C3 --single-config --chunk-size 768 --chunk-overlap 96 --limit 1 --embed-batch-size 8
```

## Evaluasi generation terbatas dengan Ollama

Pastikan Ollama aktif dan model tersedia:

```powershell
ollama list
ollama pull llama3.2:3b
```

Jalankan evaluasi generation terbatas:

```powershell
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results_full --run-generation-eval --ollama-model llama3.2:3b --generation-sample-size 30 --generation-top-k 5 --embed-batch-size 8
```

Generation evaluation hanya dilakukan pada konfigurasi retrieval terbaik.

## Output utama

File output utama berada di folder `results_full/`:

| File | Fungsi |
|---|---|
| `summary_metrics.csv` | Ringkasan metrik per metode, konfigurasi chunking, dan K. |
| `scenario_matrix.csv` | Matriks ringkas 9 skenario retrieval. |
| `summary_by_source.csv` | Analisis performa berdasarkan sumber korpus. |
| `best_configuration.csv` | Peringkat konfigurasi terbaik. |
| `failed_queries.csv` | Query yang gagal menemukan gold context. |
| `per_query_metrics.csv` | Metrik per query. |
| `retrieval_details_top10.csv` | Detail hasil Top-10 retrieval. |
| `generation_eval_limited.csv` | Hasil evaluasi generation terbatas. |
| `generation_eval_summary.csv` | Ringkasan evaluasi generation terbatas. |
| `experiment_config.json` | Konfigurasi eksperimen yang digunakan. |

CSV adalah output utama. File Excel `summary_and_details.xlsx` bersifat tambahan; jika Excel gagal karena karakter ilegal metadata, CSV tetap aman digunakan.

## Ringkasan hasil full run

Eksperimen final menggunakan **499 query**, mode `grid_3x3`, model embedding `BAAI/bge-m3`, `candidate_k = 50`, `rrf_k = 10`, dan `embed_batch_size = 8`.

Konfigurasi retrieval terbaik pada K = 10:

| Chunking | Metode | Success@10 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|
| C1 256/32 | BGE_M3_FAISS | 0.7174 | 0.5179 | 0.5660 |

Hasil generation evaluation terbatas:

| Konfigurasi | LLM | Sampel | Context alignment | Ground truth alignment | Completeness | Outside context rate |
|---|---|---:|---:|---:|---:|---:|
| C1 256/32 + BGE_M3_FAISS | llama3.2:3b | 30 | 4.53 | 4.67 | 4.17 | 0.00 |

## Catatan metodologis

- Evaluasi retrieval menggunakan pencocokan `context_hash` antara gold context dan hasil retrieval.
- `--limit` hanya membatasi jumlah query evaluasi, bukan jumlah dokumen/chunk yang diindeks.
- FAISS index disimpan di `storage/faiss_index_cache/`, sehingga embedding tidak perlu dihitung ulang pada run berikutnya.
- Generation evaluation bersifat tambahan dan tidak mengubah fokus utama penelitian, yaitu evaluasi retrieval.

## Troubleshooting singkat

### `torch.load` meminta torch minimal 2.6

Upgrade PyTorch:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Ollama port 11434 sudah dipakai

Artinya Ollama sudah berjalan. Cek:

```powershell
ollama list
```

### Proses embedding lambat

Pastikan CUDA aktif:

```powershell
nvidia-smi -l 1
```

Gunakan `--embed-batch-size 8` untuk GPU 4 GB. Jika aman, coba 12 atau 16.

## Status project

Project ini sudah siap digunakan sebagai implementasi eksperimen revisi proposal SHAVIRA dan siap diunggah ke GitHub setelah dataset mentah dan cache lokal dikecualikan dari commit.
