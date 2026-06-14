# SHAVIRA LlamaIndex Retrieval Experiment — Versi Revisi Pembimbing

Project ini merevisi eksperimen lama agar sesuai dengan arahan pembimbing: evaluasi **BM25**, **FAISS BGE-M3**, dan **Hybrid Retrieval RRF** pada **tiga konfigurasi chunking**.

## 1. Desain eksperimen

Default project menjalankan 9 skenario:

| Kode | chunk_size | chunk_overlap | Kategori | Metode |
|---|---:|---:|---|---|
| C1 | 256 | 32 | pendek | BM25, BGE_M3_FAISS, HYBRID_RRF |
| C2 | 512 | 64 | sedang/baseline | BM25, BGE_M3_FAISS, HYBRID_RRF |
| C3 | 768 | 96 | panjang | BM25, BGE_M3_FAISS, HYBRID_RRF |

Metrik utama:
- `success_at_k`
- `mrr_at_k`
- `ndcg_at_k`

Nilai K:
- 5
- 10

## 2. Struktur folder

```text
shavira-llamaindex-project/
├─ data/
│  └─ raw/
│     ├─ jdih_undiksha_ac_id.jsonl
│     ├─ upttik_undiksha_ac_id.jsonl
│     ├─ undiksha_ac_id_pmb.jsonl
│     ├─ undiksha_ac_id_tentang_undiksha.jsonl
│     └─ Dataset_Validasi_Retrieval_SHAVIRA.xlsx
├─ results/
├─ storage/
├─ src/
│  ├─ check_dataset.py
│  ├─ evaluate_retrieval.py
│  └─ shavira_retrieval/
└─ requirements.txt
```

## 3. Instalasi

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Catatan CUDA:
- Project memakai `BAAI/bge-m3` melalui `HuggingFaceEmbedding`.
- CUDA akan dipakai otomatis jika PyTorch CUDA aktif dan `torch.cuda.is_available()` bernilai True.
- Jika belum aktif, install PyTorch CUDA sesuai versi driver/NVIDIA Anda dari situs resmi PyTorch.

## 4. Cek Ollama

Evaluasi generation terbatas memakai Ollama melalui LlamaIndex. Jalankan:

```bash
ollama serve
ollama pull llama3.1:8b
ollama list
```

Model bisa diganti, misalnya:

```bash
ollama pull qwen2.5:7b
```

Lalu gunakan `--ollama-model qwen2.5:7b`.

## 5. Cek dataset

Letakkan lima file dataset di `data/raw/`, lalu jalankan:

```bash
python src/check_dataset.py --data-dir data/raw
```

## 6. Jalankan retrieval grid 9 skenario

```bash
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results
```

Uji cepat:

```bash
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results --limit 10
```

## 7. Jalankan retrieval + generation terbatas

Generation hanya dilakukan setelah konfigurasi retrieval terbaik ditentukan.

```bash
python src/evaluate_retrieval.py ^
  --data-dir data/raw ^
  --output-dir results ^
  --run-generation-eval ^
  --ollama-model llama3.1:8b ^
  --generation-sample-size 50 ^
  --generation-top-k 5
```

Linux/Mac:

```bash
python src/evaluate_retrieval.py \
  --data-dir data/raw \
  --output-dir results \
  --run-generation-eval \
  --ollama-model llama3.1:8b \
  --generation-sample-size 50 \
  --generation-top-k 5
```

## 8. Output

Folder `results/` menghasilkan:

1. `summary_metrics.csv`  
   Ringkasan metrik per konfigurasi chunking, metode, dan K.

2. `scenario_matrix.csv`  
   Matriks 9 skenario untuk tabel pengujian.

3. `summary_by_source.csv`  
   Analisis berdasarkan sumber korpus: JDIH, UPA TIK, PMB, dan Tentang Undiksha.

4. `failed_queries.csv`  
   Query yang gagal menemukan gold context pada K maksimum beserta kategori awal penyebab.

5. `best_configuration.csv`  
   Ranking konfigurasi terbaik berdasarkan skor komposit dari Success@K, MRR@K, dan nDCG@K.

6. `per_query_metrics.csv`  
   Metrik detail per query.

7. `retrieval_details_top10.csv`  
   Detail hasil Top-10 untuk audit manual.

8. `generation_eval_limited.csv`  
   Output evaluasi generation terbatas jika `--run-generation-eval` diaktifkan.

9. `summary_and_details.xlsx`  
   Semua output utama dalam satu file Excel.

## 9. Mode single config

Jika hanya ingin menjalankan konfigurasi lama 512/64:

```bash
python src/evaluate_retrieval.py --single-config --chunk-size 512 --chunk-overlap 64
```

Namun untuk proposal revisi, gunakan mode default grid 3x3.
