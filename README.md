# SHAVIRA Retrieval Evaluation

Repositori ini berisi implementasi eksperimen evaluasi retrieval untuk sistem **SHAVIRA Knowledge Search**. Project ini membandingkan metode **BM25**, **BGE-M3 + FAISS**, dan **Hybrid Retrieval menggunakan Reciprocal Rank Fusion (RRF)** pada beberapa konfigurasi chunking untuk mengukur relevansi konteks hasil pencarian.

Project ini disusun untuk mendukung revisi proposal penelitian dengan fokus utama pada tahap retrieval. Tahap generation tidak dijadikan fokus utama, tetapi digunakan sebagai evaluasi lanjutan terbatas setelah konfigurasi retrieval terbaik diperoleh.

---

## Judul Penelitian

**Evaluasi Komparatif BM25, FAISS, dan Hybrid Retrieval pada Berbagai Konfigurasi Chunking terhadap Relevansi Knowledge Search SHAVIRA**

---

## Latar Belakang Singkat

SHAVIRA merupakan sistem virtual assistant Undiksha yang membutuhkan mekanisme pencarian konteks yang relevan dari knowledge base berbasis dokumen. Dalam sistem berbasis Retrieval-Augmented Generation (RAG), kualitas tahap retrieval sangat menentukan kualitas konteks yang diberikan kepada model bahasa.

Project ini mengevaluasi performa beberapa pendekatan retrieval untuk mengetahui kombinasi metode retrieval dan konfigurasi chunking yang paling sesuai dalam mengambil konteks relevan dari korpus SHAVIRA.

---

## Tujuan Project

Project ini bertujuan untuk:

1. Membandingkan performa BM25, BGE-M3 + FAISS, dan Hybrid RRF pada knowledge search SHAVIRA.
2. Menguji pengaruh konfigurasi chunking terhadap performa retrieval.
3. Menentukan kombinasi metode retrieval dan konfigurasi chunking terbaik.
4. Menyediakan analisis hasil berdasarkan metrik Success@K, MRR@K, dan nDCG@K.
5. Menyediakan analisis per sumber korpus dan analisis query gagal.
6. Melakukan evaluasi generation terbatas menggunakan LLM lokal melalui Ollama.

---

## Desain Eksperimen

Eksperimen menggunakan desain grid **3 × 3**, yaitu tiga konfigurasi chunking dan tiga metode retrieval.

### Konfigurasi Chunking

| Kode | Chunk Size | Chunk Overlap | Kategori          |
| ---- | ---------: | ------------: | ----------------- |
| C1   |        256 |            32 | Pendek            |
| C2   |        512 |            64 | Sedang / baseline |
| C3   |        768 |            96 | Panjang           |

Ketiga konfigurasi mempertahankan rasio overlap sebesar 12,5%. Dengan demikian, eksperimen dapat membandingkan pengaruh panjang chunk secara lebih terkontrol.

### Metode Retrieval

| Metode         | Deskripsi                                                              |
| -------------- | ---------------------------------------------------------------------- |
| BM25           | Lexical retrieval berbasis kecocokan istilah antara query dan dokumen  |
| BGE-M3 + FAISS | Dense retrieval berbasis embedding BAAI/bge-m3 dan vector search FAISS |
| Hybrid RRF     | Penggabungan hasil BM25 dan FAISS menggunakan Reciprocal Rank Fusion   |

### Skenario Pengujian

| Kode      | Chunking | Metode         |
| --------- | -------- | -------------- |
| C1-BM25   | 256/32   | BM25           |
| C1-FAISS  | 256/32   | BGE-M3 + FAISS |
| C1-HYBRID | 256/32   | Hybrid RRF     |
| C2-BM25   | 512/64   | BM25           |
| C2-FAISS  | 512/64   | BGE-M3 + FAISS |
| C2-HYBRID | 512/64   | Hybrid RRF     |
| C3-BM25   | 768/96   | BM25           |
| C3-FAISS  | 768/96   | BGE-M3 + FAISS |
| C3-HYBRID | 768/96   | Hybrid RRF     |

---

## Dataset

Project ini menggunakan:

1. Empat file korpus JSONL SHAVIRA:
   - `jdih_undiksha_ac_id.jsonl`
   - `upttik_undiksha_ac_id.jsonl`
   - `undiksha_ac_id_pmb.jsonl`
   - `undiksha_ac_id_tentang_undiksha.jsonl`

2. Dataset validasi:
   - `Dataset_Validasi_Retrieval_SHAVIRA.xlsx`

Pada dataset validasi:

| Kolom    | Fungsi                                        |
| -------- | --------------------------------------------- |
| Question | Query evaluasi                                |
| Context  | Gold context                                  |
| Answer   | Pembanding untuk evaluasi generation terbatas |

Dataset asli tidak disertakan dalam repositori ini. File dataset diletakkan secara lokal pada folder:

```text
data/raw/
```

Folder `data/raw/` dikecualikan dari GitHub melalui `.gitignore`.

---

## Struktur Project

```text
shavira-retrieval-evaluation/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── GITHUB_UPLOAD_STEPS.md
│   └── README_REVISI_AWAL.md
├── results_sample/
│   ├── best_configuration.csv
│   ├── experiment_config.json
│   ├── failed_queries_sample.csv
│   ├── generation_eval_limited.csv
│   ├── generation_eval_summary.csv
│   ├── scenario_matrix.csv
│   ├── summary_by_source.csv
│   └── summary_metrics.csv
└── src/
    ├── check_dataset.py
    ├── evaluate_retrieval.py
    └── shavira_retrieval/
        ├── __init__.py
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

Folder berikut tidak diunggah ke GitHub:

```text
.venv/
data/raw/
storage/
results/
results_full/
```

---

## Instalasi

### 1. Clone repository

```bash
git clone https://github.com/BagusPramana2315101062/shavira-retrieval-evaluation.git
cd shavira-retrieval-evaluation
```

### 2. Buat virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependency

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Siapkan dataset

Buat folder:

```bash
mkdir data
mkdir data/raw
```

Letakkan file berikut ke dalam `data/raw/`:

```text
data/raw/jdih_undiksha_ac_id.jsonl
data/raw/upttik_undiksha_ac_id.jsonl
data/raw/undiksha_ac_id_pmb.jsonl
data/raw/undiksha_ac_id_tentang_undiksha.jsonl
data/raw/Dataset_Validasi_Retrieval_SHAVIRA.xlsx
```

---

## Pemeriksaan Dataset

Sebelum eksperimen dijalankan, cek struktur dataset:

```bash
python src/check_dataset.py --data-dir data/raw
```

Script ini memeriksa keberadaan file JSONL, file validasi, kolom penting, dan ringkasan dataset.

---

## Menjalankan Eksperimen Retrieval

### Uji cepat

Gunakan `--limit` untuk mencoba sebagian query terlebih dahulu:

```bash
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results --limit 10
```

### Full retrieval

```bash
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results
```

Eksperimen full akan menjalankan 9 skenario:

```text
3 konfigurasi chunking × 3 metode retrieval
```

Output akan disimpan pada folder:

```text
results/
```

---

## Menjalankan Evaluasi Generation Terbatas

Tahap generation hanya digunakan sebagai evaluasi lanjutan terbatas. Evaluasi ini dilakukan setelah konfigurasi retrieval terbaik ditentukan.

### 1. Pastikan Ollama aktif

```bash
ollama list
```

Jika belum memiliki model, gunakan model ringan:

```bash
ollama pull llama3.2:3b
```

### 2. Jalankan evaluasi generation

```bash
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results --run-generation-eval --ollama-model llama3.2:3b --generation-sample-size 30 --generation-top-k 5 --embed-batch-size 8
```

Parameter:

| Parameter                  | Fungsi                                    |
| -------------------------- | ----------------------------------------- |
| `--run-generation-eval`    | Mengaktifkan evaluasi generation terbatas |
| `--ollama-model`           | Model Ollama yang digunakan               |
| `--generation-sample-size` | Jumlah sampel pertanyaan                  |
| `--generation-top-k`       | Jumlah context retrieval yang digunakan   |
| `--embed-batch-size`       | Batch size embedding                      |

---

## Output Eksperimen

Output utama yang dihasilkan:

| File                          | Isi                                                       |
| ----------------------------- | --------------------------------------------------------- |
| `summary_metrics.csv`         | Ringkasan performa setiap metode dan konfigurasi chunking |
| `scenario_matrix.csv`         | Matriks seluruh skenario pengujian                        |
| `summary_by_source.csv`       | Analisis performa berdasarkan sumber korpus               |
| `best_configuration.csv`      | Konfigurasi terbaik berdasarkan metrik utama              |
| `failed_queries.csv`          | Daftar query yang gagal menemukan gold context            |
| `per_query_metrics.csv`       | Metrik per query                                          |
| `retrieval_details_top10.csv` | Detail hasil Top-10 retrieval                             |
| `generation_eval_limited.csv` | Hasil evaluasi generation terbatas                        |
| `generation_eval_summary.csv` | Ringkasan evaluasi generation                             |
| `experiment_config.json`      | Konfigurasi eksperimen yang dijalankan                    |

Pada repository ini, folder `results/` tidak disertakan secara penuh. Repository hanya menyertakan `results_sample/` yang berisi ringkasan hasil dan sampel output agar repository tetap ringan.

---

## Ringkasan Hasil Final

Eksperimen full dilakukan pada 499 query validasi. Konfigurasi terbaik pada K = 10 adalah:

| Chunking  | Metode         | Success@10 | MRR@10 | nDCG@10 |
| --------- | -------------- | ---------: | -----: | ------: |
| C1 256/32 | BGE-M3 + FAISS |     0,7174 | 0,5179 |  0,5660 |

Konfigurasi terbaik pada K = 5 juga sama:

| Chunking  | Metode         | Success@5 |  MRR@5 | nDCG@5 |
| --------- | -------------- | --------: | -----: | -----: |
| C1 256/32 | BGE-M3 + FAISS |    0,6473 | 0,5082 | 0,5430 |

Hasil ini menunjukkan bahwa dense retrieval menggunakan BGE-M3 + FAISS dengan chunking pendek 256/32 memberikan performa paling konsisten dalam menemukan gold context pada knowledge search SHAVIRA.

---

## Ringkasan Generation Evaluation

Evaluasi generation terbatas dilakukan menggunakan:

| Komponen          | Nilai                      |
| ----------------- | -------------------------- |
| Retrieval terbaik | C1 256/32 + BGE-M3 + FAISS |
| LLM lokal         | llama3.2:3b                |
| Jumlah sampel     | 30                         |
| Top-K context     | 5                          |

Hasil rata-rata:

| Aspek Evaluasi         | Rata-rata |
| ---------------------- | --------: |
| Context alignment      |  4,53 / 5 |
| Ground truth alignment |  4,67 / 5 |
| Completeness           |  4,17 / 5 |
| Outside context rate   |      0,00 |

Hasil tersebut menunjukkan bahwa konteks retrieval terbaik dapat digunakan oleh LLM lokal untuk menghasilkan jawaban yang cukup selaras dengan konteks dan ground truth. Namun, evaluasi generation tetap diposisikan sebagai evaluasi lanjutan terbatas, bukan fokus utama penelitian.

---

## Metrik Evaluasi

### Success@K

Success@K bernilai 1 jika gold context ditemukan dalam Top-K hasil retrieval, dan 0 jika tidak ditemukan.

### MRR@K

MRR@K mengukur posisi kemunculan pertama gold context dalam Top-K hasil retrieval. Semakin tinggi nilainya, semakin baik posisi ranking konteks relevan.

### nDCG@K

nDCG@K mengukur kualitas ranking hasil retrieval dengan memperhatikan posisi dokumen relevan. Nilai lebih tinggi menunjukkan bahwa konteks relevan muncul pada posisi yang lebih atas.

### Precision@K dan Recall@K

Precision@K dan Recall@K digunakan sebagai metrik pendukung untuk melihat proporsi hasil relevan dalam Top-K serta kemampuan sistem menemukan konteks relevan.

---

## Catatan Implementasi

Project ini menggunakan:

- Python
- LlamaIndex
- BM25 Retriever
- HuggingFace Embedding
- BAAI/bge-m3
- FAISS
- Ollama
- Pandas
- OpenPyXL
- PyTorch

Embedding BGE-M3 akan menggunakan CUDA jika PyTorch mendeteksi GPU. Jika CUDA tidak tersedia, embedding tetap dapat berjalan menggunakan CPU, tetapi proses akan lebih lambat.

---

## Troubleshooting

### 1. Torch harus versi 2.6 atau lebih baru

Jika muncul error:

```text
ValueError: Due to a serious vulnerability issue in torch.load...
```

Upgrade PyTorch:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. CUDA tidak terdeteksi

Cek CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Jika hasilnya `False`, eksperimen tetap dapat berjalan di CPU, tetapi lebih lambat.

### 3. Ollama port sudah digunakan

Jika muncul:

```text
bind: Only one usage of each socket address is normally permitted
```

Artinya Ollama sudah berjalan. Cek dengan:

```bash
ollama list
```

### 4. Export Excel gagal karena karakter ilegal

Jika `summary_and_details.xlsx` gagal dibuat, hasil CSV tetap dapat digunakan. Project sudah membersihkan karakter ilegal pada output Excel, tetapi pada kasus tertentu hasil lengkap tetap lebih aman dianalisis dari file CSV.

### 5. GitHub menolak file besar

Jangan upload folder `results/`, `storage/`, `data/raw/`, atau `.venv/`. Upload hanya source code, README, dokumentasi, dan `results_sample/`.

---

## Reproduksibilitas

Untuk mereplikasi eksperimen:

```bash
python src/check_dataset.py --data-dir data/raw
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results
```

Untuk menjalankan retrieval dan generation evaluation:

```bash
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results --run-generation-eval --ollama-model llama3.2:3b --generation-sample-size 30 --generation-top-k 5 --embed-batch-size 8
```

Hasil retrieval dapat berbeda sedikit bergantung pada versi library, perangkat keras, dan konfigurasi runtime. Namun, struktur eksperimen, konfigurasi chunking, metode retrieval, dan metrik evaluasi tetap mengikuti desain yang sama.

---

## Lisensi

Project ini disusun untuk keperluan akademik dan penelitian. Dataset mentah tidak disertakan dalam repository karena bersifat lokal dan digunakan khusus untuk evaluasi knowledge search SHAVIRA.
