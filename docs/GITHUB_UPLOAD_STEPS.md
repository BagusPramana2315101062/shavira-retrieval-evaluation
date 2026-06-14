# Langkah Upload ke GitHub

## 1. Pastikan file lokal yang tidak perlu tidak ikut commit

Folder/file berikut jangan diunggah:

- `.venv/`
- `data/raw/`
- `storage/`
- `results/`
- `results_full/`
- `__pycache__/`

Semuanya sudah diatur di `.gitignore`.

## 2. Inisialisasi Git

```powershell
git init
git status
```

## 3. Tambahkan file project

```powershell
git add README.md requirements.txt .gitignore src docs results_summary
```

Cek kembali:

```powershell
git status
```

Pastikan tidak ada file besar seperti cache FAISS, raw JSONL, atau virtual environment.

## 4. Commit awal

```powershell
git commit -m "Finalize SHAVIRA retrieval evaluation project"
```

## 5. Hubungkan ke repository GitHub

Ganti URL berikut dengan URL repo Anda:

```powershell
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```

Jika remote sudah ada:

```powershell
git remote -v
git remote set-url origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```

## 6. Jika file besar tidak sengaja masuk staging

```powershell
git reset
```

Lalu ulangi `git add` secara selektif.
