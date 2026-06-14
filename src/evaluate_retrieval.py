"""Backward-compatible wrapper.

Command lama tetap dapat digunakan:

    python src/evaluate_retrieval.py --data-dir data/raw --output-dir results
"""
from shavira_retrieval.cli import main


if __name__ == "__main__":
    main()
