"""Create the SQLite databases the crawler reads.

- labels.db: the ProdCrawl judgments, one row per page, in the table the crawler
  queries (labels(url TEXT PRIMARY KEY, label TEXT) with label 'True' or 'False').
  Pages without a judgment count as non-relevant.
- embeddings.db and embeddings_finetuned.db: empty anchor-text embedding caches, one
  per encoder. The crawler fills them through the embedding API.

Usage: python scripts/data/setup_databases.py --judgments judgments/*.tsv --out-dir data
"""
import argparse
import csv
import os
import sqlite3
import sys

CACHE_SCHEMA = ("CREATE TABLE IF NOT EXISTS embeddings "
                "(anchor_text TEXT UNIQUE NOT NULL, embedding BLOB NOT NULL)")


def build_labels(judgments_paths, db_path: str) -> int:
    if os.path.exists(db_path):
        sys.exit(f"{db_path} already exists; remove it first")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE labels(url TEXT PRIMARY KEY, label TEXT)")
    rows = 0
    for judgments_path in judgments_paths:
        with open(judgments_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            if next(reader) != ["url", "label"]:
                sys.exit(f"{judgments_path}: expected a header row 'url<TAB>label'")
            batch = []
            for url, label in reader:
                if label not in ("True", "False"):
                    sys.exit(f"unexpected label {label!r} for {url}")
                batch.append((url, label))
                if len(batch) == 100_000:
                    conn.executemany("INSERT INTO labels VALUES (?, ?)", batch)
                    rows += len(batch)
                    batch = []
            conn.executemany("INSERT INTO labels VALUES (?, ?)", batch)
            rows += len(batch)
    conn.commit()
    conn.close()
    return rows


def create_cache(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(CACHE_SCHEMA)
    conn.commit()
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True, nargs="+",
                    help="judgments TSV files, e.g. judgments/train.tsv judgments/validation.tsv judgments/test.tsv")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n = build_labels(args.judgments, os.path.join(args.out_dir, "labels.db"))
    print(f"labels.db: {n} judgments")
    for name in ("embeddings.db", "embeddings_finetuned.db"):
        create_cache(os.path.join(args.out_dir, name))
        print(f"{name}: ready")


if __name__ == "__main__":
    main()
