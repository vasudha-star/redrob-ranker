"""
precompute.py
Pre-computes candidate embeddings offline before ranking.

Run this ONCE before rank.py:
    python precompute.py --candidates ./data/candidates.jsonl

This saves embeddings to data/candidate_embeddings.npy
rank.py loads them automatically during ranking.

Why pre-computation:
Embedding 100K candidates takes ~10 minutes on CPU.
The ranking step itself takes ~45 seconds.
Separating pre-computation from ranking keeps the
ranking step well within time constraints.
"""

import json
import time
import argparse
import sys

sys.path.insert(0, ".")
from src.embeddings import precompute_embeddings


def load_candidates(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    candidates = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            candidates.append(json.loads(line))
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    start = time.time()
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates in {time.time()-start:.1f}s")

    print("\nPre-computing embeddings...")
    start = time.time()
    precompute_embeddings(candidates)
    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print("Embeddings saved. You can now run rank.py.")


if __name__ == "__main__":
    main()