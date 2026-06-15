"""
rank.py — Main entry point for the Redrob candidate ranking system.
Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

Architecture: Hybrid Rules + TF-IDF (Architecture 2)
Pipeline:
    1. Load candidates
    2. TF-IDF score on career descriptions
    3. Rule-based feature scoring
    4. Honeypot elimination
    5. Behavioral multiplier
    6. Combine into final score
    7. Output top 100 as CSV
"""

import json
import csv
import time
import argparse
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sys.path.insert(0, ".")
from src.scorer import score_candidate
from src.reasoning import generate_reasoning

# ─────────────────────────────────────────────────────────────────────────────
# JD QUERY DOCUMENT
# This is the "ideal candidate" description derived from the JD.
# TF-IDF compares every candidate's career descriptions against this.
# We use a curated version — not the raw JD — to emphasize core requirements.
# ─────────────────────────────────────────────────────────────────────────────
JD_QUERY = """
Senior AI engineer with production experience building embedding based retrieval
systems and vector search infrastructure. Shipped ranking systems recommendation
systems and search systems to real users at product companies. Experience with
FAISS pinecone weaviate qdrant milvus opensearch elasticsearch hybrid search
dense retrieval semantic search. Strong python skills. Designed evaluation
frameworks for ranking systems including NDCG MRR MAP offline evaluation
online evaluation AB testing. Experience with sentence transformers hugging face
transformers fine tuning LLMs LoRA PEFT. Learning to rank XGBoost LightGBM
feature engineering. Applied machine learning NLP information retrieval at scale.
Product company experience not consulting. Shipped to real users in production.
"""


def get_career_text(candidate: dict) -> str:
    """
    Concatenate career descriptions + summary for TF-IDF.
    Recent roles weighted more by repetition.
    """
    parts = []
    history = candidate.get("career_history", [])
    for i, job in enumerate(history):
        desc = job.get("description", "")
        if i == 0:
            parts.extend([desc, desc, desc])
        elif i == 1:
            parts.extend([desc, desc])
        else:
            parts.append(desc)
    summary = candidate["profile"].get("summary", "")
    parts.append(summary)
    return " ".join(parts)


def compute_tfidf_scores(candidates: list) -> np.ndarray:
    """
    Compute TF-IDF cosine similarity between each candidate's
    career text and the JD query document.

    Returns array of scores (0.0 to 1.0) for each candidate.

    Why TF-IDF on descriptions and not skills:
    Skills are self-reported. Descriptions are harder to fake.
    A keyword stuffer adds FAISS to skills but their description
    still talks about customer support — TF-IDF catches this.
    """
    print("  Building career text corpus...")
    corpus = [get_career_text(c) for c in candidates]

    # Add JD query as the last document
    corpus.append(JD_QUERY)

    print("  Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,      # Top 5000 terms — balances speed and coverage
        ngram_range=(1, 2),     # Unigrams + bigrams catches "vector search" etc
        min_df=2,               # Term must appear in at least 2 docs
        sublinear_tf=True       # Log normalization — reduces impact of repeated terms
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # JD vector is the last row
    jd_vector = tfidf_matrix[-1]
    candidate_matrix = tfidf_matrix[:-1]

    print("  Computing cosine similarities...")
    similarities = cosine_similarity(candidate_matrix, jd_vector).flatten()
    return similarities


def load_candidates(filepath: str) -> list:
    """Load candidates from a JSONL file or a JSON array file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    # JSON array (sample_candidates.json)
    if content.startswith("["):
        return json.loads(content)
    
    # JSONL format (candidates.jsonl)
    candidates = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            candidates.append(json.loads(line))
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True,
                        help="Path to output CSV file")
    args = parser.parse_args()

    total_start = time.time()

    # ── Step 1: Load candidates ───────────────────────────────────────────
    print(f"[1/5] Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"      Loaded {len(candidates)} candidates in "
          f"{time.time() - total_start:.1f}s")

    # ── Step 2: TF-IDF scores ─────────────────────────────────────────────
    print("[2/5] Computing TF-IDF scores...")
    t = time.time()
    tfidf_scores = compute_tfidf_scores(candidates)
    print(f"      Done in {time.time() - t:.1f}s")

    # ── Step 3: Rule-based scores ─────────────────────────────────────────
    print("[3/5] Computing rule-based scores...")
    t = time.time()
    rule_scores = []
    feature_dicts = []
    for c in candidates:
        score, features = score_candidate(c)
        rule_scores.append(score)
        feature_dicts.append(features)
    print(f"      Done in {time.time() - t:.1f}s")

    # ── Step 4: Combine scores ────────────────────────────────────────────
    print("[4/5] Combining scores...")
    # TF-IDF weight: 0.35 — career description relevance
    # Rule weight:   0.65 — structured feature scoring
    # Why this split: Rules are more reliable on synthetic data
    # because descriptions are templated. TF-IDF still adds signal
    # for candidates with rich description text.
    TFIDF_WEIGHT = 0.35
    RULE_WEIGHT = 0.65

    final_scores = []
    for i, c in enumerate(candidates):
        rule_score = rule_scores[i]

        # Honeypot candidates stay at 0.001 regardless of TF-IDF
        if rule_score <= 0.001:
            final_scores.append(0.001)
            continue

        combined = (TFIDF_WEIGHT * tfidf_scores[i] +
                    RULE_WEIGHT * rule_score)
        final_scores.append(combined)

    # ── Step 5: Rank and output ───────────────────────────────────────────
    print("[5/5] Ranking and writing output...")

    # Sort by score descending, then by candidate_id ascending for ties
    ranked = sorted(
        enumerate(candidates),
        key=lambda x: (-final_scores[x[0]], x[1]["candidate_id"])
    )

    top_100 = ranked[:100]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (idx, candidate) in enumerate(top_100, 1):
            score = round(final_scores[idx], 6)
            reasoning = generate_reasoning(candidate, rank, final_scores[idx],
                                           feature_dicts[idx])
            writer.writerow([
                candidate["candidate_id"],
                rank,
                score,
                reasoning
            ])

    elapsed = time.time() - total_start
    print(f"\n✅ Done in {elapsed:.1f}s")
    print(f"   Output: {args.out}")
    print(f"   Top candidate: {top_100[0][1]['candidate_id']} "
          f"(score={final_scores[top_100[0][0]]:.4f})")
    print(f"   Rank {len(top_100)} score: {final_scores[top_100[-1][0]]:.4f}")


if __name__ == "__main__":
    main()