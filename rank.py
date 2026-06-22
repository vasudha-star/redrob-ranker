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
from src.embeddings import compute_embedding_scores

# ─────────────────────────────────────────────────────────────────────────────
# JD QUERY DOCUMENT
# Curated ideal candidate description for TF-IDF matching.
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

    Why TF-IDF on descriptions and not skills:
    Skills are self-reported. Descriptions are harder to fake.
    A keyword stuffer adds FAISS to skills but their description
    still talks about customer support — TF-IDF catches this.
    """
    print("  Building career text corpus...")
    corpus = [get_career_text(c) for c in candidates]
    corpus.append(JD_QUERY)

    print("  Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[-1]
    candidate_matrix = tfidf_matrix[:-1]

    print("  Computing cosine similarities...")
    similarities = cosine_similarity(candidate_matrix, jd_vector).flatten()
    return similarities


def load_candidates(filepath: str) -> list:
    """Load candidates from a JSONL file or a JSON array file."""
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
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True,
                        help="Path to output CSV file")
    args = parser.parse_args()

    total_start = time.time()

    # ── Step 1: Load candidates ───────────────────────────────────────────
    print(f"[1/6] Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"      Loaded {len(candidates)} candidates in "
          f"{time.time() - total_start:.1f}s")

    # ── Step 2: Embedding scores ──────────────────────────────────────────
    print("[2/6] Computing semantic embedding scores...")
    t = time.time()
    embedding_scores = compute_embedding_scores(candidates)
    print(f"      Done in {time.time() - t:.1f}s")

    # ── Step 3: TF-IDF scores ─────────────────────────────────────────────
    print("[3/6] Computing TF-IDF scores...")
    t = time.time()
    tfidf_scores = compute_tfidf_scores(candidates)
    print(f"      Done in {time.time() - t:.1f}s")

    # ── Step 4: Rule-based scores ─────────────────────────────────────────
    print("[4/6] Computing rule-based scores...")
    t = time.time()
    rule_scores = []
    feature_dicts = []
    for c in candidates:
        score, features = score_candidate(c)
        rule_scores.append(score)
        feature_dicts.append(features)
    print(f"      Done in {time.time() - t:.1f}s")

# ── Step 5: Weighted Reciprocal Rank Fusion ───────────────────────────
    print("[5/6] Applying Weighted Reciprocal Rank Fusion...")
    #
    # Why weighted RRF over linear interpolation:
    # Three signals have incompatible score distributions:
    #   Rule scores:      0.08 – 0.72  (wide, reliable)
    #   Embedding scores: 0.20 – 0.42  (narrow, partially degraded by templates)
    #   TF-IDF scores:    0.01 – 0.18  (sparse, lowest magnitude)
    # Linear combination suppresses TF-IDF to ~2% effective contribution
    # despite its 15% nominal weight. RRF fixes this via rank-based aggregation.
    #
    # Why weighted RRF over equal-weight RRF:
    # Rule signal directly encodes 7 of 8 JD requirement categories from
    # structured fields (title, company type, years, location, assessments).
    # TF-IDF reads the same templated descriptions as embeddings with lower
    # expressiveness. Equal weights over-represent TF-IDF relative to its
    # actual predictive value for this JD.
    #
    # Weight derivation (JD-derived, not fitted to any proxy data):
    #   Rule   0.50 — structured JD requirements from reliable fields
    #   Embed  0.30 — semantic career narrative, partially template-degraded
    #   TF-IDF 0.20 — keyword retrieval matching, marginal over rules+embeddings
    #
    # k=60: standard RRF constant from Cormack et al. 2009. Not tuned.
    # Equal k across signals — signal reliability is captured by the weights,
    # not by adjusting the rank decay curve.
    #
    RRF_K = 60
    RULE_W   = 0.50
    EMBED_W  = 0.30
    TFIDF_W  = 0.20

    # Exclude honeypots from all three rankings
    valid_indices = [i for i, r in enumerate(rule_scores) if r > 0.001]

    # Three independent ranked lists — each purely by its own signal
    rule_ranking = sorted(
        valid_indices, key=lambda i: -rule_scores[i]
    )
    embedding_ranking = sorted(
        valid_indices, key=lambda i: -float(embedding_scores[i])
    )
    tfidf_ranking = sorted(
        valid_indices, key=lambda i: -float(tfidf_scores[i])
    )

    # Weighted RRF accumulation
    rrf_scores = {idx: 0.0 for idx in valid_indices}
    for ranked_list, weight in [
        (rule_ranking,      RULE_W),
        (embedding_ranking, EMBED_W),
        (tfidf_ranking,     TFIDF_W),
    ]:
        for rank, idx in enumerate(ranked_list, 1):
            rrf_scores[idx] += weight / (RRF_K + rank)

    # Final scores — honeypots remain at 0.001
    final_scores = [0.001] * len(candidates)
    for idx, score in rrf_scores.items():
        final_scores[idx] = score


    # ── Step 6: Rank and output ───────────────────────────────────────────
    print("[6/6] Ranking and writing output...")

    ranked = sorted(
        enumerate(candidates),
        key=lambda x: (-round(final_scores[x[0]], 6), x[1]["candidate_id"])
    )

    top_100 = ranked[:100]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (idx, candidate) in enumerate(top_100, 1):
            score = round(final_scores[idx], 6)
            reasoning = generate_reasoning(
                candidate, rank,
                final_scores[idx],
                feature_dicts[idx]
            )
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
    print(f"   Rank {len(top_100)} score: "
          f"{final_scores[top_100[-1][0]]:.4f}")


if __name__ == "__main__":
    main()