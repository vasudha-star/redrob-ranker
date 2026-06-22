"""
src/embeddings.py

Semantic embedding scoring using sentence-transformers.
Model: all-MiniLM-L6-v2 (22MB, CPU-friendly)

Two-phase design:
  Phase 1 — precompute.py (one-time, ~80 minutes on CPU):
      Downloads and caches the model if not already cached.
      Encodes all 100,000 candidates and saves to data/*.npy.

  Phase 2 — rank.py (every run, ~2 seconds for this module):
      Loads pre-computed .npy files from disk.
      Encodes the JD text only (one forward pass, <1 second).
      Computes dot products — no network access, no model download.

No forced TRANSFORMERS_OFFLINE flag is set here. After precompute.py
has run once, both the model cache and the .npy files exist locally,
so rank.py is naturally offline without needing the flag. The flag
would also block the model download during precompute itself, which
defeats its purpose on a fresh machine.
"""

import os
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "data/candidate_embeddings.npy"
IDS_PATH = "data/candidate_ids.npy"


def _load_model():
    """
    Load the sentence-transformer model with a clear, actionable error
    message if it is not available locally.

    During precompute.py: downloads and caches the model if not present.
    During rank.py: reads from local cache — no network access needed
    because precompute.py will have already cached it.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        raise RuntimeError(
            f"\nCould not load '{MODEL_NAME}'.\n"
            f"Run precompute.py first to download and cache the model:\n"
            f"  python precompute.py --candidates data/candidates.jsonl\n"
            f"This also generates the pre-computed embedding files that\n"
            f"rank.py depends on. Original error: {e}"
        ) from e


def get_candidate_text(candidate: dict) -> str:
    """
    Build the text representation used for embedding each candidate.
    Current title repeated 4x — strongest single signal.
    Most recent role repeated 4x — strongest career evidence.
    Only advanced/expert skills with real usage duration included.
    """
    parts = []
    profile = candidate["profile"]

    title = profile.get("current_title", "")
    parts.extend([title] * 4)

    parts.append(profile.get("summary", ""))

    history = candidate.get("career_history", [])
    for i, job in enumerate(history):
        desc = job.get("description", "")
        job_title = job.get("title", "")
        company = job.get("company", "")
        industry = job.get("industry", "")
        role_text = f"{job_title} at {company} in {industry}. {desc}"

        if i == 0:
            parts.extend([role_text] * 4)
        elif i == 1:
            parts.extend([role_text] * 2)
        else:
            parts.append(role_text)

    skills = [
        f"{s['name']} {s.get('proficiency', '')}"
        for s in candidate.get("skills", [])
        if s.get("duration_months", 0) > 12
        and s.get("proficiency") in ["advanced", "expert"]
    ]
    parts.append(" ".join(skills))

    return " ".join(parts)


JD_TEXT = """
Senior AI Engineer with production experience building embedding based retrieval
systems vector search infrastructure ranking systems recommendation systems
search systems deployed to real users at product companies.
Experience with FAISS pinecone weaviate qdrant milvus opensearch elasticsearch
hybrid search dense retrieval semantic search approximate nearest neighbor.
Built evaluation frameworks for ranking systems NDCG MRR MAP offline evaluation
online evaluation AB testing relevance judgment offline online correlation.
Shipped learning to rank models XGBoost LightGBM feature engineering.
Strong Python skills. Experience with sentence transformers hugging face
transformers fine tuning LLMs LoRA PEFT QLoRA.
NLP information retrieval at scale production deployment real users.
Product company experience not consulting services.
Applied machine learning engineer who ships systems not researcher.
Collaborative filtering matrix factorization recommendation engine.
Query understanding query expansion reranking retrieval augmented generation.
"""


def precompute_embeddings(candidates: list) -> tuple:
    """
    Encode all candidates and save to disk. Called only by precompute.py.
    Downloads the model on first run; reads from local cache on subsequent runs.
    """
    print(f"  Loading model: {MODEL_NAME}")
    model = _load_model()

    print(f"  Building candidate texts for {len(candidates)} candidates...")
    texts = [get_candidate_text(c) for c in candidates]
    ids = [c["candidate_id"] for c in candidates]

    print(f"  Computing embeddings (this takes ~80 minutes on CPU)...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(IDS_PATH, np.array(ids))
    print(f"  Saved: {EMBEDDINGS_PATH}")
    print(f"  Saved: {IDS_PATH}")

    return embeddings, ids


def load_embeddings() -> tuple:
    """Load pre-computed embeddings from disk."""
    embeddings = np.load(EMBEDDINGS_PATH)
    ids = np.load(IDS_PATH, allow_pickle=True).tolist()
    return embeddings, ids


def compute_embedding_scores(candidates: list) -> np.ndarray:
    """
    Compute cosine similarity between each candidate and the JD.

    Requires:
      - data/candidate_embeddings.npy  (from precompute.py)
      - data/candidate_ids.npy         (from precompute.py)
      - all-MiniLM-L6-v2 model in local HuggingFace cache

    The model is only used here to encode the single JD text string.
    All candidate embeddings are read from disk. No network access
    is required once precompute.py has been run.
    """
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(IDS_PATH):
        raise FileNotFoundError(
            f"\nPre-computed embedding files not found:\n"
            f"  {EMBEDDINGS_PATH}\n"
            f"  {IDS_PATH}\n"
            f"Run precompute.py first:\n"
            f"  python precompute.py --candidates data/candidates.jsonl"
        )

    model = _load_model()

    jd_embedding = model.encode(
        [JD_TEXT],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    print("  Loading pre-computed embeddings...")
    embeddings, stored_ids = load_embeddings()
    id_to_idx = {cid: i for i, cid in enumerate(stored_ids)}

    scores = np.zeros(len(candidates))
    for i, c in enumerate(candidates):
        cid = c["candidate_id"]
        if cid in id_to_idx:
            scores[i] = float(np.dot(embeddings[id_to_idx[cid]], jd_embedding))

    return scores