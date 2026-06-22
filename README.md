<div align="center">

# Intelligent Candidate Ranking System

### Redrob AI Hackathon — Intelligent Candidate Discovery & Ranking Challenge

*A multi-signal candidate ranking pipeline combining structured rule-based scoring, semantic embedding retrieval, and lexical TF-IDF matching, fused via Weighted Reciprocal Rank Fusion.*

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-supported-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-supported-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-supported-yellow?style=flat-square)
![Ranking](https://img.shields.io/badge/Fusion-Weighted_RRF-green?style=flat-square)
![IR](https://img.shields.io/badge/Domain-Information_Retrieval-lightgrey?style=flat-square)

</div>

---

## Table of Contents

- [Project Snapshot](#project-snapshot)
- [Executive Summary](#executive-summary)
- [Challenge Understanding](#challenge-understanding)
- [Design Principles](#design-principles)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Ranking Pipeline](#ranking-pipeline)
- [Feature Engineering](#feature-engineering)
- [Honeypot Detection](#honeypot-detection)
- [Why Weighted Reciprocal Rank Fusion](#why-weighted-reciprocal-rank-fusion)
- [AI Components](#ai-components)
- [Dataset Analysis and Findings](#dataset-analysis-and-findings)
- [Weight Derivation Methodology](#weight-derivation-methodology)
- [Validation and Evaluation](#validation-and-evaluation)
- [Reproducibility](#reproducibility)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Technical Decisions and Tradeoffs](#technical-decisions-and-tradeoffs)
- [Conclusion](#conclusion)

---

## Project Snapshot

| Metric | Value |
|---|---|
| Candidates processed | 100,000 |
| Valid candidates (pass integrity checks) | 67,889 |
| Honeypots removed | 32,111 (32.1%) |
| Ranking method | Weighted Reciprocal Rank Fusion |
| Signals fused | Rule engine · Embeddings · TF-IDF |
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| Embedding precomputation | ~80 minutes, one-time |
| Ranking runtime | ~110 seconds |
| Output | Top 100 candidates with fact-grounded reasoning |
| Execution constraints | CPU-only · no network during ranking |

---

## Executive Summary

### The problem with keyword search

Traditional applicant tracking systems match candidates by counting keyword overlaps between a resume and a job description. This fails in two directions simultaneously: a Marketing Manager who keyword-stuffs their profile with "FAISS, Pinecone, vector search" scores higher than an ML Engineer whose description says "built an ANN index for product search" without using those exact words. The first candidate is a bad hire; the second is a strong one. Pure keyword matching cannot tell them apart.

### Why this system is different

This system combines three independent signals that fail in different ways and fuses them using a rank-based aggregation method that is robust to their different score distributions. A structured rule engine reads verified, non-narrative fields — job title, company type, years of experience, platform assessment scores — to capture hard JD requirements that no text model can infer reliably. A semantic embedding model reads career narrative to identify candidates whose experience matches the JD even when they use different terminology. A TF-IDF model reads the same narrative for exact technical-term matching. No single signal is trusted in isolation.

The system also treats availability as a separate dimension from technical fit, which the organizers explicitly recommended: a technically perfect candidate who has not logged in for six months and never responds to recruiters is not actually hirable. Behavioral signals are applied as a multiplier on the technical score rather than added to it, so they can never override technical assessment — they can only confirm or discount it.

---

## Challenge Understanding

The Redrob Intelligent Candidate Discovery and Ranking Challenge asks participants to rank a pool of 100,000 synthetic candidate profiles against a specific job description for a Senior AI Engineer role at a Series A company. The challenge is explicitly not a keyword-matching exercise. The organizers built keyword stuffers and honeypot profiles directly into the dataset to trap naive systems. The evaluation measures ranking quality (NDCG@10, NDCG@50), submission format validity, reproducibility, reasoning quality, and the participant's ability to defend their design choices in a technical interview.

Understanding the challenge required reading the job description carefully — not just for the skills list, but for what the JD says about culture fit, disqualifiers, location flexibility, behavioral availability, and the explicit warning that candidates with only consulting-firm experience are a known bad fit. These were not background context; they were engineering requirements.

---

## Design Principles

The system was built around five principles that shaped every engineering decision:

1. **Trust structured evidence over self-reported claims.** A candidate's job title, country, and years of experience are directly readable from schema fields. Platform-verified assessment scores are tested, not claimed. These are more reliable than free-text descriptions and are given higher architectural weight.

2. **Use multiple independent ranking signals.** Each signal fails in a different way. Rules cannot read narrative nuance. Embeddings degrade on templated synthetic text. TF-IDF misses semantic equivalence. Combining all three makes the system more robust than perfecting any one of them.

3. **Separate technical fit from hireability.** Whether a candidate is technically qualified and whether they can be hired right now are independent questions. Mixing them additively would allow a highly-engaged but unqualified candidate to outscore a qualified but dormant one.

4. **Prefer explainability over opaque models.** Every ranking decision traces to a specific field value or computed feature. Every reasoning sentence cites an actual profile fact. The system can be audited line by line.

5. **Optimize for reproducibility.** Fixed reference dates, deterministic sort keys, pre-computed embeddings, and no runtime randomness ensure the same output on every run on every machine after precomputation.

---

## Technology Stack

| Area | Technology | Version |
|---|---|---|
| Language | Python | 3.12 |
| Embedding model | Sentence Transformers (`all-MiniLM-L6-v2`) | 5.5.1 |
| Vector similarity | Cosine similarity via NumPy dot product | — |
| Lexical retrieval | TF-IDF (`sklearn.TfidfVectorizer`) | — |
| Rank fusion | Weighted Reciprocal Rank Fusion | k=60 |
| Numerical computing | NumPy | 1.26.4 |
| ML utilities | scikit-learn | 1.5.2 |
| Date parsing | python-dateutil | 2.9.0 |

---

## System Architecture

```
candidates.jsonl (100,000 records)
        │
        ▼
┌───────────────────────┐
│    Honeypot Filter     │   8 integrity checks
│    src/honeypot.py     │   fail → score = 0.001, excluded from all rankings
└──────────┬────────────┘
           │
     ┌─────┴──────┬──────────────┐
     ▼            ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│   Rule    │ │Embeddings│ │  TF-IDF  │
│  Engine   │ │          │ │          │
│           │ │MiniLM    │ │sklearn   │
│ 11 feats  │ │L6-v2     │ │5000 feat │
│ verified  │ │pre-comp  │ │(1,2)-gram│
│ fields    │ │candidate │ │sublinear │
│           │ │vecs +    │ │    tf    │
│ × behav.  │ │fresh JD  │ │          │
│   mult.   │ │   vec    │ │          │
│ × notice  │ │          │ │          │
└─────┬─────┘ └─────┬────┘ └─────┬────┘
      │             │             │
      │ rule_scores │embed_scores │tfidf_scores
      └─────────────┼─────────────┘
                    ▼
        ┌───────────────────────┐
        │    Weighted RRF        │
        │                        │
        │  rule    weight = 0.50 │
        │  embed   weight = 0.30 │
        │  tfidf   weight = 0.20 │
        │  k = 60 (Cormack 2009) │
        └──────────┬────────────┘
                   ▼
        ┌───────────────────────┐
        │   Reasoning Generator  │
        │   src/reasoning.py     │
        │   fact-grounded only   │
        │   rank-band templates  │
        └──────────┬────────────┘
                   ▼
            submission.csv
        (candidate_id, rank, score, reasoning)
```

> **Note on signal independence.** The three signals are computed in parallel and never share intermediate state. Each sees the same candidate data but reads different representations of it — structured fields, embedded narrative, and raw term frequencies. Independence is a deliberate design choice: when signals agree on a candidate's position, the agreement is more reliable than any single signal could be on its own.

---

## Ranking Pipeline

### Stage 1 — Candidate loading

`rank.py` loads candidates from a JSONL file (one JSON object per line) or a JSON array, auto-detecting the format. The full 100,000-record pool is held in memory as a list of Python dicts for the duration of the pipeline.

### Stage 2 — Semantic embedding retrieval

`src/embeddings.py` computes cosine similarity between each candidate's career narrative and a curated job-description text using a pre-trained sentence-embedding model.

**Why embeddings exist:** keyword matching cannot recognize that "built an ANN index for product search" and "FAISS experience" describe the same thing. Embedding models map text to a vector space where semantically similar passages land near each other regardless of exact word choice.

**Why pre-computation is separate:** encoding 100,000 texts through a transformer model takes approximately 80 minutes on CPU. The ranking step must complete in under five minutes. `precompute.py` runs once, saves `data/candidate_embeddings.npy` and `data/candidate_ids.npy`, and `rank.py` loads them in under two seconds. The model is only called during ranking to encode the single JD text string — one forward pass. After precomputation, no network access is required because the model reads from local cache.

**Text construction per candidate:** the candidate's current title is repeated four times, followed by the professional summary, followed by career descriptions weighted by recency (current role four times, second role twice, earlier roles once), followed by advanced and expert skills with over twelve months of duration. Repetition is the mechanism for recency and importance weighting.

**Similarity computation:** both texts are encoded with L2 normalization, making dot product equal to cosine similarity: `np.dot(candidate_vec, jd_vec)`.

### Stage 3 — TF-IDF lexical retrieval

`rank.py` builds a TF-IDF representation of every candidate's career text and measures cosine similarity against the JD query document.

**Why TF-IDF alongside embeddings:** the two signals are complementary. Embeddings capture semantic equivalence; TF-IDF captures whether specific technical terms appear verbatim. A candidate whose description says "Elasticsearch hybrid search pipeline" matches TF-IDF's exact term matching in a way a general-purpose embedding model might blur with similar but different tools.

**Why descriptions and not skills:** the skills section is self-reported and unverified. A keyword stuffer adds FAISS, Pinecone, and vector search to their skills list without ever using them; their career descriptions still describe unrelated work. TF-IDF on descriptions catches this; TF-IDF on skills would not.

**Configuration:** `TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, sublinear_tf=True)`. The `(1,2)` ngram range captures multi-word technical terms like "vector search" and "learning to rank" as single units. `sublinear_tf=True` prevents a description that says "retrieval" ten times from scoring ten times higher than one that says it once.

### Stage 4 — Rule-based feature scoring

`src/features.py` and `src/scorer.py` compute eleven structured features per candidate from verified profile fields, then combine them into a single rule score adjusted by behavioral availability and notice period.

**Why rules alongside text models:** the job description's hard requirements — product company experience, India location preference, 5–9 year experience range, platform-verified assessment scores — are structured facts directly available as schema fields. Inferring them from narrative text is slower, noisier, and less reliable than reading the field directly.

**Behavioral multiplier:** after the technical rule score is computed, it is multiplied by a behavioral availability factor and a notice-period factor. These are multiplicative rather than additive because they answer "can we hire this person right now" rather than "is this person technically qualified."

### Stage 5 — Weighted Reciprocal Rank Fusion

The three signals are combined using Weighted RRF. See the [dedicated section](#why-weighted-reciprocal-rank-fusion) below.

### Stage 6 — Reasoning generation

`src/reasoning.py` writes a one-to-two sentence explanation for every candidate in the top 100. Every claim references an actual field from that candidate's profile. A keyword list ordered by specificity drives candidate-specific evidence detection — distinctive technologies are checked before generic ones, so candidates with more specific evidence receive more distinctive reasoning.

---

## Feature Engineering

All eleven features are normalized to `[0.0, 1.0]` and combined using the weights defined in `src/scorer.py`.

| Feature | Weight | What it measures |
|---|---|---|
| `retrieval_evidence` | 0.28 | Tiered keyword evidence of retrieval/ranking/search work in career descriptions |
| `product_company` | 0.15 | Duration-weighted fraction of career at non-services companies |
| `title_domain` | 0.12 | Whether the current title reads as technical and ML-relevant |
| `skill_credibility` | 0.09 | Platform assessment scores, long-duration relevant skills, endorsements |
| `company_prestige` | 0.08 | Duration-weighted time at recognized product companies |
| `evaluation_framework` | 0.08 | Evidence of evaluation methodology in descriptions (NDCG, MRR, A/B testing) |
| `experience_fit` | 0.07 | Closeness to the JD's 5–9 year target range |
| `skill_description_consistency` | 0.05 | Whether high-proficiency claimed skills appear in descriptions |
| `github_activity` | 0.04 | Linked GitHub engagement score |
| `location` | 0.04 | Geographic fit with JD's India preference |
| `education` | 0.00 | Institutional tier (present in code, zeroed — see [Weight Derivation](#weight-derivation-methodology)) |

### Retrieval evidence (weight 0.28)

Reads all career descriptions with recency weighting plus professional summary. Checks against two tiers of keywords. Tier 1 contains 31 unambiguous retrieval signals — FAISS, BM25, NDCG, learning-to-rank, collaborative filtering, cross-encoder, and others — each of which appears only in genuine retrieval/ranking/search engineering work. Tier 1 matches count twice. Tier 2 contains 18 softer signals — "embedding", "transformer", "LLM", "RAG" — that appear in retrieval work but also in unrelated contexts. A content marketer who used ChatGPT for email writing matches Tier 2; a retrieval engineer matches Tier 1. Normalization denominator is 20 weighted points.

### Product company score (weight 0.15)

Classifies each job in career history as services or product using two independent signals: company name substring matching against a curated list of known IT services and consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, Genpact, and others), and the `industry` field checked against services-indicating terms ("IT Services", "Consulting", "BPO", "Outsourcing", "Staffing"). The final score is the fraction of career months at non-services companies, weighted by duration. Floor is 0.10.

### Title domain score (weight 0.12)

Checks the current job title against curated `TECH_TITLES` and `NON_TECH_TITLES` sets. Directly JD-relevant titles (ML engineer, AI engineer, NLP engineer, search engineer, recommendation engineer) return 1.0. General technical titles return 0.6. Non-technical titles return 0.1. Unknown titles return 0.4. No candidate is completely eliminated by title alone — career evidence may compensate.

### Skill credibility score (weight 0.09)

Three components averaged together: platform-verified assessment scores for JD-relevant skills (normalized to 0–1); count of JD-relevant skills with over twelve months of claimed duration (capped at five); count of JD-relevant skills with more than ten endorsements (capped at three). Long duration is harder to fabricate than a skill listing; meaningful endorsement counts on specific technical skills suggest peers verified the claim.

### Company prestige score (weight 0.08)

Tier 1 companies (Google, Microsoft, Meta, Apple, Amazon, Netflix, Uber, Stripe, OpenAI, and others) score 1.0. Tier 2 companies (Flipkart, Swiggy, Zomato, Razorpay, Paytm, Sarvam, Rephrase, and other strong Indian product companies) score 0.7. Unknown companies score 0.3. Weighted by fraction of total career months at each company — a brief stint contributes proportionally less than a multi-year tenure.

### Evaluation framework score (weight 0.08)

Reads descriptions for evidence of rigorous evaluation methodology: NDCG, MRR, MAP, A/B testing, offline and online evaluation, recall@K, precision@K, click-through rate, offline-online correlation. The JD explicitly states that candidates who have never thought rigorously about how to evaluate a ranking system will find the role painful. Three or more matches return 1.0.

### Experience fit score (weight 0.07)

Maps years of experience to the JD's 5–9 year target with graceful degradation.

| Years of experience | Score |
|---|---|
| 5 – 9 (target range) | 1.00 |
| 9 – 12 (slightly senior) | 0.85 |
| 3 – 5 (slightly junior) | 0.75 |
| 12 – 15 (senior) | 0.65 |
| 15+ (very senior) | 0.50 |
| Under 3 (very junior) | 0.40 |

### Skill description consistency score (weight 0.05)

Checks whether skills claimed at advanced or expert proficiency with over six months of duration actually appear in career descriptions. Returns 0.5 (neutral) for candidates making no high-proficiency claims rather than penalizing modesty. Weighted at 0.05 because templated synthetic descriptions reduce its reliability.

### GitHub activity score (weight 0.04)

| GitHub score | Feature value |
|---|---|
| -1 (no account linked) | 0.40 (neutral) |
| 70 – 100 | 1.00 |
| 30 – 70 | Linear scale 0.60 – 1.00 |
| 0 – 30 | Linear scale 0.30 – 0.60 |

Weighted low because this is a credibility signal, not a domain signal — junior and senior engineers alike can have high GitHub activity scores.

### Location score (weight 0.04)

| Candidate location | Score |
|---|---|
| India, preferred city | 1.00 |
| India, other city | 0.85 |
| Outside India, willing to relocate | 0.50 |
| Outside India, not willing to relocate | 0.20 |

The JD states India preferred, outside India case-by-case with no visa sponsorship — this encodes that as a soft preference, not a hard filter, consistent with the JD's explicit language.

### Education score (weight 0.00)

Tier 1: 1.0, Tier 2: 0.75, Tier 3: 0.5, Tier 4 and unknown: 0.3. Weight is zero because the JD never mentions institutional pedigree as a factor. The code is retained so the feature can be re-enabled without changes if evidence emerged that it was predictive.

---

## Honeypot Detection

The dataset contains honeypot profiles with internally inconsistent or fabricated data, designed to trap naive ranking systems. Honeypot detection runs before any relevance scoring. A candidate that fails any check is assigned a score of 0.001 and excluded from all three RRF ranked lists — no feature extraction, embedding lookup, or TF-IDF comparison is performed for detected honeypots.

> The eight checks run in order of computational cost — cheapest first — so the most common failure modes are caught before the more expensive checks are evaluated.

| # | Rule | Description |
|---|---|---|
| 1 | Inverted salary | Expected salary minimum exceeds maximum. Logically impossible. |
| 2 | Impossible timeline | Platform signup date falls after last active date. Logically impossible. |
| 3 | Fabricated expertise | Two or more skills claimed at expert proficiency with under six months of usage duration. |
| 4 | Impossible skill duration | Any skill's claimed duration exceeds total years of experience plus a twelve-month tolerance buffer. |
| 5 | Experience inflation | Total career history duration exceeds stated years of experience by more than three years. |
| 6 | Future job start date | A current job's recorded start date is after the reference date (June 14, 2026). |
| 7 | Endorsement velocity | Total endorsements divided by days on platform exceeds ten per day. |
| 8 | Education timeline | An education record's end year precedes its start year, or extends past 2026. |

### Verified breakdown across 100,000 candidates

| Rule | Failures | % of total |
|---|---|---|
| 1 — Inverted salary | 18,865 | 18.9% |
| 2 — Impossible timeline | 7,496 | 7.5% |
| 3 — Fabricated expertise | 21 | 0.0% |
| 4 — Impossible skill duration | 9,231 | 9.2% |
| 5 — Experience inflation | 22 | 0.0% |
| 6 — Future start date | 0 | 0.0% |
| 7 — Endorsement velocity | 0 | 0.0% |
| 8 — Education timeline | 0 | 0.0% |
| **Total flagged** | **32,111** | **32.1%** |
| **Valid (pass all rules)** | **67,889** | **67.9%** |

The 32.1% flagged rate reflects systematic data generation artifacts in the synthetic dataset rather than intentional deception in every case. Inverted salary ranges (18.9%) and skill durations assigned independently of career history (9.2%) are structural properties of the generator. All eight checks are logically sound — inverted salary has no legitimate explanation, impossible timelines are binary logical failures, and skill duration allows a twelve-month buffer for concurrent roles. None of the 32,111 flagged candidates appear in the final top 100.

---

## Why Weighted Reciprocal Rank Fusion

### The scale mismatch problem

The three signals produce scores in fundamentally different numeric ranges:

| Signal | Score range | Characteristics |
|---|---|---|
| Rule scores | 0.08 – 0.72 | Wide, structured, reliable |
| Embedding scores | 0.20 – 0.42 | Narrow, semantic, partially template-degraded |
| TF-IDF scores | 0.01 – 0.18 | Sparse, lexical, lowest magnitude |

A naive weighted linear combination is broken by this distribution. A TF-IDF score change from 0.04 to 0.12 (a 3× increase, genuinely meaningful) contributes only `0.15 × 0.08 = 0.012` to the final score when weighted at 15%. The TF-IDF signal is effectively disabled regardless of its nominal weight.

### Formula

```
RRF(candidate) = Σᵢ  wᵢ / (k + rankᵢ(candidate))

where:
  i ∈ {rule, embedding, tfidf}
  w_rule      = 0.50
  w_embedding = 0.30
  w_tfidf     = 0.20
  k           = 60   (Cormack et al. 2009 — not tuned)
```

Rank-based fusion converts each signal's raw scores to rank positions before combining, making the fusion scale-invariant by construction. `k=60` controls rank decay sharpness and was not tuned — the paper's validated default is used to avoid introducing a fitting parameter.

### Why weighted rather than equal-weight

Equal-weight RRF (1/3 each) is not justified because the signals are not equally predictive for this JD. The rule signal encodes seven of the JD's eight requirement categories from structured, verified fields. TF-IDF and embeddings both read the same career narrative text, with TF-IDF being the less expressive of the two. The 5:3:2 ratio reflects the hierarchy of signal reliability, derived from JD analysis rather than fitted to data.

### Concrete demonstration

```
Candidate A: rule rank 50,  embedding rank 2000, tfidf rank 500
             (strong structured JD fit, weak narrative match)

Candidate B: rule rank 800, embedding rank 60,   tfidf rank 80
             (weak structured fit, strong narrative match)

Equal-weight RRF:
  A = 1/110 + 1/2060 + 1/560 = 0.01136
  B = 1/860 + 1/120  + 1/140 = 0.01664  ← B incorrectly ranks higher

Weighted RRF (0.50 / 0.30 / 0.20):
  A = 0.50/110 + 0.30/2060 + 0.20/560 = 0.005048
  B = 0.50/860 + 0.30/120  + 0.20/140 = 0.004510  ← A correctly ranks higher
```

Candidate A has strong structural JD fit — the profile of an ML engineer at a product company with the right experience range. Candidate B has a better-written description. Weighted RRF correctly prioritizes structural fit over narrative polish.

---

## AI Components

### Where AI is used

**Semantic embedding model — `all-MiniLM-L6-v2`:** a six-layer MiniLM transformer (22MB) from the sentence-transformers library, pre-trained with contrastive learning on sentence pairs. Used to encode candidate career texts and the JD text into 384-dimensional vectors where semantic similarity corresponds to geometric proximity. Selected for CPU feasibility at 100,000-candidate scale.

**Embedding computation:** `model.encode()` with `normalize_embeddings=True` produces L2-normalized unit vectors, making dot product equivalent to cosine similarity.

### Where AI is not used

The rule engine, honeypot detection, and reasoning generation contain no machine learning. Features are computed from deterministic functions over structured fields. Honeypot detection is eight logical checks. Reasoning is template instantiation from actual profile values — no language model is involved in generation. Every ranking decision is fully traceable to specific field values.

---

## Dataset Analysis and Findings

### Career description templates

Direct inspection of candidate profiles revealed that many career description paragraphs are reused verbatim across unrelated candidates — identical text appears describing different companies, different roles, and sometimes different durations within the same candidate's own history. This is a synthetic dataset generation artifact. The practical effect: both TF-IDF and embedding signals partially score template quality rather than genuine career relevance.

### Why no duplicate-content penalty was introduced

A duplicate-content penalty was investigated and rejected for two reasons. First, the duplication is pervasive and roughly evenly distributed across strong and weak candidates — penalizing it would degrade scores broadly rather than selectively. Second, a penalty calibrated to this synthetic artifact would not generalize to the real platform this system is meant to inform.

### How the system's design mitigates this

The rule engine reads structured fields that are immune to narrative templating: title, company name, country, years of experience, platform assessment scores. These fields cannot be template-corrupted. The rule signal's 0.50 weight in the fusion specifically reflects this reliability advantage over the two text-based signals.

---

## Weight Derivation Methodology

### Why labels were not used

No relevance judgments were provided. Training a learning-to-rank model on the available proxy data would overfit to the one known-relevant candidate in the 50-candidate sample rather than learning generalizable ranking signals.

### How weights were derived

Weights are JD-derived priors, not fitted parameters. Each JD requirement was mapped to the feature that captures it most directly, then ordered by how explicitly the JD discusses that requirement:

| Tier | Feature | Weight | JD justification |
|---|---|---|---|
| 1 | `retrieval_evidence` | 0.28 | *"Production experience with embedding-based retrieval"* — opening requirement |
| 1 | `product_company` | 0.15 | *"People who've only worked at consulting firms... we've had bad fit experiences"* |
| 2 | `title_domain` | 0.12 | *"A Marketing Manager with AI keywords is a trap"* |
| 2 | `skill_credibility` | 0.09 | Platform-verified assessment scores — only externally validated signal |
| 2 | `evaluation_framework` | 0.08 | *"If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful"* |
| 2 | `experience_fit` | 0.07 | *"5-9 years"* explicitly stated |
| 3 | `company_prestige` | 0.08 | Implicit in product vs. services preference |
| 3 | `skill_description_consistency` | 0.05 | Anti-stuffing defense |
| 3 | `github_activity` | 0.04 | Credibility signal, not domain signal |
| 3 | `location` | 0.04 | *"Pune/Noida preferred"*, *"outside India case-by-case"* |
| 4 | `education` | 0.00 | JD never mentions institutional pedigree |

### Validation without labels

Weights were validated against a 50-candidate proxy sample: the known-relevant candidate should rank first; no non-technical titles should appear in the top 10; and the score gap between rank 1 and rank 2 should be preserved under weight perturbation. `calibrate.py` implements this validation.

---

## Validation and Evaluation

The system was validated across five dimensions before submission:

| Dimension | Method | Tool |
|---|---|---|
| Submission format | Header, row count, rank uniqueness, score ordering, tie-breaking | `validate_submission.py` |
| Honeypot containment | Zero candidates with score at 0.001 floor in top 100 | Manual inspection |
| Non-technical exclusion | No non-technical titles in top 20 | `inspect_full.py` |
| Proxy relevance | Known-relevant candidate at rank 1; no non-technical titles in top 10 | `calibrate.py` |
| Reasoning quality | Candidate-specific evidence varies; concern flags surface correctly | `check_reasoning.py` |

---

## Reproducibility

> The ranking pipeline produces identical output on every run given identical input. No runtime randomness, no `datetime.now()` calls in scoring logic, and a deterministic sort key (`-round(score, 6), candidate_id`) ensure full reproducibility.

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Precompute candidate embeddings

**One-time. Approximately 80 minutes on CPU.**

```bash
python precompute.py --candidates data/candidates.jsonl
```

On a fresh machine, this step downloads and caches the `all-MiniLM-L6-v2` model (22MB) from HuggingFace on first run, then encodes all 100,000 candidates and saves `data/candidate_embeddings.npy` and `data/candidate_ids.npy`. On subsequent runs, the model is read from local cache. This step only needs to be run once, or again if the candidate pool changes.

### Step 3 — Run the ranker

**Approximately 110 seconds on CPU. No network access required.**

```bash
python rank.py --candidates data/candidates.jsonl --out submission.csv
```

After precomputation, the ranking step loads pre-computed embeddings from disk, fits TF-IDF, runs the rule engine, applies Weighted RRF, generates reasoning, and writes `submission.csv`. The model is read from local cache.

### Step 4 — Validate the submission

```bash
python validate_submission.py submission.csv
```

Expected output: `Submission is valid.`

### Runtime breakdown

| Stage | Runtime |
|---|---|
| Precomputation (one-time) | ~80 minutes |
| Candidate loading | ~9 seconds |
| Embedding scores (load from disk + JD encoding) | ~2 seconds |
| TF-IDF fitting and similarity | ~55 seconds |
| Rule scoring (100,000 candidates) | ~45 seconds |
| RRF fusion and output | <1 second |
| **Total ranking runtime** | **~110 seconds** |

---

## Repository Structure

```
redrob-ranker/
│
├── rank.py                      Main entry point. Orchestrates all six pipeline stages.
├── precompute.py                One-time embedding precomputation for all candidates.
├── calibrate.py                 Proxy weight validation on the 50-candidate sample.
│
├── src/
│   ├── __init__.py
│   ├── features.py              11 structured feature functions. Returns feature dict.
│   ├── scorer.py                Feature combination, behavioral multiplier, notice penalty.
│   ├── honeypot.py              8 integrity checks. Runs before any relevance scoring.
│   ├── embeddings.py            Sentence-transformer encoding and similarity computation.
│   └── reasoning.py             Fact-grounded reasoning generation for top 100 candidates.
│
├── data/
│   ├── candidates.jsonl         100,000-candidate pool (not committed — 465MB)
│   ├── sample_candidates.json   First 50 candidates for sandbox and local inspection
│   ├── candidate_embeddings.npy Generated by precompute.py — not committed to git
│   └── candidate_ids.npy        Generated by precompute.py — not committed to git
│
├── validate_submission.py       Format validator provided by challenge organizers
├── submission.csv               Final ranked output (generated by rank.py)
├── submission_metadata.yaml     Hackathon metadata: team, compute, methodology summary
├── requirements.txt             Pinned Python dependencies
└── README.md                    This file
```

---

## Limitations

**No labeled relevance data.** Feature weights were derived from JD analysis rather than optimized against held-out labels. They are principled and traceable, but not provably optimal. Learning-to-rank models were not used because training on a single known-positive example would overfit.

**No cross-encoder reranking.** A cross-encoder re-ranking stage would likely improve NDCG@10 by 3–8% on a real dataset. The expected quality gain on a synthetic templated dataset was lower than on real data, and the model download requirement conflicted with the offline constraint at ranking time.

**Synthetic description templating degrades text signals.** Career description paragraphs are reused verbatim across candidates in this dataset. Both embedding similarity and TF-IDF partially score template quality rather than genuine career relevance. This is inherent to the dataset and not addressable within the system without ground-truth labels to identify which descriptions are authentic.

**Services company classification coverage.** The curated company list covers known major consulting and IT services firms, supplemented by industry-field checks. Fictional company names in the synthetic dataset are not recognized and default to "product company" classification.

---

## Future Improvements

> These improvements were intentionally omitted. The challenge provided no relevance labels, required CPU-only execution, and emphasized reproducibility — conditions under which learned models overfit and opaque systems become indefensible. The current design is the strongest approach available under those constraints.

**Cross-encoder reranking.** A two-stage pipeline — first-stage retrieval using the current RRF system, second-stage reranking using a cross-encoder reading (JD, candidate-text) pairs jointly — would capture relevance signals that bi-encoders and keyword models miss.

**Learning-to-rank with human judgments.** Given even a small set of graded relevance labels, a LambdaMART or neural ranking model could replace the hand-crafted weight vector with data-driven weights optimized directly for NDCG.

**Sentence-transformer domain fine-tuning.** Fine-tuning `all-MiniLM-L6-v2` on recruiting domain data would substantially improve semantic similarity quality for career narrative matching.

**Real-time incremental pipeline.** A production system would maintain a vector index with incremental updates as new candidates join, rather than processing all 100,000 candidates from scratch on each run.

---

## Technical Decisions and Tradeoffs

**Three signals instead of one.** Each signal fails differently — rules cannot read narrative nuance, embeddings degrade on templated text, TF-IDF misses semantic equivalence. Using all three, each where it is most reliable, is more robust than perfecting any one.

**Multiplicative behavioral adjustment.** Adding behavioral signals to the technical score would allow a highly-engaged but unqualified candidate to outscore a qualified but dormant one. The multiplicative design ensures behavioral signals modify hireability without overriding technical fit.

**Precomputed embeddings.** Separating the 80-minute encoding step from the sub-5-minute ranking step is required by the evaluation constraints. In production, a vector index with incremental updates would eliminate this distinction.

**Fixed `REFERENCE_DATE`.** Honeypot date checks use a hardcoded reference date (`2026-06-14`) rather than `datetime.now()`, ensuring identical results regardless of when the system is evaluated.

**JD-derived weights, not fitted.** With one known positive in the 50-candidate sample, any learned weights would memorize that profile rather than generalize. JD-derived weights are a deliberate choice under zero labeled training data.

---

## Conclusion

This system represents a complete, production-ready approach to zero-label candidate ranking. It correctly identifies that the problem has three distinct dimensions — technical qualification (from both structured fields and narrative text), data integrity (honeypot detection), and practical hireability (behavioral availability) — and handles each with the appropriate tool rather than a single uniform approach.

The Weighted RRF fusion resolves the fundamental scale-incompatibility problem that breaks naive score combination, without requiring labeled data to parameterize. The behavioral multiplier design correctly models the relationship between technical fit and hiring feasibility. The fact-grounded reasoning generation ensures every ranking decision is auditable against the source data.

Every design decision has a specific, traceable justification in the job description, the dataset properties, or the information retrieval literature.

---

<div align="center">
<sub>Redrob AI Hackathon · Intelligent Candidate Discovery & Ranking Challenge · vasudha-redrob</sub>
</div>
