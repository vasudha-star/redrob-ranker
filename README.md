# Redrob Intelligent Candidate Discovery & Ranking Challenge

## Approach Summary
Hybrid TF-IDF + Rule-based ranking system for matching candidates
to a Senior AI Engineer job description across 100,000 profiles.

### Architecture
- **Stage 1 — Honeypot Filter:** Eliminates candidates with impossible
  signals (inverted salary, impossible timeline, fabricated expertise)
- **Stage 2 — TF-IDF Scoring:** Cosine similarity between candidate
  career descriptions and a curated JD query document
- **Stage 3 — Rule-based Scoring:** Eight hand-crafted features derived
  directly from JD requirements
- **Stage 4 — Behavioral Multiplier:** Availability modifier based on
  recency, response rate, and interview completion
- **Stage 5 — Combined Score:** Weighted combination of TF-IDF (35%)
  and rule-based (65%) scores

### Feature Weights
| Feature | Weight | Rationale |
|---------|--------|-----------|
| Retrieval evidence | 35% | Core JD requirement |
| Product company experience | 20% | JD explicitly disqualifies services-only |
| Title domain alignment | 15% | Guards against keyword stuffers |
| Skill credibility | 10% | Platform-verified assessments prioritized |
| Experience fit (5-9 yrs) | 8% | JD target range |
| Location (India preferred) | 7% | JD location requirement |
| Education tier | 3% | Minor signal |
| Evaluation framework | 2% | JD hard requirement |

## Project Structure
redrob-ranker/

├── data/                          # Place candidates.jsonl here

├── src/

│   ├── features.py                # Feature engineering functions

│   ├── honeypot.py                # Honeypot detection

│   ├── scorer.py                  # Scoring framework

│   └── reasoning.py               # Reasoning generation

├── rank.py                        # Main entry point

├── validate_submission.py         # Format validator

├── requirements.txt

└── README.md


## Setup

```bash
git clone https://github.com/YOUR_USERNAME/redrob-ranker
cd redrob-ranker
pip install -r requirements.txt
```

Place `candidates.jsonl` in the `data/` folder.

## Reproduction Command

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

## Runtime
- Full 100K candidates: ~97 seconds
- CPU only, no GPU required
- No network calls during ranking
- Memory: ~4GB peak

## Validation
```bash
python validate_submission.py submission.csv
```

## AI Tools Used
- Claude (architecture discussion, code review)

## Compute Environment
- Python 3.12
- CPU only
- 16GB RAM