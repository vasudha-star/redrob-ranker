"""
calibrate.py
Proxy weight calibration on 50-sample candidates.
Validates proposed weights against current weights.
Run: python calibrate.py
"""

import json
import sys
sys.path.insert(0, ".")
import src.scorer as scorer_mod

with open("data/sample_candidates.json") as f:
    sample = json.load(f)

NON_TECH = {
    "marketing manager", "operations manager", "hr manager", "accountant",
    "civil engineer", "mechanical engineer", "business analyst",
    "project manager", "customer support", "customer service",
    "graphic designer", "content writer", "sales executive"
}

CURRENT_WEIGHTS = {
    "retrieval_evidence":            0.27,
    "product_company":               0.14,
    "title_domain":                  0.11,
    "skill_credibility":             0.09,
    "company_prestige":              0.10,
    "skill_description_consistency": 0.07,
    "experience_fit":                0.07,
    "evaluation_framework":          0.08,
    "github_activity":               0.05,
    "location":                      0.02,
    "education":                     0.00,
}

PROPOSED_WEIGHTS = {
    "retrieval_evidence":            0.28,
    "product_company":               0.15,
    "title_domain":                  0.12,
    "skill_credibility":             0.09,
    "company_prestige":              0.08,
    "skill_description_consistency": 0.05,
    "experience_fit":                0.07,
    "evaluation_framework":          0.08,
    "github_activity":               0.04,
    "location":                      0.04,
    "education":                     0.00,
}


def evaluate(weights: dict, label: str) -> dict:
    original = scorer_mod.WEIGHTS.copy()
    scorer_mod.WEIGHTS = weights

    results = []
    for c in sample:
        score, _ = scorer_mod.score_candidate(c)
        results.append({
            "id": c["candidate_id"],
            "score": score,
            "title": c["profile"]["current_title"],
            "company": c["profile"]["current_company"],
            "country": c["profile"]["country"],
        })

    scorer_mod.WEIGHTS = original
    results.sort(key=lambda x: (-x["score"], x["id"]))

    rank1_correct = results[0]["id"] == "CAND_0000031"
    score_gap = results[0]["score"] - results[1]["score"] if len(results) > 1 else 0
    non_tech_top10 = sum(
        1 for r in results[:10]
        if any(nt in r["title"].lower() for nt in NON_TECH)
    )

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  CAND_0000031 rank 1 : {rank1_correct}")
    print(f"  Score gap (r1 - r2) : {score_gap:.4f}")
    print(f"  Non-tech in top 10  : {non_tech_top10}")
    print(f"\n  TOP 10:")
    for i, r in enumerate(results[:10], 1):
        marker = " ← CAND_0000031" if r["id"] == "CAND_0000031" else ""
        non_india = " [non-India]" if r["country"] != "India" else ""
        print(f"  #{i:<3} {r['id']}  {r['title'][:28]:<28}  "
              f"{r['company'][:18]:<18}{non_india}{marker}")

    return {"rank1_correct": rank1_correct, "gap": score_gap,
            "non_tech": non_tech_top10}


r_current = evaluate(CURRENT_WEIGHTS, "CURRENT WEIGHTS")
r_proposed = evaluate(PROPOSED_WEIGHTS, "PROPOSED WEIGHTS")

print(f"\n{'='*55}")
print("  VERDICT")
print(f"{'='*55}")
gap_improved = r_proposed["gap"] >= r_current["gap"] * 0.85
rank1_ok = r_proposed["rank1_correct"]
non_tech_ok = r_proposed["non_tech"] <= r_current["non_tech"]

if rank1_ok and gap_improved and non_tech_ok:
    print("  ✅ PROPOSED WEIGHTS VALIDATED — apply to scorer.py")
else:
    print("  ❌ PROPOSED WEIGHTS FAILED — investigate before applying")
    if not rank1_ok:
        print("     Reason: CAND_0000031 no longer rank 1")
    if not gap_improved:
        print(f"     Reason: Score gap reduced "
              f"({r_current['gap']:.4f} → {r_proposed['gap']:.4f})")
    if not non_tech_ok:
        print("     Reason: More non-tech candidates in top 10")