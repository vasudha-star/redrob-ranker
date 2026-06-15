"""
src/scorer.py
Combines features and behavioral signals into a final score.
"""

from datetime import datetime
from dateutil.parser import parse
from src.features import extract_features
from src.honeypot import is_honeypot

REFERENCE_DATE = datetime(2026, 6, 14)

# Feature weights — must sum to 1.0
# Retrieval evidence is highest because it is the core JD requirement.
WEIGHTS = {
    "retrieval_evidence":   0.35,
    "product_company":      0.20,
    "title_domain":         0.15,
    "skill_credibility":    0.10,
    "experience_fit":       0.08,
    "location":             0.07,
    "education":            0.03,
    "evaluation_framework": 0.02,
}


def behavioral_multiplier(signals: dict) -> float:
    """
    Converts behavioral signals into a multiplier (0.3 to 1.3).
    
    Why multiplier and not additive:
    Behavioral signals modify HIREABILITY, not skill fit.
    A great candidate who never responds is unhireable.
    A weak candidate who responds instantly is still weak.
    
    Components:
    - Recency: how recently they logged in
    - Response rate: do they reply to recruiters
    - Interview completion: do they show up
    - Open to work: explicit availability signal
    """
    # Recency score
    last_active = parse(signals["last_active_date"])
    days_inactive = (REFERENCE_DATE - last_active).days
    if days_inactive <= 30:
        recency = 1.0
    elif days_inactive <= 90:
        recency = 0.8
    elif days_inactive <= 180:
        recency = 0.6
    else:
        recency = 0.3

    response_rate = signals["recruiter_response_rate"]
    interview_rate = signals["interview_completion_rate"]
    open_bonus = 1.1 if signals["open_to_work_flag"] else 1.0

    composite = (
        0.4 * recency +
        0.3 * response_rate +
        0.3 * interview_rate
    )

    return max(0.3, min(1.3, composite)) * open_bonus


def notice_period_multiplier(signals: dict) -> float:
    """
    Small penalty for long notice periods.
    JD says sub-30 days preferred, 30 days buyable, 30+ = higher bar.
    """
    days = signals.get("notice_period_days", 90)
    if days <= 30:
        return 1.0
    elif days <= 60:
        return 0.95
    elif days <= 90:
        return 0.88
    else:
        return 0.80


def score_candidate(candidate: dict) -> tuple[float, dict]:
    """
    Master scoring function.
    
    Returns:
        (final_score, feature_dict)
    
    Pipeline:
    1. Honeypot check — if flagged, return 0.001 immediately
    2. Extract features
    3. Compute weighted base score
    4. Apply behavioral multiplier
    5. Apply notice period multiplier
    """
    # Step 1: Honeypot check
    flagged, reason = is_honeypot(candidate)
    if flagged:
        return 0.001, {"honeypot_reason": reason}

    # Step 2: Extract features
    features = extract_features(candidate)

    # Step 3: Weighted base score
    base_score = sum(
        WEIGHTS[feature] * features[feature]
        for feature in WEIGHTS
    )

    # Step 4: Behavioral multiplier
    signals = candidate["redrob_signals"]
    b_mult = behavioral_multiplier(signals)

    # Step 5: Notice period multiplier
    n_mult = notice_period_multiplier(signals)

    final_score = base_score * b_mult * n_mult

    return final_score, features