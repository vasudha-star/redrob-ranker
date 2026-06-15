"""
src/honeypot.py
Detects honeypot candidates before scoring.
A honeypot gets score=0.001 and never enters the top 100.
"""

from dateutil.parser import parse


def is_honeypot(candidate: dict) -> tuple[bool, str]:
    """
    Returns (True, reason) if candidate is a honeypot, else (False, "").
    
    Three detection rules:
    1. Inverted salary (min > max) — logically impossible
    2. Impossible timeline (signup_date > last_active_date) — logically impossible  
    3. Expert proficiency on 3+ skills with duration < 6 months — fabricated expertise
    """
    signals = candidate["redrob_signals"]

    # Rule 1: Inverted salary
    sal = signals["expected_salary_range_inr_lpa"]
    if sal["min"] > sal["max"]:
        return True, "inverted_salary"

    # Rule 2: Impossible timeline
    signup = parse(signals["signup_date"])
    last_active = parse(signals["last_active_date"])
    if signup > last_active:
        return True, "impossible_timeline"

    # Rule 3: Expert claims with near-zero experience
    skills = candidate.get("skills", [])
    suspicious_expert = sum(
        1 for s in skills
        if s.get("proficiency") == "expert"
        and s.get("duration_months", 0) < 6
    )
    if suspicious_expert >= 3:
        return True, "fabricated_expertise"

    return False, ""