"""
src/honeypot.py
Honeypot detection — eliminates candidates with impossible profiles.
A honeypot gets score=0.001 and never enters the top 100.

Detection rules in order of cheapest to most expensive:
1. Inverted salary (min > max) — logically impossible
2. Impossible timeline (signup > last_active) — logically impossible
3. Expert + near-zero duration on 2+ skills — fabricated expertise
4. Skill duration exceeding total experience — impossible
5. Experience inflation (history >> stated years) — fabricated history
6. Future job start date — logically impossible
7. Endorsement velocity too high for platform age — manipulation
8. Education timeline impossible — logically impossible
"""

from datetime import datetime
from dateutil.parser import parse

REFERENCE_DATE = datetime(2026, 6, 14)


def _check_inverted_salary(candidate: dict) -> tuple[bool, str]:
    sal = candidate["redrob_signals"]["expected_salary_range_inr_lpa"]
    if sal["min"] > sal["max"]:
        return True, "inverted_salary"
    return False, ""


def _check_impossible_timeline(candidate: dict) -> tuple[bool, str]:
    signals = candidate["redrob_signals"]
    signup = parse(signals["signup_date"])
    last_active = parse(signals["last_active_date"])
    if signup > last_active:
        return True, "impossible_timeline"
    return False, ""


def _check_fabricated_expertise(candidate: dict) -> tuple[bool, str]:
    """
    Expert proficiency with near-zero usage duration.
    Threshold lowered to 2 skills (was 3) — genuine experts almost
    always have real duration behind the claim, so 2 simultaneous
    near-zero-duration "expert" claims is already a strong signal.
    """
    skills = candidate.get("skills", [])
    suspicious = sum(
        1 for s in skills
        if s.get("proficiency") == "expert"
        and s.get("duration_months", 0) < 6
    )
    if suspicious >= 2:
        return True, "fabricated_expertise"
    return False, ""


def _check_skill_duration_impossible(candidate: dict) -> tuple[bool, str]:
    """
    A skill cannot have been used longer than the candidate's total
    career length. 12-month tolerance for overlapping roles / rounding.
    """
    yoe_months = candidate["profile"].get("years_of_experience", 0) * 12
    skills = candidate.get("skills", [])
    for skill in skills:
        duration = skill.get("duration_months", 0)
        if duration > yoe_months + 12:
            return True, "impossible_skill_duration"
    return False, ""


def _check_experience_inflation(candidate: dict) -> tuple[bool, str]:
    """
    Career history duration should not vastly exceed stated experience.
    3-year tolerance for overlapping roles.
    """
    stated_years = candidate["profile"].get("years_of_experience", 0)
    history = candidate.get("career_history", [])
    total_months = sum(job.get("duration_months", 0) for job in history)
    total_years = total_months / 12.0
    if total_years > stated_years + 3:
        return True, "experience_inflation"
    return False, ""


def _check_future_start_date(candidate: dict) -> tuple[bool, str]:
    """Current job cannot have started in the future."""
    for job in candidate.get("career_history", []):
        if job.get("is_current"):
            try:
                start = parse(job["start_date"])
                if start > REFERENCE_DATE:
                    return True, "future_start_date"
            except Exception:
                pass
    return False, ""


def _check_endorsement_velocity(candidate: dict) -> tuple[bool, str]:
    """
    Endorsements cannot accumulate faster than ~10/day on average.
    A candidate with 500 endorsements who signed up 30 days ago
    is not realistic on a real platform.
    """
    signals = candidate["redrob_signals"]
    try:
        signup = parse(signals["signup_date"])
        days_on_platform = max(1, (REFERENCE_DATE - signup).days)
        total_endorsements = signals.get("endorsements_received", 0)
        if total_endorsements / days_on_platform > 10:
            return True, "impossible_endorsement_velocity"
    except Exception:
        pass
    return False, ""


def _check_education_timeline(candidate: dict) -> tuple[bool, str]:
    """
    Education end year must be after start year and not in the future.
    """
    for edu in candidate.get("education", []):
        start = edu.get("start_year", 0)
        end = edu.get("end_year", 0)
        if end < start:
            return True, "impossible_education_timeline"
        if end > 2026:
            return True, "future_graduation"
    return False, ""


def is_honeypot(candidate: dict) -> tuple[bool, str]:
    """
    Master honeypot detection function.
    Runs all checks in order of cheapest to most expensive.
    Returns (True, reason) on first match, else (False, "").
    """
    checks = [
        _check_inverted_salary,
        _check_impossible_timeline,
        _check_fabricated_expertise,
        _check_skill_duration_impossible,
        _check_experience_inflation,
        _check_future_start_date,
        _check_endorsement_velocity,
        _check_education_timeline,
    ]

    for check in checks:
        flagged, reason = check(candidate)
        if flagged:
            return True, reason

    return False, ""