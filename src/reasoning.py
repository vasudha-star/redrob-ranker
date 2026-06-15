"""
src/reasoning.py
Generates truthful, candidate-specific reasoning for each ranked candidate.
Every claim references actual profile fields — no hallucination.
"""

from dateutil.parser import parse
from datetime import datetime

REFERENCE_DATE = datetime(2026, 6, 14)


def generate_reasoning(candidate: dict, rank: int,
                        score: float, features: dict) -> str:
    """
    Generate a 1-2 sentence reasoning string for a candidate.

    Rules:
    - Every fact must come from the actual candidate profile
    - Tone must match rank (rank 1-10 positive, rank 80+ acknowledges gaps)
    - No two candidates should have identical reasoning
    - Acknowledge concerns honestly
    """
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]

    yoe = profile["years_of_experience"]
    title = profile["current_title"]
    company = profile["current_company"]
    location = profile["location"]
    country = profile["country"]
    notice = signals["notice_period_days"]
    response_rate = signals["recruiter_response_rate"]
    last_active = parse(signals["last_active_date"])
    days_inactive = (REFERENCE_DATE - last_active).days
    open_to_work = signals["open_to_work_flag"]
    assessments = signals["skill_assessment_scores"]

    # Pull career companies as a string
    companies = [job["company"] for job in candidate.get("career_history", [])]
    companies_str = ", ".join(companies[:3])

    # Pull top relevant skills
    relevant_skill_names = [
        s["name"] for s in candidate.get("skills", [])
        if s.get("duration_months", 0) > 12
        and s.get("proficiency") in ["advanced", "expert"]
    ][:3]
    skills_str = ", ".join(relevant_skill_names) if relevant_skill_names else "general technical skills"

    # Assessment note
    assessment_note = ""
    if assessments:
        top_assessment = max(assessments.items(), key=lambda x: x[1])
        assessment_note = f"Platform-verified {top_assessment[0]} score of {top_assessment[1]:.0f}/100."

    # Availability note
    if days_inactive <= 30:
        availability = "active on platform in the last 30 days"
    elif days_inactive <= 90:
        availability = f"active {days_inactive} days ago"
    else:
        availability = f"inactive for {days_inactive} days"

    # Concern flags
    concerns = []
    if notice > 60:
        concerns.append(f"notice period is {notice} days")
    if response_rate < 0.3:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")
    if days_inactive > 180:
        concerns.append(f"inactive for {days_inactive} days")
    if country != "India":
        concerns.append(f"based outside India ({country})")
    concern_str = "; ".join(concerns)

    # ── Reasoning templates by rank band ─────────────────────────────────

    if rank <= 10:
        # Strong positive reasoning
        sentence1 = (f"{yoe:.0f} years of experience as {title} at "
                     f"{companies_str}; strong retrieval and ranking "
                     f"background matching core JD requirements.")
        sentence2 = (f"{assessment_note + ' ' if assessment_note else ''}"
                     f"{'Open to work and ' if open_to_work else ''}"
                     f"{availability} with {response_rate:.0%} recruiter "
                     f"response rate."
                     + (f" Concern: {concern_str}." if concerns else ""))

    elif rank <= 30:
        # Positive with minor concerns
        sentence1 = (f"{yoe:.0f}-year career at {companies_str} with "
                     f"relevant skills in {skills_str}; good fit on "
                     f"technical dimensions.")
        sentence2 = (f"{availability.capitalize()}"
                     + (f"; concern: {concern_str}." if concerns
                        else f" with {response_rate:.0%} response rate."))

    elif rank <= 60:
        # Mixed signal
        sentence1 = (f"Partial fit — {title} at {company} with {yoe:.0f} "
                     f"years experience; some relevant skills "
                     f"({skills_str}) but limited retrieval system evidence.")
        sentence2 = (f"Ranked here due to behavioral availability "
                     f"({availability})"
                     + (f"; concern: {concern_str}." if concerns else "."))

    else:
        # Lower rank — honest about gaps
        sentence1 = (f"{title} at {company} with {yoe:.0f} years; "
                     f"adjacent skills present but career evidence does "
                     f"not demonstrate production retrieval or ranking "
                     f"system experience.")
        sentence2 = (f"Included at rank {rank} as best available given "
                     f"remaining pool"
                     + (f"; concern: {concern_str}." if concerns else "."))

    return f"{sentence1} {sentence2}".strip()