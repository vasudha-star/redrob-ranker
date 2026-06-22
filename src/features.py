"""
src/features.py
Feature engineering for the Redrob candidate ranking system.
Each function takes a candidate dictionary and returns a float 0.0-1.0
unless stated otherwise.
"""

import re
from dateutil.parser import parse

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Companies that are pure IT services / consulting.
# Candidates whose ENTIRE career is at these companies are down-ranked.
# Source: JD explicitly names these as a disqualifier.
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "techmahindra", "mindtree", "mphasis",
    "hexaware", "niit", "patni", "mastech", "syntel", "l&t infotech",
    "ltimindtree", "persistent", "mphasis", "birlasoft", "sonata",
    "zensar", "cyient", "kpit", "happiest minds", "coforge",
    "genpact" 
}

# Titles that are clearly non-technical.
# A candidate currently holding these titles gets a near-zero title score
# UNLESS their career descriptions contain strong retrieval evidence.
NON_TECH_TITLES = {
    "marketing manager", "operations manager", "hr manager",
    "human resources", "accountant", "civil engineer",
    "mechanical engineer", "graphic designer", "content writer",
    "sales executive", "business analyst", "project manager",
    "customer support", "customer service", "brand manager",
    "supply chain", "logistics", "finance manager", "legal"
}

# Titles that are clearly technical and relevant to the JD.
TECH_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "data scientist", "nlp engineer", "search engineer",
    "recommendation", "ranking engineer", "retrieval",
    "applied scientist", "research engineer", "data engineer",
    "software engineer", "backend engineer", "full stack",
    "fullstack", "platform engineer", "infrastructure engineer",
    "devops engineer", "cloud engineer", "python developer",
    "java developer", ".net developer", "mobile developer",
    "frontend engineer", "qa engineer"
}

# Keywords in career descriptions that signal retrieval/ranking/search work.
# These are the core JD requirements.
# Tier 1 — hard retrieval signals, unambiguous
RETRIEVAL_KEYWORDS_TIER1 = [
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "vector search", "dense retrieval",
    "semantic search", "hybrid search", "bm25", "inverted index",
    "learning to rank", "learning-to-rank", "ltr", "ndcg", "mrr",
    "mean average precision", "information retrieval",
    "sentence transformer", "bi-encoder", "cross-encoder",
    "approximate nearest neighbor", "reranking", "re-ranking",
    "recommendation system", "recommender system",
    "collaborative filtering", "matrix factorization",
    "query expansion", "query understanding",
    "a/b test", "offline evaluation", "online evaluation",
    "relevance judgment", "click-through rate", "solr",
    "recall@", "precision@", "offline-online"
]

# Tier 2 — soft signals, present in retrieval work but not exclusive
RETRIEVAL_KEYWORDS_TIER2 = [
    "embedding", "embeddings", "transformer", "bert",
    "hugging face", "fine-tuning", "fine-tune", "lora", "peft",
    "llm", "gpt", "rag", "langchain", "ranking", "retrieval",
    "feature engineering", "mlflow", "mlops"
]

# Keywords for evaluation framework experience specifically.
# The JD calls this out as a hard requirement.
EVALUATION_KEYWORDS = [
    "ndcg", "mrr", "map", "a/b test", "a/b testing",
    "offline evaluation", "online evaluation", "relevance",
    "recall@", "precision@", "click-through rate", "ctr",
    "engagement metric", "offline-online correlation"
]
# Tier 1 product companies — highest signal
# These companies are known for strong ML/AI engineering culture
PRESTIGE_COMPANIES_TIER1 = {
    "google", "microsoft", "meta", "apple", "amazon", "netflix",
    "uber", "airbnb", "linkedin", "twitter", "stripe", "openai",
    "anthropic", "deepmind", "nvidia", "salesforce"
}

# Tier 2 — strong Indian product companies
PRESTIGE_COMPANIES_TIER2 = {
    "flipkart", "swiggy", "zomato", "razorpay", "paytm", "ola",
    "cred", "meesho", "phonepe", "freshworks", "zerodha", "groww",
    "nykaa", "bigbasket", "dunzo", "urban company", "sharechat",
    "dream11", "unacademy", "byju", "upgrad", "lenskart",
    "policybazaar", "cars24", "spinny", "sarvam", "krutrim",
    "mad street den", "aganitha", "rephrase", "niramai"
}
# GitHub activity threshold — candidates with real engineering activity
GITHUB_ACTIVE_THRESHOLD = 30.0
# India locations preferred by JD.
PREFERRED_LOCATIONS = {
    "hyderabad", "pune", "noida", "bangalore", "bengaluru",
    "mumbai", "delhi", "chennai", "gurgaon", "gurugram",
    "kolkata", "ahmedabad", "india"
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def company_prestige_score(candidate: dict) -> float:
    """
    Score based on prestige of companies in career history,
    weighted by time spent — 1 month at Google should not score
    the same as 5 years at Google.

    Why duration-weighted:
    A candidate who interned at Google for 1 month a decade ago
    should not score identically to a current Google Staff Engineer.
    Weighting by duration reflects actual depth of experience.
    """
    history = candidate.get("career_history", [])
    if not history:
        return 0.3

    total_months = sum(job.get("duration_months", 0) for job in history)
    if total_months == 0:
        return 0.3

    weighted_score = 0.0
    for job in history:
        company = job.get("company", "").lower()
        duration = job.get("duration_months", 0)
        weight = duration / total_months

        company_score = 0.3  # default for unknown companies
        for t1 in PRESTIGE_COMPANIES_TIER1:
            if t1 in company:
                company_score = 1.0
                break
        else:
            for t2 in PRESTIGE_COMPANIES_TIER2:
                if t2 in company:
                    company_score = 0.7
                    break

        weighted_score += company_score * weight

    # Normalize: a career entirely at Tier-1 companies = 1.0
    # a career entirely at unknown companies = 0.3
    return min(1.0, weighted_score / 0.5)

def _normalize(text: str) -> str:
    """Lowercase and strip a string for consistent comparison."""
    return text.lower().strip()


def _get_all_descriptions(candidate: dict) -> str:
    """
    Concatenate all career descriptions into one string.
    We weight recent roles more by repeating them.
    Why: Recent experience matters more than old experience.
    """
    history = candidate.get("career_history", [])
    parts = []
    for i, job in enumerate(history):
        desc = job.get("description", "")
        # Most recent role (index 0) repeated 3x, next 2x, rest 1x
        if i == 0:
            parts.extend([desc, desc, desc])
        elif i == 1:
            parts.extend([desc, desc])
        else:
            parts.append(desc)
    return " ".join(parts).lower()


def _is_services_company(company_name: str) -> bool:
    """Return True if company is a pure IT services firm."""
    name = _normalize(company_name)
    for svc in SERVICES_COMPANIES:
        if svc in name:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1: TITLE DOMAIN SCORE
# ─────────────────────────────────────────────────────────────────────────────

def title_domain_score(candidate: dict) -> float:
    """
    Score based on current job title.
    
    Why this matters: The JD explicitly says a Marketing Manager with AI
    keywords is a trap. Title is the first signal of domain fit.
    
    Score breakdown:
    - Clearly relevant ML/AI/search title → 1.0
    - General tech title (SWE, backend, data) → 0.6
    - Non-technical title → 0.1
    
    We do NOT return 0.0 for non-tech titles because career history
    may redeem them. 0.1 keeps them alive but heavily down-ranked.
    """
    title = _normalize(candidate["profile"]["current_title"])

    # Check for highly relevant titles first
    for t in TECH_TITLES:
        if t in title:
            # Extra boost for directly relevant titles
            if any(kw in title for kw in
                   ["ml", "machine learning", "ai ", "nlp", "search",
                    "recommendation", "ranking", "retrieval", "data scientist"]):
                return 1.0
            return 0.6

    # Check for non-technical titles
    for t in NON_TECH_TITLES:
        if t in title:
            return 0.1

    # Unknown title — give benefit of the doubt
    return 0.4


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2: PRODUCT COMPANY SCORE
# ─────────────────────────────────────────────────────────────────────────────

# Industry keywords that indicate a services/consulting company,
# used as a backup check when the company name isn't on our list
SERVICES_INDUSTRY_KEYWORDS = {
    "it services", "consulting", "bpo", "outsourcing", "staffing"
}


def _is_services_by_industry(industry: str) -> bool:
    """Catches services companies not on our name list, using the
    industry field that's already in every job entry."""
    industry_lower = industry.lower()
    return any(kw in industry_lower for kw in SERVICES_INDUSTRY_KEYWORDS)


def product_company_score(candidate: dict) -> float:
    history = candidate.get("career_history", [])
    if not history:
        return 0.3

    total_months = 0
    product_months = 0

    for job in history:
        duration = job.get("duration_months", 0)
        company = job.get("company", "")
        industry = job.get("industry", "")
        total_months += duration

        is_services = _is_services_company(company) or _is_services_by_industry(industry)
        if not is_services:
            product_months += duration

    if total_months == 0:
        return 0.3

    ratio = product_months / total_months
    return max(0.1, ratio)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 3: RETRIEVAL EVIDENCE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_evidence_score(candidate: dict) -> float:
    """
    Score based on retrieval/ranking/search keywords in career descriptions.

    Why tiered keywords:
    Tier 1 = unambiguous retrieval signals (FAISS, BM25, NDCG)
    Tier 2 = general ML signals (transformer, LLM, RAG)
    A candidate who used GPT for content generation should not score
    the same as one who built a FAISS index in production.

    Tier 1 keywords worth 2x Tier 2.
    Normalize: 20 weighted points = 1.0
    """
    text = _get_all_descriptions(candidate)
    summary = candidate["profile"].get("summary", "").lower()
    combined = text + " " + summary

    tier1_matches = sum(1 for kw in RETRIEVAL_KEYWORDS_TIER1 if kw in combined)
    tier2_matches = sum(1 for kw in RETRIEVAL_KEYWORDS_TIER2 if kw in combined)

    weighted = (tier1_matches * 2) + (tier2_matches * 1)
    return min(1.0, weighted / 20.0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 4: EVALUATION FRAMEWORK SCORE
# ─────────────────────────────────────────────────────────────────────────────

def evaluation_framework_score(candidate: dict) -> float:
    """
    Score based on evaluation framework keywords.
    Uses Tier 1 keywords only — evaluation signals are hard signals.
    """
    text = _get_all_descriptions(candidate)
    summary = candidate["profile"].get("summary", "").lower()
    combined = text + " " + summary

    matched = sum(1 for kw in EVALUATION_KEYWORDS if kw in combined)
    return min(1.0, matched / 3.0)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 5: EXPERIENCE FIT SCORE
# ─────────────────────────────────────────────────────────────────────────────

def experience_fit_score(candidate: dict) -> float:
    """
    Score based on years of experience vs. JD target range (5-9 years).
    
    Why this matters: JD says 5-9 years is the target. Very junior (<3 years)
    or very senior (15+ years in management) candidates are lower priority.
    
    Score breakdown:
    - 5-9 years → 1.0 (sweet spot)
    - 3-5 years → 0.75 (slightly junior but workable)
    - 9-12 years → 0.85 (slightly senior but fine)
    - 12-15 years → 0.65 (getting senior)
    - 0-3 years → 0.4 (too junior)
    - 15+ years → 0.5 (too senior, likely management)
    """
    yoe = candidate["profile"].get("years_of_experience", 0)

    if 5 <= yoe <= 9:
        return 1.0
    elif 3 <= yoe < 5:
        return 0.75
    elif 9 < yoe <= 12:
        return 0.85
    elif 12 < yoe <= 15:
        return 0.65
    elif yoe < 3:
        return 0.4
    else:  # 15+
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 6: LOCATION SCORE
# ─────────────────────────────────────────────────────────────────────────────

def location_score(candidate: dict) -> float:
    """
    Score based on location fit with JD requirements.
    
    Why this matters: JD says Pune/Noida preferred, India-based required,
    no visa sponsorship. Outside India = case-by-case.
    
    Score breakdown:
    - India, Tier-1 city → 1.0
    - India, any city → 0.85
    - Outside India, willing to relocate → 0.5
    - Outside India, not willing to relocate → 0.2
    """
    country = _normalize(candidate["profile"].get("country", ""))
    location = _normalize(candidate["profile"].get("location", ""))
    willing = candidate["redrob_signals"].get("willing_to_relocate", False)

    if country == "india":
        # Check if in a preferred city
        for city in PREFERRED_LOCATIONS:
            if city in location:
                return 1.0
        return 0.85
    else:
        # Outside India
        if willing:
            return 0.5
        else:
            return 0.2


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 7: SKILL CREDIBILITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def skill_credibility_score(candidate: dict) -> float:
    """
    Score based on verified skill signals.
    
    Why this matters: Self-reported skills are unverified. We trust:
    1. Platform assessment scores (strongest signal — tested by Redrob)
    2. Skills with long duration (hard to fake time)
    3. Skills with decent endorsements
    
    We specifically look for JD-relevant skills only.
    """
    skills = candidate.get("skills", [])
    assessment_scores = candidate["redrob_signals"].get("skill_assessment_scores", {})

    # JD-relevant skill names
    relevant_skills = {
        "embedding", "embeddings", "faiss", "pinecone", "weaviate",
        "qdrant", "milvus", "opensearch", "elasticsearch",
        "sentence transformers", "hugging face transformers",
        "information retrieval", "recommendation systems",
        "machine learning", "nlp", "deep learning", "pytorch",
        "tensorflow", "scikit-learn", "mlflow", "mlops",
        "langchain", "vector search", "bm25", "ranking",
        "feature engineering", "python", "data science"
    }

    score_components = []

    # Component 1: Assessment scores for relevant skills
    # These are the most trustworthy signal
    relevant_assessments = [
        v for k, v in assessment_scores.items()
        if any(rs in k.lower() for rs in relevant_skills)
    ]
    if relevant_assessments:
        avg_assessment = sum(relevant_assessments) / len(relevant_assessments)
        # Normalize from 0-100 to 0-1
        score_components.append(avg_assessment / 100.0)

    # Component 2: Relevant skills with long duration
    # duration_months > 12 means they've actually used it
    long_duration_relevant = [
        s for s in skills
        if any(rs in s["name"].lower() for rs in relevant_skills)
        and s.get("duration_months", 0) > 12
    ]
    if skills:
        duration_ratio = min(1.0, len(long_duration_relevant) / 5.0)
        score_components.append(duration_ratio)

    # Component 3: Endorsements on relevant skills
    relevant_endorsed = [
        s for s in skills
        if any(rs in s["name"].lower() for rs in relevant_skills)
        and s.get("endorsements", 0) > 10
    ]
    if skills:
        endorsed_ratio = min(1.0, len(relevant_endorsed) / 3.0)
        score_components.append(endorsed_ratio)

    if not score_components:
        return 0.1

    return sum(score_components) / len(score_components)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 8: EDUCATION SCORE
# ─────────────────────────────────────────────────────────────────────────────

def education_score(candidate: dict) -> float:
    """
    Score based on education tier.
    
    Why this matters: Minor signal. Tier-1 institutions (IITs, IISc,
    top global universities) suggest strong fundamentals. But this is
    NOT a primary signal — career evidence matters far more.
    
    Score:
    - tier_1 → 1.0
    - tier_2 → 0.75
    - tier_3 → 0.5
    - tier_4 → 0.3
    - unknown → 0.3
    """
    education = candidate.get("education", [])
    if not education:
        return 0.3

    tier_scores = {
        "tier_1": 1.0,
        "tier_2": 0.75,
        "tier_3": 0.5,
        "tier_4": 0.3,
        "unknown": 0.3
    }

    # Take the best education tier across all degrees
    best = 0.3
    for edu in education:
        tier = edu.get("tier", "unknown")
        best = max(best, tier_scores.get(tier, 0.3))

    return best

def skill_description_consistency_score(candidate: dict) -> float:
    """
    Checks if skills claimed actually appear in career descriptions.
    
    Why this matters:
    A keyword stuffer adds FAISS, Pinecone, and vector search to their
    skills list. But if their career descriptions never mention these,
    the claims are unverified. This catches sophisticated stuffers that
    TF-IDF and embeddings might miss.
    
    Method:
    For each high-proficiency relevant skill, check if any form of
    that skill name appears in career descriptions.
    Ratio of verified to claimed = consistency score.
    """
    skills = candidate.get("skills", [])
    descriptions = _get_all_descriptions(candidate)

    # Only check advanced/expert skills — these are the claims worth verifying
    high_claims = [
        s for s in skills
        if s.get("proficiency") in ["advanced", "expert"]
        and s.get("duration_months", 0) > 6
    ]

    if not high_claims:
        return 0.5  # No strong claims = neutral, not penalized

    verified = 0
    for skill in high_claims:
        skill_name = skill["name"].lower()
        # Check if skill name or any part of it appears in descriptions
        parts = skill_name.split()
        if any(part in descriptions for part in parts if len(part) > 3):
            verified += 1

    return verified / len(high_claims)
# ─────────────────────────────────────────────────────────────────────────────
# MASTER FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
def github_activity_score(candidate: dict) -> float:
    """
    Score based on GitHub activity.

    Why this matters:
    GitHub activity is one of the few independently verifiable signals
    in the dataset. A candidate with real commits, PRs, and stars is
    demonstrably building things — not just listing skills.

    Score breakdown:
    - -1 means no GitHub linked → neutral (not penalized, not rewarded)
    - 0-30 → low activity → small penalty
    - 30-70 → moderate activity → moderate score
    - 70+ → high activity → full score

    Why not make this high-weight:
    Junior developers can have high GitHub scores too.
    This is a credibility signal, not a domain signal.
    """
    score = candidate["redrob_signals"].get("github_activity_score", -1)

    if score < 0:
        return 0.4  # No GitHub linked — neutral

    if score >= 70:
        return 1.0
    elif score >= 30:
        return 0.6 + 0.4 * ((score - 30) / 40)
    else:
        return 0.3 + 0.3 * (score / 30)
    
def extract_features(candidate: dict) -> dict:
    return {
        "title_domain":                  title_domain_score(candidate),
        "product_company":               product_company_score(candidate),
        "retrieval_evidence":            retrieval_evidence_score(candidate),
        "evaluation_framework":          evaluation_framework_score(candidate),
        "experience_fit":                experience_fit_score(candidate),
        "location":                      location_score(candidate),
        "skill_credibility":             skill_credibility_score(candidate),
        "education":                     education_score(candidate),
        "skill_description_consistency": skill_description_consistency_score(candidate),
        "company_prestige":              company_prestige_score(candidate),
        "github_activity":               github_activity_score(candidate),
    }