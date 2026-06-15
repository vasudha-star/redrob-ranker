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
    "zensar", "cyient", "kpit", "happiest minds", "coforge"
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
RETRIEVAL_KEYWORDS = [
    # Vector / embedding systems
    "embedding", "embeddings", "vector search", "dense retrieval",
    "semantic search", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "opensearch", "elasticsearch", "solr",
    "sentence transformer", "sentence-transformer", "bi-encoder",
    "cross-encoder", "ann", "approximate nearest neighbor",
    # Ranking systems
    "ranking", "ranker", "learning to rank", "learning-to-rank",
    "ltr", "xgboost rank", "lightgbm rank", "pointwise", "pairwise",
    "listwise", "ndcg", "mrr", "map@", "mean average precision",
    # Recommendation systems
    "recommendation", "recommender", "collaborative filtering",
    "content-based filtering", "matrix factorization",
    # Search infrastructure
    "retrieval", "information retrieval", "hybrid search",
    "bm25", "tf-idf", "tfidf", "inverted index", "query expansion",
    "reranking", "re-ranking", "query understanding",
    # Evaluation
    "a/b test", "a/b testing", "offline evaluation", "online evaluation",
    "relevance judgment", "click-through", "engagement metric",
    "recall@", "precision@", "offline-online",
    # LLM / modern AI (secondary)
    "rag", "retrieval augmented", "langchain", "llm", "fine-tuning",
    "fine-tune", "lora", "qlora", "peft", "hugging face",
    "transformer", "bert", "gpt"
]

# Keywords for evaluation framework experience specifically.
# The JD calls this out as a hard requirement.
EVALUATION_KEYWORDS = [
    "ndcg", "mrr", "map", "a/b test", "a/b testing",
    "offline evaluation", "online evaluation", "relevance",
    "recall@", "precision@", "click-through rate", "ctr",
    "engagement metric", "offline-online correlation"
]

# India locations preferred by JD.
PREFERRED_LOCATIONS = {
    "hyderabad", "pune", "noida", "bangalore", "bengaluru",
    "mumbai", "delhi", "chennai", "gurgaon", "gurugram",
    "kolkata", "ahmedabad", "india"
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

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

def product_company_score(candidate: dict) -> float:
    """
    Score based on what fraction of career was at product companies.
    
    Why this matters: JD explicitly disqualifies people whose entire
    career is at TCS/Infosys/Wipro etc. Product company experience
    is a strong signal of real engineering ownership.
    
    Score breakdown:
    - 100% product company career → 1.0
    - Mixed career → proportional
    - 100% services career → 0.1 (not 0.0, skills may still exist)
    """
    history = candidate.get("career_history", [])
    if not history:
        return 0.3

    total_months = 0
    product_months = 0

    for job in history:
        duration = job.get("duration_months", 0)
        company = job.get("company", "")
        total_months += duration
        if not _is_services_company(company):
            product_months += duration

    if total_months == 0:
        return 0.3

    ratio = product_months / total_months

    # Apply a floor so pure-services candidates aren't completely zeroed
    return max(0.1, ratio)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 3: RETRIEVAL EVIDENCE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_evidence_score(candidate: dict) -> float:
    """
    Score based on retrieval/ranking/search keywords in career descriptions.
    
    Why this matters: This is the CORE JD requirement. We look at career
    descriptions (not skills list) because descriptions are harder to fake.
    A keyword stuffer can add FAISS to their skills, but if their
    descriptions talk about customer support management, the score stays low.
    
    Why descriptions and not skills:
    Skills are self-reported. Descriptions describe what you actually did.
    
    Scoring:
    - Count unique retrieval keyword matches in weighted descriptions
    - Normalize to 0.0-1.0 by dividing by a reasonable max (15 keywords)
    - Cap at 1.0
    """
    text = _get_all_descriptions(candidate)

    matched = set()
    for kw in RETRIEVAL_KEYWORDS:
        if kw in text:
            matched.add(kw)

    # Also check summary — gives some weight to candidate's own framing
    summary = candidate["profile"].get("summary", "").lower()
    for kw in RETRIEVAL_KEYWORDS:
        if kw in summary:
            matched.add(kw)

    # Normalize: 15+ keyword matches → score of 1.0
    # This threshold means a candidate needs substantial retrieval
    # language in their history to score highly
    score = min(1.0, len(matched) / 15.0)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 4: EVALUATION FRAMEWORK SCORE
# ─────────────────────────────────────────────────────────────────────────────

def evaluation_framework_score(candidate: dict) -> float:
    """
    Score based on evaluation framework keywords in career descriptions.
    
    Why this matters: JD says 'if you've never thought about how to evaluate
    a ranking system rigorously, this role will be very painful.'
    Evaluation experience is a HARD requirement, so we score it separately.
    
    Scoring:
    - 3+ evaluation keywords → 1.0
    - 1-2 → partial credit
    - 0 → 0.0
    """
    text = _get_all_descriptions(candidate)
    summary = candidate["profile"].get("summary", "").lower()
    combined = text + " " + summary

    matched = set()
    for kw in EVALUATION_KEYWORDS:
        if kw in combined:
            matched.add(kw)

    # 3+ matches → full score
    return min(1.0, len(matched) / 3.0)


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


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(candidate: dict) -> dict:
    """
    Run all feature functions on a candidate and return a dictionary
    of feature name → score.
    
    This is the single function called by the scorer.
    """
    return {
        "title_domain":         title_domain_score(candidate),
        "product_company":      product_company_score(candidate),
        "retrieval_evidence":   retrieval_evidence_score(candidate),
        "evaluation_framework": evaluation_framework_score(candidate),
        "experience_fit":       experience_fit_score(candidate),
        "location":             location_score(candidate),
        "skill_credibility":    skill_credibility_score(candidate),
        "education":            education_score(candidate),
    }