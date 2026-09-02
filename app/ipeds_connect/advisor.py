from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StudentProfile:
    interests: tuple[str, ...]
    degree_level: str | None = None
    risk_tolerance: str = "balanced"
    career_priority: str = "balanced"
    location: str | None = None
    max_annual_cost: int | None = None


INTEREST_TO_CIP_KEYWORDS = {
    "technology": ("Computer", "Engineering", "Mathematics"),
    "healthcare": ("Health", "Biological", "Psychology"),
    "business": ("Business", "Management", "Marketing"),
    "education": ("Education",),
    "public-service": ("Public Administration", "Social Service", "Homeland"),
    "creative": ("Visual", "Performing", "Communication", "Journalism"),
    "science": ("Physical Sciences", "Natural Resources", "Biological"),
    "trades": ("Construction", "Mechanic", "Precision", "Transportation"),
}

PROMPT_ALIASES = {
    "data science": ("technology", "science"),
    "python": ("technology", "science"),
    "sql": ("technology", "business"),
    "statistics": ("technology", "science"),
    "excel": ("business", "technology"),
    "analytics": ("technology", "business"),
    "dashboard": ("technology", "business"),
    "machine learning": ("technology", "science"),
    "artificial intelligence": ("technology", "science"),
    "ai": ("technology", "science"),
    "software": ("technology",),
    "developer": ("technology",),
    "programming": ("technology",),
    "cybersecurity": ("technology",),
    "security analyst": ("technology", "public-service"),
    "nursing": ("healthcare",),
    "medicine": ("healthcare",),
    "patient": ("healthcare",),
    "clinical": ("healthcare",),
    "public health": ("healthcare", "public-service"),
    "finance": ("business",),
    "accounting": ("business",),
    "management": ("business",),
    "marketing": ("business", "creative"),
    "teacher": ("education",),
    "teaching": ("education",),
    "policy": ("public-service",),
    "social work": ("public-service",),
    "design": ("creative",),
    "media": ("creative",),
    "construction": ("trades",),
    "electrician": ("trades",),
    "mechanic": ("trades",),
}

DEGREE_ALIASES = {
    "associate": "associate",
    "bachelor": "bachelor",
    "undergraduate": "bachelor",
    "master": "master",
    "graduate": "master",
    "doctoral": "doctoral",
    "phd": "doctoral",
    "certificate": "certificate",
}

PROMPT_TO_CIP_BOOST = {
    "data science": ("Computer", "Mathematics", "Statistics"),
    "python": ("Computer", "Mathematics", "Statistics"),
    "sql": ("Computer", "Information Sciences"),
    "machine learning": ("Computer", "Mathematics", "Engineering"),
    "artificial intelligence": ("Computer", "Engineering", "Mathematics"),
    "ai": ("Computer", "Engineering", "Mathematics"),
    "software": ("Computer",),
    "cybersecurity": ("Computer", "Homeland"),
    "nursing": ("Health",),
    "public health": ("Health", "Public Administration"),
    "finance": ("Business",),
    "accounting": ("Business",),
}

JOB_INTEREST_KEYWORDS = {
    "technology": ("computer", "software", "data", "database", "cyber", "information systems", "web developer"),
    "healthcare": ("health", "medical", "nurs", "clinical", "physician", "therap", "dental", "pharmac"),
    "business": ("business", "management", "manager", "account", "financial", "marketing", "sales", "analyst"),
    "education": ("teacher", "education", "instruction", "school", "postsecondary"),
    "public-service": ("public", "social", "community", "law enforcement", "emergency", "government"),
    "creative": ("artist", "design", "media", "writer", "editor", "music", "film", "communication"),
    "science": ("scientist", "science", "research", "statistic", "biological", "chemist", "physic"),
    "trades": ("construction", "mechanic", "repair", "electric", "machin", "transport", "installer"),
}


def degree_matches(program: dict, degree_level: str | None) -> bool:
    """Treat an explicitly requested credential as a constraint, not a score hint."""
    if not degree_level:
        return True
    award = str(program.get("awlevel_name", "")).lower()
    aliases = {
        "associate": ("associate",),
        "bachelor": ("bachelor's degree", "bachelors degree"),
        "master": ("master",),
        "doctoral": ("doctor", "professional degree"),
        "certificate": ("certificate", "award <"),
    }
    return any(token in award for token in aliases.get(degree_level.lower(), (degree_level.lower(),)))


def _text_match_score(program: dict, interests: Iterable[str]) -> float:
    name = str(program.get("cip2_name", "")).lower()
    score = 0.0
    for interest in interests:
        for keyword in INTEREST_TO_CIP_KEYWORDS.get(interest, (interest,)):
            if keyword.lower() in name:
                score += 1.0
    return score


def _level_score(program: dict, degree_level: str | None) -> float:
    if not degree_level:
        return 0.0
    return 1.0 if degree_level.lower() in str(program.get("awlevel_name", "")).lower() else 0.0


def score_program(program: dict, profile: StudentProfile) -> float:
    """Rank programs for exploration without hiding the research caveats."""
    risk_label = str(program.get("sunset_label", ""))
    alignment = str(program.get("alignment", ""))
    net_change = float(program.get("program_net_pct_change") or 0)
    bls_growth = float(program.get("bls_growth_by_degree") or program.get("bls_occupational_growth") or 0)
    openings = float(program.get("bls_annual_openings_mapped") or 0)

    score = 45.0
    score += _text_match_score(program, profile.interests) * 18.0
    score += _level_score(program, profile.degree_level) * 10.0
    score += min(max(bls_growth, -20.0), 40.0) * 0.7
    score += min(openings / 100000.0, 8.0)

    if risk_label == "High Risk":
        score -= 24.0 if profile.risk_tolerance != "adventurous" else 10.0
    elif risk_label == "Moderate":
        score -= 10.0 if profile.risk_tolerance == "cautious" else 5.0
    elif risk_label == "Growth/Stable":
        score += 8.0

    if alignment == "Misaligned":
        score -= 12.0
    elif alignment in {"Strong", "Moderate"}:
        score += 8.0

    if profile.career_priority == "demand":
        score += min(max(bls_growth, -10.0), 30.0) * 0.6
    elif profile.career_priority == "stability":
        score += min(max(net_change, -20.0), 20.0) * 0.4

    return round(max(0.0, min(100.0, score)), 1)


def advisor_reason(program: dict) -> str:
    risk = program.get("sunset_label") or "Unlabeled"
    alignment = program.get("alignment") or "Insufficient alignment data"
    net = program.get("program_net_pct_change")
    bls = program.get("bls_growth_by_degree") or program.get("bls_occupational_growth")

    parts = [f"{risk} program trend", f"{alignment} labor-market alignment"]
    if net is not None:
        parts.append(f"{float(net):+.1f}% completions change")
    if bls is not None:
        parts.append(f"{float(bls):+.1f}% projected BLS growth")
    return "; ".join(parts) + "."


def _job_relevance_score(job: dict, prompt: str) -> float:
    title = str(job.get("title", "")).lower()
    prompt_lower = prompt.lower()
    score = 0.0
    for token in ("data", "science", "scientist", "analyst", "analytics", "statistic", "machine learning", "database", "computer"):
        if token in prompt_lower and token in title:
            score += 18.0
    if "data science" in prompt_lower and "data scientist" in title:
        score += 45.0
    if "analyst" in prompt_lower and "analyst" in title:
        score += 30.0
    growth = job.get("projected_growth")
    openings = job.get("annual_openings")
    if isinstance(growth, (int, float)):
        score += min(float(growth), 40.0)
    if isinstance(openings, (int, float)):
        score += min(float(openings) / 10000.0, 20.0)
    return score


def recommend(programs: list[dict], profile: StudentProfile, limit: int = 8) -> list[dict]:
    ranked = []
    for program in programs:
        if not degree_matches(program, profile.degree_level):
            continue
        item = dict(program)
        item["advisor_score"] = score_program(item, profile)
        item["advisor_reason"] = advisor_reason(item)
        ranked.append(item)
    ranked.sort(key=lambda row: row["advisor_score"], reverse=True)
    return ranked[:limit]


def profile_from_prompt(prompt: str) -> tuple[StudentProfile, list[str]]:
    text = prompt.lower()
    interests: list[str] = []
    reasons: list[str] = []

    for phrase, mapped in PROMPT_ALIASES.items():
        if phrase in text:
            for interest in mapped:
                if interest not in interests:
                    interests.append(interest)
            reasons.append(f"Mapped '{phrase}' to {', '.join(mapped)}.")

    for interest in INTEREST_TO_CIP_KEYWORDS:
        if interest in text and interest not in interests:
            interests.append(interest)
            reasons.append(f"Used explicit interest '{interest}'.")

    degree = None
    for phrase, mapped_degree in DEGREE_ALIASES.items():
        if phrase in text:
            degree = mapped_degree
            reasons.append(f"Detected degree target '{mapped_degree}'.")
            break

    risk = "balanced"
    if any(word in text for word in ("stable", "safe", "low risk", "secure")):
        risk = "cautious"
        reasons.append("Prioritized lower-risk program trends.")
    elif any(word in text for word in ("emerging", "new", "experimental", "pivot")):
        risk = "adventurous"
        reasons.append("Allowed emerging or less stable pathways.")

    priority = "balanced"
    if any(word in text for word in ("job", "jobs", "demand", "salary", "career", "employment")):
        priority = "demand"
        reasons.append("Prioritized labor-market demand.")
    elif any(word in text for word in ("stable program", "program stability", "not declining")):
        priority = "stability"
        reasons.append("Prioritized completion stability.")

    if not interests:
        interests = ["technology", "business", "healthcare"]
        reasons.append("No specific field detected, so started with broad high-demand areas.")

    location = None
    location_match = re.search(
        r"(?:near|around)\s+([A-Za-z][A-Za-z .'-]+?)(?:[,.]|\s+(?:with|for|and|that|where|under|max|budget)\b|$)",
        prompt,
        flags=re.IGNORECASE,
    )
    if location_match:
        location = location_match.group(1).strip()
        reasons.append(f"Detected location preference '{location}'.")

    max_annual_cost = None
    budget_match = re.search(r"(?:under|max(?:imum)?|budget(?: of)?)\s*\$?([\d,]+)", text)
    if budget_match:
        max_annual_cost = int(budget_match.group(1).replace(",", ""))
        reasons.append(f"Detected annual cost ceiling ${max_annual_cost:,}.")

    return StudentProfile(tuple(interests), degree, risk, priority, location, max_annual_cost), reasons


def _intent_from_prompt(prompt: str) -> str:
    text = prompt.lower()
    if any(phrase in text for phrase in ("list the jobs", "what jobs", "job titles", "job designations", "can be done")):
        return "job-list"
    if any(word in text for word in ("skill", "skills", "good at", "experience with", "i know")):
        return "skills-to-degrees"
    if any(word in text for word in ("job", "career", "role", "occupation", "in demand", "salary")):
        return "job-to-degree"
    if any(word in text for word in ("choose", "compare", "path", "degree", "institution", "college", "university")):
        return "path-analysis"
    return "general-coaching"


def agentic_recommend(programs: list[dict], prompt: str, limit: int = 8) -> dict:
    profile, reasons = profile_from_prompt(prompt)
    intent = _intent_from_prompt(prompt)
    intent_copy = {
        "job-list": "I treated this as a job-designation question and looked for BLS occupations linked to relevant degree fields.",
        "skills-to-degrees": "I treated this as a skills-to-degree question and looked for related academic fields.",
        "job-to-degree": "I treated this as a job-to-degree question and emphasized labor-market demand signals.",
        "path-analysis": "I treated this as a path comparison question and balanced demand, trend stability, and alignment.",
        "general-coaching": "I treated this as an open advising question and started with the closest grounded pathways.",
    }
    boost_terms: list[str] = []
    prompt_lower = prompt.lower()
    for phrase, terms in PROMPT_TO_CIP_BOOST.items():
        if phrase in prompt_lower:
            boost_terms.extend(terms)

    ranked = []
    for program in programs:
        if not degree_matches(program, profile.degree_level):
            continue
        item = dict(program)
        score = score_program(item, profile)
        name = str(item.get("cip2_name", ""))
        prompt_match = bool(boost_terms and any(term.lower() in name.lower() for term in boost_terms))
        if prompt_match:
            score = min(100.0, score + 16.0)
        item["advisor_score"] = round(score, 1)
        item["advisor_reason"] = advisor_reason(item)
        item["prompt_match"] = prompt_match
        ranked.append(item)
    ranked.sort(key=lambda row: (row["prompt_match"], row["advisor_score"]), reverse=True)
    selected = ranked[:limit]

    job_candidates: list[dict] = []
    seen_jobs: set[str] = set()
    explicit_interests = set(profile.interests)
    for program in selected:
        for job in program.get("job_designations", []) or []:
            title = str(job.get("title", "")).strip()
            title_lower = title.lower()
            recognized_domains = {
                domain for domain, keywords in JOB_INTEREST_KEYWORDS.items() if any(keyword in title_lower for keyword in keywords)
            }
            if explicit_interests and recognized_domains and not (recognized_domains & explicit_interests):
                continue
            if title and title not in seen_jobs:
                seen_jobs.add(title)
                job_candidates.append(
                    {
                        **job,
                        "related_field": program.get("cip2_name"),
                        "related_award": program.get("awlevel_name"),
                        "relevance_score": _job_relevance_score(job, prompt),
                    }
                )
    job_candidates.sort(
        key=lambda job: (
            job.get("relevance_score") or 0,
            job.get("projected_growth") or 0,
            job.get("annual_openings") or 0,
        ),
        reverse=True,
    )
    job_designations = job_candidates[:12]

    return {
        "intent": intent,
        "profile": {
            "interests": profile.interests,
            "degree_level": profile.degree_level,
            "risk_tolerance": profile.risk_tolerance,
            "career_priority": profile.career_priority,
            "location": profile.location,
            "max_annual_cost": profile.max_annual_cost,
        },
        "reasoning": [intent_copy[intent], *reasons],
        "recommendations": selected,
        "job_designations": job_designations,
    }
