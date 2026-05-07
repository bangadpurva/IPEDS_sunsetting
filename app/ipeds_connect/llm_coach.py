from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .advisor import agentic_recommend


def _llm_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL"))


def _chat_completion(messages: list[dict[str, str]]) -> str:
    if os.getenv("OLLAMA_BASE_URL"):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        payload = {"model": model, "messages": messages, "stream": False}
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 700,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _program_context(recommendations: list[dict[str, Any]]) -> str:
    rows = []
    for row in recommendations[:6]:
        rows.append(
            {
                "field": row.get("cip2_name"),
                "award": row.get("awlevel_name"),
                "score": row.get("advisor_score"),
                "trend": row.get("program_net_pct_change"),
                "sunset": row.get("sunset_label"),
                "alignment": row.get("alignment"),
                "bls_growth": row.get("bls_growth_by_degree") or row.get("bls_occupational_growth"),
            }
        )
    return json.dumps(rows, indent=2)


def _job_context(jobs: list[dict[str, Any]]) -> str:
    rows = []
    for job in jobs[:10]:
        rows.append(
            {
                "title": job.get("title"),
                "related_field": job.get("related_field"),
                "projected_growth": job.get("projected_growth"),
                "annual_openings": job.get("annual_openings"),
                "median_wage": job.get("median_wage"),
                "typical_education": job.get("typical_education"),
            }
        )
    return json.dumps(rows, indent=2)


def coach_with_optional_llm(programs: list[dict], prompt: str) -> dict:
    base = agentic_recommend(programs, prompt)
    base["mode"] = "rules"
    base["llm_available"] = _llm_configured()

    if not _llm_configured():
        base["coach_answer"] = (
            "LLM is not configured, so I used the local rules-based coach grounded in the IPEDS/BLS dataset."
        )
        return base

    messages = [
        {
            "role": "system",
            "content": (
                "You are an education and career coach. Use only the provided IPEDS/BLS recommendation "
                "context for specific degree/program claims. If the student asks about current job-market "
                "facts beyond the data, say that it should be verified with live labor-market sources. "
                "Be concise, practical, and student-friendly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Student prompt: {prompt}\n\n"
                f"Interpreted profile: {json.dumps(base['profile'])}\n\n"
                f"Recommendation context:\n{_program_context(base['recommendations'])}\n\n"
                f"Job designation context:\n{_job_context(base.get('job_designations', []))}\n\n"
                "If the prompt asks for jobs, list relevant job designations first. Otherwise give: "
                "1) best-fit direction, 2) why, 3) what to compare next."
            ),
        },
    ]

    try:
        base["coach_answer"] = _chat_completion(messages).strip()
        base["mode"] = "llm"
    except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as exc:
        base["coach_answer"] = (
            "LLM call failed, so I used the local rules-based coach. "
            f"Reason: {exc}"
        )
        base["mode"] = "rules"
    return base
