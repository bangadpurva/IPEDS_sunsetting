from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .research_engine import ROOT

BASE_URL = "https://api-v2.onetcenter.org"
DEFAULT_CACHE = ROOT / "data_cache" / "onet.json"


def onet_soc(soc: str) -> str:
    value = str(soc).strip()
    return value if not value or "." in value else f"{value}.00"


def _request(path: str, api_key: str) -> Any:
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        headers={"X-API-Key": api_key, "Accept": "application/json", "User-Agent": "IPEDS-Student-Pathways/1.0"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def sync_onet_cache(api_key: str, programs_path: Path, output_path: Path = DEFAULT_CACHE, limit: int = 100) -> Path:
    """Cache O*NET descriptions and skills for occupations already present in the research data."""
    if not api_key:
        raise ValueError("A free O*NET API key is required. Set ONET_API_KEY.")
    programs = json.loads(programs_path.read_text(encoding="utf-8")).get("programs", [])
    socs: list[str] = []
    for program in programs:
        for job in program.get("job_designations", []) or []:
            code = onet_soc(job.get("soc", ""))
            if code and code not in socs:
                socs.append(code)

    occupations: dict[str, Any] = {}
    for code in socs[:limit]:
        try:
            overview = _request(f"/online/occupations/{code}/", api_key)
            skills = _request(f"/online/occupations/{code}/summary/skills", api_key)
            technology = _request(f"/online/occupations/{code}/hot_technology?start=1&end=10", api_key)
        except urllib.error.HTTPError as exc:
            if exc.code in {404, 422}:
                continue
            raise
        occupations[code] = {
            "description": overview.get("description"),
            "bright_outlook": bool(overview.get("bright_outlook")),
            "skills": [item.get("name") for item in skills.get("element", []) if item.get("name")],
            "technology": [item.get("title") for item in technology.get("example", []) if item.get("title")],
        }

    document = {
        "source": BASE_URL,
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "occupations": occupations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return output_path


def load_onet_cache(path: Path = DEFAULT_CACHE) -> dict[str, Any]:
    if not path.exists():
        return {"source": BASE_URL, "retrieved_at": None, "occupations": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache free O*NET occupation and skill data.")
    parser.add_argument("--programs", type=Path, default=ROOT / "web" / "data" / "programs.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    path = sync_onet_cache(os.getenv("ONET_API_KEY", ""), args.programs, args.output, args.limit)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
