from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .research_engine import ROOT

BASE_URL = "https://api.data.gov/ed/collegescorecard/v1/schools"
DEFAULT_CACHE = ROOT / "data_cache" / "college_scorecard.json"
FIELDS = (
    "id",
    "school.name",
    "school.city",
    "school.state",
    "latest.student.size",
    "latest.admissions.admission_rate.overall",
    "latest.cost.avg_net_price.overall",
    "latest.cost.tuition.in_state",
    "latest.cost.tuition.out_of_state",
    "latest.completion.rate_suppressed.overall",
    "latest.earnings.10_yrs_after_entry.median",
    "latest.aid.median_debt.completers.overall",
)


def _get(row: dict[str, Any], key: str) -> Any:
    return row.get(key)


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "unitid": str(_get(row, "id")),
        "scorecard_name": _get(row, "school.name"),
        "scorecard_city": _get(row, "school.city"),
        "scorecard_state": _get(row, "school.state"),
        "student_size": _get(row, "latest.student.size"),
        "admission_rate": _get(row, "latest.admissions.admission_rate.overall"),
        "average_net_price": _get(row, "latest.cost.avg_net_price.overall"),
        "tuition_in_state": _get(row, "latest.cost.tuition.in_state"),
        "tuition_out_of_state": _get(row, "latest.cost.tuition.out_of_state"),
        "completion_rate": _get(row, "latest.completion.rate_suppressed.overall"),
        "median_earnings_10yr": _get(row, "latest.earnings.10_yrs_after_entry.median"),
        "median_debt_completers": _get(row, "latest.aid.median_debt.completers.overall"),
    }


def sync_scorecard_cache(api_key: str, output_path: Path = DEFAULT_CACHE) -> Path:
    """Download institution-level Scorecard data once for fast, repeatable app builds."""
    if not api_key:
        raise ValueError("A free College Scorecard API key is required. Set COLLEGE_SCORECARD_API_KEY.")

    rows: list[dict[str, Any]] = []
    page = 0
    total = None
    while total is None or len(rows) < total:
        query = urllib.parse.urlencode(
            {"api_key": api_key, "fields": ",".join(FIELDS), "per_page": 100, "page": page}
        )
        request = urllib.request.Request(f"{BASE_URL}?{query}", headers={"User-Agent": "IPEDS-Student-Pathways/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
        total = int(payload.get("metadata", {}).get("total", 0))
        page_rows = payload.get("results", [])
        if not page_rows:
            break
        rows.extend(_normalize(row) for row in page_rows)
        page += 1

    document = {
        "source": BASE_URL,
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "institutions": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return output_path


def load_scorecard_cache(path: Path = DEFAULT_CACHE) -> dict[str, Any]:
    if not path.exists():
        return {"source": BASE_URL, "retrieved_at": None, "institutions": []}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache free College Scorecard institution data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    path = sync_scorecard_cache(os.getenv("COLLEGE_SCORECARD_API_KEY", ""), args.output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
