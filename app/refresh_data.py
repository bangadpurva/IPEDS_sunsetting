"""Validate a new annual IPEDS release before rebuilding Viascope data."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {"UNITID", "CIPCODE", "AWLEVEL", "CTOTALT"}


def discover_releases(data_dir: Path) -> dict[int, Path]:
    releases: dict[int, Path] = {}
    for path in data_dir.glob("c????_a.csv"):
        match = re.fullmatch(r"c(\d{4})_a\.csv", path.name, re.IGNORECASE)
        if match:
            releases[int(match.group(1))] = path
    if len(releases) < 3:
        raise ValueError("At least three annual IPEDS completion files are required.")
    years = sorted(releases)
    missing = sorted(set(range(years[0], years[-1] + 1)) - set(years))
    if missing:
        raise ValueError(f"Missing annual IPEDS files: {missing}")
    return {year: releases[year] for year in years}


def validate_headers(releases: dict[int, Path]) -> None:
    import pandas as pd

    for year, path in releases.items():
        columns = set(pd.read_csv(path, nrows=0).columns)
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"{path.name} ({year}) is missing columns: {sorted(missing)}")


def status(data_dir: Path = ROOT / "data_uni") -> dict:
    releases = discover_releases(data_dir)
    validate_headers(releases)
    years = sorted(releases)
    directory = next(iter(sorted(data_dir.glob("[hH][dD]????.csv"), reverse=True)), None)
    return {"ready": True, "study_window": f"{years[0]}–{years[-1]}", "years": years,
            "latest_completions_file": releases[years[-1]].name,
            "latest_directory_file": directory.name if directory else None,
            "next_expected_year": years[-1] + 1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Viascope annual IPEDS inputs.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_uni")
    args = parser.parse_args()
    print(json.dumps(status(args.data_dir), indent=2))


if __name__ == "__main__":
    main()
