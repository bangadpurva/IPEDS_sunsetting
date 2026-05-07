from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .research_engine import ROOT, load_research_module

GENDER_COLUMNS = {
    "men": "CTOTALM",
    "women": "CTOTALW",
}

RACE_COLUMNS = {
    "American Indian/Alaska Native": "CAIANT",
    "Asian": "CASIAT",
    "Black/African American": "CBKAAT",
    "Hispanic/Latino": "CHISPT",
    "Native Hawaiian/Pacific Islander": "CNHPIT",
    "White": "CWHITT",
    "Two or more races": "C2MORT",
    "Unknown": "CUNKNT",
    "Nonresident alien": "CNRALT",
}

DIRECTORY_CANDIDATES = (
    "hd2024.csv",
    "HD2024.csv",
    "data_uni/hd2024.csv",
    "data_uni/HD2024.csv",
    "data/hd2024.csv",
    "data/HD2024.csv",
)


def _clean_number(value: Any) -> float | int | str | None:
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return round(float(value), 4)
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    return [
        {str(key).lower(): _clean_number(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _directory_lookup() -> pd.DataFrame:
    for candidate in DIRECTORY_CANDIDATES:
        path = ROOT / candidate
        if path.exists():
            hd = pd.read_csv(path, dtype=str, low_memory=False)
            rename = {
                "UNITID": "UNITID",
                "INSTNM": "institution_name",
                "CITY": "city",
                "STABBR": "state",
                "SECTOR": "sector",
                "CONTROL": "control",
                "COUNTYNM": "county",
                "LATITUDE": "latitude",
                "LONGITUD": "longitude",
            }
            keep = [c for c in rename if c in hd.columns]
            if "UNITID" not in keep:
                return pd.DataFrame(columns=["UNITID"])
            out = hd[keep].rename(columns=rename)
            out["UNITID"] = out["UNITID"].astype(str).str.strip()
            return out
    return pd.DataFrame(columns=["UNITID", "institution_name", "city", "state", "sector", "control"])


def _load_a_files() -> pd.DataFrame:
    research = load_research_module()
    frames = []
    for year, path in getattr(research, "FILES_BY_YEAR").items():
        df = pd.read_csv(ROOT / path, low_memory=False)
        df["YEAR"] = year
        df["UNITID"] = df["UNITID"].astype(str).str.strip()
        df["CIP2"] = df["CIPCODE"].astype("string").str.extract(r"^(\d{2})", expand=False)
        df["CIP2_Name"] = df["CIP2"].apply(research.cip2_name)
        df["AWLEVEL_Name"] = df["AWLEVEL"].astype(str).apply(research.awlevel_name)
        if getattr(research, "KEEP_PRIMARY_MAJOR_ONLY", True) and "MAJORNUM" in df.columns:
            df["MAJORNUM"] = pd.to_numeric(df["MAJORNUM"], errors="coerce")
            df = df[df["MAJORNUM"] == getattr(research, "PRIMARY_MAJOR_VALUE", 1)].copy()
        if getattr(research, "REMOVE_AWLEVEL_1_AND_2", True) and "AWLEVEL" in df.columns:
            remove_set = getattr(research, "AWLEVEL_REMOVE_SET", {"01", "1", "02", "2"})
            df = df[~df["AWLEVEL"].astype(str).isin(remove_set)].copy()
        if getattr(research, "EXCLUDE_CIP2_99_FROM_DECLINE_ANALYSIS", True):
            df = df[df["CIP2"] != getattr(research, "CIP2_EXCLUDED_FOR_DECLINE", "99")].copy()
        numeric_cols = ["CTOTALT", *GENDER_COLUMNS.values(), *RACE_COLUMNS.values()]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        keep_cols = [
            "UNITID",
            "CIPCODE",
            "CIP2",
            "CIP2_Name",
            "AWLEVEL",
            "AWLEVEL_Name",
            "YEAR",
            "CTOTALT",
            *GENDER_COLUMNS.values(),
            *RACE_COLUMNS.values(),
        ]
        frames.append(df[[c for c in keep_cols if c in df.columns]])
    return pd.concat(frames, ignore_index=True)


def _institution_trends(a_files: pd.DataFrame, directory: pd.DataFrame) -> pd.DataFrame:
    years = list(range(2019, 2025))
    grouped = (
        a_files.groupby(["UNITID", "CIP2", "CIP2_Name", "AWLEVEL", "AWLEVEL_Name", "YEAR"], as_index=False)
        .agg(completions=("CTOTALT", "sum"))
    )
    wide = (
        grouped.pivot_table(
            index=["UNITID", "CIP2", "CIP2_Name", "AWLEVEL", "AWLEVEL_Name"],
            columns="YEAR",
            values="completions",
            aggfunc="sum",
        )
        .reindex(columns=years)
        .fillna(0)
        .reset_index()
    )
    wide["change_2019_2024"] = wide[2024] - wide[2019]
    wide["pct_change_2019_2024"] = np.where(
        wide[2019] > 0,
        (wide["change_2019_2024"] / wide[2019]) * 100,
        np.nan,
    )
    wide["baseline_avg_2019_2021"] = wide[[2019, 2020, 2021]].mean(axis=1)
    wide["trend_direction"] = np.select(
        [wide["change_2019_2024"] <= -5, wide["change_2019_2024"] >= 5],
        ["Declining", "Increasing"],
        default="Flat/small change",
    )

    if not directory.empty:
        wide = wide.merge(directory, on="UNITID", how="left")

    for year in years:
        wide[str(year)] = wide[year]
        wide = wide.drop(columns=[year])

    ranked = pd.concat(
        [
            wide.sort_values("change_2019_2024").head(150),
            wide.sort_values("change_2019_2024", ascending=False).head(150),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["UNITID", "CIP2", "AWLEVEL"])
    return ranked


def _demographics(a_files: pd.DataFrame) -> dict[str, list[dict]]:
    latest = a_files[a_files["YEAR"] == 2024].copy()
    group_cols = ["CIP2", "CIP2_Name", "AWLEVEL", "AWLEVEL_Name"]

    gender = latest.groupby(group_cols, as_index=False).agg(
        total=("CTOTALT", "sum"),
        men=("CTOTALM", "sum"),
        women=("CTOTALW", "sum"),
    )
    gender["women_share"] = np.where(gender["total"] > 0, gender["women"] / gender["total"], np.nan)
    gender["men_share"] = np.where(gender["total"] > 0, gender["men"] / gender["total"], np.nan)
    gender = gender.sort_values("total", ascending=False).head(250)

    race_aggs = {label: (col, "sum") for label, col in RACE_COLUMNS.items() if col in latest.columns}
    race = latest.groupby(group_cols, as_index=False).agg(total=("CTOTALT", "sum"), **race_aggs)
    race = race.sort_values("total", ascending=False).head(250)
    return {"gender": _records(gender), "race": _records(race)}


def build_dimensions_dataset() -> dict:
    a_files = _load_a_files()
    directory = _directory_lookup()
    institution_trends = _institution_trends(a_files, directory)
    demographics = _demographics(a_files)
    has_geography = "state" in institution_trends.columns and institution_trends["state"].notna().any()

    return {
        "summary": {
            "study_window": "2019-2024",
            "raw_records": int(len(a_files)),
            "institutions": int(a_files["UNITID"].nunique()),
            "program_rows_ranked": int(len(institution_trends)),
            "has_institution_directory": bool(has_geography),
            "geography_note": (
                "Institution names/geography loaded from IPEDS directory file."
                if has_geography
                else "Add hd2024.csv/HD2024.csv to enable institution names, state, city, sector, and geography filters."
            ),
        },
        "institution_trends": _records(institution_trends),
        "demographics": demographics,
        "dictionary": {
            "UNITID": "IPEDS institution identifier. Join to HD directory files for name/geography.",
            "CIPCODE": "Detailed academic program code.",
            "CIP2": "Two-digit CIP field family used in the research pipeline.",
            "AWLEVEL": "IPEDS award level.",
            "CTOTALT": "Total completions.",
            "CTOTALM": "Male completions.",
            "CTOTALW": "Female completions.",
            **{col: f"{label} completions." for label, col in RACE_COLUMNS.items()},
        },
    }


def export_dimensions_json(output_path: Path = ROOT / "web" / "data" / "dimensions.json") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_dimensions_dataset(), indent=2), encoding="utf-8")
    return output_path
