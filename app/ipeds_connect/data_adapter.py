from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .onet import load_onet_cache, onet_soc
from .research_engine import (
    ROOT,
    load_research_module,
    output_freshness,
    research_config,
    resolve_research_outputs,
    run_research_pipeline,
)


def _clean_number(value: Any) -> float | int | None:
    if isinstance(value, list):
        return [_clean_number(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_number(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return round(value, 4)
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    return [
        {str(key).lower(): _clean_number(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _occupation_designations(bls_path: Path) -> dict[str, list[dict]]:
    research = load_research_module()
    bls_raw = pd.read_excel(bls_path, sheet_name="BLS_Projections_Raw")
    if "SOC" not in bls_raw.columns or "SOC_Title" not in bls_raw.columns:
        return {}

    map_df = research.expand_cip2_soc_map(getattr(research, "CIP2_TO_SOC_MAPPING", {}))
    if map_df.empty:
        return {}

    keep_cols = [
        "SOC",
        "SOC_Title",
        "BLS_Projected_Pct_Change",
        "BLS_Annual_Openings",
        "Median_Wage",
        "BLS_Typical_Education",
    ]
    merged = map_df.merge(bls_raw[[c for c in keep_cols if c in bls_raw.columns]], on="SOC", how="left")
    merged = merged.dropna(subset=["SOC_Title"])
    merged = merged[~merged["SOC_Title"].astype(str).str.lower().str.contains("all occupations|occupations$", na=False)]

    onet_occupations = load_onet_cache().get("occupations", {})
    designations: dict[str, list[dict]] = {}
    for cip2, group in merged.groupby("CIP2"):
        sort_col = "BLS_Annual_Openings" if "BLS_Annual_Openings" in group.columns else "BLS_Projected_Pct_Change"
        top = group.sort_values(sort_col, ascending=False).head(8)
        records = _records(
            top.rename(
                columns={
                    "SOC": "soc",
                    "SOC_Title": "title",
                    "BLS_Projected_Pct_Change": "projected_growth",
                    "BLS_Annual_Openings": "annual_openings",
                    "Median_Wage": "median_wage",
                    "BLS_Typical_Education": "typical_education",
                }
            )[["soc", "title", "projected_growth", "annual_openings", "median_wage", "typical_education"]]
        )
        for record in records:
            record.update(onet_occupations.get(onet_soc(record.get("soc", "")), {}))
        designations[str(cip2).zfill(2)] = records
    return designations


def load_student_dataset(
    program_workbook: Path | None = None,
    bls_workbook: Path | None = None,
) -> dict:
    outputs = resolve_research_outputs()
    program_path = program_workbook or outputs.program_workbook
    bls_path = bls_workbook or outputs.bls_workbook

    programs = pd.read_excel(program_path)
    cip_bls = pd.read_excel(bls_path, sheet_name="CIP2_BLS")
    degree_bls = pd.read_excel(bls_path, sheet_name="CIP2_AWLEVEL_BLS")
    mismatches = pd.read_excel(bls_path, sheet_name="Mismatches")
    lag = pd.read_excel(bls_path, sheet_name="Lag_Analysis")

    programs = programs.rename(
        columns={
            "CIP2": "cip2",
            "CIP2_Name": "cip2_name",
            "AWLEVEL": "awlevel",
            "AWLEVEL_Name": "awlevel_name",
            "baseline_avg_2019_2021": "baseline_avg_2019_2021",
            "net_pct_change_2019_2024": "program_net_pct_change",
            "sunset_label": "sunset_label",
        }
    )
    degree_bls = degree_bls.rename(
        columns={
            "CIP2": "cip2",
            "AWLEVEL": "awlevel",
            "BLS_Growth_by_Degree": "bls_growth_by_degree",
            "BLS_Annual_Openings_Mapped": "bls_annual_openings_mapped",
            "Gap_Program_minus_BLS": "gap_program_minus_bls",
            "Cohens_d": "cohens_d",
            "Alignment": "alignment",
        }
    )
    cip_bls = cip_bls.rename(
        columns={
            "CIP2": "cip2",
            "BLS_Occupational_Growth": "bls_occupational_growth",
            "Correlation_Direction": "correlation_direction",
        }
    )

    merged = programs.merge(
        degree_bls[
            [
                "cip2",
                "awlevel",
                "bls_growth_by_degree",
                "bls_annual_openings_mapped",
                "gap_program_minus_bls",
                "cohens_d",
                "alignment",
            ]
        ],
        on=["cip2", "awlevel"],
        how="left",
    ).merge(
        cip_bls[["cip2", "bls_occupational_growth", "correlation_direction"]],
        on="cip2",
        how="left",
    )

    designations = _occupation_designations(bls_path)
    merged["job_designations"] = merged["cip2"].astype(str).str.zfill(2).map(lambda cip2: designations.get(cip2, []))

    for year in range(2019, 2025):
        if year in merged.columns:
            merged[str(year)] = merged[year]
            merged = merged.drop(columns=[year])

    summary = {
        "program_count": int(len(merged)),
        "fields": int(merged["cip2"].nunique()),
        "high_risk_count": int((merged["sunset_label"] == "High Risk").sum()),
        "moderate_count": int((merged["sunset_label"] == "Moderate").sum()),
        "source_program_workbook": str(program_path.relative_to(ROOT)),
        "source_bls_workbook": str(bls_path.relative_to(ROOT)),
        "using_fallback_outputs": outputs.used_fallback,
        "freshness": output_freshness([program_path, bls_path]),
    }

    return {
        "summary": summary,
        "research": research_config(),
        "programs": _records(merged),
        "mismatches": _records(mismatches),
        "lag_analysis": _records(lag),
    }


def export_json(
    output_path: Path = ROOT / "web" / "data" / "programs.json",
    refresh_research: bool = False,
) -> Path:
    if refresh_research:
        run_research_pipeline()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_student_dataset()
    output_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    return output_path
