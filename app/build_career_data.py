"""Build the compact occupation dataset consumed by the hosted Viascope app."""

from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
crosswalk = pd.read_excel(ROOT / "CIP2020_SOC2018_Crosswalk.xlsx", sheet_name="CIP-SOC")
bls = pd.read_excel(ROOT / "data_uni/bls_correlation_analysis.xlsx", sheet_name="BLS_Projections_Raw")

crosswalk["cip2"] = crosswalk["CIP2020Code"].astype(str).str.split(".").str[0].astype(int)
crosswalk["SOC2018Code"] = crosswalk["SOC2018Code"].astype(str).str.strip()
bls["SOC"] = bls["SOC"].astype(str).str.strip()

merged = crosswalk.merge(bls, left_on="SOC2018Code", right_on="SOC", how="inner")
merged = merged[merged["Occupation_Type"].astype(str).str.lower().ne("summary")]
merged = merged.drop_duplicates(["cip2", "SOC"])

def clean(value):
    if pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value.item() if hasattr(value, "item") else value

records = {}
for cip2, group in merged.groupby("cip2"):
    group = group.assign(
        rank_openings=pd.to_numeric(group["BLS_Annual_Openings"], errors="coerce").fillna(0),
        rank_growth=pd.to_numeric(group["BLS_Projected_Pct_Change"], errors="coerce").fillna(-100),
    ).sort_values(["rank_openings", "rank_growth"], ascending=False).head(12)
    records[str(int(cip2))] = [{
        "soc": clean(row.SOC),
        "title": clean(row.SOC_Title),
        "growth": clean(row.BLS_Projected_Pct_Change),
        "annual_openings": clean(row.BLS_Annual_Openings),
        "median_wage": clean(row.Median_Wage),
        "education": clean(row.BLS_Typical_Education),
        "experience": clean(row.Work_Experience),
        "training": clean(row.On_The_Job_Training),
    } for row in group.itertuples()]

output = ROOT / "site/public/data/occupations.json"
output.write_text(json.dumps({"source": "BLS Employment Projections 2024–2034", "by_cip2": records}, separators=(",", ":")), encoding="utf-8")
print(f"Wrote {sum(map(len, records.values()))} occupation mappings to {output}")
