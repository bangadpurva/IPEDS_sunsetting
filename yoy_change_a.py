import pandas as pd
from pathlib import Path
from typing import Dict, Union

# =========================
# CONFIG
# =========================

FILES_BY_YEAR: Dict[int, Union[str, Path]] = {
    2019: "data_uni/c2019_a.csv",
    2020: "data_uni/c2020_a.csv",
    2021: "data_uni/c2021_a.csv",
    2022: "data_uni/c2022_a.csv",
    2023: "data_uni/c2023_a.csv",
    2024: "data_uni/c2024_a.csv",
}

COL_CIP = "CIPCODE"
COL_AWLEVEL = "AWLEVEL"
COL_TOTAL = "CTOTALT"
COL_MAJORNUM = "MAJORNUM"

KEEP_PRIMARY_MAJOR_ONLY = True
PRIMARY_MAJOR_VALUE = 1

YEARS = list(range(2019, 2025))

OUT_FILE = "data_uni/cip_grouped_awlevel_yoy_students_2019_2024.xlsx"

# =========================
# 1) LOAD & COMBINE
# =========================

frames = []

for year, path in FILES_BY_YEAR.items():
    df = pd.read_csv(path, low_memory=False)

    # Clean
    df[COL_CIP] = df[COL_CIP].astype("string").str.strip()
    df[COL_AWLEVEL] = df[COL_AWLEVEL].astype("string").str.strip()
    df[COL_TOTAL] = pd.to_numeric(df[COL_TOTAL], errors="coerce").fillna(0)

    # Keep primary major only (optional)
    if KEEP_PRIMARY_MAJOR_ONLY and COL_MAJORNUM in df.columns:
        df[COL_MAJORNUM] = pd.to_numeric(df[COL_MAJORNUM], errors="coerce")
        df = df[df[COL_MAJORNUM] == PRIMARY_MAJOR_VALUE]

    # =========================
    # CREATE 2-DIGIT CIP (BROAD FIELD)
    # =========================
    # Works for "01.0999", "01", "52.0201", etc.
    df["CIP2"] = (
        df[COL_CIP]
        .astype("string")
        .str.extract(r"^(\d{2})", expand=False)
    )

    # Drop rows where CIP2 missing (rare but possible)
    df = df[df["CIP2"].notna()]

    df["YEAR"] = year
    frames.append(df[["CIP2", COL_AWLEVEL, COL_TOTAL, "YEAR"]])

combined = pd.concat(frames, ignore_index=True)

# =========================
# 2) AGGREGATE (CIP2 × AWLEVEL × YEAR)
# =========================

agg = (
    combined.groupby(["CIP2", COL_AWLEVEL, "YEAR"], as_index=False)
            .agg(total_students=(COL_TOTAL, "sum"))
)

# Pivot to wide
wide = (
    agg.pivot_table(
        index=["CIP2", COL_AWLEVEL],
        columns="YEAR",
        values="total_students",
        aggfunc="sum"
    )
    .reindex(columns=YEARS)
    .fillna(0)
    .reset_index()
)

# =========================
# 3) YOY COUNTS & YOY %
# =========================

for i in range(1, len(YEARS)):
    y_prev, y_cur = YEARS[i - 1], YEARS[i]
    diff_col = f"{y_cur}-{str(y_prev)[-2:]}"
    pct_col = f"{diff_col}%"

    wide[diff_col] = wide[y_cur] - wide[y_prev]
    wide[pct_col] = (wide[diff_col] / wide[y_prev].replace({0: pd.NA})) * 100

pct_cols = [c for c in wide.columns if isinstance(c, str) and c.endswith("%")]
wide[pct_cols] = wide[pct_cols].round(2)

# =========================
# 4) SAVE OUTPUT
# =========================

wide.to_excel(OUT_FILE, index=False)
print(f"Saved output → {OUT_FILE}")
