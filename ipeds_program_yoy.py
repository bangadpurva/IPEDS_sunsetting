"""
IPEDS Completions Trend Builder (CIPCODE x AWLEVEL)
---------------------------------------------------
Goal:
  For each (CIPCODE, AWLEVEL):
    - Total completions per year (CTOTALT)
    - YoY change (absolute + percent)
    - Institutions producing >=1 award (unique UNITID with CTOTALT>0)
    - Simple “declining/rising” flags

Works with:
  A) Separate files per year (CSV or Excel)
  B) One combined file with a YEAR column

Edit ONLY the CONFIG section to match your files.
"""
#%%
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Optional

# =========================
# CONFIG (EDIT THIS)
# =========================
MODE = "separate_files"   # "separate_files" OR "single_file"

# --- If MODE = "separate_files" ---
FILES_BY_YEAR: Dict[int, Union[str, Path]] = {
    2022: "data/c2022_a.csv",  # can be .csv or .xlsx
    2023: "data/c2023_a.csv",
    2024: "data/c2024_a.csv",
}

# If your Excel file has a specific sheet name, set it; else None (uses first sheet)
EXCEL_SHEET_NAME: Optional[str] = None

# --- If MODE = "single_file" ---
SINGLE_FILE_PATH = "data/completions_all_years.csv"  # can be .csv or .xlsx
YEAR_COLUMN_NAME = "YEAR"  # adjust if your combined file uses a different name

# Common options
KEEP_PRIMARY_MAJOR_ONLY = True   # MAJORNUM == 1 (recommended to avoid double counting)
PRIMARY_MAJOR_VALUE = "1"        # some files store 1 as int; we normalize to string anyway

# Column names from IPEDS (your headers match these)
COL_UNITID = "UNITID"
COL_CIP = "CIPCODE"
COL_AWLEVEL = "AWLEVEL"
COL_MAJORNUM = "MAJORNUM"
COL_TOTAL = "CTOTALT"  # total completions (all students, total awards)

# Outputs
OUT_LONG = "program_awlevel_long.csv"
OUT_WIDE = "program_awlevel_wide.csv"
OUT_LATEST_DYING = "program_awlevel_latest_declining.csv"
OUT_LATEST_RISING = "program_awlevel_latest_rising.csv"

#%%
# =========================
# HELPERS
# =========================
def read_any(path: Union[str, Path], excel_sheet: Optional[str] = None) -> pd.DataFrame:
    """Read CSV or Excel into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=excel_sheet)
    elif path.suffix.lower() in [".csv", ".txt"]:
        # dtype later; read flexibly
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure key identifier columns are treated consistently."""
    for c in [COL_UNITID, COL_CIP, COL_AWLEVEL, COL_MAJORNUM]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    if COL_TOTAL in df.columns:
        df[COL_TOTAL] = pd.to_numeric(df[COL_TOTAL], errors="coerce").fillna(0)
    return df

def enforce_primary_major(df: pd.DataFrame) -> pd.DataFrame:
    """Optional: Keep only MAJORNUM == 1 if present."""
    if KEEP_PRIMARY_MAJOR_ONLY and (COL_MAJORNUM in df.columns):
        return df[df[COL_MAJORNUM].fillna("").astype("string").str.strip() == PRIMARY_MAJOR_VALUE]
    return df

def required_cols_present(df: pd.DataFrame, required) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")

#%%
# =========================
# LOAD DATA
# =========================
dfs = []

if MODE == "separate_files":
    for year, fp in FILES_BY_YEAR.items():
        df = read_any(fp, excel_sheet=EXCEL_SHEET_NAME)
        df = standardize_types(df)
        df = enforce_primary_major(df)

        required_cols_present(df, [COL_UNITID, COL_CIP, COL_AWLEVEL, COL_TOTAL])

        df["YEAR"] = int(year)
        dfs.append(df[["YEAR", COL_UNITID, COL_CIP, COL_AWLEVEL, COL_TOTAL]])

    all_years = pd.concat(dfs, ignore_index=True)

elif MODE == "single_file":
    df = read_any(SINGLE_FILE_PATH, excel_sheet=EXCEL_SHEET_NAME)
    df = standardize_types(df)
    df = enforce_primary_major(df)

    required_cols_present(df, [YEAR_COLUMN_NAME, COL_UNITID, COL_CIP, COL_AWLEVEL, COL_TOTAL])

    df["YEAR"] = pd.to_numeric(df[YEAR_COLUMN_NAME], errors="coerce")
    df = df.dropna(subset=["YEAR"])
    df["YEAR"] = df["YEAR"].astype(int)

    all_years = df[["YEAR", COL_UNITID, COL_CIP, COL_AWLEVEL, COL_TOTAL]].copy()

else:
    raise ValueError("MODE must be 'separate_files' or 'single_file'")

# Basic cleaning
all_years[COL_CIP] = all_years[COL_CIP].astype("string").str.strip()
all_years[COL_AWLEVEL] = all_years[COL_AWLEVEL].astype("string").str.strip()
all_years[COL_UNITID] = all_years[COL_UNITID].astype("string").str.strip()
all_years[COL_TOTAL] = pd.to_numeric(all_years[COL_TOTAL], errors="coerce").fillna(0)

#%%
# =========================
# AGGREGATE: (YEAR, CIPCODE, AWLEVEL)
# =========================
# Total completions
# Institutions_with_awards: count unique UNITID where CTOTALT > 0
agg = (
    all_years
    .groupby(["YEAR", COL_CIP, COL_AWLEVEL], as_index=False)
    .agg(
        total_completed=(COL_TOTAL, "sum"),
        institutions_with_awards=(COL_UNITID, lambda s: s[all_years.loc[s.index, COL_TOTAL] > 0].nunique())
    )
)

#%%
# =========================
# YOY CHANGE
# =========================
agg = agg.sort_values([COL_CIP, COL_AWLEVEL, "YEAR"]).reset_index(drop=True)

agg["yoy_change"] = agg.groupby([COL_CIP, COL_AWLEVEL])["total_completed"].diff()
agg["prev_year_total"] = agg.groupby([COL_CIP, COL_AWLEVEL])["total_completed"].shift(1)

# Percent change: avoid divide-by-zero
agg["yoy_pct"] = (agg["yoy_change"] / agg["prev_year_total"].replace({0: pd.NA})) * 100

# =========================
# "RISING / DECLINING" HEURISTICS (optional, meeting-friendly)
# =========================
# 1) big move flags (threshold adjustable)
BIG_DROP_PCT = -25
BIG_RISE_PCT = 25

agg["big_drop"] = agg["yoy_pct"].apply(lambda x: pd.notna(x) and x <= BIG_DROP_PCT)
agg["big_rise"] = agg["yoy_pct"].apply(lambda x: pd.notna(x) and x >= BIG_RISE_PCT)

# 2) consecutive decline/rise flags (2-year streak)
def streak_flag(series: pd.Series, direction: str, window: int = 2) -> pd.Series:
    diffs = series.diff()
    if direction == "down":
        hits = (diffs < 0).astype(int)
    elif direction == "up":
        hits = (diffs > 0).astype(int)
    else:
        raise ValueError("direction must be 'down' or 'up'")
    return hits.rolling(window).sum() == window

agg["declining_2y"] = (
    agg.groupby([COL_CIP, COL_AWLEVEL])["total_completed"]
       .apply(lambda s: streak_flag(s, "down", window=2))
       .reset_index(level=[0, 1], drop=True)
)

agg["rising_2y"] = (
    agg.groupby([COL_CIP, COL_AWLEVEL])["total_completed"]
       .apply(lambda s: streak_flag(s, "up", window=2))
       .reset_index(level=[0, 1], drop=True)
)

#%%
# =========================
# WIDE TABLE (easy to show in meeting / export to Excel)
# =========================
years_sorted = sorted(agg["YEAR"].unique())

wide = agg.pivot_table(
    index=[COL_CIP, COL_AWLEVEL],
    columns="YEAR",
    values="total_completed",
    aggfunc="first"
).reset_index()

# Add YoY columns for each adjacent year pair
for i in range(1, len(years_sorted)):
    y_prev, y_cur = years_sorted[i-1], years_sorted[i]
    wide[f"yoy_change_{y_cur}_{y_prev}"] = wide.get(y_cur, 0) - wide.get(y_prev, 0)
    wide[f"yoy_pct_{y_cur}_{y_prev}"] = (
        wide[f"yoy_change_{y_cur}_{y_prev}"] / wide.get(y_prev, pd.Series([pd.NA]*len(wide))).replace({0: pd.NA})
    ) * 100

# =========================
# LATEST YEAR: top decliners / risers
# =========================
latest_year = max(years_sorted)

latest = agg[agg["YEAR"] == latest_year].copy()

# Define "declining" vs "rising" as:
# - big drop/rise OR 2-year streak
declining_latest = latest[latest["big_drop"] | latest["declining_2y"]].copy()
rising_latest = latest[latest["big_rise"] | latest["rising_2y"]].copy()

# Sort by severity (most negative first / most positive first)
declining_latest = declining_latest.sort_values(["yoy_pct", "yoy_change"], ascending=[True, True])
rising_latest = rising_latest.sort_values(["yoy_pct", "yoy_change"], ascending=[False, False])

# =========================
# SAVE OUTPUTS
# =========================
agg.to_csv(OUT_LONG, index=False)
wide.to_csv(OUT_WIDE, index=False)
declining_latest.to_csv(OUT_LATEST_DYING, index=False)
rising_latest.to_csv(OUT_LATEST_RISING, index=False)

# =========================
# PRINT QUICK SUMMARY
# =========================
print("\n=== DONE ===")
print(f"Years found: {years_sorted}")
print(f"Saved long table:  {OUT_LONG}")
print(f"Saved wide table:  {OUT_WIDE}")
print(f"Saved decliners:   {OUT_LATEST_DYING} (latest year={latest_year})")
print(f"Saved risers:      {OUT_LATEST_RISING} (latest year={latest_year})")

print("\nTop 10 declining (latest year):")
print(declining_latest[[COL_CIP, COL_AWLEVEL, "total_completed", "yoy_change", "yoy_pct", "institutions_with_awards"]].head(10))

print("\nTop 10 rising (latest year):")
print(rising_latest[[COL_CIP, COL_AWLEVEL, "total_completed", "yoy_change", "yoy_pct", "institutions_with_awards"]].head(10))
