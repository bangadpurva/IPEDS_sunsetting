"""
This script processes the "A" files from the IPEDS Completions dataset to analyze year-over-year changes in program completions by broad field (CIP2) and award level (AWLEVEL). It produces an Excel file with detailed metrics and two key visualizations: a sunset matrix scatter plot and a heatmap of median net percentage change.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union
import matplotlib.pyplot as plt

# =========================
# CONFIG (edit paths)
# =========================
FILES_BY_YEAR: Dict[int, Union[str, Path]] = {
    2019: "data_uni/c2019_a.csv",
    2020: "data_uni/c2020_a.csv",
    2021: "data_uni/c2021_a.csv",
    2022: "data_uni/c2022_a.csv",
    2023: "data_uni/c2023_a.csv",
    2024: "data_uni/c2024_a.csv",
}

COL_UNITID = "UNITID"
COL_CIP = "CIPCODE"
COL_AWLEVEL = "AWLEVEL"
COL_TOTAL = "CTOTALT"      # In A files: total completions for that UNITID x CIP x AWLEVEL
COL_MAJORNUM = "MAJORNUM"

YEARS = list(range(2019, 2025))
OUT_XLSX = "data_uni/cip_grouped_awlevel_yoy_students_2019_2024.xlsx"

# Optional professor-driven cleanup controls
KEEP_PRIMARY_MAJOR_ONLY = True
PRIMARY_MAJOR_VALUE = 1

# Many CIP/AWLEVEL combinations are tiny; they can visually dominate (sea of red) even if not meaningful.
# Use a minimum baseline filter for "risk labeling" and for cleaner plots.
MIN_BASELINE_FOR_LABELS = 20   # baseline avg 2019-2021 must be >= this to be labeled Growth/Moderate/High Risk
DROP_TINY_ROWS_FOR_PLOTS = True

# AWLEVEL anomaly check:
# If AWLEVEL == "01" has data in 2019 and 0 after, it may be a reporting/definition issue rather than a true collapse.
# Set this to True to exclude AWLEVEL 01 entirely (certificate < 1 year) from outputs/plots.
EXCLUDE_AWLEVEL_01 = False

# "Outlier control" for visualization:
# Percent changes can explode when baseline is small (e.g., 2 -> 40 = 1900%).
# Clipping keeps direction but prevents a few extreme points from ruining the scale of the plot/heatmap.
CLIP_PCT_FOR_PLOTS = True
PCT_CLIP_LOW, PCT_CLIP_HIGH = -100, 100

# =========================
# CIP2 -> Field name mapping
# =========================
# "Automatically" can mean using the official CIP taxonomy file (not included here).
# This is a built-in mapping for common 2-digit CIP families.
# You can expand it, or replace it with a lookup from an external crosswalk file if you have one.
CIP2_TO_NAME = {
    "01": "Agriculture & Related Sciences",
    "03": "Natural Resources & Conservation",
    "04": "Architecture & Related Services",
    "05": "Area, Ethnic, Cultural & Gender Studies",
    "09": "Communication & Journalism",
    "10": "Communications Technologies",
    "11": "Computer & Information Sciences",
    "12": "Personal & Culinary Services",
    "13": "Education",
    "14": "Engineering",
    "15": "Engineering Technologies",
    "16": "Foreign Languages",
    "19": "Family & Consumer Sciences",
    "22": "Legal Professions",
    "23": "English Language & Literature",
    "24": "Liberal Arts & Sciences",
    "25": "Library Science",
    "26": "Biological & Biomedical Sciences",
    "27": "Mathematics & Statistics",
    "30": "Multi/Interdisciplinary Studies",
    "31": "Parks, Recreation, Fitness",
    "38": "Philosophy & Religious Studies",
    "40": "Physical Sciences",
    "42": "Psychology",
    "43": "Homeland Security & Law Enforcement",
    "44": "Public Administration & Social Service",
    "45": "Social Sciences",
    "46": "Construction Trades",
    "47": "Mechanic & Repair Technologies",
    "48": "Precision Production",
    "49": "Transportation & Materials Moving",
    "50": "Visual & Performing Arts",
    "51": "Health Professions",
    "52": "Business, Management, Marketing",
    "54": "History",
}

def cip2_name(cip2: str) -> str:
    cip2 = str(cip2).zfill(2)
    return CIP2_TO_NAME.get(cip2, "Unknown/Other")

# =========================
# Helper: extract CIP2
# =========================
def extract_cip2(val) -> str:
    # Works for "01.0999", "01", "52.0201", etc.
    s = str(val).strip()
    m = pd.Series([s]).str.extract(r"^(\d{2})", expand=False).iloc[0]
    return m if pd.notna(m) else None

# =========================
# 1) LOAD + COMBINE ALL YEARS
# =========================
frames = []

for year, path in FILES_BY_YEAR.items():
    df = pd.read_csv(path, low_memory=False)

    # Basic cleaning
    df[COL_CIP] = df[COL_CIP].astype("string").str.strip()
    df[COL_AWLEVEL] = df[COL_AWLEVEL].astype("string").str.strip()

    # CTOTALT is numeric program completions
    df[COL_TOTAL] = pd.to_numeric(df[COL_TOTAL], errors="coerce").fillna(0)

    # Optional: only keep primary major to avoid double counting across MAJORNUM categories
    if KEEP_PRIMARY_MAJOR_ONLY and COL_MAJORNUM in df.columns:
        df[COL_MAJORNUM] = pd.to_numeric(df[COL_MAJORNUM], errors="coerce")
        df = df[df[COL_MAJORNUM] == PRIMARY_MAJOR_VALUE]

    # Optional exclusion: AWLEVEL "01"
    if EXCLUDE_AWLEVEL_01:
        df = df[df[COL_AWLEVEL] != "01"]

    # Create CIP2
    df["CIP2"] = df[COL_CIP].apply(extract_cip2)
    df = df[df["CIP2"].notna()]  # drop malformed CIP rows

    df["YEAR"] = year

    frames.append(df[["CIP2", COL_AWLEVEL, COL_TOTAL, "YEAR"]])

combined = pd.concat(frames, ignore_index=True)

# =========================
# 2) AGGREGATE: CIP2 x AWLEVEL x YEAR
# =========================
agg = (
    combined.groupby(["CIP2", COL_AWLEVEL, "YEAR"], as_index=False)
            .agg(total_completed=(COL_TOTAL, "sum"))
)

# Wide table: rows are CIP2 x AWLEVEL, columns are years
wide = (
    agg.pivot_table(index=["CIP2", COL_AWLEVEL], columns="YEAR", values="total_completed", aggfunc="sum")
       .reindex(columns=YEARS)
       .fillna(0)
       .reset_index()
)

# Add field names
wide["CIP2_Name"] = wide["CIP2"].apply(cip2_name)

# =========================
# 3) YOY count and YOY %
# =========================
for i in range(1, len(YEARS)):
    y_prev, y_cur = YEARS[i-1], YEARS[i]
    diff_col = f"{y_cur}-{str(y_prev)[-2:]}"     # e.g., "2020-19"
    pct_col = f"{diff_col}%"

    wide[diff_col] = wide[y_cur] - wide[y_prev]
    wide[pct_col] = (wide[diff_col] / wide[y_prev].replace({0: pd.NA})) * 100

pct_cols = [c for c in wide.columns if isinstance(c, str) and c.endswith("%")]
wide[pct_cols] = wide[pct_cols].round(2)

# =========================
# 4) Baseline + Net % change for Sunset Matrix
# =========================
# Baseline size:
# We use avg(2019-2021) to represent "normal" pre-shift conditions
# (smoother than using 2019 alone).
wide["baseline_avg_2019_2021"] = wide[[2019, 2020, 2021]].mean(axis=1)

# net_pct_change:
# This is overall change across the study window (2019 -> 2024):
#   net_pct_change = ((2024 - 2019) / 2019) * 100
#
# What it gives you:
# - a single, interpretable summary of 5-year direction and magnitude
# - helps classify "growth vs decline" without being overly sensitive to one noisy YoY step
#
# Why it helps:
# - YoY can be volatile; net_pct_change captures the longer-run shift.
wide["net_pct_change_2019_2024"] = (
    (wide[2024] - wide[2019]) / wide[2019].replace({0: pd.NA}) * 100
)

# Flag unstable base (2019=0) because % change is undefined or misleading
wide["unstable_base_2019_zero"] = (wide[2019] == 0)

# =========================
# 5) Data-driven threshold labeling (quantiles)
# =========================
# Instead of "guessing" cutoffs (e.g., -15% = high risk), we derive thresholds from the data:
# - This is defensible in methodology: thresholds are empirical (based on distribution).
#
# Typical approach:
# - Use only "stable" rows (2019 > 0) and meaningful size (baseline >= MIN_BASELINE_FOR_LABELS).
# - Then define:
#   High Risk  = bottom 20% net_pct_change
#   Growth     = top 20% net_pct_change
#   Moderate   = everything in between
#
# This avoids tiny programs (baseline 1-5) creating huge percentages that distort labels.
mask_label_pool = (
    (~wide["unstable_base_2019_zero"])
    & (wide["baseline_avg_2019_2021"] >= MIN_BASELINE_FOR_LABELS)
    & wide["net_pct_change_2019_2024"].notna()
)

label_pool = wide.loc[mask_label_pool, "net_pct_change_2019_2024"].astype(float)

if len(label_pool) > 0:
    q20 = float(label_pool.quantile(0.20))
    q80 = float(label_pool.quantile(0.80))
else:
    q20, q80 = -10.0, 10.0  # fallback if something is wrong

def label_row(r):
    if r["unstable_base_2019_zero"]:
        return "Unstable (2019=0)"
    if r["baseline_avg_2019_2021"] < MIN_BASELINE_FOR_LABELS:
        return "Too small to label"
    v = r["net_pct_change_2019_2024"]
    if pd.isna(v):
        return "Unstable/NA"
    if v <= q20:
        return "High Risk"
    if v >= q80:
        return "Growth"
    return "Moderate"

wide["sunset_label"] = wide.apply(label_row, axis=1)

# -------------------------
# Statistical threshold model (weighted + outlier control)
# -------------------------
USE_WEIGHTED_THRESHOLDS = True

# Outlier control for threshold estimation (NOT for the raw results table)
# Winsorize net_pct_change into a reasonable range so a few extreme rows don't set the bar.
WINSOR_LOW, WINSOR_HIGH = -100.0, 300.0

# Sigma cutoffs (common, defensible choices)
Z_HIGH_RISK = -2.0     # <= -2σ below mean
Z_MODERATE  = -1.0     # between -1σ and -2σ

def safe_pct_change(curr: pd.Series, prev: pd.Series) -> pd.Series:
    """Percent change with protection against divide-by-zero."""
    prev_safe = prev.replace({0: np.nan})
    return ((curr - prev) / prev_safe) * 100

def weighted_mean_std(x: np.ndarray, w: np.ndarray):
    """Weighted mean + weighted std (population-style)."""
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.sum(w * x) / np.sum(w)
    v = np.sum(w * (x - m) ** 2) / np.sum(w)
    return m, np.sqrt(v)

def apply_stat_threshold_labels(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns:
      - baseline_avg_2019_2021
      - net_pct_change_2019_2024
    Returns wide with:
      - net_pct_winsor (for threshold estimation)
      - z_score_weighted
      - sunset_label_stat
    """
    out = wide.copy()

    # 1) Winsorize net % change for estimating thresholds
    out["net_pct_winsor"] = out["net_pct_change_2019_2024"].clip(WINSOR_LOW, WINSOR_HIGH)

    # 2) Choose weights (baseline size); ignore rows with missing/0 baseline for model stats
    mask = out["baseline_avg_2019_2021"].notna() & (out["baseline_avg_2019_2021"] > 0) & out["net_pct_winsor"].notna()
    x = out.loc[mask, "net_pct_winsor"].astype(float).to_numpy()
    w = out.loc[mask, "baseline_avg_2019_2021"].astype(float).to_numpy()

    if len(x) < 10:
        # Too few rows to estimate stable thresholds
        out["z_score_weighted"] = np.nan
        out["sunset_label_stat"] = "Unstable/NA"
        return out

    mu, sigma = weighted_mean_std(x, w)
    sigma = max(sigma, 1e-9)  # avoid divide-by-zero

    # 3) Weighted z-score per row (based on the winsorized distribution)
    out["z_score_weighted"] = (out["net_pct_winsor"] - mu) / sigma

    # 4) Labels from z-score
    def label_from_z(row):
        if pd.isna(row["z_score_weighted"]):
            return "Unstable/NA"
        if row["z_score_weighted"] <= Z_HIGH_RISK:
            return "High Risk"
        if row["z_score_weighted"] <= Z_MODERATE:
            return "Moderate"
        return "Growth/Stable"

    out["sunset_label_stat"] = out.apply(label_from_z, axis=1)
    return out

# After you create baseline_avg_2019_2021 and net_pct_change_2019_2024:
wide = apply_stat_threshold_labels(wide)

# Use this label column in plots / tables:
wide["sunset_label"] = wide["sunset_label_stat"]
# =========================
# 6) Save Excel output (your main deliverable)
# =========================
# Keep a clean, professor-readable column order:
yoy_count_cols = [f"{YEARS[i]}-{str(YEARS[i-1])[-2:]}" for i in range(1, len(YEARS))]
yoy_pct_cols = [c + "%" for c in yoy_count_cols]

out_cols = (
    ["CIP2", "CIP2_Name", COL_AWLEVEL]
    + YEARS
    + yoy_count_cols
    + yoy_pct_cols
    + ["baseline_avg_2019_2021", "net_pct_change_2019_2024", "unstable_base_2019_zero", "sunset_label"]
)

wide[out_cols].to_excel(OUT_XLSX, index=False)
print(f"Saved Excel → {OUT_XLSX}")

# =========================
# 7) SUNSET MATRIX SCATTER PLOT (NA-safe)
# =========================
plot_df = wide.copy()

# Ensure the plot columns are numeric (pd.NA -> np.nan automatically via errors="coerce")
plot_df["baseline_avg_2019_2021"] = pd.to_numeric(plot_df["baseline_avg_2019_2021"], errors="coerce")
plot_df["net_pct_change_2019_2024"] = pd.to_numeric(plot_df["net_pct_change_2019_2024"], errors="coerce")

# Optionally drop tiny rows from plots to reduce “sea of red”
if DROP_TINY_ROWS_FOR_PLOTS:
    plot_df = plot_df[plot_df["baseline_avg_2019_2021"] >= MIN_BASELINE_FOR_LABELS].copy()

# For visualization only: clip extreme percent values (outlier control)
plot_df["net_pct_for_plot"] = plot_df["net_pct_change_2019_2024"].copy()

# Replace inf/-inf (can happen if someone computed % using division by ~0)
plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

if CLIP_PCT_FOR_PLOTS:
    plot_df["net_pct_for_plot"] = plot_df["net_pct_for_plot"].clip(PCT_CLIP_LOW, PCT_CLIP_HIGH)

# CRITICAL: drop rows that still have missing plot values
plot_df = plot_df.dropna(subset=["baseline_avg_2019_2021", "net_pct_for_plot"]).copy()

# Size threshold line: median baseline among plotted rows (data-derived)
if len(plot_df) > 0:
    size_cut = float(plot_df["baseline_avg_2019_2021"].median())
else:
    size_cut = 0.0

plt.figure(figsize=(14, 8))

# Plot each label group separately so legend is meaningful
for label in ["Growth", "Moderate", "High Risk", "Unstable (2019=0)", "Too small to label", "Unstable/NA"]:
    sub = plot_df[plot_df["sunset_label"] == label]
    if len(sub) == 0:
        continue

    plt.scatter(
        sub["baseline_avg_2019_2021"].astype(float),
        sub["net_pct_for_plot"].astype(float),
        alpha=0.8,
        label=label
    )

plt.axhline(0, linestyle="--")             # 0% line: decline vs growth
plt.axvline(size_cut, linestyle="--")      # data-derived split: small vs large

# log scale requires strictly positive x values
plt.xscale("log")
plt.title("Sunset Matrix: Program Size vs Net % Change (CIP2 × AWLEVEL, 2019–2024)")
plt.xlabel("Baseline Size (Avg completions 2019–2021, log scale)")
plt.ylabel("Net % Change (2019→2024)")

plt.legend()
plt.tight_layout()

scatter_path = "data_uni/sunset_matrix_scatter.png"
plt.savefig(scatter_path, dpi=300)
plt.show()
print(f"Saved scatter → {scatter_path}")

# =========================
# 8) HEATMAP (decline/growth by Field × AWLEVEL)
# =========================
# Heatmap shows patterns across:
# - rows = broad fields (CIP2_Name)
# - columns = AWLEVEL
# - cell value = average/median net % change
#
# Median is usually more stable than mean when there are outliers.
heat_source = wide.copy()

# Optionally remove tiny baselines from heatmap too
if DROP_TINY_ROWS_FOR_PLOTS:
    heat_source = heat_source[heat_source["baseline_avg_2019_2021"] >= MIN_BASELINE_FOR_LABELS].copy()

heat_source["net_pct_for_heat"] = heat_source["net_pct_change_2019_2024"].copy()
if CLIP_PCT_FOR_PLOTS:
    heat_source["net_pct_for_heat"] = heat_source["net_pct_for_heat"].clip(PCT_CLIP_LOW, PCT_CLIP_HIGH)

heat = heat_source.pivot_table(
    index="CIP2_Name",
    columns=COL_AWLEVEL,
    values="net_pct_for_heat",
    aggfunc="median"
).fillna(0)

plt.figure(figsize=(14, 10))
plt.imshow(heat.values, aspect="auto")  # default colormap (no custom colors)

plt.colorbar(label="Median Net % Change (2019→2024)")

plt.title("Heatmap: Median Net % Change by Field (CIP2) and AWLEVEL (2019–2024)")
plt.xlabel("AWLEVEL")
plt.ylabel("Field (CIP2)")

plt.xticks(range(len(heat.columns)), heat.columns, rotation=0)
plt.yticks(range(len(heat.index)), heat.index)

plt.tight_layout()

heatmap_path = "data_uni/sunset_matrix_heatmap.png"
plt.savefig(heatmap_path, dpi=300)
plt.show()
print(f"Saved heatmap → {heatmap_path}")

# =========================
# 9) Quick quadrant interpretation helper (optional)
# =========================
# Large Declining: baseline >= size_cut AND net_pct < 0
# Small Declining: baseline < size_cut AND net_pct < 0
# Large Growing:  baseline >= size_cut AND net_pct > 0
# Small Growing:  baseline < size_cut AND net_pct > 0
#
# This is how you translate the scatter into “which fields live where”.
quad_df = plot_df.copy()
quad_df["quadrant"] = np.where(
    quad_df["baseline_avg_2019_2021"] >= size_cut,
    np.where(quad_df["net_pct_for_plot"] < 0, "Large Declining", "Large Growing"),
    np.where(quad_df["net_pct_for_plot"] < 0, "Small Declining", "Small Growing"),
)

# Example: top 15 “Large Declining” rows by biggest absolute baseline (most impactful)
top_large_declining = (
    quad_df[quad_df["quadrant"] == "Large Declining"]
    .sort_values(by="baseline_avg_2019_2021", ascending=False)
    .head(15)[["CIP2", "CIP2_Name", COL_AWLEVEL, "baseline_avg_2019_2021", "net_pct_change_2019_2024", "sunset_label"]]
)

print("\nTop Large Declining (most impactful) examples:")
print(top_large_declining.to_string(index=False))