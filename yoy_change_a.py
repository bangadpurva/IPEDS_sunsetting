"""
IPEDS A-Files: CIP2 × AWLEVEL YoY + Sunset Matrix (2019–2024)
============================================================

Goal
----
Use IPEDS Completions "A" files (cYYYY_a.csv) to measure how program completions
change over time by:
  - Broad field (CIP2 = first 2 digits of CIPCODE)
  - Award level (AWLEVEL)

Outputs
-------
1) Excel:
   data_uni/cip_grouped_awlevel_yoy_students_2019_2024.xlsx
   Contains:
     - Yearly totals 2019–2024
     - YoY change counts and YoY % for each year step
     - Baseline size (avg completions 2019–2021)
     - Net % change (2019→2024)
     - A quantitative decline label ("Growth/Stable", "Moderate", "High Risk")

2) Scatter plot (Sunset Matrix):
   data_uni/sunset_matrix_scatter.png
   X = baseline size (avg 2019–2021, log scale)
   Y = net % change (2019→2024), optionally clipped for readability

3) Heatmap:
   data_uni/sunset_matrix_heatmap.png
   Rows = CIP2_Name, Columns = AWLEVEL, Values = median net % change (2019→2024)

Important Notes
---------------
- A-files contain CIPCODE + AWLEVEL + CTOTALT for each UNITID (institution).
  CTOTALT represents completions in that program/award level at that institution.
- If KEEP_PRIMARY_MAJOR_ONLY=True and MAJORNUM exists, we keep MAJORNUM==1 to avoid double counting.
- If EXCLUDE_AWLEVEL_01=True, removes AWLEVEL "01" if it is anomalous or out-of-scope.

Formulas
--------
YoY Count:
  YoY_count(y) = Total(y) - Total(y-1)

YoY %:
  YoY_pct(y) = ((Total(y) - Total(y-1)) / Total(y-1)) * 100

Baseline (size proxy):
  baseline_avg_2019_2021 = mean(Total(2019), Total(2020), Total(2021))

Net % change (long-run shift):
  net_pct_change_2019_2024 = ((Total(2024) - Total(2019)) / Total(2019)) * 100
  (Undefined when Total(2019)=0; treated as unstable_base_2019_zero.)

Thresholding (Quantitative)
---------------------------
We label decline using a weighted z-score model:
- Compute net_pct_change for each CIP2×AWLEVEL
- Winsorize net_pct_change into [-100, 300] ONLY for estimating distribution parameters
- Weight rows by baseline size, so tiny programs do not dominate thresholds
- Compute weighted mean and std, then z-score:
    z = (winsor_net_pct - weighted_mean) / weighted_std
- Label:
    z <= -2.0  -> High Risk
    z <= -1.0  -> Moderate
    else       -> Growth/Stable
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Union


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
COL_TOTAL = "CTOTALT"      # A files: completions for that UNITID × CIPCODE × AWLEVEL
COL_MAJORNUM = "MAJORNUM"

YEARS = list(range(2019, 2025))

OUT_XLSX = "data_uni/cip_grouped_awlevel_yoy_students_2019_2024.xlsx"
SCATTER_PATH = "data_uni/sunset_matrix_scatter.png"
HEATMAP_PATH = "data_uni/sunset_matrix_heatmap.png"


# -------------------------
# Cleanup controls
# -------------------------
KEEP_PRIMARY_MAJOR_ONLY = True
PRIMARY_MAJOR_VALUE = 1

# Minimum baseline for labeling & plots (reduces “sea of red” effect)
MIN_BASELINE_FOR_LABELS = 20
DROP_TINY_ROWS_FOR_PLOTS = True

# Optional exclusion if AWLEVEL 01 is anomalous/out-of-scope
EXCLUDE_AWLEVEL_01 = False

# Outlier control for visualization (NOT for raw saved values)
CLIP_PCT_FOR_PLOTS = True
PCT_CLIP_LOW, PCT_CLIP_HIGH = -100, 100


# =========================
# CIP2 -> Field name mapping
# =========================
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
    c = str(cip2).zfill(2)
    return CIP2_TO_NAME.get(c, "Unknown/Other")


# =========================
# AWLEVEL name helper (best-effort)
# =========================
# Note: AWLEVEL detailed definitions can vary by IPEDS year; this is a practical label layer.
AWLEVEL_TO_NAME = {
    "01": "Award <1 academic year",
    "1":  "Award <1 academic year",
    "02": "Award 1 to <4 academic years",
    "2":  "Award 1 to <4 academic years",
    "03": "Associate's degree",
    "3":  "Associate's degree",
    "04": "Award < Bachelor's",
    "4":  "Award < Bachelor's",
    "05": "Bachelor's degree",
    "5":  "Bachelor's degree",
    "06": "Postbaccalaureate certificate",
    "6":  "Postbaccalaureate certificate",
    "07": "Master's degree",
    "7":  "Master's degree",
    "08": "Post-master's certificate",
    "8":  "Post-master's certificate",
    "09": "Doctor's degree",
    "9":  "Doctor's degree",
    "17": "First professional degree",
    "18": "Graduate/Professional certificate (other)",
    "19": "Graduate/Professional certificate (other)",
    "20": "Graduate/Professional certificate (other)",
    "21": "Graduate/Professional certificate (other)",
}

def awlevel_name(code: str) -> str:
    s = str(code).strip()
    return AWLEVEL_TO_NAME.get(s, "Unknown/Other")


# =========================
# Helper: extract CIP2 (fast)
# =========================
def extract_cip2_series(cip_series: pd.Series) -> pd.Series:
    """
    Extract first 2 digits of CIPCODE.
    Works for: '01.0999', '01', '52.0201', etc.
    """
    return cip_series.astype("string").str.strip().str.extract(r"^(\d{2})", expand=False)


# =========================
# Statistical threshold model (weighted + winsorization)
# =========================
USE_WEIGHTED_THRESHOLDS = True
WINSOR_LOW, WINSOR_HIGH = -100.0, 300.0
Z_HIGH_RISK = -2.0
Z_MODERATE = -1.0

def weighted_mean_std(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.sum(w * x) / np.sum(w)
    v = np.sum(w * (x - m) ** 2) / np.sum(w)
    return float(m), float(np.sqrt(v))

def apply_stat_threshold_labels(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Expects:
      - baseline_avg_2019_2021
      - net_pct_change_2019_2024
    Returns wide with:
      - net_pct_winsor (for threshold estimation only)
      - z_score_weighted
      - sunset_label_stat
      - (prints derived mean/std and cutoffs)
    """
    out = wide.copy()

    out["net_pct_winsor"] = pd.to_numeric(out["net_pct_change_2019_2024"], errors="coerce").clip(WINSOR_LOW, WINSOR_HIGH)

    mask = (
        out["baseline_avg_2019_2021"].notna()
        & (out["baseline_avg_2019_2021"] > 0)
        & out["net_pct_winsor"].notna()
        & (~out["unstable_base_2019_zero"])
        & (out["baseline_avg_2019_2021"] >= MIN_BASELINE_FOR_LABELS)
    )

    x = out.loc[mask, "net_pct_winsor"].astype(float).to_numpy()
    w = out.loc[mask, "baseline_avg_2019_2021"].astype(float).to_numpy()

    if len(x) < 10:
        out["z_score_weighted"] = np.nan
        out["sunset_label_stat"] = "Unstable/NA"
        print("\n[WARN] Too few rows to estimate weighted thresholds. Labels set to Unstable/NA.")
        return out

    mu, sigma = weighted_mean_std(x, w)
    sigma = max(sigma, 1e-9)

    out["z_score_weighted"] = (out["net_pct_winsor"] - mu) / sigma

    def label_from_z(z):
        if pd.isna(z):
            return "Unstable/NA"
        if z <= Z_HIGH_RISK:
            return "High Risk"
        if z <= Z_MODERATE:
            return "Moderate"
        return "Growth/Stable"

    out["sunset_label_stat"] = out["z_score_weighted"].apply(label_from_z)

    # Print model diagnostics
    print("\n[Threshold Model] Weighted + winsorized distribution (baseline-weighted)")
    print(f"  Winsor range: [{WINSOR_LOW}, {WINSOR_HIGH}] (used only to estimate μ,σ)")
    print(f"  Weighted mean (μ): {mu:.2f}")
    print(f"  Weighted std  (σ): {sigma:.2f}")
    print(f"  High Risk cutoff (z <= {Z_HIGH_RISK}): net_pct ≲ {mu + Z_HIGH_RISK*sigma:.2f}")
    print(f"  Moderate cutoff (z <= {Z_MODERATE}): net_pct ≲ {mu + Z_MODERATE*sigma:.2f}")

    return out


# =========================
# MAIN
# =========================
def main():
    # Ensure output directory exists
    Path("data_uni").mkdir(parents=True, exist_ok=True)

    # 1) LOAD + COMBINE
    frames = []
    print("[LOAD] Reading IPEDS A files...")
    for year in YEARS:
        if year not in FILES_BY_YEAR:
            raise ValueError(f"Missing year in FILES_BY_YEAR config: {year}")

        path = Path(FILES_BY_YEAR[year])
        if not path.exists():
            raise FileNotFoundError(f"Missing file for {year}: {path}")

        df = pd.read_csv(path, low_memory=False)

        # Validate required columns
        required = [COL_CIP, COL_AWLEVEL, COL_TOTAL]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        # Clean types
        df[COL_CIP] = df[COL_CIP].astype("string").str.strip()
        df[COL_AWLEVEL] = df[COL_AWLEVEL].astype("string").str.strip()
        df[COL_TOTAL] = pd.to_numeric(df[COL_TOTAL], errors="coerce").fillna(0)

        # Optional: primary major only
        if KEEP_PRIMARY_MAJOR_ONLY and COL_MAJORNUM in df.columns:
            df[COL_MAJORNUM] = pd.to_numeric(df[COL_MAJORNUM], errors="coerce")
            df = df[df[COL_MAJORNUM] == PRIMARY_MAJOR_VALUE]

        # Optional: exclude AWLEVEL 01
        if EXCLUDE_AWLEVEL_01:
            df = df[df[COL_AWLEVEL] != "01"]

        # CIP2 extraction
        df["CIP2"] = extract_cip2_series(df[COL_CIP])
        df = df[df["CIP2"].notna()].copy()

        df["YEAR"] = year

        frames.append(df[["CIP2", COL_AWLEVEL, COL_TOTAL, "YEAR"]])

        # Print quick inventory
        uniq_aw = sorted(df[COL_AWLEVEL].dropna().unique().tolist())
        print(f"  - {year}: rows={len(df):,} | unique AWLEVEL={uniq_aw}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n[COMBINE] Combined rows (all years): {len(combined):,}")

    # 2) AGGREGATE: CIP2 × AWLEVEL × YEAR
    agg = (
        combined.groupby(["CIP2", COL_AWLEVEL, "YEAR"], as_index=False)
                .agg(total_completed=(COL_TOTAL, "sum"))
    )

    wide = (
        agg.pivot_table(
            index=["CIP2", COL_AWLEVEL],
            columns="YEAR",
            values="total_completed",
            aggfunc="sum"
        )
        .reindex(columns=YEARS)
        .fillna(0)
        .reset_index()
    )

    # Add names
    wide["CIP2_Name"] = wide["CIP2"].apply(cip2_name)
    wide["AWLEVEL_Name"] = wide[COL_AWLEVEL].apply(awlevel_name)

    # Ensure year columns numeric
    for y in YEARS:
        wide[y] = pd.to_numeric(wide[y], errors="coerce").fillna(0)

    # 3) YOY count and YOY %
    yoy_count_cols = []
    yoy_pct_cols = []
    for i in range(1, len(YEARS)):
        y_prev, y_cur = YEARS[i - 1], YEARS[i]
        diff_col = f"{y_cur}-{str(y_prev)[-2:]}"  # e.g., "2020-19"
        pct_col = f"{diff_col}%"

        wide[diff_col] = wide[y_cur] - wide[y_prev]
        wide[pct_col] = (wide[diff_col] / wide[y_prev].replace({0: np.nan})) * 100

        yoy_count_cols.append(diff_col)
        yoy_pct_cols.append(pct_col)

    wide[yoy_pct_cols] = wide[yoy_pct_cols].apply(pd.to_numeric, errors="coerce").round(2)

    # 4) Baseline + Net % change for Sunset Matrix
    wide["baseline_avg_2019_2021"] = wide[[2019, 2020, 2021]].mean(axis=1)

    # Net % change (2019→2024)
    wide["net_pct_change_2019_2024"] = ((wide[2024] - wide[2019]) / wide[2019].replace({0: np.nan})) * 100
    wide["net_pct_change_2019_2024"] = pd.to_numeric(wide["net_pct_change_2019_2024"], errors="coerce")

    wide["unstable_base_2019_zero"] = (wide[2019] == 0)

    # 5) Quantitative labeling (weighted z-score)
    if USE_WEIGHTED_THRESHOLDS:
        wide = apply_stat_threshold_labels(wide)
        wide["sunset_label"] = wide["sunset_label_stat"]
    else:
        wide["sunset_label"] = "Unlabeled"

    # Print label counts
    print("\n[LABELS] Sunset label counts:")
    print(wide["sunset_label"].value_counts(dropna=False).to_string())

    # 6) Save Excel output
    out_cols = (
        ["CIP2", "CIP2_Name", COL_AWLEVEL, "AWLEVEL_Name"]
        + YEARS
        + yoy_count_cols
        + yoy_pct_cols
        + ["baseline_avg_2019_2021", "net_pct_change_2019_2024", "unstable_base_2019_zero", "sunset_label"]
    )

    wide[out_cols].to_excel(OUT_XLSX, index=False)
    print(f"\n[SAVE] Excel saved → {OUT_XLSX}")

    # =========================
    # 7) SUNSET MATRIX SCATTER PLOT (NA-safe)
    # =========================
    plot_df = wide.copy()

    # Coerce to numeric + remove inf/NA
    plot_df["baseline_avg_2019_2021"] = pd.to_numeric(plot_df["baseline_avg_2019_2021"], errors="coerce")
    plot_df["net_pct_change_2019_2024"] = pd.to_numeric(plot_df["net_pct_change_2019_2024"], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

    # Drop tiny rows for plots (optional)
    if DROP_TINY_ROWS_FOR_PLOTS:
        plot_df = plot_df[plot_df["baseline_avg_2019_2021"] >= MIN_BASELINE_FOR_LABELS].copy()

    # Clip percent for plotting (optional)
    plot_df["net_pct_for_plot"] = plot_df["net_pct_change_2019_2024"]
    if CLIP_PCT_FOR_PLOTS:
        plot_df["net_pct_for_plot"] = plot_df["net_pct_for_plot"].clip(PCT_CLIP_LOW, PCT_CLIP_HIGH)

    # Drop rows still missing plot values
    plot_df = plot_df.dropna(subset=["baseline_avg_2019_2021", "net_pct_for_plot"]).copy()

    # Need strictly positive baseline for log scale
    plot_df = plot_df[plot_df["baseline_avg_2019_2021"] > 0].copy()

    # Data-derived size split (median)
    size_cut = float(plot_df["baseline_avg_2019_2021"].median()) if len(plot_df) else 0.0

    plt.figure(figsize=(14, 8))
    label_order = ["Growth/Stable", "Moderate", "High Risk", "Unstable (2019=0)", "Unstable/NA"]

    for label in label_order:
        sub = plot_df[plot_df["sunset_label"] == label]
        if len(sub) == 0:
            continue
        plt.scatter(
            sub["baseline_avg_2019_2021"].astype(float),
            sub["net_pct_for_plot"].astype(float),
            alpha=0.8,
            label=label
        )

    plt.axhline(0, linestyle="--")
    plt.axvline(size_cut, linestyle="--")
    plt.xscale("log")
    plt.title("Sunset Matrix: Size vs Net % Change (CIP2 × AWLEVEL, 2019–2024)")
    plt.xlabel("Baseline Size (Avg completions 2019–2021, log scale)")
    plt.ylabel("Net % Change (2019→2024)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCATTER_PATH, dpi=300)
    plt.show()
    print(f"[SAVE] Scatter saved → {SCATTER_PATH}")

    # =========================
    # 8) HEATMAP (median net % change by Field × AWLEVEL)
    # =========================
    heat_source = wide.copy()

    heat_source["baseline_avg_2019_2021"] = pd.to_numeric(heat_source["baseline_avg_2019_2021"], errors="coerce")
    heat_source["net_pct_change_2019_2024"] = pd.to_numeric(heat_source["net_pct_change_2019_2024"], errors="coerce")
    heat_source = heat_source.replace([np.inf, -np.inf], np.nan)

    if DROP_TINY_ROWS_FOR_PLOTS:
        heat_source = heat_source[heat_source["baseline_avg_2019_2021"] >= MIN_BASELINE_FOR_LABELS].copy()

    heat_source["net_pct_for_heat"] = heat_source["net_pct_change_2019_2024"]
    if CLIP_PCT_FOR_PLOTS:
        heat_source["net_pct_for_heat"] = heat_source["net_pct_for_heat"].clip(PCT_CLIP_LOW, PCT_CLIP_HIGH)

    heat = heat_source.pivot_table(
        index="CIP2_Name",
        columns=COL_AWLEVEL,
        values="net_pct_for_heat",
        aggfunc="median"
    )

    # Sort rows for a stable view (alphabetical)
    heat = heat.sort_index()

    plt.figure(figsize=(14, 10))
    plt.imshow(heat.fillna(0).values, aspect="auto")
    plt.colorbar(label="Median Net % Change (2019→2024)")
    plt.title("Heatmap: Median Net % Change by Field (CIP2) and AWLEVEL (2019–2024)")
    plt.xlabel("AWLEVEL")
    plt.ylabel("Field (CIP2)")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=0)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=300)
    plt.show()
    print(f"[SAVE] Heatmap saved → {HEATMAP_PATH}")

    # =========================
    # 9) Optional: quadrant interpretation (printed)
    # =========================
    quad_df = plot_df.copy()
    quad_df["quadrant"] = np.where(
        quad_df["baseline_avg_2019_2021"] >= size_cut,
        np.where(quad_df["net_pct_for_plot"] < 0, "Large Declining", "Large Growing"),
        np.where(quad_df["net_pct_for_plot"] < 0, "Small Declining", "Small Growing"),
    )

    top_large_declining = (
        quad_df[quad_df["quadrant"] == "Large Declining"]
        .sort_values(by="baseline_avg_2019_2021", ascending=False)
        .head(15)[["CIP2", "CIP2_Name", COL_AWLEVEL, "AWLEVEL_Name", "baseline_avg_2019_2021", "net_pct_change_2019_2024", "sunset_label"]]
    )

    print("\n[INSIGHT] Top 15 Large Declining (highest baseline impact):")
    if len(top_large_declining):
        print(top_large_declining.to_string(index=False))
    else:
        print("  None found (after filters).")

    # =========================
    # 10) HEATMAP (Large Declining highlight)
    # =========================
    # This heatmap is binary:
    #   1 = Large Declining  (baseline >= size_cut AND net % change < 0)
    #   0 = everything else
    #
    # Benefit: makes “sunset candidates” visually obvious, avoids a “sea of red”
    # from small programs with noisy percent changes.

    highlight_df = plot_df.copy()

    # Define large declining based on the same scatter thresholds
    highlight_df["is_large_declining"] = (
        (highlight_df["baseline_avg_2019_2021"] >= size_cut) &
        (highlight_df["net_pct_for_plot"] < 0)
    ).astype(int)

    # Build a matrix: rows=field, cols=AWLEVEL, values=% of rows in that cell that are large declining
    # Using mean() on 0/1 gives a proportion (0 to 1). Multiply by 100 for % if desired.
    highlight_heat = highlight_df.pivot_table(
        index="CIP2_Name",
        columns=COL_AWLEVEL,
        values="is_large_declining",
        aggfunc="mean"
    ).fillna(0).sort_index()

    plt.figure(figsize=(14, 10))
    plt.imshow(highlight_heat.values, aspect="auto")
    plt.colorbar(label="Share of Large Declining (0–1)")
    plt.title("Heatmap: Large Declining Highlight (CIP2 × AWLEVEL, 2019–2024)")
    plt.xlabel("AWLEVEL")
    plt.ylabel("Field (CIP2)")

    plt.xticks(range(len(highlight_heat.columns)), highlight_heat.columns, rotation=0)
    plt.yticks(range(len(highlight_heat.index)), highlight_heat.index)

    plt.tight_layout()

    highlight_path = "data_uni/sunset_matrix_heatmap_large_declining.png"
    plt.savefig(highlight_path, dpi=300)
    plt.show()
    print(f"[SAVE] Large Declining heatmap saved → {highlight_path}")


if __name__ == "__main__":
    main()