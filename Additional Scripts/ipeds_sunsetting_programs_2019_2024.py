#%%
import pandas as pd
import numpy as np
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

COL_UNITID = "UNITID"
COL_CIP = "CIPCODE"
COL_AWLEVEL = "AWLEVEL"
COL_MAJORNUM = "MAJORNUM"
COL_TOTAL = "CTOTALT"   # program-level total completions

KEEP_PRIMARY_MAJOR_ONLY = True
PRIMARY_MAJOR_VALUE = 1

YEARS = list(range(2019, 2025))

OUT_EXCEL = "data_uni/ipeds_sunsetting_programs_2019_2024.xlsx"
OUT_COMBINED = "data_uni/completions_programs_2019_2024.csv"

# "Sunsetting" thresholds (tune if needed)
MIN_HIST_AVG = 10      # historical average completions threshold
LOW_LATEST = 2         # near-zero in latest year
NEG_YOY_COUNT_MIN = 3  # at least N negative YoY deltas in window

# =========================
# 1) LOAD + COMBINE
# =========================
frames = []
for year, path in FILES_BY_YEAR.items():
    df = pd.read_csv(path, low_memory=False)

    for c in [COL_UNITID, COL_CIP, COL_AWLEVEL, COL_MAJORNUM]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    df[COL_TOTAL] = pd.to_numeric(df[COL_TOTAL], errors="coerce").fillna(0)
    df["YEAR"] = year

    keep_cols = [COL_UNITID, COL_CIP, COL_AWLEVEL, COL_TOTAL, "YEAR"]
    if COL_MAJORNUM in df.columns:
        keep_cols.insert(3, COL_MAJORNUM)

    df = df[keep_cols]
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)

if KEEP_PRIMARY_MAJOR_ONLY and COL_MAJORNUM in combined.columns:
    combined[COL_MAJORNUM] = pd.to_numeric(combined[COL_MAJORNUM], errors="coerce")
    combined = combined[combined[COL_MAJORNUM] == PRIMARY_MAJOR_VALUE]

Path("data_uni").mkdir(exist_ok=True)
combined.to_csv(OUT_COMBINED, index=False)

# =========================
# 2) HELPERS
# =========================
def add_yoy_and_trend(wide: pd.DataFrame, year_cols: list[int]) -> pd.DataFrame:
    """Adds YoY delta, YoY %, and a linear trend slope across years."""
    out = wide.copy()

    # YoY columns
    yoy_delta_cols = []
    yoy_pct_cols = []

    for i in range(1, len(year_cols)):
        y_prev, y_cur = year_cols[i - 1], year_cols[i]
        delta_col = f"{str(y_cur)[-2:]}-{str(y_prev)[-2:]}"
        pct_col = f"{delta_col}%"

        out[delta_col] = out[y_cur] - out[y_prev]
        out[pct_col] = (out[delta_col] / out[y_prev].replace({0: np.nan})) * 100

        yoy_delta_cols.append(delta_col)
        yoy_pct_cols.append(pct_col)

    # Trend slope (linear regression on year index)
    x = np.arange(len(year_cols), dtype=float)

    def slope_row(row_vals: np.ndarray) -> float:
        y = row_vals.astype(float)
        # if all zeros, slope = 0
        if np.allclose(y, 0):
            return 0.0
        # np.polyfit returns [slope, intercept]
        return float(np.polyfit(x, y, 1)[0])

    out["trend_slope_2019_2024"] = out[year_cols].apply(lambda r: slope_row(r.values), axis=1)

    # Round YoY %
    pct_cols = [c for c in out.columns if isinstance(c, str) and c.endswith("%")]
    out[pct_cols] = out[pct_cols].round(2)

    return out, yoy_delta_cols, yoy_pct_cols

def classify_sunsetting(df: pd.DataFrame, year_cols: list[int], yoy_delta_cols: list[str]) -> pd.DataFrame:
    """Adds interpretable decline flags for 'sunsetting'."""
    out = df.copy()

    out["total_2019"] = out[2019]
    out["total_2024"] = out[2024]
    out["net_change_24_vs_19"] = out[2024] - out[2019]

    # negative YoY count
    out["neg_yoy_count"] = (out[yoy_delta_cols] < 0).sum(axis=1)

    # historical average (2019-2021)
    out["avg_2019_2021"] = out[[2019, 2020, 2021]].mean(axis=1)

    # Flags
    out["flag_declining_trend"] = out["trend_slope_2019_2024"] < 0
    out["flag_persistent_decline"] = (out["neg_yoy_count"] >= NEG_YOY_COUNT_MIN) & (out["net_change_24_vs_19"] < 0)
    out["flag_fallen_off_rail"] = (out["avg_2019_2021"] >= MIN_HIST_AVG) & (out[2024] <= LOW_LATEST)

    # Overall sunsetting label
    out["sunsetting_candidate"] = out["flag_declining_trend"] & (out["flag_persistent_decline"] | out["flag_fallen_off_rail"])

    return out

# =========================
# 3) UNIVERSITY-LEVEL: UNITID x CIPCODE x AWLEVEL
# =========================
uni_agg = (
    combined.groupby([COL_UNITID, COL_CIP, COL_AWLEVEL, "YEAR"], as_index=False)
            .agg(total_completed=(COL_TOTAL, "sum"))
)

uni_wide = (
    uni_agg.pivot_table(
        index=[COL_UNITID, COL_CIP, COL_AWLEVEL],
        columns="YEAR",
        values="total_completed",
        aggfunc="sum"
    )
    .reindex(columns=YEARS)
    .fillna(0)
    .reset_index()
)

uni_wide, uni_yoy_delta_cols, uni_yoy_pct_cols = add_yoy_and_trend(uni_wide, YEARS)
uni_final = classify_sunsetting(uni_wide, YEARS, uni_yoy_delta_cols)

# Keep a "decliners only" view too
uni_decliners = uni_final[uni_final["sunsetting_candidate"]].copy()
uni_decliners = uni_decliners.sort_values(
    by=["flag_fallen_off_rail", "trend_slope_2019_2024", "net_change_24_vs_19"],
    ascending=[False, True, True]
)

# =========================
# 4) NATIONAL: CIPCODE x AWLEVEL (sum over UNITID)
#    + institutions producing >=1 completion each year
# =========================
nat_agg = (
    combined.groupby([COL_CIP, COL_AWLEVEL, "YEAR"], as_index=False)
            .agg(total_completed=(COL_TOTAL, "sum"))
)

nat_wide = (
    nat_agg.pivot_table(
        index=[COL_CIP, COL_AWLEVEL],
        columns="YEAR",
        values="total_completed",
        aggfunc="sum"
    )
    .reindex(columns=YEARS)
    .fillna(0)
    .reset_index()
)

# institution availability: count of UNITIDs with >=1 completion
inst_avail = (
    combined.assign(has_award=(combined[COL_TOTAL] > 0).astype(int))
            .groupby([COL_CIP, COL_AWLEVEL, "YEAR"], as_index=False)
            .agg(institutions_awarding=("has_award", "sum"))
)

inst_wide = (
    inst_avail.pivot_table(
        index=[COL_CIP, COL_AWLEVEL],
        columns="YEAR",
        values="institutions_awarding",
        aggfunc="sum"
    )
    .reindex(columns=YEARS)
    .fillna(0)
    .reset_index()
)

nat_wide, nat_yoy_delta_cols, nat_yoy_pct_cols = add_yoy_and_trend(nat_wide, YEARS)
nat_final = classify_sunsetting(nat_wide, YEARS, nat_yoy_delta_cols)

# merge availability columns in a readable way
inst_wide = inst_wide.rename(columns={y: f"inst_{y}" for y in YEARS})
nat_final = nat_final.merge(inst_wide, on=[COL_CIP, COL_AWLEVEL], how="left")

nat_decliners = nat_final[nat_final["sunsetting_candidate"]].copy()
nat_decliners = nat_decliners.sort_values(
    by=["flag_fallen_off_rail", "trend_slope_2019_2024", "net_change_24_vs_19"],
    ascending=[False, True, True]
)

# =========================
# 5) WRITE OUTPUTS
# =========================
with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
    uni_final.to_excel(writer, sheet_name="Uni_All", index=False)
    uni_decliners.to_excel(writer, sheet_name="Uni_Sunsetting", index=False)

    nat_final.to_excel(writer, sheet_name="National_All", index=False)
    nat_decliners.to_excel(writer, sheet_name="National_Sunsetting", index=False)

print(f"Saved outputs → {OUT_EXCEL}")
print(f"Saved combined program file → {OUT_COMBINED}")

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
IN_EXCEL = Path("data_uni/ipeds_sunsetting_programs_2019_2024.xlsx")  # output from your script
OUT_DIR = Path("viz_sunsetting")
OUT_DIR.mkdir(exist_ok=True)

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

COL_CIP = "CIPCODE"
COL_AWLEVEL = "AWLEVEL"
COL_UNITID = "UNITID"

# =========================
# LOAD OUTPUTS
# =========================
nat_all = pd.read_excel(IN_EXCEL, sheet_name="National_All")
nat_sunset = pd.read_excel(IN_EXCEL, sheet_name="National_Sunsetting")
uni_all = pd.read_excel(IN_EXCEL, sheet_name="Uni_All")
uni_sunset = pd.read_excel(IN_EXCEL, sheet_name="Uni_Sunsetting")

# Ensure numeric year columns
for df in [nat_all, nat_sunset, uni_all, uni_sunset]:
    for y in YEARS:
        if y in df.columns:
            df[y] = pd.to_numeric(df[y], errors="coerce").fillna(0)

# =========================
# 1) TOP NATIONAL DECLINERS (BAR)
# =========================
top_n = 20
nat_sunset["label"] = nat_sunset[COL_CIP].astype(str) + " | " + nat_sunset[COL_AWLEVEL].astype(str)

top_decliners = (
    nat_sunset.sort_values("net_change_24_vs_19")  # most negative first
             .head(top_n)
             .copy()
)

plt.figure()
plt.barh(top_decliners["label"], top_decliners["net_change_24_vs_19"])
plt.title(f"Top {top_n} National Declines (Net change 2024 vs 2019)")
plt.xlabel("Net Change in Completions (2024 - 2019)")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_top_national_decliners_bar.png", dpi=200)
plt.close()

# =========================
# 2) TRAJECTORY LINES FOR TOP DECLINERS
# =========================
# Pick 10 most negative and plot their year-by-year trend
line_n = 10
line_df = top_decliners.head(line_n)

plt.figure()
for _, row in line_df.iterrows():
    yvals = [row[y] for y in YEARS]
    plt.plot(YEARS, yvals, marker="o", label=row["label"])
plt.title(f"Trajectories of Top {line_n} National Decliners (2019–2024)")
plt.xlabel("Year")
plt.ylabel("Completions")
plt.legend(fontsize=7, loc="best")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_top_decliners_trajectories.png", dpi=200)
plt.close()

# =========================
# 3) HEATMAP OF YOY % FOR NATIONAL SUNSETTING
# =========================
# Collect YoY % cols like "20-19%" etc
pct_cols = [c for c in nat_sunset.columns if isinstance(c, str) and c.endswith("%")]
heat = nat_sunset.copy()
heat["label"] = heat[COL_CIP].astype(str) + " | " + heat[COL_AWLEVEL].astype(str)

# Keep top 30 most negative net change for readability
heat = heat.sort_values("net_change_24_vs_19").head(30)
heat_mat = heat[pct_cols].to_numpy(dtype=float)

plt.figure()
plt.imshow(heat_mat, aspect="auto")
plt.yticks(range(len(heat)), heat["label"])
plt.xticks(range(len(pct_cols)), pct_cols, rotation=45, ha="right")
plt.title("YoY % Change Heatmap (Top 30 National Sunsetting Candidates)")
plt.colorbar(label="YoY % Change")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_yoy_pct_heatmap_top30.png", dpi=200)
plt.close()

# =========================
# 4) SUPPLY vs AVAILABILITY SCATTER (2024)
# =========================
# Uses inst_2024 (institutions awarding in 2024) and 2024 completions
if "inst_2024" in nat_all.columns:
    plot_df = nat_all.copy()
    plot_df["is_sunsetting"] = plot_df.get("sunsetting_candidate", False).astype(int)

    plt.figure()
    plt.scatter(plot_df["inst_2024"], plot_df[2024])
    plt.title("2024 Completions vs Institutions Awarding (National)")
    plt.xlabel("Institutions awarding in 2024 (inst_2024)")
    plt.ylabel("Total completions in 2024")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_supply_vs_availability_scatter_2024.png", dpi=200)
    plt.close()

# =========================
# 5) UNIVERSITY RISK: COUNT SUNSETTING PROGRAMS PER UNITID
# =========================
uni_counts = (
    uni_sunset.groupby(COL_UNITID, as_index=False)
             .size()
             .rename(columns={"size": "sunsetting_program_count"})
             .sort_values("sunsetting_program_count", ascending=False)
             .head(25)
)

plt.figure()
plt.bar(uni_counts[COL_UNITID].astype(str), uni_counts["sunsetting_program_count"])
plt.title("Top 25 Institutions by Count of Sunsetting Candidates")
plt.xlabel("UNITID")
plt.ylabel("# Sunsetting program candidates")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(OUT_DIR / "05_top_institutions_sunsetting_counts.png", dpi=200)
plt.close()

print(f"Saved charts in: {OUT_DIR.resolve()}")


# %%
