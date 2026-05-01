from __future__ import annotations

import io
import re
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
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
COL_TOTAL = "CTOTALT"
COL_MAJORNUM = "MAJORNUM"

YEARS = list(range(2019, 2025))

OUT_XLSX = "data_ipeds_bls/cip_grouped_awlevel_yoy_students_2019_2024.xlsx"
SCATTER_PATH = "data_ipeds_bls/sunset_matrix_scatter.png"
HEATMAP_PATH = "data_ipeds_bls/sunset_matrix_heatmap.png"
HIGHLIGHT_HEATMAP_PATH = "data_ipeds_bls/sunset_matrix_heatmap_large_declining.png"
BLS_EXCEL_PATH = "data_ipeds_bls/bls_correlation_analysis.xlsx"

# -------------------------
# Cleanup controls
# -------------------------
KEEP_PRIMARY_MAJOR_ONLY = True
PRIMARY_MAJOR_VALUE = 1
MIN_BASELINE_FOR_LABELS = 20
DROP_TINY_ROWS_FOR_PLOTS = True
EXCLUDE_AWLEVEL_01 = False
REMOVE_AWLEVEL_1_AND_2 = True
AWLEVEL_REMOVE_SET = {"01", "1", "02", "2"}
EXCLUDE_CIP2_99_FROM_DECLINE_ANALYSIS = True
CIP2_EXCLUDED_FOR_DECLINE = "99"
CLIP_PCT_FOR_PLOTS = True
PCT_CLIP_LOW, PCT_CLIP_HIGH = -100, 100

# -------------------------
# Thresholding controls
# -------------------------
USE_WEIGHTED_THRESHOLDS = True
WINSOR_LOW, WINSOR_HIGH = -100.0, 300.0
Z_HIGH_RISK = -2.0
Z_MODERATE = -1.0

# -------------------------
# BLS sources
# -------------------------
BLS_TIMEOUT = 60
BLS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
        "*/*;q=0.8"
    ),
    "Referer": "https://www.bls.gov/",
}
BLS_PROJ_HTML = "https://www.bls.gov/emp/tables/occupational-projections-and-characteristics.htm"
BLS_OEWS_TABLES_PAGE = "https://www.bls.gov/oes/tables.htm"

# ============================================================
# CIP2 / AWLEVEL LABELS
# ============================================================
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

AWLEVEL_TO_NAME = {
    "01": "Award <1 academic year",
    "1": "Award <1 academic year",
    "02": "Award 1 to <4 academic years",
    "2": "Award 1 to <4 academic years",
    "03": "Associate's degree",
    "3": "Associate's degree",
    "04": "Award < Bachelor's",
    "4": "Award < Bachelor's",
    "05": "Bachelor's degree",
    "5": "Bachelor's degree",
    "06": "Postbaccalaureate certificate",
    "6": "Postbaccalaureate certificate",
    "07": "Master's degree",
    "7": "Master's degree",
    "08": "Post-master's certificate",
    "8": "Post-master's certificate",
    "09": "Doctor's degree",
    "9": "Doctor's degree",
    "17": "First professional degree",
    "18": "Graduate/Professional certificate (other)",
    "19": "Graduate/Professional certificate (other)",
    "20": "Graduate/Professional certificate (other)",
    "21": "Graduate/Professional certificate (other)",
}


def cip2_name(cip2: str) -> str:
    return CIP2_TO_NAME.get(str(cip2).zfill(2), "Unknown/Other")


def awlevel_name(code: str) -> str:
    return AWLEVEL_TO_NAME.get(str(code).strip(), "Unknown/Other")


# ============================================================
# HELPERS
# ============================================================
def extract_cip2_series(cip_series: pd.Series) -> pd.Series:
    return cip_series.astype("string").str.strip().str.extract(r"^(\d{2})", expand=False)


def weighted_mean_std(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mean = np.sum(w * x) / np.sum(w)
    var = np.sum(w * (x - mean) ** 2) / np.sum(w)
    return float(mean), float(np.sqrt(var))


def _clean_soc(code: str) -> str:
    if pd.isna(code):
        return np.nan
    s = str(code).strip()
    m = re.search(r"(\d{2}-\d{4})", s)
    return m.group(1) if m else s


def _weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))


def _award_bucket_from_awlevel(aw: str) -> str:
    s = str(aw).strip().zfill(2)
    if s in {"03", "04"}:
        return "Associate_or_sub_bachelor"
    if s == "05":
        return "Bachelors"
    if s == "06":
        return "Postbacc_certificate"
    if s == "07":
        return "Masters"
    if s == "08":
        return "Postmasters_certificate"
    if s in {"09", "17"}:
        return "Doctoral_or_professional"
    if s in {"18", "19", "20", "21"}:
        return "Graduate_certificate_other"
    return "Other"


def _bls_education_bucket(entry_edu: str) -> str:
    s = str(entry_edu).strip().lower()
    if "associate" in s or "postsecondary nondegree award" in s:
        return "Associate_or_sub_bachelor"
    if "bachelor" in s:
        return "Bachelors"
    if "master" in s:
        return "Masters"
    if "doctoral" in s or "professional" in s:
        return "Doctoral_or_professional"
    return "Other"


# ============================================================
# THRESHOLD LABELING
# ============================================================
def apply_stat_threshold_labels(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()

    out["net_pct_winsor"] = pd.to_numeric(
        out["net_pct_change_2019_2024"], errors="coerce"
    ).clip(WINSOR_LOW, WINSOR_HIGH)

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

    print("\n[Threshold Model] Weighted + winsorized distribution (baseline-weighted)")
    print(f"  Winsor range: [{WINSOR_LOW}, {WINSOR_HIGH}] (used only to estimate μ,σ)")
    print(f"  Weighted mean (μ): {mu:.2f}")
    print(f"  Weighted std  (σ): {sigma:.2f}")
    print(f"  High Risk cutoff (z <= {Z_HIGH_RISK}): net_pct ≲ {mu + Z_HIGH_RISK * sigma:.2f}")
    print(f"  Moderate cutoff (z <= {Z_MODERATE}): net_pct ≲ {mu + Z_MODERATE * sigma:.2f}")

    return out


# ============================================================
# CIP ↔ SOC CROSSWALK
# ============================================================
def load_cip2_to_soc_map(
    xlsx_path: Union[str, Path] = "CIP2020_SOC2018_Crosswalk.xlsx",
    sheet: str = "CIP-SOC",
) -> Dict[str, List[str]]:
    path = Path(xlsx_path)
    if not path.exists():
        print(f"[WARN] crosswalk file not found ({path}); BLS analysis will be empty.")
        return {}

    df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    for col in ("CIP2020Code", "SOC2018Code"):
        if col not in df.columns:
            raise ValueError(f"expected column {col} in {xlsx_path}")

    df["CIP2"] = df["CIP2020Code"].str.extract(r"^(\d{2})", expand=False)
    df["SOC2018Code"] = df["SOC2018Code"].map(_clean_soc)
    df = df.dropna(subset=["CIP2", "SOC2018Code"])

    return (
        df.groupby("CIP2")["SOC2018Code"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )


def expand_cip2_soc_map(cip2_to_soc_mapping: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for cip2, socs in cip2_to_soc_mapping.items():
        for soc in socs:
            soc_clean = _clean_soc(soc)
            if pd.notna(soc_clean):
                rows.append({"CIP2": str(cip2).zfill(2), "SOC": soc_clean})
    if not rows:
        return pd.DataFrame(columns=["CIP2", "SOC"])
    return pd.DataFrame(rows).drop_duplicates()


CIP2_TO_SOC_MAPPING = load_cip2_to_soc_map("CIP2020_SOC2018_Crosswalk.xlsx", "CIP-SOC")


# ============================================================
# BLS PROJECTIONS PIPELINE
# ============================================================
def _normalize_projection_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        lc = c.lower()
        if "matrix code" in lc or lc in {"code", "occupation code"} or lc.endswith("code"):
            rename[c] = "SOC"
        elif "matrix title" in lc or lc in {"title", "occupation title"} or lc.endswith("title"):
            rename[c] = "SOC_Title"
        elif "occupation type" in lc:
            rename[c] = "Occupation_Type"
        elif "employment, 2024" in lc:
            rename[c] = "BLS_Base_Employment"
        elif "employment, 2034" in lc:
            rename[c] = "BLS_Projected_Employment"
        elif "employment change, numeric" in lc:
            rename[c] = "BLS_Employment_Change_Numeric"
        elif "employment change, percent" in lc:
            rename[c] = "BLS_Projected_Pct_Change"
        elif "occupational openings" in lc:
            rename[c] = "BLS_Annual_Openings"
        elif "median annual wage" in lc:
            rename[c] = "Median_Wage"
        elif "typical education needed for entry" in lc:
            rename[c] = "BLS_Typical_Education"
        elif "work experience in a related occupation" in lc:
            rename[c] = "Work_Experience"
        elif "on-the-job training" in lc:
            rename[c] = "On_The_Job_Training"

    df = df.rename(columns=rename)

    required = {
        "SOC",
        "SOC_Title",
        "BLS_Base_Employment",
        "BLS_Projected_Employment",
        "BLS_Projected_Pct_Change",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after normalization: {sorted(missing)}")

    df["SOC"] = df["SOC"].map(_clean_soc)

    numeric_cols = [
        "BLS_Base_Employment",
        "BLS_Projected_Employment",
        "BLS_Employment_Change_Numeric",
        "BLS_Projected_Pct_Change",
        "BLS_Annual_Openings",
        "Median_Wage",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("—", "", regex=False)
                .str.replace("--", "", regex=False)
                .str.strip()
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "BLS_Typical_Education" not in df.columns:
        df["BLS_Typical_Education"] = np.nan

    df["BLS_Edu_Bucket"] = df["BLS_Typical_Education"].map(_bls_education_bucket)
    df = df[df["SOC"].astype(str).str.match(r"^\d{2}-\d{4}$", na=False)].copy()
    df = df.drop_duplicates(subset=["SOC"])
    return df


def load_bls_employment_projections_html() -> pd.DataFrame:
    resp = requests.get(BLS_PROJ_HTML, headers=BLS_HEADERS, timeout=BLS_TIMEOUT)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    if not tables:
        raise ValueError("No HTML tables found on BLS projections page.")

    chosen = None
    best_score = -1
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        score = 0
        if any("code" in c for c in cols):
            score += 1
        if any("title" in c for c in cols):
            score += 1
        if any("employment change, percent" in c for c in cols):
            score += 2
        if any("typical education needed for entry" in c for c in cols):
            score += 1
        if any("employment, 2024" in c for c in cols):
            score += 1
        if any("employment, 2034" in c for c in cols):
            score += 1
        if score > best_score:
            best_score = score
            chosen = t

    if chosen is None:
        raise ValueError("Could not identify the occupational projections table from HTML.")

    return _normalize_projection_columns(chosen)


def load_oews_national_files(years: List[int]) -> pd.DataFrame:
    """
    Load OEWS national files for 2019-2024 from the official BLS national ZIP downloads.

    Official pattern resolved from the OEWS tables page:
      2024 -> https://www.bls.gov/oes/special-requests/oesm24nat.zip
      2023 -> https://www.bls.gov/oes/special-requests/oesm23nat.zip
      ...
      2019 -> https://www.bls.gov/oes/special-requests/oesm19nat.zip
    """
    records = []

    for year in years:
        yy = str(year)[-2:]
        zip_url = f"https://www.bls.gov/oes/special-requests/oesm{yy}nat.zip"

        try:
            r = requests.get(zip_url, headers=BLS_HEADERS, timeout=BLS_TIMEOUT)
            r.raise_for_status()

            zf = zipfile.ZipFile(io.BytesIO(r.content))
            names = zf.namelist()

            # Prefer xlsx, but accept xls/csv if BLS changes packaging
            target = None
            for ext in (".xlsx", ".xls", ".csv"):
                target = next((n for n in names if n.lower().endswith(ext)), None)
                if target is not None:
                    break

            if target is None:
                print(f"[WARN] No spreadsheet file found inside {zip_url}. Contents: {names}")
                continue

            with zf.open(target) as f:
                if target.lower().endswith(".csv"):
                    raw = pd.read_csv(f, low_memory=False)
                else:
                    xls_bytes = io.BytesIO(f.read())
                    xls = pd.ExcelFile(xls_bytes)

                    raw = None
                    for sheet in xls.sheet_names:
                        tmp = pd.read_excel(xls, sheet_name=sheet)
                        cols = [str(c).strip().lower() for c in tmp.columns]

                        has_occ = any(c in cols for c in ["occ_code", "occupation code"])
                        has_emp = any(c in cols for c in ["tot_emp", "employment"])
                        if has_occ and has_emp:
                            raw = tmp
                            break

                    if raw is None:
                        # fallback to first sheet
                        raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

            raw.columns = [str(c).strip() for c in raw.columns]

            col_soc = next((c for c in raw.columns if c.lower() in {"occ_code", "occupation code"}), None)
            if col_soc is None:
                col_soc = next((c for c in raw.columns if "code" in c.lower()), None)

            col_title = next((c for c in raw.columns if c.lower() in {"occ_title", "occupation title"}), None)
            if col_title is None:
                col_title = next((c for c in raw.columns if "title" in c.lower()), None)

            col_emp = next((c for c in raw.columns if c.lower() in {"tot_emp", "employment"}), None)
            if col_emp is None:
                col_emp = next((c for c in raw.columns if "tot_emp" in c.lower() or "employment" in c.lower()), None)

            if col_soc is None or col_emp is None:
                print(f"[WARN] Could not identify SOC/employment columns for {year}. Columns: {raw.columns.tolist()}")
                continue

            tmp = raw[[c for c in [col_soc, col_title, col_emp] if c is not None]].copy()

            if len(tmp.columns) == 2:
                tmp.columns = ["SOC", "OEWS_Employment"]
                tmp["SOC_Title"] = np.nan
            else:
                tmp.columns = ["SOC", "SOC_Title", "OEWS_Employment"]

            tmp["SOC"] = tmp["SOC"].map(_clean_soc)
            tmp["OEWS_Employment"] = (
                tmp["OEWS_Employment"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("—", "", regex=False)
                .str.replace("--", "", regex=False)
                .str.strip()
            )
            tmp["OEWS_Employment"] = pd.to_numeric(tmp["OEWS_Employment"], errors="coerce")
            tmp["YEAR"] = year

            tmp = tmp[tmp["SOC"].astype(str).str.match(r"^\d{2}-\d{4}$", na=False)].copy()
            tmp = tmp.dropna(subset=["OEWS_Employment"])

            if len(tmp) == 0:
                print(f"[WARN] No usable OEWS rows after cleaning for {year}.")
                continue

            records.append(tmp[["SOC", "SOC_Title", "YEAR", "OEWS_Employment"]])
            print(f"[OEWS] Loaded {year}: {len(tmp):,} rows from {zip_url}")

        except Exception as e:
            print(f"[WARN] Could not load OEWS year {year} from {zip_url}: {e}")

    if not records:
        raise ValueError("Could not load OEWS historical national files from official BLS ZIP downloads.")

    out = pd.concat(records, ignore_index=True)
    out = out.groupby(["SOC", "SOC_Title", "YEAR"], as_index=False)["OEWS_Employment"].mean()
    return out

# ============================================================
# BLS ANALYSIS
# ============================================================
def correlate_cip_to_bls(
    wide: pd.DataFrame,
    bls_proj: pd.DataFrame,
    cip2_to_soc_mapping: Dict[str, List[str]],
) -> pd.DataFrame:
    map_df = expand_cip2_soc_map(cip2_to_soc_mapping)
    if map_df.empty:
        return pd.DataFrame()

    prog = (
        wide.groupby(["CIP2", "CIP2_Name"], as_index=False)
        .agg(
            Program_2019=(2019, "sum"),
            Program_2024=(2024, "sum"),
            Program_Baseline=("baseline_avg_2019_2021", "sum"),
        )
    )
    prog["Program_Net_Pct_Change"] = np.where(
        prog["Program_2019"] > 0,
        ((prog["Program_2024"] - prog["Program_2019"]) / prog["Program_2019"]) * 100,
        np.nan,
    )

    merged = map_df.merge(bls_proj, on="SOC", how="left")
    bls_agg = (
        merged.groupby("CIP2", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "Mapped_SOC_Count": g["SOC"].nunique(),
                    "BLS_Occupational_Growth": _weighted_mean(
                        g["BLS_Projected_Pct_Change"], g["BLS_Base_Employment"]
                    ),
                    "BLS_Base_Employment_Mapped": g["BLS_Base_Employment"].sum(min_count=1),
                    "BLS_Annual_Openings_Mapped": g["BLS_Annual_Openings"].sum(min_count=1),
                }
            )
        )
        .reset_index(drop=True)
    )

    out = prog.merge(bls_agg, on="CIP2", how="left")
    valid = out.dropna(subset=["Program_Net_Pct_Change", "BLS_Occupational_Growth"]).copy()

    if len(valid) >= 3:
        r_p, p_p = pearsonr(valid["Program_Net_Pct_Change"], valid["BLS_Occupational_Growth"])
        r_s, p_s = spearmanr(valid["Program_Net_Pct_Change"], valid["BLS_Occupational_Growth"])
    else:
        r_p = p_p = r_s = p_s = np.nan

    out["Global_Pearson_r"] = r_p
    out["Global_Pearson_p"] = p_p
    out["Global_Spearman_rho"] = r_s
    out["Global_Spearman_p"] = p_s
    out["Correlation_Direction"] = np.where(
        np.sign(out["Program_Net_Pct_Change"]) == np.sign(out["BLS_Occupational_Growth"]),
        "Aligned",
        "Misaligned",
    )
    return out


def correlate_cip_awlevel_to_bls(
    wide: pd.DataFrame,
    bls_proj: pd.DataFrame,
    cip2_to_soc_mapping: Dict[str, List[str]],
) -> pd.DataFrame:
    map_df = expand_cip2_soc_map(cip2_to_soc_mapping)
    if map_df.empty:
        return pd.DataFrame()

    base = wide.copy()
    base["Award_Bucket"] = base[COL_AWLEVEL].map(_award_bucket_from_awlevel)
    merged_soc = map_df.merge(bls_proj, on="SOC", how="left")
    yoy_pct_cols = [c for c in base.columns if isinstance(c, str) and c.endswith("%")]

    rows = []
    for _, r in base.iterrows():
        cip2 = r["CIP2"]
        award_bucket = r["Award_Bucket"]
        sub_soc = merged_soc[
            (merged_soc["CIP2"] == cip2) & (merged_soc["BLS_Edu_Bucket"] == award_bucket)
        ].copy()

        bls_growth = _weighted_mean(sub_soc["BLS_Projected_Pct_Change"], sub_soc["BLS_Base_Employment"])
        bls_base_emp = sub_soc["BLS_Base_Employment"].sum(min_count=1)
        bls_openings = sub_soc["BLS_Annual_Openings"].sum(min_count=1)

        program_net = r["net_pct_change_2019_2024"]
        gap = program_net - bls_growth if pd.notna(program_net) and pd.notna(bls_growth) else np.nan

        prog_sd = pd.to_numeric(pd.Series(r[yoy_pct_cols]), errors="coerce").std()
        bls_sd = pd.to_numeric(sub_soc["BLS_Projected_Pct_Change"], errors="coerce").std()
        denom = np.sqrt(np.nanmean([prog_sd ** 2, bls_sd ** 2])) if not (pd.isna(prog_sd) and pd.isna(bls_sd)) else np.nan
        cohens_d = abs(gap) / denom if pd.notna(gap) and pd.notna(denom) and denom > 0 else np.nan

        if pd.isna(cohens_d):
            alignment = "Insufficient data"
        elif cohens_d < 0.2:
            alignment = "Strong"
        elif cohens_d < 0.5:
            alignment = "Moderate"
        elif cohens_d < 0.8:
            alignment = "Weak"
        else:
            alignment = "Misaligned"

        rows.append(
            {
                "CIP2": cip2,
                "CIP2_Name": r["CIP2_Name"],
                "AWLEVEL": r[COL_AWLEVEL],
                "AWLEVEL_Name": r["AWLEVEL_Name"],
                "Award_Bucket": award_bucket,
                "Program_Net_Pct_Change": program_net,
                "BLS_Growth_by_Degree": bls_growth,
                "BLS_Base_Employment_Mapped": bls_base_emp,
                "BLS_Annual_Openings_Mapped": bls_openings,
                "Gap_Program_minus_BLS": gap,
                "Cohens_d": cohens_d,
                "Alignment": alignment,
            }
        )

    out = pd.DataFrame(rows)
    valid = out.dropna(subset=["Program_Net_Pct_Change", "BLS_Growth_by_Degree"]).copy()

    if len(valid) >= 3:
        r_p, p_p = pearsonr(valid["Program_Net_Pct_Change"], valid["BLS_Growth_by_Degree"])
        r_s, p_s = spearmanr(valid["Program_Net_Pct_Change"], valid["BLS_Growth_by_Degree"])
    else:
        r_p = p_p = r_s = p_s = np.nan

    out["Global_Pearson_r"] = r_p
    out["Global_Pearson_p"] = p_p
    out["Global_Spearman_rho"] = r_s
    out["Global_Spearman_p"] = p_s
    return out


def identify_mismatches(cip_bls_corr: pd.DataFrame) -> pd.DataFrame:
    """
    Find program areas where education supply does not match labor demand.

    This version is more robust than the earlier strict studentized-residual
    approach and avoids returning an empty sheet in most realistic cases.
    It uses:
      1) weighted regression to estimate expected program growth from BLS growth
      2) standardized residuals
      3) a softer threshold
      4) fallback to top absolute deviations if nothing crosses the threshold
    """
    df = cip_bls_corr.dropna(
        subset=["Program_Net_Pct_Change", "BLS_Occupational_Growth", "Program_Baseline"]
    ).copy()

    if len(df) < 5:
        return pd.DataFrame()

    X = sm.add_constant(df["BLS_Occupational_Growth"]).astype(float)
    y = df["Program_Net_Pct_Change"].astype(float)
    w = np.maximum(df["Program_Baseline"].astype(float), 1.0)

    # Weighted regression so larger programs matter more
    model = sm.WLS(y, X, weights=w).fit()

    df["Predicted_Program_Change"] = model.predict(X)
    df["Residual"] = y - df["Predicted_Program_Change"]

    # Standardize residuals manually
    resid_std = df["Residual"].std(ddof=1)
    if pd.isna(resid_std) or resid_std == 0:
        df["Studentized_Residual"] = np.nan
    else:
        df["Studentized_Residual"] = df["Residual"] / resid_std

    # Softer cutoff than ±2 so useful mismatches are surfaced
    threshold = 1.25

    df["Mismatch_Type"] = np.where(
        df["Studentized_Residual"] <= -threshold,
        "Under-supplying vs demand",
        np.where(
            df["Studentized_Residual"] >= threshold,
            "Over-supplying vs demand",
            "Within expected range",
        ),
    )

    out = df[df["Mismatch_Type"] != "Within expected range"].copy()

    # Fallback: if no rows pass the threshold, still return the largest gaps
    if out.empty:
        out = df.loc[df["Residual"].abs().sort_values(ascending=False).head(10).index].copy()
        out["Mismatch_Type"] = np.where(
            out["Residual"] < 0,
            "Under-supplying vs demand (largest gap)",
            "Over-supplying vs demand (largest gap)",
        )

    out = out.sort_values("Residual")
    out["Regression_R2"] = model.rsquared
    out["Regression_p_BLS"] = model.pvalues.get("BLS_Occupational_Growth", np.nan)

    return out

def analyze_lag_time(
    wide: pd.DataFrame,
    oews_hist: pd.DataFrame,
    cip2_to_soc_mapping: Dict[str, List[str]],
    max_lag_years: int = 3,
) -> pd.DataFrame:
    map_df = expand_cip2_soc_map(cip2_to_soc_mapping)
    if map_df.empty:
        return pd.DataFrame()

    prog = (
        wide.groupby(["CIP2", "CIP2_Name"], as_index=False)[[2019, 2020, 2021, 2022, 2023, 2024]]
        .sum()
    )

    prog_yoy_rows = []
    for _, r in prog.iterrows():
        vals = pd.Series({2019: r[2019], 2020: r[2020], 2021: r[2021], 2022: r[2022], 2023: r[2023], 2024: r[2024]})
        yoy = vals.pct_change() * 100
        for year, pct in yoy.items():
            if year == 2019:
                continue
            prog_yoy_rows.append(
                {
                    "CIP2": r["CIP2"],
                    "CIP2_Name": r["CIP2_Name"],
                    "YEAR": year,
                    "IPEDS_YoY_Pct": pct,
                }
            )
    prog_yoy = pd.DataFrame(prog_yoy_rows)

    merged = map_df.merge(oews_hist, on="SOC", how="left")
    cip_oews_year = (
        merged.groupby(["CIP2", "YEAR"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "OEWS_Employment": _weighted_mean(
                        g["OEWS_Employment"], pd.Series(np.ones(len(g)), index=g.index)
                    )
                }
            )
        )
        .reset_index(drop=True)
    )
    cip_oews_year = cip_oews_year.sort_values(["CIP2", "YEAR"])
    cip_oews_year["OEWS_YoY_Pct"] = cip_oews_year.groupby("CIP2")["OEWS_Employment"].pct_change() * 100

    out_rows = []
    for cip2, g_prog in prog_yoy.groupby("CIP2"):
        g_oews = cip_oews_year[cip_oews_year["CIP2"] == cip2].copy()
        if g_oews.empty:
            continue

        best = None
        for lag in range(0, max_lag_years + 1):
            tmp = g_prog.copy()
            tmp["OEWS_YEAR"] = tmp["YEAR"] - lag
            merged_lag = tmp.merge(
                g_oews[["YEAR", "OEWS_YoY_Pct"]].rename(columns={"YEAR": "OEWS_YEAR"}),
                on="OEWS_YEAR",
                how="left",
            ).dropna(subset=["IPEDS_YoY_Pct", "OEWS_YoY_Pct"])

            if len(merged_lag) < 3:
                continue

            r, p = pearsonr(merged_lag["IPEDS_YoY_Pct"], merged_lag["OEWS_YoY_Pct"])
            if best is None or abs(r) > abs(best["Lag_Correlation_Strength"]):
                best = {
                    "Lag_Months": lag * 12,
                    "Lag_Correlation_Strength": r,
                    "Lag_p_value": p,
                    "N_Overlap": len(merged_lag),
                    "Recent_YoY_Change_2023_2024": merged_lag.loc[
                        merged_lag["YEAR"] == 2024, "IPEDS_YoY_Pct"
                    ].mean(),
                    "Lagged_BLS_Growth": merged_lag.loc[
                        merged_lag["YEAR"] == 2024, "OEWS_YoY_Pct"
                    ].mean(),
                }

        if best is None:
            out_rows.append(
                {
                    "CIP2": cip2,
                    "CIP2_Name": g_prog["CIP2_Name"].iloc[0],
                    "Recent_YoY_Change_2023_2024": np.nan,
                    "Lagged_BLS_Growth": np.nan,
                    "Estimated_Lag_Response": "Insufficient data",
                    "Lag_Months": np.nan,
                    "Lag_Correlation_Strength": np.nan,
                    "Lag_p_value": np.nan,
                    "N_Overlap": 0,
                }
            )
        else:
            if abs(best["Lag_Correlation_Strength"]) >= 0.6 and best["Lag_p_value"] < 0.05:
                response = "Responsive"
            elif abs(best["Lag_Correlation_Strength"]) >= 0.3:
                response = "Partially Responsive"
            else:
                response = "Weak/Delayed"

            out_rows.append(
                {
                    "CIP2": cip2,
                    "CIP2_Name": g_prog["CIP2_Name"].iloc[0],
                    "Recent_YoY_Change_2023_2024": best["Recent_YoY_Change_2023_2024"],
                    "Lagged_BLS_Growth": best["Lagged_BLS_Growth"],
                    "Estimated_Lag_Response": response,
                    "Lag_Months": best["Lag_Months"],
                    "Lag_Correlation_Strength": best["Lag_Correlation_Strength"],
                    "Lag_p_value": best["Lag_p_value"],
                    "N_Overlap": best["N_Overlap"],
                }
            )

    return pd.DataFrame(out_rows)


def run_bls_analysis(wide: pd.DataFrame, cip2_to_soc_mapping: Dict[str, List[str]]):
    print("[BLS] Loading projections from official HTML table...")
    bls_proj = load_bls_employment_projections_html()

    print("[BLS] Loading OEWS historical files for lag analysis...")
    oews_hist = load_oews_national_files([2019, 2020, 2021, 2022, 2023, 2024])

    print("[BLS] CIP2-level correlation...")
    cip_bls_corr = correlate_cip_to_bls(wide, bls_proj, cip2_to_soc_mapping)

    print("[BLS] CIP2 x AWLEVEL correlation...")
    cip_awlevel_bls_corr = correlate_cip_awlevel_to_bls(wide, bls_proj, cip2_to_soc_mapping)

    print("[BLS] Residual-based mismatch detection...")
    mismatches = identify_mismatches(cip_bls_corr) if not cip_bls_corr.empty else pd.DataFrame()

    print("[BLS] Lag analysis...")
    lag_analysis = analyze_lag_time(wide, oews_hist, cip2_to_soc_mapping)

    return bls_proj, cip_bls_corr, cip_awlevel_bls_corr, mismatches, lag_analysis


# ============================================================
# OPTIONAL BLS VISUALS
# ============================================================
def _create_bls_visualizations(
    cip_bls_corr: pd.DataFrame,
    cip_awlevel_bls_corr: pd.DataFrame,
    mismatches: pd.DataFrame,
    lag_analysis: pd.DataFrame,
):
    out_dir = Path("data_ipeds_bls")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cip_bls_corr.empty:
        df = cip_bls_corr.dropna(subset=["BLS_Occupational_Growth", "Program_Net_Pct_Change"]).copy()
        if len(df) > 0:
            plt.figure(figsize=(12, 8))
            aligned = df["Correlation_Direction"] == "Aligned"
            plt.scatter(
                df.loc[aligned, "BLS_Occupational_Growth"],
                df.loc[aligned, "Program_Net_Pct_Change"],
                alpha=0.75,
                label="Aligned",
            )
            plt.scatter(
                df.loc[~aligned, "BLS_Occupational_Growth"],
                df.loc[~aligned, "Program_Net_Pct_Change"],
                alpha=0.75,
                label="Misaligned",
            )
            lims = [
                np.nanmin([df["BLS_Occupational_Growth"].min(), df["Program_Net_Pct_Change"].min()]),
                np.nanmax([df["BLS_Occupational_Growth"].max(), df["Program_Net_Pct_Change"].max()]),
            ]
            plt.plot(lims, lims, linestyle="--")
            plt.xlabel("BLS Occupational Growth (%)")
            plt.ylabel("IPEDS Program Net % Change (%)")
            plt.title("IPEDS vs BLS Alignment Scatter")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "bls_alignment_scatter.png", dpi=300)
            plt.close()

    if not mismatches.empty:
        top = mismatches.copy().sort_values("Residual").head(10)
        plt.figure(figsize=(12, 8))
        plt.barh(top["CIP2_Name"], top["Residual"])
        plt.xlabel("Residual: Program Change - Predicted Change")
        plt.ylabel("CIP2")
        plt.title("Top Mismatch Gaps")
        plt.tight_layout()
        plt.savefig(out_dir / "bls_mismatch_gaps.png", dpi=300)
        plt.close()

    if not cip_awlevel_bls_corr.empty:
        heat = cip_awlevel_bls_corr.pivot_table(
            index="CIP2_Name", columns="AWLEVEL_Name", values="Cohens_d", aggfunc="median"
        )
        if not heat.empty:
            plt.figure(figsize=(14, 10))
            plt.imshow(heat.fillna(0).values, aspect="auto")
            plt.colorbar(label="Cohen's d")
            plt.title("CIP2 × Degree Level Misalignment Heatmap")
            plt.xlabel("Degree Level")
            plt.ylabel("CIP2")
            plt.xticks(range(len(heat.columns)), heat.columns, rotation=90)
            plt.yticks(range(len(heat.index)), heat.index)
            plt.tight_layout()
            plt.savefig(out_dir / "bls_cohens_d_heatmap.png", dpi=300)
            plt.close()

    if not lag_analysis.empty:
        lag_counts = lag_analysis["Estimated_Lag_Response"].value_counts(dropna=False)
        if len(lag_counts) > 0:
            plt.figure(figsize=(8, 8))
            plt.pie(lag_counts.values, labels=lag_counts.index, autopct="%1.1f%%")
            plt.title("Lag Response Classification")
            plt.tight_layout()
            plt.savefig(out_dir / "bls_lag_response_pie.png", dpi=300)
            plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    Path("data_ipeds_bls").mkdir(parents=True, exist_ok=True)

    frames = []
    print("[LOAD] Reading IPEDS A files...")
    for year in YEARS:
        path = Path(FILES_BY_YEAR[year])
        if not path.exists():
            raise FileNotFoundError(f"Missing file for {year}: {path}")

        df = pd.read_csv(path, low_memory=False)
        required = [COL_CIP, COL_AWLEVEL, COL_TOTAL]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        df[COL_CIP] = df[COL_CIP].astype("string").str.strip()
        df[COL_AWLEVEL] = df[COL_AWLEVEL].astype("string").str.strip()
        df[COL_TOTAL] = pd.to_numeric(df[COL_TOTAL], errors="coerce").fillna(0)

        if KEEP_PRIMARY_MAJOR_ONLY and COL_MAJORNUM in df.columns:
            df[COL_MAJORNUM] = pd.to_numeric(df[COL_MAJORNUM], errors="coerce")
            df = df[df[COL_MAJORNUM] == PRIMARY_MAJOR_VALUE]

        if EXCLUDE_AWLEVEL_01:
            df = df[df[COL_AWLEVEL] != "01"]

        if REMOVE_AWLEVEL_1_AND_2:
            df = df[~df[COL_AWLEVEL].isin(AWLEVEL_REMOVE_SET)].copy()

        df["CIP2"] = extract_cip2_series(df[COL_CIP])
        df = df[df["CIP2"].notna()].copy()
        df["YEAR"] = year
        frames.append(df[["CIP2", COL_AWLEVEL, COL_TOTAL, "YEAR"]])

        uniq_aw = sorted(df[COL_AWLEVEL].dropna().unique().tolist())
        print(f"  - {year}: rows={len(df):,} | unique AWLEVEL={uniq_aw}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n[COMBINE] Combined rows (all years): {len(combined):,}")

    agg = (
        combined.groupby(["CIP2", COL_AWLEVEL, "YEAR"], as_index=False)
        .agg(total_completed=(COL_TOTAL, "sum"))
    )

    wide = (
        agg.pivot_table(
            index=["CIP2", COL_AWLEVEL],
            columns="YEAR",
            values="total_completed",
            aggfunc="sum",
        )
        .reindex(columns=YEARS)
        .fillna(0)
        .reset_index()
    )

    wide["CIP2_Name"] = wide["CIP2"].apply(cip2_name)
    wide["AWLEVEL_Name"] = wide[COL_AWLEVEL].apply(awlevel_name)

    for y in YEARS:
        wide[y] = pd.to_numeric(wide[y], errors="coerce").fillna(0)

    yoy_count_cols = []
    yoy_pct_cols = []
    for i in range(1, len(YEARS)):
        y_prev, y_cur = YEARS[i - 1], YEARS[i]
        diff_col = f"{y_cur}-{str(y_prev)[-2:]}"
        pct_col = f"{diff_col}%"
        wide[diff_col] = wide[y_cur] - wide[y_prev]
        wide[pct_col] = (wide[diff_col] / wide[y_prev].replace({0: np.nan})) * 100
        yoy_count_cols.append(diff_col)
        yoy_pct_cols.append(pct_col)

    wide[yoy_pct_cols] = wide[yoy_pct_cols].apply(pd.to_numeric, errors="coerce").round(2)

    wide["baseline_avg_2019_2021"] = wide[[2019, 2020, 2021]].mean(axis=1)
    wide["net_pct_change_2019_2024"] = (
        (wide[2024] - wide[2019]) / wide[2019].replace({0: np.nan})
    ) * 100
    wide["net_pct_change_2019_2024"] = pd.to_numeric(wide["net_pct_change_2019_2024"], errors="coerce")
    wide["unstable_base_2019_zero"] = wide[2019] == 0

    if USE_WEIGHTED_THRESHOLDS:
        if EXCLUDE_CIP2_99_FROM_DECLINE_ANALYSIS:
            wide_for_labeling = wide[wide["CIP2"] != CIP2_EXCLUDED_FOR_DECLINE].copy()
        else:
            wide_for_labeling = wide.copy()

        wide_for_labeling = apply_stat_threshold_labels(wide_for_labeling)
        wide_for_labeling["sunset_label"] = wide_for_labeling["sunset_label_stat"]

        wide = wide.merge(
            wide_for_labeling[["CIP2", COL_AWLEVEL, "sunset_label_stat", "sunset_label"]],
            on=["CIP2", COL_AWLEVEL],
            how="left",
        )

        if EXCLUDE_CIP2_99_FROM_DECLINE_ANALYSIS:
            wide.loc[wide["CIP2"] == CIP2_EXCLUDED_FOR_DECLINE, "sunset_label"] = "Excluded (CIP2=99)"
    else:
        wide["sunset_label"] = "Unlabeled"

    print("\n[LABELS] Sunset label counts:")
    print(wide["sunset_label"].value_counts(dropna=False).to_string())

    out_cols = (
        ["CIP2", "CIP2_Name", COL_AWLEVEL, "AWLEVEL_Name"]
        + YEARS
        + yoy_count_cols
        + yoy_pct_cols
        + ["baseline_avg_2019_2021", "net_pct_change_2019_2024", "unstable_base_2019_zero", "sunset_label"]
    )
    wide[out_cols].to_excel(OUT_XLSX, index=False)
    print(f"\n[SAVE] Excel saved → {OUT_XLSX}")

    plot_df = wide.copy()
    if EXCLUDE_CIP2_99_FROM_DECLINE_ANALYSIS:
        plot_df = plot_df[plot_df["CIP2"] != CIP2_EXCLUDED_FOR_DECLINE].copy()

    plot_df["baseline_avg_2019_2021"] = pd.to_numeric(plot_df["baseline_avg_2019_2021"], errors="coerce")
    plot_df["net_pct_change_2019_2024"] = pd.to_numeric(plot_df["net_pct_change_2019_2024"], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

    if DROP_TINY_ROWS_FOR_PLOTS:
        plot_df = plot_df[plot_df["baseline_avg_2019_2021"] >= MIN_BASELINE_FOR_LABELS].copy()

    plot_df["net_pct_for_plot"] = plot_df["net_pct_change_2019_2024"]
    if CLIP_PCT_FOR_PLOTS:
        plot_df["net_pct_for_plot"] = plot_df["net_pct_for_plot"].clip(PCT_CLIP_LOW, PCT_CLIP_HIGH)

    plot_df = plot_df.dropna(subset=["baseline_avg_2019_2021", "net_pct_for_plot"]).copy()
    plot_df = plot_df[plot_df["baseline_avg_2019_2021"] > 0].copy()
    size_cut = float(plot_df["baseline_avg_2019_2021"].median()) if len(plot_df) else 0.0

    plt.figure(figsize=(14, 8))
    label_order = ["Growth/Stable", "Moderate", "High Risk", "Unstable/NA", "Excluded (CIP2=99)"]
    for label in label_order:
        sub = plot_df[plot_df["sunset_label"] == label]
        if len(sub) == 0:
            continue
        plt.scatter(
            sub["baseline_avg_2019_2021"].astype(float),
            sub["net_pct_for_plot"].astype(float),
            alpha=0.8,
            label=label,
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
    plt.close()
    print(f"[SAVE] Scatter saved → {SCATTER_PATH}")

    heat_source = wide.copy()
    if EXCLUDE_CIP2_99_FROM_DECLINE_ANALYSIS:
        heat_source = heat_source[heat_source["CIP2"] != CIP2_EXCLUDED_FOR_DECLINE].copy()

    heat_source["baseline_avg_2019_2021"] = pd.to_numeric(heat_source["baseline_avg_2019_2021"], errors="coerce")
    heat_source["net_pct_change_2019_2024"] = pd.to_numeric(heat_source["net_pct_change_2019_2024"], errors="coerce")
    heat_source = heat_source.replace([np.inf, -np.inf], np.nan)

    if DROP_TINY_ROWS_FOR_PLOTS:
        heat_source = heat_source[heat_source["baseline_avg_2019_2021"] >= MIN_BASELINE_FOR_LABELS].copy()

    heat_source["net_pct_for_heat"] = heat_source["net_pct_change_2019_2024"]
    if CLIP_PCT_FOR_PLOTS:
        heat_source["net_pct_for_heat"] = heat_source["net_pct_for_heat"].clip(PCT_CLIP_LOW, PCT_CLIP_HIGH)

    heat = heat_source.pivot_table(
        index="CIP2_Name", columns=COL_AWLEVEL, values="net_pct_for_heat", aggfunc="median"
    ).sort_index()

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
    plt.close()
    print(f"[SAVE] Heatmap saved → {HEATMAP_PATH}")

    quad_df = plot_df.copy()
    quad_df["quadrant"] = np.where(
        quad_df["baseline_avg_2019_2021"] >= size_cut,
        np.where(quad_df["net_pct_for_plot"] < 0, "Large Declining", "Large Growing"),
        np.where(quad_df["net_pct_for_plot"] < 0, "Small Declining", "Small Growing"),
    )

    top_large_declining = (
        quad_df[quad_df["quadrant"] == "Large Declining"]
        .sort_values(by="baseline_avg_2019_2021", ascending=False)
        .head(15)[
            [
                "CIP2",
                "CIP2_Name",
                COL_AWLEVEL,
                "AWLEVEL_Name",
                "baseline_avg_2019_2021",
                "net_pct_change_2019_2024",
                "sunset_label",
            ]
        ]
    )

    print("\n[INSIGHT] Top 15 Large Declining (highest baseline impact):")
    if len(top_large_declining):
        print(top_large_declining.to_string(index=False))
    else:
        print("  None found (after filters).")

    highlight_df = plot_df.copy()
    highlight_df["is_large_declining"] = (
        (highlight_df["baseline_avg_2019_2021"] >= size_cut) & (highlight_df["net_pct_for_plot"] < 0)
    ).astype(int)

    highlight_heat = highlight_df.pivot_table(
        index="CIP2_Name", columns=COL_AWLEVEL, values="is_large_declining", aggfunc="mean"
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
    plt.savefig(HIGHLIGHT_HEATMAP_PATH, dpi=300)
    plt.close()
    print(f"[SAVE] Large Declining heatmap saved → {HIGHLIGHT_HEATMAP_PATH}")

    print("\n[BLS] Running official BLS alignment analysis...")
    bls_proj, cip_bls_corr, cip_awlevel_bls_corr, mismatches, lag_analysis = run_bls_analysis(
        wide=wide,
        cip2_to_soc_mapping=CIP2_TO_SOC_MAPPING,
    )

    print("\n[STEP 1] CIP2 ↔ BLS")
    if cip_bls_corr.empty:
        print("  No CIP2↔SOC mapping available or no usable BLS rows.")
    else:
        print(
            cip_bls_corr[
                [
                    "CIP2",
                    "CIP2_Name",
                    "Program_Net_Pct_Change",
                    "BLS_Occupational_Growth",
                    "Mapped_SOC_Count",
                    "BLS_Annual_Openings_Mapped",
                    "Correlation_Direction",
                    "Global_Pearson_r",
                    "Global_Pearson_p",
                    "Global_Spearman_rho",
                    "Global_Spearman_p",
                ]
            ].to_string(index=False)
        )

    print("\n[STEP 2] CIP2 × AWLEVEL ↔ BLS by degree level")
    if cip_awlevel_bls_corr.empty:
        print("  No usable degree-level alignment rows.")
    else:
        print(
            cip_awlevel_bls_corr[
                [
                    "CIP2_Name",
                    "AWLEVEL_Name",
                    "Award_Bucket",
                    "Program_Net_Pct_Change",
                    "BLS_Growth_by_Degree",
                    "Gap_Program_minus_BLS",
                    "Cohens_d",
                    "Alignment",
                ]
            ].to_string(index=False)
        )

    print("\n[STEP 3] Non-correlating / mismatch instances")
    if mismatches.empty:
        print("  → No strong residual mismatches detected.")
    else:
        print(
            mismatches[
                [
                    "CIP2",
                    "CIP2_Name",
                    "Program_Net_Pct_Change",
                    "BLS_Occupational_Growth",
                    "Predicted_Program_Change",
                    "Residual",
                    "Studentized_Residual",
                    "Mismatch_Type",
                ]
            ].to_string(index=False)
        )

    print("\n[STEP 4] Lag time")
    if lag_analysis.empty:
        print("  No lag analysis results available.")
    else:
        print(
            lag_analysis[
                [
                    "CIP2",
                    "CIP2_Name",
                    "Recent_YoY_Change_2023_2024",
                    "Lagged_BLS_Growth",
                    "Estimated_Lag_Response",
                    "Lag_Months",
                    "Lag_Correlation_Strength",
                    "Lag_p_value",
                    "N_Overlap",
                ]
            ].to_string(index=False)
        )

    with pd.ExcelWriter(BLS_EXCEL_PATH) as writer:
        bls_proj.to_excel(writer, sheet_name="BLS_Projections_Raw", index=False)
        cip_bls_corr.to_excel(writer, sheet_name="CIP2_BLS", index=False)
        cip_awlevel_bls_corr.to_excel(writer, sheet_name="CIP2_AWLEVEL_BLS", index=False)
        mismatches.to_excel(writer, sheet_name="Mismatches", index=False)
        lag_analysis.to_excel(writer, sheet_name="Lag_Analysis", index=False)
    print(f"\n[SAVE] BLS analysis saved → {BLS_EXCEL_PATH}")

    _create_bls_visualizations(cip_bls_corr, cip_awlevel_bls_corr, mismatches, lag_analysis)


if __name__ == "__main__":
    main()