"""
IPEDS Program Intelligence  —  Student Career & Program Explorer
================================================================
Streamlit app that lets students:
  1. See which academic programs are sunsetting vs peaking
  2. Browse institutions, programs, and degree levels
  3. Get AI-powered career advice (agentic, with skill packs)
  4. Discover trending jobs linked to BLS projections
"""

from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path
from typing import Any

import openai
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPEDS Program Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
YEARS = list(range(2019, 2025))

CIP2_TO_NAME: dict[str, str] = {
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
    "31": "Parks, Recreation & Fitness",
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
    "52": "Business, Management & Marketing",
    "54": "History",
}

AWLEVEL_TO_NAME: dict[str, str] = {
    "03": "Associate's",    "3": "Associate's",
    "04": "< Bachelor's",   "4": "< Bachelor's",
    "05": "Bachelor's",     "5": "Bachelor's",
    "06": "Post-Bacc Cert", "6": "Post-Bacc Cert",
    "07": "Master's",       "7": "Master's",
    "08": "Post-Master's",  "8": "Post-Master's",
    "09": "Doctoral",       "9": "Doctoral",
    "17": "1st Professional",
    "18": "Grad Cert", "19": "Grad Cert", "20": "Grad Cert", "21": "Grad Cert",
}

# Career field → common job titles (used when BLS data is not loaded)
CIP2_TO_JOBS: dict[str, list[str]] = {
    "11": ["Software Engineer", "Data Scientist", "Cloud Architect", "ML Engineer", "DevOps Engineer"],
    "14": ["Civil Engineer", "Mechanical Engineer", "Electrical Engineer", "Aerospace Engineer"],
    "51": ["Registered Nurse", "Physician Assistant", "Physical Therapist", "Health Informatics Analyst"],
    "52": ["Financial Analyst", "Marketing Manager", "Operations Manager", "Business Analyst", "MBA Graduate"],
    "27": ["Data Analyst", "Statistician", "Actuary", "Quantitative Analyst"],
    "26": ["Biomedical Researcher", "Clinical Lab Scientist", "Bioinformatics Analyst"],
    "13": ["K-12 Teacher", "Instructional Designer", "School Counselor", "Education Administrator"],
    "45": ["Social Worker", "Policy Analyst", "Urban Planner", "Research Analyst"],
    "42": ["Clinical Psychologist", "Counselor", "UX Researcher", "HR Specialist"],
    "40": ["Research Scientist", "Environmental Scientist", "Lab Analyst"],
    "09": ["Public Relations Specialist", "Journalist", "Content Strategist", "Media Analyst"],
    "43": ["Law Enforcement Officer", "Cybersecurity Analyst", "Emergency Manager"],
    "15": ["Electrical Technician", "Industrial Technologist", "Quality Control Engineer"],
    "03": ["Environmental Consultant", "Conservation Scientist", "Wildlife Biologist"],
    "01": ["Agricultural Engineer", "Food Scientist", "Farm Manager", "Agronomist"],
    "50": ["Graphic Designer", "UX/UI Designer", "Art Director", "Multimedia Artist"],
    "44": ["Social Worker", "Community Organizer", "Nonprofit Manager", "Government Analyst"],
    "22": ["Paralegal", "Legal Assistant", "Compliance Officer"],
    "04": ["Architect", "Urban Designer", "Interior Designer", "Landscape Architect"],
}

# Skills the student can enable
SKILLS: dict[str, dict] = {
    "program_explorer": {
        "label": "📚 Program Explorer",
        "desc": "Find & compare academic programs by growth trend",
        "color": "#1E88E5",
    },
    "career_pathfinder": {
        "label": "🛣️ Career Pathfinder",
        "desc": "Map your degree to real career outcomes",
        "color": "#43A047",
    },
    "job_scout": {
        "label": "💼 Job Scout",
        "desc": "Discover trending jobs aligned with your studies",
        "color": "#FB8C00",
    },
    "market_analyst": {
        "label": "📊 Market Intelligence",
        "desc": "Identify shortage areas and supply-demand gaps",
        "color": "#8E24AA",
    },
}

STATUS_COLORS = {
    "🔴 High Risk (Sunsetting)": "#ef5350",
    "🟡 Moderate Decline":       "#FFA726",
    "🟢 Growing / Stable":       "#66BB6A",
    "ℹ️ Insufficient Data":      "#90A4AE",
}

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading IPEDS data…")
def load_raw() -> pd.DataFrame | None:
    path = Path("out/combined_A_completions_2019_2024.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df["CIPCODE"] = df["CIPCODE"].astype(str).str.strip()
    df["AWLEVEL"]  = df["AWLEVEL"].astype(str).str.strip()
    df["UNITID"]   = df["UNITID"].astype(str).str.strip()
    df["CTOTALT"]  = pd.to_numeric(df["CTOTALT"], errors="coerce").fillna(0)
    df["YEAR"]     = pd.to_numeric(df["YEAR"], errors="coerce")
    df["CIP2"]     = df["CIPCODE"].str.extract(r"^(\d{2})", expand=False).str.zfill(2)
    df["CIP2_Name"]    = df["CIP2"].map(lambda x: CIP2_TO_NAME.get(x, "Unknown/Other"))
    df["AWLEVEL_Name"] = df["AWLEVEL"].map(lambda x: AWLEVEL_TO_NAME.get(x, f"Level {x}"))
    if "MAJORNUM" in df.columns:
        df = df[pd.to_numeric(df["MAJORNUM"], errors="coerce") == 1]
    return df


@st.cache_data(show_spinner="Computing program trends…")
def build_wide(df: pd.DataFrame) -> pd.DataFrame:
    """National CIP2 × AWLEVEL wide table with trend stats and sunset labels."""
    agg = (
        df.groupby(["CIP2", "CIP2_Name", "AWLEVEL", "AWLEVEL_Name", "YEAR"])["CTOTALT"]
        .sum().reset_index()
    )
    wide = (
        agg.pivot_table(
            index=["CIP2", "CIP2_Name", "AWLEVEL", "AWLEVEL_Name"],
            columns="YEAR", values="CTOTALT", aggfunc="sum",
        )
        .fillna(0).reset_index()
    )
    for y in YEARS:
        if y not in wide.columns:
            wide[y] = 0.0
    wide = wide.reindex(columns=["CIP2", "CIP2_Name", "AWLEVEL", "AWLEVEL_Name"] + YEARS)

    wide["baseline"] = wide[[2019, 2020, 2021]].mean(axis=1)
    wide["completions_2024"] = wide[2024]
    wide["net_pct"] = ((wide[2024] - wide[2019]) / wide[2019].replace({0: np.nan})) * 100
    wide["net_pct"] = pd.to_numeric(wide["net_pct"], errors="coerce")

    x = np.arange(len(YEARS), dtype=float)
    def _slope(row: pd.Series) -> float:
        y = np.array([row[yr] for yr in YEARS], dtype=float)
        return 0.0 if np.allclose(y, 0) else float(np.polyfit(x, y, 1)[0])
    wide["slope"] = wide.apply(_slope, axis=1)

    # Weighted z-score labeling (mirrors ipeds_bls_projections.py)
    valid_mask = (wide["baseline"] >= 20) & wide["net_pct"].notna() & (wide[2019] > 0)
    valid = wide[valid_mask]
    if len(valid) >= 10:
        clipped = valid["net_pct"].clip(-100, 300)
        mu = float(np.average(clipped, weights=valid["baseline"]))
        sigma = float(np.sqrt(np.average((clipped - mu) ** 2, weights=valid["baseline"])))
        sigma = max(sigma, 1e-9)
        wide["z"] = (wide["net_pct"].clip(-100, 300) - mu) / sigma
    else:
        wide["z"] = np.nan

    def _label(row: pd.Series) -> str:
        if row["baseline"] < 20 or pd.isna(row["net_pct"]) or row[2019] == 0:
            return "ℹ️ Insufficient Data"
        z = row.get("z", np.nan)
        if pd.isna(z):                return "ℹ️ Insufficient Data"
        if z <= -2.0:                 return "🔴 High Risk (Sunsetting)"
        if z <= -1.0:                 return "🟡 Moderate Decline"
        return "🟢 Growing / Stable"

    wide["status"] = wide.apply(_label, axis=1)
    return wide


@st.cache_data(show_spinner="Loading institution profiles…")
def build_inst_wide(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["UNITID", "CIP2", "CIP2_Name", "AWLEVEL", "AWLEVEL_Name", "YEAR"])["CTOTALT"]
        .sum().reset_index()
    )
    wide = (
        agg.pivot_table(
            index=["UNITID", "CIP2", "CIP2_Name", "AWLEVEL", "AWLEVEL_Name"],
            columns="YEAR", values="CTOTALT", aggfunc="sum",
        )
        .fillna(0).reset_index()
    )
    for y in YEARS:
        if y not in wide.columns:
            wide[y] = 0.0
    wide = wide.reindex(columns=["UNITID", "CIP2", "CIP2_Name", "AWLEVEL", "AWLEVEL_Name"] + YEARS)
    wide["net_pct"] = ((wide[2024] - wide[2019]) / wide[2019].replace({0: np.nan})) * 100
    wide["completions_2024"] = wide[2024]
    return wide


@st.cache_data
def load_bls() -> pd.DataFrame | None:
    for p in ["data_ipeds_bls/bls_correlation_analysis.xlsx",
              "data_uni/bls_correlation_analysis.xlsx"]:
        if Path(p).exists():
            try:
                return pd.read_excel(p, sheet_name="CIP2_BLS")
            except Exception:
                pass
    return None


@st.cache_data(show_spinner="Loading institution names…")
def load_institution_names() -> dict[str, str]:
    """Return UNITID → institution name. Downloads IPEDS HD file if not cached locally."""
    import io, zipfile, requests as _req

    local = Path("data_uni/hd2024.csv")
    if local.exists():
        try:
            hd = pd.read_csv(local, usecols=["UNITID", "INSTNM"], dtype=str, encoding="latin-1")
            return dict(zip(hd["UNITID"].str.strip(), hd["INSTNM"].str.strip()))
        except Exception:
            pass

    for year in (2024, 2023):
        url = f"https://nces.ed.gov/ipeds/datacenter/data/HD{year}.zip"
        try:
            r = _req.get(url, timeout=30)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                fname = next(n for n in z.namelist() if n.lower().endswith(".csv"))
                with z.open(fname) as f:
                    hd = pd.read_csv(f, usecols=["UNITID", "INSTNM"], dtype=str, encoding="latin-1")
            mapping = dict(zip(hd["UNITID"].str.strip(), hd["INSTNM"].str.strip()))
            # Save locally for next run
            hd.to_csv(local, index=False)
            return mapping
        except Exception:
            continue
    return {}


@st.cache_data
def load_crosswalk() -> dict[str, list[str]]:
    path = Path("CIP2020_SOC2018_Crosswalk.xlsx")
    if not path.exists():
        return {}
    try:
        df = pd.read_excel(path, sheet_name="CIP-SOC", dtype=str)
        df["CIP2"] = df["CIP2020Code"].str.extract(r"^(\d{2})", expand=False).str.zfill(2)
        df = df.dropna(subset=["CIP2", "SOC2018Code"])
        return (
            df.groupby("CIP2")["SOC2018Code"]
            .apply(lambda s: sorted(set(s.dropna().tolist())))
            .to_dict()
        )
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# AGENT TOOLS  (pure functions querying the loaded dataframes)
# ──────────────────────────────────────────────────────────────────────────────
def _tool_search_programs(inp: dict, wide: pd.DataFrame) -> str:
    field      = inp.get("field", "").strip().lower()
    degree     = inp.get("degree_level", "").strip().lower()
    status_f   = inp.get("status_filter", "all").lower()
    min_stud   = int(inp.get("min_students", 0))

    df = wide.copy()
    if field:
        df = df[df["CIP2_Name"].str.lower().str.contains(field, na=False)]
    if degree:
        df = df[df["AWLEVEL_Name"].str.lower().str.contains(degree, na=False)]
    if status_f == "growing":
        df = df[df["status"] == "🟢 Growing / Stable"]
    elif status_f == "declining":
        df = df[df["status"].isin(["🔴 High Risk (Sunsetting)", "🟡 Moderate Decline"])]
    if min_stud:
        df = df[df["baseline"] >= min_stud]

    df = df.sort_values("net_pct", ascending=False).head(15)
    if df.empty:
        return json.dumps({"error": "No programs matched the filters."})

    records = df[["CIP2_Name", "AWLEVEL_Name", "completions_2024", "net_pct", "status"]].rename(
        columns={"CIP2_Name": "field", "AWLEVEL_Name": "degree", "completions_2024": "graduates_2024",
                 "net_pct": "net_pct_change_2019_to_2024"}
    ).round({"net_pct_change_2019_to_2024": 1}).to_dict(orient="records")
    return json.dumps({"programs": records, "count": len(records)})


def _tool_program_detail(inp: dict, wide: pd.DataFrame) -> str:
    field  = inp.get("field", "").strip().lower()
    degree = inp.get("degree_level", "").strip().lower()

    df = wide.copy()
    if field:
        df = df[df["CIP2_Name"].str.lower().str.contains(field, na=False)]
    if degree:
        df = df[df["AWLEVEL_Name"].str.lower().str.contains(degree, na=False)]
    if df.empty:
        return json.dumps({"error": "No match found."})

    row = df.iloc[0]
    year_data = {str(y): int(row[y]) for y in YEARS}
    return json.dumps({
        "field": row["CIP2_Name"],
        "degree": row["AWLEVEL_Name"],
        "completions_by_year": year_data,
        "net_pct_change": round(float(row["net_pct"]) if pd.notna(row["net_pct"]) else 0, 1),
        "status": row["status"],
        "baseline_avg_2019_2021": round(float(row["baseline"]), 0),
    })


def _tool_get_career_paths(inp: dict, crosswalk: dict, bls: pd.DataFrame | None) -> str:
    field = inp.get("field", "").strip().lower()
    # Find CIP2 code
    cip2 = next(
        (k for k, v in CIP2_TO_NAME.items() if field in v.lower()),
        None
    )
    jobs = CIP2_TO_JOBS.get(cip2, ["Data Analyst", "Project Manager", "Business Analyst"])

    result: dict[str, Any] = {
        "field_of_study": CIP2_TO_NAME.get(cip2, field.title()),
        "common_career_paths": jobs,
    }

    if bls is not None and cip2:
        row = bls[bls["CIP2"].astype(str).str.zfill(2) == cip2]
        if not row.empty:
            r = row.iloc[0]
            result["bls_occupational_growth_pct"] = round(float(r.get("BLS_Occupational_Growth", 0)), 1)
            result["mapped_job_openings_annual"] = int(r.get("BLS_Annual_Openings_Mapped", 0) or 0)

    if cip2 and cip2 in crosswalk:
        result["soc_codes"] = crosswalk[cip2][:8]

    return json.dumps(result)


def _tool_trending_jobs(inp: dict, wide: pd.DataFrame, bls: pd.DataFrame | None) -> str:
    field = inp.get("field", "").strip().lower()

    results = []
    if bls is not None:
        df = bls.copy()
        if field:
            df = df[df["CIP2_Name"].str.lower().str.contains(field, na=False)]
        df = df.dropna(subset=["BLS_Occupational_Growth"]).sort_values("BLS_Occupational_Growth", ascending=False).head(10)
        for _, r in df.iterrows():
            results.append({
                "field": r["CIP2_Name"],
                "bls_growth_pct": round(float(r["BLS_Occupational_Growth"]), 1),
                "annual_openings": int(r.get("BLS_Annual_Openings_Mapped", 0) or 0),
            })

    # Augment with known job titles
    if not results:
        cip2 = next((k for k, v in CIP2_TO_NAME.items() if field in v.lower()), None)
        if cip2:
            titles = CIP2_TO_JOBS.get(cip2, [])
        else:
            # Top growing fields by static knowledge
            titles = ["Software Engineer", "Nurse Practitioner", "Data Scientist",
                      "Cybersecurity Analyst", "Wind Turbine Technician", "Physician Assistant"]
        results = [{"job_title": t, "trend": "High Demand"} for t in titles]

    return json.dumps({"trending_jobs": results})


def _tool_job_links(inp: dict) -> str:
    titles = inp.get("job_titles", [])
    location = inp.get("location", "United States")
    links = []
    for t in titles[:6]:
        q = urllib.parse.quote_plus(t)
        loc = urllib.parse.quote_plus(location)
        links.append({
            "title": t,
            "indeed": f"https://www.indeed.com/jobs?q={q}&l={loc}",
            "linkedin": f"https://www.linkedin.com/jobs/search/?keywords={q}&location={loc}",
            "bls_ooh": f"https://www.bls.gov/ooh/search.htm?q={q}",
        })
    return json.dumps({"job_links": links,
                       "note": "Click links to search live job postings."})


def _tool_supply_demand(inp: dict, wide: pd.DataFrame, bls: pd.DataFrame | None) -> str:
    field = inp.get("field", "").strip().lower()
    cip2 = next((k for k, v in CIP2_TO_NAME.items() if field in v.lower()), None)

    prog_row = None
    if cip2:
        df = wide[wide["CIP2"] == cip2]
        if not df.empty:
            prog_row = df.groupby("CIP2")[YEARS + ["net_pct"]].mean().iloc[0]

    bls_row = None
    if bls is not None and cip2:
        b = bls[bls["CIP2"].astype(str).str.zfill(2) == cip2]
        if not b.empty:
            bls_row = b.iloc[0]

    result: dict[str, Any] = {"field": CIP2_TO_NAME.get(cip2, field.title())}
    if prog_row is not None:
        result["program_net_pct_change"] = round(float(prog_row["net_pct"]) if pd.notna(prog_row["net_pct"]) else 0, 1)
    if bls_row is not None:
        result["bls_occupational_growth_pct"] = round(float(bls_row.get("BLS_Occupational_Growth", 0)), 1)
        result["annual_job_openings"] = int(bls_row.get("BLS_Annual_Openings_Mapped", 0) or 0)
        prog_c = result.get("program_net_pct_change", 0)
        bls_c  = result["bls_occupational_growth_pct"]
        if bls_c > 5 and prog_c < -5:
            result["gap_signal"] = "⚠️ Supply shortage: programs declining while jobs grow."
        elif bls_c < -5 and prog_c > 5:
            result["gap_signal"] = "⚠️ Oversupply risk: programs growing while jobs shrink."
        else:
            result["gap_signal"] = "✅ Supply and demand appear broadly aligned."
    return json.dumps(result)


def _tool_top_sunsetting(inp: dict, wide: pd.DataFrame) -> str:
    n = int(inp.get("top_n", 10))
    df = wide[wide["status"] == "🔴 High Risk (Sunsetting)"].copy()
    df = df.sort_values("net_pct").head(n)
    records = df[["CIP2_Name", "AWLEVEL_Name", "completions_2024", "net_pct"]].rename(
        columns={"CIP2_Name": "field", "AWLEVEL_Name": "degree",
                 "completions_2024": "graduates_2024", "net_pct": "net_pct_change"}
    ).round({"net_pct_change": 1}).to_dict(orient="records")
    return json.dumps({"sunsetting_programs": records})


def _tool_top_growing(inp: dict, wide: pd.DataFrame) -> str:
    n = int(inp.get("top_n", 10))
    df = wide[wide["status"] == "🟢 Growing / Stable"].copy()
    df = df[df["baseline"] >= 50].sort_values("net_pct", ascending=False).head(n)
    records = df[["CIP2_Name", "AWLEVEL_Name", "completions_2024", "net_pct"]].rename(
        columns={"CIP2_Name": "field", "AWLEVEL_Name": "degree",
                 "completions_2024": "graduates_2024", "net_pct": "net_pct_change"}
    ).round({"net_pct_change": 1}).to_dict(orient="records")
    return json.dumps({"growing_programs": records})


# ──────────────────────────────────────────────────────────────────────────────
# LLM PROVIDER PRESETS
# ──────────────────────────────────────────────────────────────────────────────
LLM_PROVIDERS: dict[str, dict] = {
    "🦙 Ollama (Local — Free)": {
        "base_url":      "http://localhost:11434/v1",
        "default_key":   "ollama",
        "default_model": "llama3.1:8b",
        "needs_key":     False,
    },
    "⚡ Groq (Free Cloud)": {
        "base_url":      "https://api.groq.com/openai/v1",
        "default_key":   "",
        "default_model": "llama-3.1-8b-instant",
        "needs_key":     True,
    },
    "🤝 Together AI": {
        "base_url":      "https://api.together.xyz/v1",
        "default_key":   "",
        "default_model": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "needs_key":     True,
    },
    "🔧 LM Studio / Custom": {
        "base_url":      "http://localhost:1234/v1",
        "default_key":   "none",
        "default_model": "",
        "needs_key":     False,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS  (OpenAI function-calling schema)
# ──────────────────────────────────────────────────────────────────────────────
ALL_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "search_programs",
            "description": "Search IPEDS data for academic programs. Returns trend status (sunsetting/growing), completion counts, and net % change 2019-2024.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field":         {"type": "string",  "description": "Partial field name, e.g. 'Computer', 'Nursing', 'Business'"},
                    "degree_level":  {"type": "string",  "description": "Degree level substring, e.g. 'Bachelor', 'Master', 'Associate'"},
                    "status_filter": {"type": "string",  "enum": ["all", "growing", "declining"], "description": "Filter by trend status"},
                    "min_students":  {"type": "integer", "description": "Minimum baseline enrollment (default 0)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "program_detail",
            "description": "Get year-by-year completion counts and trend stats for a specific program and degree level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field":        {"type": "string", "description": "Field of study substring"},
                    "degree_level": {"type": "string", "description": "Degree level substring"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_career_paths",
            "description": "Return common career paths and BLS job outlook for a field of study. Includes SOC codes and annual openings where available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "Field of study, e.g. 'Computer Science', 'Health Professions'"},
                },
                "required": ["field"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trending_jobs",
            "description": "Return the top trending occupations (by BLS growth) in a given field, or across all fields if no field specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "Optional field filter, e.g. 'Engineering', 'Health'"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "job_links",
            "description": "Generate direct job search links (Indeed, LinkedIn, BLS OOH) for a list of job titles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_titles": {"type": "array", "items": {"type": "string"}, "description": "List of job titles"},
                    "location":   {"type": "string", "description": "Location filter (default: United States)"},
                },
                "required": ["job_titles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "supply_demand_gap",
            "description": "Analyze the gap between educational supply (IPEDS program growth) and labor demand (BLS projections) for a field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "Field of study to analyze"},
                },
                "required": ["field"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_sunsetting",
            "description": "Return the top N programs with the steepest decline (sunsetting).",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "How many programs to return (default 10)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_growing",
            "description": "Return the top N growing academic programs by net % change 2019-2024.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "How many programs to return (default 10)"},
                },
            },
        },
    },
]

# Which tools each skill unlocks
SKILL_TOOLS: dict[str, list[str]] = {
    "program_explorer": ["search_programs", "program_detail", "top_sunsetting", "top_growing"],
    "career_pathfinder": ["get_career_paths", "supply_demand_gap"],
    "job_scout":  ["trending_jobs", "job_links"],
    "market_analyst": ["supply_demand_gap", "top_sunsetting", "top_growing"],
}


def active_tool_defs(active_skills: list[str]) -> list[dict]:
    allowed = set()
    for s in active_skills:
        allowed.update(SKILL_TOOLS.get(s, []))
    allowed.add("search_programs")
    return [t for t in ALL_TOOL_DEFS if t["function"]["name"] in allowed]


def dispatch_tool(name: str, inp: dict, wide: pd.DataFrame,
                  bls: pd.DataFrame | None, crosswalk: dict) -> str:
    if name == "search_programs":      return _tool_search_programs(inp, wide)
    if name == "program_detail":       return _tool_program_detail(inp, wide)
    if name == "get_career_paths":     return _tool_get_career_paths(inp, crosswalk, bls)
    if name == "trending_jobs":        return _tool_trending_jobs(inp, wide, bls)
    if name == "job_links":            return _tool_job_links(inp)
    if name == "supply_demand_gap":    return _tool_supply_demand(inp, wide, bls)
    if name == "top_sunsetting":       return _tool_top_sunsetting(inp, wide)
    if name == "top_growing":          return _tool_top_growing(inp, wide)
    return json.dumps({"error": f"Unknown tool: {name}"})


# ──────────────────────────────────────────────────────────────────────────────
# AGENT LOOP
# ──────────────────────────────────────────────────────────────────────────────
def build_system_prompt(active_skills: list[str]) -> str:
    skill_labels = [SKILLS[s]["label"] for s in active_skills if s in SKILLS]
    skill_lines  = "\n".join(f"  • {l}" for l in skill_labels) if skill_labels else "  • None"
    return f"""You are an intelligent academic and career advisor for university students.
You have access to real IPEDS (Integrated Postsecondary Education Data System) data
from 2019–2024 covering hundreds of thousands of US program completions.

Active skills:
{skill_lines}

Guidelines:
- Be warm, encouraging, and student-friendly — avoid jargon.
- When citing numbers, translate them into plain language (e.g. "This field dropped 23% — more than 1 in 5 students left it").
- Always call a tool to get real data before making claims about specific programs or job numbers.
- When suggesting career paths, use the job_links tool to surface actionable job search links.
- Flag sunsetting programs clearly so students can make informed choices.
- End responses with one concrete next step the student can take today.
"""


def run_agent(
    user_msg: str,
    history: list[dict],
    active_skills: list[str],
    wide: pd.DataFrame,
    bls: pd.DataFrame | None,
    crosswalk: dict,
    llm_cfg: dict,
) -> str:
    client = openai.OpenAI(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"] or "none",
    )
    tools   = active_tool_defs(active_skills)
    system  = build_system_prompt(active_skills)
    model   = llm_cfg["model"]

    messages: list[dict] = [{"role": "system", "content": system}]
    messages += history
    messages.append({"role": "user", "content": user_msg})

    for _ in range(8):
        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            tools=tools,
            messages=messages,
        )
        choice = response.choices[0]
        msg    = choice.message

        if choice.finish_reason == "stop" or not msg.tool_calls:
            return msg.content or ""

        # Append the assistant's tool-call turn
        messages.append({
            "role":       "assistant",
            "content":    msg.content,
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each tool and append results
        for tc in msg.tool_calls:
            inp        = json.loads(tc.function.arguments)
            result_str = dispatch_tool(tc.function.name, inp, wide, bls, crosswalk)
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result_str,
            })

    return "I wasn't able to complete the analysis. Please try rephrasing your question."


# ──────────────────────────────────────────────────────────────────────────────
# DARK THEME
# ──────────────────────────────────────────────────────────────────────────────
_DARK_BG   = "#0F1117"
_DARK_PLOT = "#1A1C26"
_DARK_GRID = "#2C2F45"
_DARK_TEXT = "#E8EAF0"
_DARK_TICK = "#8B8FA8"

# Vivid palette that pops on dark backgrounds
_RED    = "#FF5252"
_ORANGE = "#FF9800"
_GREEN  = "#69F0AE"
_TEAL   = "#00BCD4"
_BLUE   = "#448AFF"

_PIE_COLORS = [
    "#448AFF", "#69F0AE", "#FF9800", "#FF5252",
    "#AB47BC", "#00BCD4", "#FFA726", "#26C6DA",
]


def _dark_layout(**extra) -> dict:
    base = dict(
        plot_bgcolor=_DARK_PLOT,
        paper_bgcolor=_DARK_BG,
        font=dict(color=_DARK_TEXT, size=12),
        xaxis=dict(
            gridcolor=_DARK_GRID, zerolinecolor=_DARK_GRID,
            tickfont=dict(color=_DARK_TICK), title_font=dict(color=_DARK_TEXT),
        ),
        yaxis=dict(
            gridcolor=_DARK_GRID, zerolinecolor=_DARK_GRID,
            tickfont=dict(color=_DARK_TICK), title_font=dict(color=_DARK_TEXT),
        ),
    )
    base.update(extra)
    return base


# ──────────────────────────────────────────────────────────────────────────────
# SHARED CHART HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def trend_sparkline(row: pd.Series, title: str = "") -> go.Figure:
    vals = [float(row[y]) for y in YEARS]
    color = _RED if vals[-1] < vals[0] else _GREEN
    fig = go.Figure(go.Scatter(
        x=YEARS, y=vals, mode="lines+markers",
        line=dict(color=color, width=3),
        marker=dict(size=8, color=color, line=dict(color=_DARK_PLOT, width=1)),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=_DARK_TEXT)),
        height=220, margin=dict(l=10, r=10, t=35, b=10),
        plot_bgcolor=_DARK_PLOT, paper_bgcolor=_DARK_BG,
        font=dict(color=_DARK_TEXT, size=12),
        xaxis=dict(showgrid=False, tickfont=dict(color=_DARK_TICK), zerolinecolor=_DARK_GRID),
        yaxis=dict(showgrid=True, gridcolor=_DARK_GRID, tickfont=dict(color=_DARK_TICK), zerolinecolor=_DARK_GRID),
    )
    return fig


def status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#90A4AE")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:12px;font-size:0.8rem">{status}</span>'


def _style_status(val: str) -> str:
    """Pandas Styler function for the Status column (dark-theme safe)."""
    if "High Risk" in str(val):  return "background-color:#5C1A1A;color:#FF8A80"
    if "Moderate"  in str(val):  return "background-color:#4A3800;color:#FFD54F"
    if "Growing"   in str(val):  return "background-color:#0D3320;color:#69F0AE"
    return ""


def _style_net_pct(val) -> str:
    """Pandas Styler function for numeric Net % Change column (dark-theme safe)."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v < -20: return "color:#FF8A80;font-weight:bold"
    if v <   0: return "color:#FFD54F"
    if v >   0: return "color:#69F0AE"
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1 — DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
def page_dashboard(wide: pd.DataFrame, inst_wide: pd.DataFrame) -> None:
    st.title("🎓 IPEDS Program Intelligence")
    st.caption("National program completion trends 2019–2024 · US Department of Education data")

    total       = len(wide)
    sunsetting  = (wide["status"] == "🔴 High Risk (Sunsetting)").sum()
    moderate    = (wide["status"] == "🟡 Moderate Decline").sum()
    growing     = (wide["status"] == "🟢 Growing / Stable").sum()
    total_grads = int(wide[2024].sum())
    num_inst    = inst_wide["UNITID"].nunique()
    pct_at_risk = (sunsetting + moderate) / max(total, 1) * 100
    nat_chg     = ((wide[2024].sum() - wide[2019].sum()) / wide[2019].sum()) * 100

    # Key insight callout
    insight_color = "#5C1A1A" if pct_at_risk > 40 else "#0D3320"
    insight_text  = (
        f"**{pct_at_risk:.0f}% of tracked programs are in decline or high risk** — "
        f"national completions shifted {nat_chg:+.1f}% from 2019 to 2024 across "
        f"{num_inst:,} institutions."
    )
    st.markdown(
        f'<div style="background:{insight_color};border-left:4px solid '
        f'{"#FF5252" if pct_at_risk > 40 else "#69F0AE"};'
        f'padding:12px 16px;border-radius:6px;color:#E8EAF0;margin-bottom:12px">'
        f'{insight_text}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Programs Tracked", f"{total:,}")
    c2.metric("Institutions", f"{num_inst:,}")
    c3.metric("🔴 High Risk", f"{sunsetting:,}")
    c4.metric("🟡 Moderate Decline", f"{moderate:,}")
    c5.metric("🟢 Growing / Stable", f"{growing:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("📉 Top Sunsetting Programs")
        top_down = (
            wide[wide["status"] == "🔴 High Risk (Sunsetting)"]
            .sort_values("net_pct").head(12)
        )
        if not top_down.empty:
            top_down["label"] = top_down["CIP2_Name"] + " · " + top_down["AWLEVEL_Name"]
            fig = px.bar(
                top_down, x="net_pct", y="label", orientation="h",
                color="net_pct", color_continuous_scale=[_RED, _ORANGE],
                labels={"net_pct": "Net % Change (2019→2024)", "label": ""},
            )
            fig.update_layout(
                height=420, margin=dict(l=5, r=5, t=5, b=5),
                showlegend=False, coloraxis_showscale=False,
                **_dark_layout(),
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("📈 Top Growing Programs")
        top_up = (
            wide[(wide["status"] == "🟢 Growing / Stable") & (wide["baseline"] >= 50)]
            .sort_values("net_pct", ascending=False).head(12)
        )
        if not top_up.empty:
            top_up["label"] = top_up["CIP2_Name"] + " · " + top_up["AWLEVEL_Name"]
            fig = px.bar(
                top_up, x="net_pct", y="label", orientation="h",
                color="net_pct", color_continuous_scale=[_TEAL, _GREEN],
                labels={"net_pct": "Net % Change (2019→2024)", "label": ""},
            )
            fig.update_layout(
                height=420, margin=dict(l=5, r=5, t=5, b=5),
                showlegend=False, coloraxis_showscale=False,
                **_dark_layout(),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🌐 National Completions Over Time")

    nat = wide.copy()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Overall Trend", "🎓 By Degree Level", "🚦 By Risk Status", "🏆 Top Fields"]
    )

    # ── Tab 1: Overall + YoY % change ───────────────────────────────────────
    with tab1:
        yr_totals = nat[YEARS].sum().reset_index()
        yr_totals.columns = ["Year", "Total Completions"]
        yr_totals["YoY %"] = yr_totals["Total Completions"].pct_change() * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yr_totals["Year"], y=yr_totals["Total Completions"],
            mode="lines+markers", name="Total Completions",
            line=dict(color=_BLUE, width=3),
            marker=dict(size=8, color=_BLUE),
            fill="tozeroy", fillcolor="rgba(68,138,255,0.15)",
            yaxis="y1",
        ))
        # YoY % change as bar on secondary axis
        bar_colors = [_GREEN if v >= 0 else _RED for v in yr_totals["YoY %"].fillna(0)]
        fig.add_trace(go.Bar(
            x=yr_totals["Year"], y=yr_totals["YoY %"].fillna(0),
            name="YoY % Change", marker_color=bar_colors,
            opacity=0.7, yaxis="y2",
        ))
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor=_DARK_PLOT, paper_bgcolor=_DARK_BG,
            font=dict(color=_DARK_TEXT, size=12),
            xaxis=dict(gridcolor=_DARK_GRID, zerolinecolor=_DARK_GRID, tickfont=dict(color=_DARK_TICK)),
            yaxis=dict(
                title="Completions", gridcolor=_DARK_GRID, zerolinecolor=_DARK_GRID,
                title_font=dict(color=_BLUE), tickfont=dict(color=_BLUE),
            ),
            yaxis2=dict(
                title="YoY % Change", overlaying="y", side="right",
                title_font=dict(color=_DARK_TICK), tickfont=dict(color=_DARK_TICK),
                showgrid=False, zeroline=True, zerolinecolor=_DARK_GRID,
            ),
            legend=dict(font=dict(color=_DARK_TEXT), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bars show year-over-year % change; line shows total completions.")

    # ── Tab 2: Stacked area by degree level ─────────────────────────────────
    with tab2:
        deg_rows = []
        for aw in nat["AWLEVEL_Name"].dropna().unique():
            sub = nat[nat["AWLEVEL_Name"] == aw][YEARS].sum()
            for yr in YEARS:
                deg_rows.append({"Year": yr, "Degree Level": aw, "Completions": int(sub[yr])})
        deg_df = pd.DataFrame(deg_rows)
        # Keep only top 6 degree levels by total volume
        top_deg = deg_df.groupby("Degree Level")["Completions"].sum().nlargest(6).index
        deg_df  = deg_df[deg_df["Degree Level"].isin(top_deg)]

        deg_colors = [_BLUE, _GREEN, _ORANGE, _RED, _TEAL, "#AB47BC"]
        fig = px.area(
            deg_df, x="Year", y="Completions", color="Degree Level",
            color_discrete_sequence=deg_colors,
        )
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(color=_DARK_TEXT), bgcolor="rgba(0,0,0,0)"),
            **_dark_layout(),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Stacked by degree level — see which award levels drive the national total.")

    # ── Tab 3: Stacked area by risk status ──────────────────────────────────
    with tab3:
        status_order = ["🟢 Growing / Stable", "🟡 Moderate Decline", "🔴 High Risk (Sunsetting)"]
        status_colors_map = {
            "🟢 Growing / Stable":        _GREEN,
            "🟡 Moderate Decline":        _ORANGE,
            "🔴 High Risk (Sunsetting)":  _RED,
        }
        risk_rows = []
        for st_label in status_order:
            sub = nat[nat["status"] == st_label][YEARS].sum()
            for yr in YEARS:
                risk_rows.append({"Year": yr, "Status": st_label, "Completions": int(sub[yr])})
        risk_df = pd.DataFrame(risk_rows)

        fig = px.area(
            risk_df, x="Year", y="Completions", color="Status",
            color_discrete_map=status_colors_map,
            category_orders={"Status": status_order},
        )
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(color=_DARK_TEXT), bgcolor="rgba(0,0,0,0)"),
            **_dark_layout(),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Watch the red band — sunsetting programs eating into the total over time.")

    # ── Tab 4: Line chart of top 8 fields ───────────────────────────────────
    with tab4:
        top_fields = (
            nat.groupby("CIP2_Name")[2019].sum()
            .nlargest(8).index.tolist()
        )
        field_colors = [_BLUE, _GREEN, _ORANGE, _RED, _TEAL, "#AB47BC", "#FFF176", "#80DEEA"]
        field_rows = []
        for field in top_fields:
            sub = nat[nat["CIP2_Name"] == field][YEARS].sum()
            for yr in YEARS:
                field_rows.append({"Year": yr, "Field": field, "Completions": int(sub[yr])})
        field_df = pd.DataFrame(field_rows)

        fig = px.line(
            field_df, x="Year", y="Completions", color="Field",
            color_discrete_sequence=field_colors, markers=True,
        )
        fig.update_traces(line_width=2.5, marker_size=7)
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(color=_DARK_TEXT, size=11), bgcolor="rgba(0,0,0,0)"),
            **_dark_layout(),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Top 8 fields by 2019 volume — see which held strong and which declined.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2 — PROGRAM EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
def page_program_explorer(wide: pd.DataFrame) -> None:
    st.title("📚 Program Explorer")
    st.caption("Browse all IPEDS programs · filter by field, degree, and trend status")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        field_opts = ["All"] + sorted(wide["CIP2_Name"].dropna().unique().tolist())
        field_sel  = st.selectbox("Field of Study", field_opts)
    with col2:
        degree_opts = ["All"] + sorted(wide["AWLEVEL_Name"].dropna().unique().tolist())
        degree_sel  = st.selectbox("Degree Level", degree_opts)
    with col3:
        status_opts = ["All", "🟢 Growing / Stable", "🟡 Moderate Decline",
                       "🔴 High Risk (Sunsetting)", "ℹ️ Insufficient Data"]
        status_sel  = st.selectbox("Status", status_opts)

    df = wide.copy()
    if field_sel != "All":
        df = df[df["CIP2_Name"] == field_sel]
    if degree_sel != "All":
        df = df[df["AWLEVEL_Name"] == degree_sel]
    if status_sel != "All":
        df = df[df["status"] == status_sel]

    st.caption(f"Showing **{len(df):,}** programs")

    # Table with color-coded status
    display = df[["CIP2_Name", "AWLEVEL_Name", 2019, 2024, "net_pct", "status"]].copy()
    display.columns = ["Field", "Degree", "Completions 2019", "Completions 2024",
                       "Net % Change", "Status"]
    display["Net % Change"] = display["Net % Change"].round(1)

    st.dataframe(
        display.style.applymap(_style_status, subset=["Status"])
                     .applymap(_style_net_pct, subset=["Net % Change"]),
        use_container_width=True, height=380,
    )

    # Detail: pick a row to see sparkline
    st.divider()
    st.subheader("Program Trend Detail")
    row_opts = df.apply(lambda r: f"{r['CIP2_Name']} — {r['AWLEVEL_Name']}", axis=1).tolist()
    if row_opts:
        selected = st.selectbox("Select a program to view trend", row_opts)
        idx = row_opts.index(selected)
        row = df.iloc[idx]
        lc, rc = st.columns([3, 1])
        with lc:
            st.plotly_chart(
                trend_sparkline(row, f"{row['CIP2_Name']} · {row['AWLEVEL_Name']}"),
                use_container_width=True,
            )
        with rc:
            st.markdown(f"**Status:**")
            st.markdown(status_badge(row["status"]), unsafe_allow_html=True)
            st.metric("2019 Completions", f"{int(row[2019]):,}")
            st.metric("2024 Completions", f"{int(row[2024]):,}")
            chg = row["net_pct"]
            st.metric("Net Change", f"{chg:+.1f}%" if pd.notna(chg) else "N/A",
                      delta=f"{chg:+.1f}%" if pd.notna(chg) else None,
                      delta_color="normal")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3 — INSTITUTION BROWSER
# ──────────────────────────────────────────────────────────────────────────────
def page_institution_browser(raw: pd.DataFrame, inst_wide: pd.DataFrame,
                              inst_names: dict[str, str]) -> None:
    st.title("🏛️ Institution Browser")
    st.caption("Explore programs offered by any US institution · 2019–2024 completion trends")

    all_unitids = sorted(inst_wide["UNITID"].unique().tolist())

    def _label(uid: str) -> str:
        name = inst_names.get(uid, "")
        return f"{name}  ({uid})" if name else uid

    uid_labels = {uid: _label(uid) for uid in all_unitids}
    label_to_uid = {v: k for k, v in uid_labels.items()}

    search = st.selectbox(
        "Search institution by name or UNITID",
        options=sorted(uid_labels.values()),
        help="Type to filter by institution name",
    )
    unitid = label_to_uid[search]

    if not unitid:
        return

    inst_display_name = inst_names.get(unitid, f"Institution {unitid}")
    st.subheader(inst_display_name)
    st.caption(f"UNITID: {unitid}")

    df = inst_wide[inst_wide["UNITID"] == unitid].copy()
    total_grads = int(df[2024].sum())
    programs = len(df)
    declining = (df["net_pct"] < -10).sum()
    growing   = (df["net_pct"] >  10).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Programs Offered", f"{programs:,}")
    c2.metric("Total Graduates (2024)", f"{total_grads:,}")
    c3.metric("📉 Declining (>10%)", f"{declining:,}")
    c4.metric("📈 Growing (>10%)", f"{growing:,}")

    st.divider()

    left, right = st.columns([3, 2])

    with left:
        st.subheader("All Programs")
        display = df[["CIP2_Name", "AWLEVEL_Name", 2019, 2024, "net_pct"]].copy()
        display.columns = ["Field", "Degree", "2019", "2024", "Net % Chg"]
        display["Net % Chg"] = display["Net % Chg"].round(1)

        st.dataframe(
            display.style.applymap(_style_net_pct, subset=["Net % Chg"]),
            use_container_width=True, height=340,
        )

    with right:
        st.subheader("Program Mix (2024)")
        by_field = df.groupby("CIP2_Name")[2024].sum().sort_values(ascending=False).head(8)
        if not by_field.empty:
            fig = px.pie(values=by_field.values, names=by_field.index,
                         color_discrete_sequence=_PIE_COLORS, hole=0.35)
            fig.update_traces(textfont=dict(color=_DARK_TEXT, size=12))
            fig.update_layout(
                height=340, margin=dict(l=0, r=0, t=0, b=0), showlegend=True,
                legend=dict(font=dict(color=_DARK_TEXT)),
                **_dark_layout(),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Trend for a selected program at this institution
    st.divider()
    st.subheader("Program Trend at this Institution")
    opts = df.apply(lambda r: f"{r['CIP2_Name']} — {r['AWLEVEL_Name']}", axis=1).tolist()
    if opts:
        sel = st.selectbox("Select program", opts, key="inst_prog_sel")
        row = df.iloc[opts.index(sel)]
        st.plotly_chart(
            trend_sparkline(row, f"{row['CIP2_Name']} · {row['AWLEVEL_Name']} — {inst_display_name}"),
            use_container_width=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4 — CAREER ADVISOR (AGENTIC)
# ──────────────────────────────────────────────────────────────────────────────
def page_career_advisor(wide: pd.DataFrame, bls: pd.DataFrame | None,
                         crosswalk: dict) -> None:
    st.title("🤖 AI Career Advisor")
    st.caption("Powered by open-source LLMs · Uses real IPEDS + BLS data")

    # ── LLM provider config ─────────────────────────────────────────────────
    with st.expander("⚙️ LLM Settings", expanded="llm_cfg" not in st.session_state):
        provider_name = st.selectbox(
            "Provider", list(LLM_PROVIDERS.keys()), key="llm_provider_name"
        )
        preset = LLM_PROVIDERS[provider_name]

        col_model, col_key = st.columns([2, 2])
        with col_model:
            model = st.text_input("Model", value=preset["default_model"], key="llm_model_input")
        with col_key:
            if preset["needs_key"]:
                api_key = st.text_input("API Key", value=preset["default_key"],
                                        type="password", key="llm_key_input")
            else:
                api_key = preset["default_key"]
                st.caption(f"No key needed for {provider_name}")

        base_url = preset["base_url"]
        st.caption(f"Endpoint: `{base_url}`")

        if st.button("✅ Apply Settings"):
            st.session_state["llm_cfg"] = {
                "base_url": base_url,
                "api_key":  api_key,
                "model":    model,
            }
            st.success("LLM settings saved.")

    llm_cfg: dict | None = st.session_state.get("llm_cfg")

    if not llm_cfg:
        st.info("Configure and apply LLM settings above to start chatting.", icon="⚙️")
        return

    # ── Skill selector ──────────────────────────────────────────────────────
    st.subheader("🧩 Your Skill Pack")
    st.caption("Enable skills to unlock specialized advisor capabilities")

    if "active_skills" not in st.session_state:
        st.session_state.active_skills = ["program_explorer", "career_pathfinder"]

    cols = st.columns(len(SKILLS))
    for i, (sid, meta) in enumerate(SKILLS.items()):
        with cols[i]:
            enabled = sid in st.session_state.active_skills
            btn_label = f"{'✅' if enabled else '⬜'} {meta['label']}"
            if st.button(btn_label, key=f"skill_{sid}", use_container_width=True):
                if enabled:
                    st.session_state.active_skills.remove(sid)
                else:
                    st.session_state.active_skills.append(sid)
                st.rerun()
            st.caption(meta["desc"])

    active = st.session_state.active_skills
    if active:
        st.success(f"Active: {', '.join(SKILLS[s]['label'] for s in active if s in SKILLS)}")
    else:
        st.info("Enable at least one skill to get started.")

    st.divider()

    # ── Chat interface ───────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.subheader("💬 Chat with your Advisor")

    # Starter prompts
    if not st.session_state.messages:
        st.caption("**Try asking:**")
        starters = [
            "Which programs are sunsetting the fastest right now?",
            "I'm interested in Computer Science — is it a good bet?",
            "What are the top growing fields I should consider?",
            "Find me trending jobs in Health Professions and share links.",
            "Is there a supply-demand gap in Engineering?",
        ]
        cols = st.columns(len(starters))
        for i, s in enumerate(starters):
            with cols[i]:
                if st.button(s, key=f"starter_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": s})
                    with st.spinner("Thinking…"):
                        reply = run_agent(s, [], active, wide, bls, crosswalk, llm_cfg)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask about programs, careers, or jobs…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build message history (last 10 turns to stay within token limits)
        history = []
        for m in st.session_state.messages[:-1][-10:]:
            history.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            with st.spinner("Researching your question…"):
                reply = run_agent(prompt, history, active, wide, bls, crosswalk, llm_cfg)
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 5 — JOB MARKET TRENDS
# ──────────────────────────────────────────────────────────────────────────────
def page_job_market(wide: pd.DataFrame, bls: pd.DataFrame | None) -> None:
    st.title("💼 Job Market Trends")
    st.caption("BLS occupational projections linked to academic fields · Click job cards to search live openings")

    # Field selector
    field_opts = ["All Fields"] + sorted(CIP2_TO_NAME.values())
    sel_field  = st.selectbox("Filter by Field of Study", field_opts)

    cip2 = None
    if sel_field != "All Fields":
        cip2 = next((k for k, v in CIP2_TO_NAME.items() if v == sel_field), None)

    # ── BLS-based view (if data available) ──────────────────────────────────
    if bls is not None:
        df = bls.copy()
        if cip2:
            df = df[df["CIP2"].astype(str).str.zfill(2) == cip2]
        df = df.dropna(subset=["BLS_Occupational_Growth"]).sort_values(
            "BLS_Occupational_Growth", ascending=False
        ).head(20)

        if not df.empty:
            st.subheader("📊 BLS Occupational Growth vs Program Trend")
            fig = px.scatter(
                df,
                x="BLS_Occupational_Growth",
                y="Program_Net_Pct_Change",
                text="CIP2_Name",
                color="Correlation_Direction" if "Correlation_Direction" in df.columns else None,
                color_discrete_map={"Aligned": _GREEN, "Misaligned": _RED},
                size_max=30,
                labels={
                    "BLS_Occupational_Growth": "BLS Job Growth (%)",
                    "Program_Net_Pct_Change": "Program Completions Change (%)",
                },
            )
            fig.add_hline(y=0, line_dash="dash", line_color=_DARK_GRID)
            fig.add_vline(x=0, line_dash="dash", line_color=_DARK_GRID)
            fig.update_traces(textposition="top center", textfont=dict(color=_DARK_TEXT, size=11))
            fig.update_layout(
                height=480,
                legend=dict(font=dict(color=_DARK_TEXT)),
                **_dark_layout(),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**Upper-left**: programs growing, jobs not following · **Lower-right**: jobs growing, programs declining (shortage risk)")

    # ── Job cards with search links ──────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Find Jobs in This Field")

    if cip2:
        job_list = CIP2_TO_JOBS.get(cip2, ["Data Analyst", "Project Manager", "Consultant"])
        field_label = sel_field
    else:
        # Show top jobs across all hot fields
        job_list = [
            "Software Engineer", "Nurse Practitioner", "Data Scientist",
            "Cybersecurity Analyst", "Physician Assistant", "Financial Analyst",
            "Physical Therapist", "Wind Turbine Technician", "Cloud Architect",
            "Health Informatics Analyst",
        ]
        field_label = "Top Trending"

    st.caption(f"**{field_label}** — click any card to search live openings")

    job_cols = st.columns(min(len(job_list), 3))
    for i, title in enumerate(job_list[:9]):
        q   = urllib.parse.quote_plus(title)
        loc = "United States"
        indeed_url  = f"https://www.indeed.com/jobs?q={q}&l={urllib.parse.quote_plus(loc)}"
        linkedin_url = f"https://www.linkedin.com/jobs/search/?keywords={q}"
        bls_url     = f"https://www.bls.gov/ooh/search.htm?q={q}"

        with job_cols[i % 3]:
            st.markdown(
                f"""
                <div style="border:1px solid #2C2F45;border-radius:12px;padding:16px;margin-bottom:12px;
                            background:#1A1C26;color:#E8EAF0;">
                  <h4 style="margin:0 0 10px 0;color:#E8EAF0;font-size:1rem">{title}</h4>
                  <a href="{indeed_url}" target="_blank"
                     style="margin-right:10px;color:#448AFF;text-decoration:none;font-size:0.85rem">🔍 Indeed</a>
                  <a href="{linkedin_url}" target="_blank"
                     style="margin-right:10px;color:#448AFF;text-decoration:none;font-size:0.85rem">💼 LinkedIn</a>
                  <a href="{bls_url}" target="_blank"
                     style="color:#448AFF;text-decoration:none;font-size:0.85rem">📈 BLS OOH</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Sunsetting vs Growing field comparison ───────────────────────────────
    st.divider()
    st.subheader("📉 Sunsetting vs 📈 Growing — At a Glance")

    sunset_fields = (
        wide[wide["status"] == "🔴 High Risk (Sunsetting)"]
        .groupby("CIP2_Name")["net_pct"].mean()
        .sort_values().head(8)
        .reset_index()
    )
    grow_fields = (
        wide[(wide["status"] == "🟢 Growing / Stable") & (wide["baseline"] >= 30)]
        .groupby("CIP2_Name")["net_pct"].mean()
        .sort_values(ascending=False).head(8)
        .reset_index()
    )

    lc, rc = st.columns(2)
    with lc:
        if not sunset_fields.empty:
            fig = px.bar(sunset_fields, x="net_pct", y="CIP2_Name", orientation="h",
                         color_discrete_sequence=[_RED],
                         labels={"net_pct": "Avg Net % Change", "CIP2_Name": ""})
            fig.update_layout(height=320, margin=dict(l=5, r=5, t=5, b=5), **_dark_layout())
            st.caption("**⚠️ Fields to reconsider**")
            st.plotly_chart(fig, use_container_width=True)
    with rc:
        if not grow_fields.empty:
            fig = px.bar(grow_fields, x="net_pct", y="CIP2_Name", orientation="h",
                         color_discrete_sequence=[_GREEN],
                         labels={"net_pct": "Avg Net % Change", "CIP2_Name": ""})
            fig.update_layout(height=320, margin=dict(l=5, r=5, t=5, b=5), **_dark_layout())
            st.caption("**✅ Fields with strong momentum**")
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR + ROUTING
# ──────────────────────────────────────────────────────────────────────────────
def sidebar_nav() -> str:
    st.sidebar.title("🎓 Program Intelligence")
    st.sidebar.caption("IPEDS 2019–2024 · BLS Projections")
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Navigate",
        options=["Dashboard", "Program Explorer", "Institution Browser",
                 "Career Advisor", "Job Market"],
        format_func=lambda x: {
            "Dashboard":            "🏠  Dashboard",
            "Program Explorer":     "📚  Program Explorer",
            "Institution Browser":  "🏛️  Institution Browser",
            "Career Advisor":       "🤖  Career Advisor",
            "Job Market":           "💼  Job Market",
        }[x],
    )
    st.sidebar.divider()
    st.sidebar.caption("Data: IPEDS Completions (A+C files)")
    st.sidebar.caption("Source: US Dept of Education + BLS")
    return page




# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    raw = load_raw()
    if raw is None:
        st.error(
            "**Data not found.** Run the analysis pipeline first:\n"
            "```bash\npython 'Additional Scripts/ipeds_sunset_labels.py'\n```\n"
            "This generates `out/combined_A_completions_2019_2024.csv`."
        )
        st.stop()

    wide       = build_wide(raw)
    inst_wide  = build_inst_wide(raw)
    bls        = load_bls()
    crosswalk  = load_crosswalk()
    inst_names = load_institution_names()

    page = sidebar_nav()

    if page == "Dashboard":
        page_dashboard(wide, inst_wide)
    elif page == "Program Explorer":
        page_program_explorer(wide)
    elif page == "Institution Browser":
        page_institution_browser(raw, inst_wide, inst_names)
    elif page == "Career Advisor":
        page_career_advisor(wide, bls, crosswalk)
    elif page == "Job Market":
        page_job_market(wide, bls)


if __name__ == "__main__":
    main()
