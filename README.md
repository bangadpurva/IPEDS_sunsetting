# IPEDS Sunsetting: Program Sunset Risk and Labor Market Alignment Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: IPEDS](https://img.shields.io/badge/Data-IPEDS%202019--2024-informational)](https://nces.ed.gov/ipeds/use-the-data)
[![Data: BLS](https://img.shields.io/badge/Data-BLS%20Projections%202024--2034-informational)](https://www.bls.gov/emp/)

A reproducible, nationally-scoped analytical pipeline that classifies U.S. academic program sunset risk and measures labor market alignment using federal public data sources — IPEDS Completions Survey, BLS Employment Projections, BLS OEWS, and the NCES CIP-SOC Crosswalk.

This repository accompanies the research paper:

> **"Mapping the Divergence: A Longitudinal Analysis of U.S. Higher Education Program Sunset Risk and Labor Market Alignment Using IPEDS Completions Data and BLS Occupational Projections (2019-2024)"**
> Institutional Research Division, 2025.

---

## Table of Contents

- [Overview](#overview)
- [Research Questions](#research-questions)
- [Data Sources](#data-sources)
- [Repository Structure](#repository-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Statistical Methods](#statistical-methods)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [AI Tool Disclosure](#ai-tool-disclosure)
- [Citation](#citation)
- [License](#license)

---

## Overview

This pipeline integrates six years of IPEDS completions data (1,354,465 records across 345 CIP2 x AWLEVEL combinations) with BLS 10-year occupational projections and historical OEWS wage data to answer two connected empirical questions:

1. Which academic programs are in **structural decline** relative to the broader portfolio?
2. Are those declines aligned with — or running against — **actual labor market demand**?

The framework is fully reproducible from publicly available federal data and is designed to be extensible to state-level, institution-level, and demographic disaggregation in future work.

---

## Research Questions

| Question | Method Used |
|---|---|
| Which programs are structurally declining? | Composite 3-metric sunset risk classifier |
| How severe is the decline relative to the portfolio? | Weighted winsorized z-score threshold model |
| Is program supply trending with occupational demand? | Pearson and Spearman correlation (CIP2 level) |
| What is the gap at the credential level? | Cohen's d degree-level alignment scoring |
| Which programs are anomalous outliers? | Weighted least-squares residual detection |
| Do institutions respond to labor market signals? | Cross-lagged correlation at 0, 12, 24, 36 months |

---

## Data Sources

All data sources are publicly available at no cost.

| Source | Description | Link |
|---|---|---|
| IPEDS Completions Survey | Annual completion records (c2019_a through c2024_a) | [NCES](https://nces.ed.gov/ipeds/use-the-data) |
| BLS Employment Projections 2024-2034 | 832 SOC-level occupational demand projections | [BLS](https://www.bls.gov/emp/tables/occupational-projections-and-characteristics.htm) |
| BLS OEWS National Files 2019-2024 | Historical occupational employment and wage data | [BLS](https://www.bls.gov/oes/tables.htm) |
| NCES CIP-SOC Crosswalk (2020-2018) | Many-to-many mapping from CIP fields to SOC occupations | [NCES](https://nces.ed.gov/ipeds/cipcode/resources.aspx?y=56) |

### Data Not Included in This Repository

IPEDS CSV files and BLS OEWS ZIP archives are not included due to file size. Download them directly from the links above and place them in the directories described in [Repository Structure](#repository-structure).

BLS Employment Projections data is fetched live from the BLS website at runtime.

---

## Repository Structure

```
IPEDS_sunsetting/
│
├── ipeds_bls_projections.py        # Main pipeline script
│
├── data_uni/                       # IPEDS annual CSV files (download separately)
│   ├── c2019_a.csv
│   ├── c2020_a.csv
│   ├── c2021_a.csv
│   ├── c2022_a.csv
│   ├── c2023_a.csv
│   └── c2024_a.csv
│
├── data_ipeds_bls/                 # Output directory (auto-created on run)
│   ├── cip_grouped_awlevel_yoy_students_2019_2024.xlsx
│   ├── bls_correlation_analysis.xlsx
│   ├── sunset_matrix_scatter.png
│   ├── sunset_matrix_heatmap.png
│   └── sunset_matrix_heatmap_large_declining.png
│
├── CIP2020_SOC2018_Crosswalk.xlsx  # NCES crosswalk file (download separately)
├── requirements.txt
└── README.md
```

---

## Pipeline Architecture

The pipeline runs as a single Python script (`ipeds_bls_projections.py`) through five sequential stages.

```
Stage 1: Data Ingestion
  Load 6x IPEDS CSV files → filter primary majors (MAJORNUM=1)
  Remove sub-associate AWLEVEL (1, 2) → extract CIP2 prefix
  Aggregate by CIP2 x AWLEVEL x YEAR

Stage 2: Feature Engineering
  Pivot to wide format (one row per CIP2 x AWLEVEL)
  Compute: baseline average (2019-2021), net % change (2019-2024)
  Compute: year-over-year % changes for each consecutive pair
  Flag: consecutive decline counts, post-pandemic recovery (2022-2024)

Stage 3: Sunset Risk Classification
  Apply 3-metric composite sunset rule per program-credential combination
  Apply weighted winsorized z-score threshold model
  Assign final composite label: High Risk / Moderate / Growth/Stable / Unstable/NA

Stage 4: BLS Alignment Analysis (4 steps)
  Step 1 — CIP2-level Pearson + Spearman correlation
  Step 2 — CIP2 x AWLEVEL Cohen's d degree-level alignment
  Step 3 — Weighted least-squares residual mismatch detection
  Step 4 — Cross-lagged correlation at 0, 12, 24, 36 months (OEWS)

Stage 5: Output
  Write structured Excel workbook (6 sheets)
  Generate scatter plots and heatmaps (PNG)
```

---

## Statistical Methods

### 1. Weighted Winsorization
Net percentage changes are clipped to `[-100, 300]` before computing the weighted distribution, using 2019-2021 baseline completions as weights. This prevents extreme outliers (e.g., Precision Production at +641%) from distorting thresholds for the broader portfolio.

- Weighted mean (mu): **1.66**
- Weighted std (sigma): **20.62**
- High Risk cutoff (z <= -2.0): net_pct <= **-39.58**
- Moderate cutoff (z <= -1.0): net_pct <= **-18.96**

### 2. Composite 3-Metric Sunset Rule

Each CIP2 x AWLEVEL combination is scored on three independent metrics:

| Metric | High Risk Threshold |
|---|---|
| Net enrollment change (2019-2024) | <= -15.6% (25th percentile) |
| Consecutive YoY declines | 3 or more years (p = 0.125 under null) |
| Post-pandemic recovery (2022-2024) | <= -5% |

- **High Risk**: 2 or more metrics flagged
- **Moderate**: 1 metric flagged
- **Low Risk**: 0 metrics flagged

### 3. Pearson and Spearman Correlation
Applied at CIP2 level to assess directional alignment between program net change and BLS occupational growth. Both coefficients reported to distinguish linear from rank-order relationships.

- Pearson r = **-0.1206** (p = 0.497) — weak, non-significant
- Spearman rho = **0.1285** (p = 0.469) — weak, non-significant

### 4. Cohen's d Degree-Level Alignment

```python
cohens_d = abs(gap) / sqrt(mean(prog_var, bls_var))
# gap = program_net_pct - bls_growth_by_degree_bucket
```

| Cohen's d | Alignment Classification |
|---|---|
| < 0.2 | Strong |
| 0.2 - 0.5 | Moderate |
| 0.5 - 0.8 | Weak |
| >= 0.8 | Misaligned |

### 5. Weighted Least-Squares Residual Detection

```python
model = sm.WLS(y, sm.add_constant(X), weights=np.maximum(baseline, 1.0)).fit()
studentized = (y - model.predict()) / residuals.std(ddof=1)
# Programs with |studentized| > 1.25 flagged as mismatch anomalies
```

Model R-squared = **0.007** — confirming near-zero aggregate fit. Outliers are genuine anomalies, not noise around a trend.

### 6. Cross-Lagged Correlation
IPEDS year-over-year completion changes cross-correlated against OEWS employment changes at lags of 0, 12, 24, and 36 months. Best lag selected by maximum absolute Pearson r.

---

## Key Findings

| Finding | Value |
|---|---|
| Total records analyzed | 1,354,465 |
| CIP2 x AWLEVEL combinations | 345 |
| High or Moderate risk combinations | 44 (12.8%) |
| Aggregate Pearson r (program vs BLS demand) | -0.12 (non-significant) |
| Engineering Bachelor's gap | -17.7 pp (supply down 8%, demand up 9.7%) |
| Public Admin Bachelor's gap | -22.4 pp (largest misalignment) |
| Precision Production growth vs BLS outlook | +641% vs -3.5% (strongest over-supply anomaly) |
| Health Professions lag correlation | r = 0.999 at 24 months |
| Business lag correlation | r = 0.999 at 24 months |
| Education lag correlation | r = -0.96 at 12 months (p = 0.041, only significant result) |

---

## Installation

### Requirements

- Python 3.10 or higher
- pip

### Steps

```bash
# Clone the repository
git clone https://github.com/bangadpurva/IPEDS_sunsetting.git
cd IPEDS_sunsetting

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas>=2.0
numpy>=1.24
scipy>=1.10
statsmodels>=0.14
matplotlib>=3.7
requests>=2.28
openpyxl>=3.1
```

---

## Usage

### 1. Download IPEDS Data

Download the six annual completions files from [NCES IPEDS](https://nces.ed.gov/ipeds/use-the-data) and place them in the `data_uni/` directory:

```
data_uni/c2019_a.csv
data_uni/c2020_a.csv
data_uni/c2021_a.csv
data_uni/c2022_a.csv
data_uni/c2023_a.csv
data_uni/c2024_a.csv
```

### 2. Download the CIP-SOC Crosswalk

Download `CIP2020_SOC2018_Crosswalk.xlsx` from [NCES](https://nces.ed.gov/ipeds/cipcode/resources.aspx?y=56) and place it in the root directory.

### 3. Run the Pipeline

```bash
python ipeds_bls_projections.py
```

BLS Employment Projections and OEWS data are fetched automatically at runtime. An active internet connection is required.

---

## Configuration

All key parameters are defined at the top of `ipeds_bls_projections.py` under the `CONFIG` section and can be adjusted without modifying the pipeline logic.

| Parameter | Default | Description |
|---|---|---|
| `KEEP_PRIMARY_MAJOR_ONLY` | `True` | Filter to MAJORNUM = 1 only |
| `REMOVE_AWLEVEL_1_AND_2` | `True` | Remove sub-associate award levels |
| `MIN_BASELINE_FOR_LABELS` | `20` | Minimum baseline completions to assign a risk label |
| `USE_WEIGHTED_THRESHOLDS` | `True` | Use baseline-weighted winsorized distribution |
| `WINSOR_LOW / WINSOR_HIGH` | `-100 / 300` | Winsorization bounds for threshold estimation |
| `Z_HIGH_RISK` | `-2.0` | Z-score cutoff for High Risk label |
| `Z_MODERATE` | `-1.0` | Z-score cutoff for Moderate label |
| `EXCLUDE_CIP2_99_FROM_DECLINE_ANALYSIS` | `True` | Exclude miscellaneous CIP2 = 99 |
| `PCT_CLIP_LOW / PCT_CLIP_HIGH` | `-100 / 100` | Clip range for visualisation only |

---

## Outputs

Running the pipeline generates the following files in `data_ipeds_bls/`:

| File | Description |
|---|---|
| `cip_grouped_awlevel_yoy_students_2019_2024.xlsx` | Full wide-format dataset with YoY changes and sunset labels |
| `bls_correlation_analysis.xlsx` | Multi-sheet BLS alignment workbook (see sheets below) |
| `sunset_matrix_scatter.png` | Scatter: baseline size vs net % change, colored by sunset label |
| `sunset_matrix_heatmap.png` | Heatmap: median net change by CIP2 field and AWLEVEL |
| `sunset_matrix_heatmap_large_declining.png` | Heatmap: large-declining share by CIP2 and AWLEVEL |

### BLS Correlation Workbook Sheets

| Sheet | Contents |
|---|---|
| `Dictionary` | Column definitions and metadata |
| `BLS_Projections_Raw` | Raw BLS occupational projection data (832 SOC rows) |
| `CIP2_BLS` | CIP2-level Pearson and Spearman correlation results |
| `CIP2_AWLEVEL_BLS` | Cohen's d degree-level alignment per CIP2 x AWLEVEL |
| `Mismatches` | WLS residual mismatch flags and studentized residuals |
| `Lag_Analysis` | Cross-lagged correlation results at 0, 12, 24, 36 months |

---

## AI Tool Disclosure

Microsoft Copilot was used to assist with **code cleanup and inline comment generation** in this pipeline. All analytical logic, statistical methodology, model design, and interpretations are the sole work of the authors. The use of AI assistance was limited to documentation and code readability improvements only.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Data used in this pipeline is sourced from U.S. federal agencies (NCES, BLS) and is in the public domain. No proprietary data is included in this repository.

---

*For questions, please open an issue on GitHub or contact the authors through the institutional research division.*