# 📊 IPEDS Program Sunsetting Analysis (2019–2024)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Status](https://img.shields.io/badge/Status-Research%20Project-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

## Overview

This project analyzes U.S. higher education program completions using IPEDS data from 2019 to 2024 to identify trends in program growth, stability, and decline. The analysis focuses on broad academic fields (2-digit CIP codes) and credential levels (AWLEVEL) to determine which programs may be expanding, shrinking, or potentially “sunsetting.”

The goal is to provide quantitative evidence to support higher education planning, workforce alignment analysis, and research into program sustainability.

---

## Data Sources

The project uses IPEDS **Completions by Program (A files)**:

- `c2019_a.csv`
- `c2020_a.csv`
- `c2021_a.csv`
- `c2022_a.csv`
- `c2023_a.csv`
- `c2024_a.csv`

### Key Variables

| Variable | Description |
|----------|-------------|
| CIPCODE | Academic program classification |
| AWLEVEL | Credential level |
| CTOTALT | Total completions |
| MAJORNUM | Primary vs secondary major indicator |
| UNITID | Institution identifier |

---

## Methodology

### 1. Data Preparation

- Combined yearly IPEDS files (2019–2024)
- Filtered to primary majors only to avoid double counting
- Aggregated completions by:

> **CIP2 (broad field) × AWLEVEL × Year**

---

### 2. Trend Metrics

#### Baseline Program Size

Average completions before disruption period:
