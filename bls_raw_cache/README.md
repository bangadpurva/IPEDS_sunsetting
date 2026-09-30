# BLS raw data cache

This folder holds frozen, unmodified snapshots of BLS source data, kept so the
CIP-BLS alignment analysis (`run_bls_analysis()` in `ipeds_bls_projections.py`)
can be reproduced without re-fetching from bls.gov every time.

## Why this exists

1. `www.bls.gov` began returning `403 Forbidden` to scripted `requests` calls
   (confirmed a real BLS-side block, not a dead link -- the page itself loads
   fine in a normal browser).
2. Separately, and more importantly for reproducibility: BLS refreshes its
   "Occupational projections and characteristics" table roughly annually. The
   analysis in the submitted manuscript used the **2024-34 projections
   vintage**. As of the 2026 refresh, the live page now serves the newer
   **2025-35 vintage** instead. Re-scraping live today would silently swap in
   different underlying BLS data than the manuscript describes, which is a
   correctness problem independent of the 403.

## Files

- `bls_projections_2024_2034.xlsx` -- the SOC-level occupational projections
  table (`Employment distribution, percent, 2024` / `..., 2034` columns,
  median wages, typical education, annual openings, etc.), recovered from
  `data_uni/bls_correlation_analysis.xlsx` (sheet `BLS_Projections_Raw`) as
  committed to this repo on 2026-03-06 (commit `55df401`), i.e. the exact
  table originally used to produce the manuscript's BLS alignment results.
  This data is BLS/SOC-side only -- it has no dependency on the CIP2 parsing
  bug that was fixed in `ipeds_bls_projections.py`, so it remains valid to
  reuse as-is.
- `oews/oesm{YY}nat/` (YY = 19..24) -- OEWS national employment files, used
  only for the lag-response analysis. Downloaded manually on 2026-09-30 from
  `https://www.bls.gov/oes/special-requests/oesm{YY}nat.zip` (browsers are
  not blocked by BLS's bot protection). Each folder is the browser's
  auto-extracted contents of that zip (macOS Safari extracts zip downloads
  by default) and contains one spreadsheet, `national_M{YEAR}_dl.xlsx`.

## How the script uses this folder

`load_bls_employment_projections_html()` and `load_oews_national_files()`
both check this folder first and use a cached/local file when present,
falling back to a live `requests.get()` only if it's missing. For OEWS, both
a `oesm{YY}nat.zip` file and an already-extracted `oesm{YY}nat/` folder are
accepted. Delete a file/folder here (and re-run) to force a fresh live fetch
of that specific input.
