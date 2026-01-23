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