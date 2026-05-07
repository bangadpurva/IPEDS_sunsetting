# Data Dimensions Available for the Student UI

The raw IPEDS completions files support more than the national CIP2 x award-level view used in the paper-facing advisor.

## Present in A Files

`data_uni/cYYYY_a.csv` includes:

- `UNITID`: institution identifier.
- `CIPCODE`: detailed academic program code.
- `MAJORNUM`: first or second major.
- `AWLEVEL`: award level.
- `CTOTALT`: total completions.
- `CTOTALM`, `CTOTALW`: completions by reported gender.
- Race/ethnicity completion columns:
  - `CAIANT`
  - `CASIAT`
  - `CBKAAT`
  - `CHISPT`
  - `CNHPIT`
  - `CWHITT`
  - `C2MORT`
  - `CUNKNT`
  - `CNRALT`

The app now builds `web/data/dimensions.json` from these files.

## Institution and Geography

The A files contain `UNITID`, but not institution name, city, state, sector, or control. To enable geography filters and readable institution names, add an IPEDS directory/header file such as:

```text
data_uni/hd2024.csv
```

Expected join columns include:

- `UNITID`
- `INSTNM`
- `CITY`
- `STABBR`
- `SECTOR`
- `CONTROL`

When this file exists, the dimensions builder will enrich institution trend rows automatically.

## Current UI Signals

The frontend now exposes:

- largest institution-program increases and declines, 2019-2024;
- 2024 gender mix by CIP2 and award level;
- a note showing whether institution geography is enabled.

