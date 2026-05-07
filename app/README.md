# IPEDS Student Pathways App

This layer turns the research output from `ipeds_bls_projections.py` into a student-facing exploration tool.

## Build Data

```bash
python3 app/build_student_data.py
```

Use the research script as the rebuild engine:

```bash
python3 app/build_student_data.py --run-research
```

The builder looks first in `data_ipeds_bls/` and then falls back to the current generated workbooks in `data_uni/`.

## Run Locally

```bash
python3 -m app.ipeds_connect.server
```

Then open `http://127.0.0.1:8000`.

The UI can refresh from existing outputs or run `ipeds_bls_projections.py` before rebuilding the app dataset. Full research refreshes need network access because the paper pipeline fetches BLS data.

## Product Direction

The web app is designed for students evaluating education and career advancement paths. It is informative, interactive, and uses an explainable advisor score. The next AI step should be a chat interface that grounds every response in `web/data/programs.json` and cites the underlying IPEDS/BLS metrics.
