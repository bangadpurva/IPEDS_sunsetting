# Repository Structure

This repo now separates the paper pipeline from student-facing product code.

```text
ipeds_bls_projections.py       # Research paper pipeline; reproducible analytical source
Additional Scripts/            # Earlier exploratory scripts retained for provenance
data_uni/, data_stu/, out/      # Raw, intermediate, and generated research artifacts
data_ipeds_bls/                 # Preferred future output directory from the paper pipeline
app/ipeds_connect/              # App-facing data adapter, recommender, and local server
web/                            # Static student-facing frontend
web/data/programs.json          # Generated app dataset, rebuilt from research workbooks
docs/STRUCTURE.md               # This map
```

## Research Core

`ipeds_bls_projections.py` remains the canonical script for the research paper. It ingests IPEDS completions, joins the NCES CIP-SOC crosswalk, fetches BLS projections/OEWS data, creates sunset risk labels, and writes analysis workbooks/figures.

The student app intentionally reads the generated workbooks instead of importing and running the whole research script on every page load. That keeps the paper pipeline reproducible while making the product surface fast, deployable, and easier to iterate.

## App Flow

1. Run or refresh the research pipeline when new IPEDS/BLS inputs are available.
2. Build the web dataset:

   ```bash
   python3 app/build_student_data.py
   ```

   To force a full rebuild from `ipeds_bls_projections.py` first:

   ```bash
   python3 app/build_student_data.py --run-research
   ```

3. Serve the student app locally:

   ```bash
   python3 -m app.ipeds_connect.server
   ```

4. Open `http://127.0.0.1:8000`.

The local server exposes:

- `/api/research` for the live research configuration imported from `ipeds_bls_projections.py`.
- `/api/recommend` for backend advisor scoring.
- `/api/refresh` to rebuild `web/data/programs.json` from research outputs.
- `/api/refresh?run=1` to run `ipeds_bls_projections.py` first, then rebuild the app data.

## AI/Agentic Layer

The current advisor is a transparent scoring agent in `app/ipeds_connect/advisor.py`. It combines:

- student interests,
- credential target,
- risk comfort,
- labor-market growth,
- openings,
- sunset label,
- alignment score.

The app imports research labels, thresholds, output paths, and configuration through `app/ipeds_connect/research_engine.py`, so the UI stays tied to the same methodology used for the paper. This gives students an explainable shortlist instead of a black-box answer. A later deployment can add an LLM chat layer on top of the same structured recommendations, with retrieval limited to the generated research dataset.

## Deployment Path

The frontend is static and can be deployed to any static host once `web/data/programs.json` exists. For dynamic updates, run the Python app as a small API service and trigger `/api/refresh?run=1` on a schedule or after new IPEDS/BLS inputs are available.
