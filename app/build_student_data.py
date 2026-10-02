import argparse
from pathlib import Path

from ipeds_connect.data_adapter import export_json
from ipeds_connect.dimensions import export_dimensions_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the student-facing JSON dataset.")
    parser.add_argument(
        "--run-research",
        action="store_true",
        help="Run ipeds_bls_projections.py before exporting the web dataset.",
    )
    args = parser.parse_args()
    for root in (Path("web/data"), Path("site/public/data")):
        path = export_json(output_path=root / "programs.json", refresh_research=args.run_research)
        print(f"Wrote {path}")
        dimensions_path = export_dimensions_json(output_path=root / "dimensions.json")
        print(f"Wrote {dimensions_path}")
