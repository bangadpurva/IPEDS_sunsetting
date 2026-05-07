import argparse

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
    path = export_json(refresh_research=args.run_research)
    print(f"Wrote {path}")
    dimensions_path = export_dimensions_json()
    print(f"Wrote {dimensions_path}")
