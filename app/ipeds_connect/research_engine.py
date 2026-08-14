from __future__ import annotations

import importlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_MODULE = "ipeds_bls_projections"


@dataclass(frozen=True)
class ResearchOutputs:
    program_workbook: Path
    bls_workbook: Path
    used_fallback: bool


def load_research_module():
    """Import the paper pipeline as the authoritative source of labels/config."""
    mpl_cache = Path("/private/tmp/ipeds-sunsetting-matplotlib")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module(RESEARCH_MODULE)


def research_config() -> dict[str, Any]:
    research = load_research_module()
    years = list(getattr(research, "YEARS", range(2019, 2025)))
    return {
        "engine": f"{RESEARCH_MODULE}.py",
        "years": years,
        "outputs": {
            "program_workbook": str(Path(getattr(research, "OUT_XLSX")).as_posix()),
            "bls_workbook": str(Path(getattr(research, "BLS_EXCEL_PATH")).as_posix()),
            "scatter": str(Path(getattr(research, "SCATTER_PATH")).as_posix()),
            "heatmap": str(Path(getattr(research, "HEATMAP_PATH")).as_posix()),
        },
        "thresholds": {
            "min_baseline_for_labels": getattr(research, "MIN_BASELINE_FOR_LABELS", None),
            "winsor_low": getattr(research, "WINSOR_LOW", None),
            "winsor_high": getattr(research, "WINSOR_HIGH", None),
            "z_high_risk": getattr(research, "Z_HIGH_RISK", None),
            "z_moderate": getattr(research, "Z_MODERATE", None),
        },
        "filters": {
            "primary_major_only": getattr(research, "KEEP_PRIMARY_MAJOR_ONLY", None),
            "remove_awlevel_1_and_2": getattr(research, "REMOVE_AWLEVEL_1_AND_2", None),
            "exclude_cip2_99": getattr(research, "EXCLUDE_CIP2_99_FROM_DECLINE_ANALYSIS", None),
        },
        "cip2_labels": getattr(research, "CIP2_TO_NAME", {}),
        "awlevel_labels": getattr(research, "AWLEVEL_TO_NAME", {}),
    }


def expected_output_paths() -> tuple[Path, Path]:
    research = load_research_module()
    return ROOT / getattr(research, "OUT_XLSX"), ROOT / getattr(research, "BLS_EXCEL_PATH")


def resolve_research_outputs() -> ResearchOutputs:
    expected_program, expected_bls = expected_output_paths()
    if expected_program.exists() and expected_bls.exists():
        return ResearchOutputs(expected_program, expected_bls, used_fallback=False)

    fallback_program = ROOT / "data_uni" / "cip_grouped_awlevel_yoy_students_2019_2024.xlsx"
    fallback_bls = ROOT / "data_uni" / "bls_correlation_analysis.xlsx"
    if fallback_program.exists() and fallback_bls.exists():
        return ResearchOutputs(fallback_program, fallback_bls, used_fallback=True)

    missing = [str(p.relative_to(ROOT)) for p in (expected_program, expected_bls) if not p.exists()]
    raise FileNotFoundError(
        "Research outputs are missing. Run ipeds_bls_projections.py first. Missing: "
        + ", ".join(missing)
    )


def run_research_pipeline() -> subprocess.CompletedProcess[str]:
    """Run the paper pipeline exactly as a researcher would run it."""
    return subprocess.run(
        [sys.executable, str(ROOT / "ipeds_bls_projections.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def stream_research_pipeline():
    """Yield output lines while the paper pipeline runs."""
    mpl_cache = Path("/private/tmp/ipeds-sunsetting-matplotlib")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [sys.executable, str(ROOT / "ipeds_bls_projections.py")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        yield line.rstrip()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process.args)


def output_freshness(paths: list[Path]) -> dict[str, str | None]:
    freshness: dict[str, str | None] = {}
    for path in paths:
        key = str(path.relative_to(ROOT))
        freshness[key] = None
        if path.exists():
            freshness[key] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return freshness
