from pathlib import Path

import pytest

from app.refresh_data import discover_releases


def _touch_years(folder: Path, years: list[int]) -> None:
    for year in years:
        (folder / f"c{year}_a.csv").write_text("UNITID,CIPCODE,AWLEVEL,CTOTALT\n", encoding="utf-8")


def test_discovers_contiguous_releases(tmp_path: Path):
    _touch_years(tmp_path, [2022, 2023, 2024])
    assert list(discover_releases(tmp_path)) == [2022, 2023, 2024]


def test_rejects_gap_in_releases(tmp_path: Path):
    _touch_years(tmp_path, [2021, 2022, 2024])
    with pytest.raises(ValueError, match="2023"):
        discover_releases(tmp_path)


def test_requires_three_year_baseline(tmp_path: Path):
    _touch_years(tmp_path, [2023, 2024])
    with pytest.raises(ValueError, match="three"):
        discover_releases(tmp_path)
