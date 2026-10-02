import pandas as pd

import ipeds_bls_projections as research


def test_extract_cip2_preserves_leading_zero_codes():
    values = pd.Series(["01.0101", "03.0201", "09.0901", "51.3801"], dtype="string")
    assert research.extract_cip2_series(values).tolist() == ["01", "03", "09", "51"]


def test_awlevel_normalizes_mixed_year_encodings():
    values = pd.Series(["01", "1", "08", "8", "17"])
    assert research.normalize_awlevel_series(values).tolist() == ["01", "01", "08", "08", "17"]


def test_new_cip_names_are_known():
    assert research.cip2_name("29") != "Unknown/Other"
    assert research.cip2_name("39") != "Unknown/Other"
    assert research.cip2_name("41") != "Unknown/Other"
