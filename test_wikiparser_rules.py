"""Offline checks of the cleaning rules (no network): header units, row/column NULL handling, popularity guard."""
import numpy as np
import pandas as pd

from wikiparser import is_unit_annotation, strip_header_parentheses, WikiTableParser


def test_unit_parentheses_are_kept_and_notes_removed():
    keep = {"Budget (US$)": "Budget (US$)", "Area (km2)": "Area (km2)", "Speed (km/h)": "Speed (km/h)",
            "Duration (min:sec)": "Duration (min:sec)", "Share (%)": "Share (%)", "Rate (per 1000)": "Rate (per 1000)",
            "GDP (in bil. US$ PPP)": "GDP (in bil. US$ PPP)", "Weight (kg) (2020)": "Weight (kg)"}
    drop = {"Rank (2020)": "Rank", "Winner(s)": "Winner", "Date of birth (age)": "Date of birth",
            "Time (Greatest) UTC": "Time UTC", "Population (2021/22 census)": "Population", "Goals (NCBI)": "Goals"}
    for h, want in {**keep, **drop}.items():
        assert strip_header_parentheses(h) == want, h
    assert strip_header_parentheses("Budget (US$)", keep_units=False) == "Budget"
    assert not is_unit_annotation("s") and not is_unit_annotation("") and is_unit_annotation("US$")


def _parser():
    p = WikiTableParser.__new__(WikiTableParser)
    import logging
    p.logger = logging.getLogger("t")
    p.url = "t"
    p.cfg = {"maximum_column_null_values": {"type": "percent", "value": 0.4}, "drop_table_on_column_removal": False}
    return p


def test_sparse_rows_are_kept_sparse_columns_dropped_all_null_rows_removed():
    p = _parser()
    p.df = pd.DataFrame({"a": ["x", "y", "z", "w", "v"], "b": [1, np.nan, 3, 4, np.nan],
                         "c": [np.nan, np.nan, np.nan, np.nan, np.nan], "d": [np.nan, np.nan, 3, np.nan, np.nan]})
    p.df.loc[4, ["a", "b", "c", "d"]] = np.nan                       # a completely empty row
    p.nulls_check()
    assert list(p.df.columns) == ["a", "b"]                          # c and d are more than 40% NULL
    assert list(p.df.index) == [0, 1, 2, 3]                          # row 1 has a NULL in b but is kept; the empty row goes
