from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_compact_figure_sources_have_final_support() -> None:
    expected = {
        "figure3_rho_path.csv": 864,
        "figure4_lp_range.csv": 51189,
        "figure5_nnproj_pairs.csv": 360,
        "figure6_var_relative_difference.csv": 12,
    }
    for name, rows in expected.items():
        assert len(pd.read_csv(ROOT / "outputs" / "figure_sources" / name)) == rows


def test_compact_lp_source_reproduces_reported_association() -> None:
    frame = pd.read_csv(ROOT / "outputs" / "figure_sources" / "figure4_lp_range.csv")
    spearman = frame["feasible_cxx_range"].corr(frame["cxx_diff"].abs(), method="spearman")
    summary = json.loads((ROOT / "outputs" / "full_rerun_summary.json").read_text(encoding="utf-8"))
    # The compact CSV is decimal serialized, so allow sub-1e-8 rounding drift.
    assert spearman == pytest.approx(summary["lp_range"]["spearman_range_vs_abs_cxx"], abs=1e-8)


def test_panel_digest_manifest_is_well_formed() -> None:
    digest, size, name = (ROOT / "data" / "input_panel.sha256").read_text(encoding="ascii").split()
    assert len(digest) == 64
    int(digest, 16)
    assert int(size) == 12_900_106
    assert name == "synthetic_call_panel.parquet"
