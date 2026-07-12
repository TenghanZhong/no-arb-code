from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "working" / "paper_repo_scripts" / "paper_hedge_persistence.py"


def load_module():
    name = "test_paper_hedge_persistence"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_signed_hedge_pair_is_scaled_by_fixed_gross_spot_notional() -> None:
    rows = []
    for arm, hedge in [("L2", 0.2), ("SMOOTH_RHO_10", 0.1)]:
        rows.append(
            {
                "source_file": "book_outputs.parquet",
                "run": "BTC_2024-01_top50_fullarb_v50",
                "currency": "BTC",
                "month": "2024-01",
                "slice_selection": "top",
                "date": 1704067200,
                "expiration_date": 1706745600,
                "book": "RISK_REVERSAL_5",
                "arm": arm,
                "underlyer_spot": 100.0,
                "hedge_underlying_units": hedge,
                "leg_qty": "[1.0, -1.0]",
                "nextday_available": True,
            }
        )

    pair = load_module().make_signed_pairs(
        pd.DataFrame(rows),
        book="RISK_REVERSAL_5",
        arm_a="L2",
        arm_b="SMOOTH10",
        target_notional_usd=1_000_000.0,
    ).iloc[0]

    assert pair["gross_units"] == pytest.approx(2.0)
    assert pair["base_notional_usd"] == pytest.approx(200.0)
    assert pair["signed_gap_bps_per_target"] == pytest.approx(500.0)
    assert pair["signed_gap_usd_per_target"] == pytest.approx(50_000.0)
