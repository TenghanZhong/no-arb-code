from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "working" / "common" / "pairwise_metrics.py"


def load_module():
    name = "test_pairwise_metrics"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_gamma_scale_uses_only_the_matched_pair() -> None:
    greeks = pd.DataFrame(
        [
            {"currency": "BTC", "date": 1, "expiration_date": 2, "strike": 90, "arm": "A", "gamma_usd": 1.0},
            {"currency": "BTC", "date": 1, "expiration_date": 2, "strike": 100, "arm": "A", "gamma_usd": 3.0},
            {"currency": "BTC", "date": 1, "expiration_date": 2, "strike": 90, "arm": "B", "gamma_usd": 2.0},
            {"currency": "BTC", "date": 1, "expiration_date": 2, "strike": 100, "arm": "B", "gamma_usd": 4.0},
            {"currency": "BTC", "date": 1, "expiration_date": 2, "strike": 90, "arm": "C", "gamma_usd": 1000.0},
            {"currency": "BTC", "date": 1, "expiration_date": 2, "strike": 100, "arm": "C", "gamma_usd": 1000.0},
        ]
    )
    strike = pd.DataFrame(
        [
            {
                "currency": "BTC",
                "date": 1,
                "expiration_date": 2,
                "strike": 90,
                "arm_a": "A",
                "arm_b": "B",
                "gamma_diff": 1.0,
                "gamma_rr": 0.5,
                "price_diff_box": 0.1,
            }
        ]
    )
    cfg = SimpleNamespace(
        price_diff_box_threshold=0.25,
        gamma_rr_threshold=0.2,
        gamma_scaled_only_threshold=0.05,
    )

    result = load_module().apply_pair_specific_gamma_scale(strike, greeks, cfg)

    assert result.loc[0, "slice_gamma_scale"] == pytest.approx(2.5)
    assert result.loc[0, "gamma_diff_scaled"] == pytest.approx(0.4)
    assert bool(result.loc[0, "small_price_large_gamma"])
    assert bool(result.loc[0, "small_price_large_gamma_scaled"])
