from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "working" / "derive_surface_diagnostics.py"


def load_module():
    name = "test_derive_surface_diagnostics"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_process_run_matches_greeks_by_slice_without_tuple_broadcast(tmp_path: Path) -> None:
    strikes = [90.0, 100.0, 110.0]
    surface_rows = []
    greek_rows = []
    prices = {"L2": [0.16, 0.10, 0.06], "SMOOTH_RHO_10": [0.15, 0.095, 0.055]}
    gammas = {"L2": [0.01, 0.02, 0.01], "SMOOTH_RHO_10": [0.012, 0.018, 0.012]}
    for arm in prices:
        for strike, price, gamma in zip(strikes, prices[arm], gammas[arm]):
            common = {
                "currency": "BTC",
                "date": 20240115,
                "expiration_date": 20240215,
                "strike": strike,
                "arm": arm,
                "core_repair_slice": True,
            }
            surface_rows.append(
                {
                    **common,
                    "C_repaired": price,
                    "forward_price": 100.0,
                    "underlyer_spot": 99.0,
                    "C_bid": max(price - 0.02, 0.0),
                    "C_ask": min(price + 0.02, 1.0),
                }
            )
            greek_rows.append({**common, "gamma_usd": gamma})
    pd.DataFrame(surface_rows).to_parquet(tmp_path / "repaired_surfaces.parquet", index=False)
    pd.DataFrame(greek_rows).to_parquet(tmp_path / "greeks.parquet", index=False)

    pieces = load_module().process_run("classical", tmp_path)

    assert len(pieces) == 1
    assert len(pieces[0]) == 1
    assert pieces[0].iloc[0]["pair"] == "L2__SMOOTH_RHO_10"


def test_bound_diagnostics_ignore_machine_precision_ratios() -> None:
    ratio, flags = load_module().bound_diagnostics(
        np.array([1.8e-12, 0.5]), np.array([1.2e-12, 1.0])
    )

    assert np.isnan(ratio[0])
    assert flags.tolist() == [False, False]
    assert ratio[1] == 0.5
