from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKING = ROOT / "code" / "working"
PIPELINES = [
    WORKING / "phase2_v38_package" / "phase2_repair_pilot_v3_8.py",
    WORKING / "phase2_v42_package" / "phase2_repair_pilot_v4_2.py",
]
LP_SCRIPT = WORKING / "paper_repo_scripts" / "run_feasible_curvature_range.py"


def load_pipeline(path: Path):
    name = f"test_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("pipeline_path", PIPELINES)
def test_active_verifier_rejects_vertical_spread_violation(pipeline_path: Path) -> None:
    mod = load_pipeline(pipeline_path)
    x = np.array([0.90, 1.00, 1.10])
    c = np.array([0.50, 0.10, 0.00])

    report = mod.verify_solution(c, np.zeros(3), np.ones(3), np.zeros(3), x)

    assert report["verified_feasible"] is False
    assert report["max_vertical_spread_violation"] > 2.9


@pytest.mark.parametrize("pipeline_path", PIPELINES)
def test_feasibility_uses_max_residual_not_sum_of_tiny_quote_residuals(
    pipeline_path: Path,
) -> None:
    mod = load_pipeline(pipeline_path)
    x = np.linspace(0.80, 1.20, 100)
    c = np.full(100, 0.30)
    bid = c + 1e-8
    ask = c + 0.01

    report = mod.verify_solution(c, bid, ask, np.zeros(100), x)

    assert report["verified_feasible"] is True
    assert report["max_box_low_violation"] < 2e-6
    assert report["box_slack_sum"] > 1e-5
    assert "box_slack_sum\"] <=" not in pipeline_path.read_text(encoding="utf-8")


def make_selection_panel() -> pd.DataFrame:
    rows = []
    date = 1_700_000_000
    specs = [
        (1_710_000_000, [0.50, 0.10, 0.00], 100.0),
        (1_720_000_000, [0.15, 0.07, 0.01], 10.0),
    ]
    strikes = [90.0, 100.0, 110.0]
    for expiration, values, danger in specs:
        for strike, value in zip(strikes, values):
            rows.append(
                {
                    "currency": "BTC",
                    "date": date,
                    "expiration_date": expiration,
                    "strike": strike,
                    "forward_price": 100.0,
                    "underlyer_spot": 100.0,
                    "tau": 0.1,
                    "DTE": 10.0,
                    "k": np.log(strike / 100.0),
                    "C_bid": max(value - 0.001, 0.0),
                    "C_ask": min(value + 0.001, 1.0),
                    "C_mid": value,
                    "danger_score": danger,
                    "danger_cell_old": False,
                    "relspread_floor": 0.02,
                    "h_norm": 0.1,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("pipeline_path", PIPELINES)
def test_selection_filters_full_infeasibility_before_ranking(
    pipeline_path: Path, tmp_path: Path
) -> None:
    mod = load_pipeline(pipeline_path)
    cfg = mod.Phase2Config(min_strikes=3, danger_slices=1, random_state=7)
    cfg.slice_selection = "top"
    cfg.full_feasibility_audit_path = str(tmp_path / "audit.csv")

    selected = mod.select_repair_slices(make_selection_panel(), cfg)
    audit = pd.read_csv(tmp_path / "audit.csv")

    assert selected["expiration_date"].nunique() == 1
    assert int(selected["expiration_date"].iloc[0]) == 1_720_000_000
    assert audit["full_feasible"].tolist() == [False, True]


def test_lp_range_uses_the_shared_full_constraint_matrix() -> None:
    lp = load_pipeline(LP_SCRIPT)
    common = load_pipeline(WORKING / "common" / "full_strike_constraints.py")
    x = np.array([0.80, 0.92, 1.05, 1.30])

    lp_A, lp_b = lp.build_lp_constraints(x)
    shared_A, shared_b, _ = common.build_full_constraint_matrix(x)

    np.testing.assert_allclose(lp_A, shared_A)
    np.testing.assert_allclose(lp_b, shared_b)


def test_lp_range_uses_the_repair_certification_tolerance() -> None:
    lp = load_pipeline(LP_SCRIPT)
    repair = load_pipeline(PIPELINES[-1])
    repair_tolerance = inspect.signature(repair.verify_solution).parameters["tol"].default

    assert lp.MATCHED_CERTIFICATION_TOL == repair_tolerance == 2e-6
    assert lp.BOX_EMPTY_TOL == 1e-12
