from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "analysis" / "audit_lp_range_tolerance.py"


def load_module():
    name = "test_audit_lp_range_tolerance"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_refinement_returns_a_tightly_feasible_curve() -> None:
    module = load_module()
    x = np.array([0.8, 1.0, 1.2])
    lower = np.zeros(3)
    upper = np.ones(3)
    approximate = np.array([0.30000002, 0.1, 0.05])

    refined, report = module.refine_feasible_curve(approximate, x, lower, upper)

    assert report["success"] is True
    assert report["parameterization"] == "nonincreasing_vertical_slopes"
    assert report["max_adjustment"] > 0.0
    assert report["max_constraint_violation"] <= module.REFINEMENT_SOLVER_TOLERANCE
    assert np.max(np.abs(refined - approximate)) == report["max_adjustment"]
