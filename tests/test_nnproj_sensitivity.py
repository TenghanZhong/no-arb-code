import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "analysis"
    / "nnproj"
    / "nnproj_sensitivity.py"
)


def load_sensitivity_module():
    spec = importlib.util.spec_from_file_location("nnproj_sensitivity_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_architecture_pair_kind_separates_seed_from_architecture():
    mod = load_sensitivity_module()

    assert mod.architecture_pair_kind("32x32", 1, "32x32", 2) == "within_arch"
    assert mod.architecture_pair_kind("32x32", 1, "64x64", 1) == "same_seed_between_arch"
    assert mod.architecture_pair_kind("32x32", 1, "64x64", 2) == "cross_seed_between_arch"


def test_analysis_criteria_use_common_success_support():
    mod = load_sensitivity_module()
    summary = pd.DataFrame(
        {
            "n_seeds_success_ge_90pct": [5, 5],
            "seed_pair_excess_over_l1_l2": [1.0, 1.0],
            "fit_10pct_seed_count": [5, 5],
            "fit_10pct_seed_pair_excess": [1.0, 1.0],
            "fit_20pct_seed_count": [5, 5],
            "fit_20pct_seed_pair_excess": [1.0, 1.0],
            "min_leave_one_seed_excess": [1.0, 1.0],
            "common_support_seed_pair_excess_over_l1_l2": [-0.1, -0.2],
            "common_support_fit_10pct_seed_pair_excess": [-0.1, -0.2],
            "common_support_fit_20pct_seed_pair_excess": [-0.1, -0.2],
            "common_support_min_leave_one_seed_excess": [-0.1, -0.2],
        }
    )

    status = mod.build_analysis_status(Path("analysis"), summary, pd.DataFrame())

    assert status["seed_excess_positive_windows"] == 2
    assert status["common_support_seed_excess_positive_windows"] == 0
    assert status["analysis_criteria_pass"] is False
