from __future__ import annotations

import subprocess
import sys
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "code" / "analysis" / "nnproj" / "audit_nnproj_heldout_fit.py"
POSTPROCESS = ROOT / "code" / "analysis" / "run_postprocessing_full.py"


def load_postprocess_module():
    name = "test_run_postprocessing_full"
    spec = importlib.util.spec_from_file_location(name, POSTPROCESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_audit_module():
    name = "test_audit_nnproj_heldout_fit"
    spec = importlib.util.spec_from_file_location(name, AUDIT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_heldout_audit_imports_from_isolated_snapshot() -> None:
    completed = subprocess.run(
        [sys.executable, str(AUDIT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_postprocessing_runs_heldout_audit_before_reported_summary() -> None:
    source = POSTPROCESS.read_text(encoding="utf-8")
    audit_position = source.find('"targeted_nn_heldout_fit"')
    summary_position = source.find('"reported_summary"')
    assert audit_position >= 0
    assert summary_position >= 0
    assert audit_position < summary_position


def test_postprocessing_recomputes_pair_scale_before_aggregates() -> None:
    source = POSTPROCESS.read_text(encoding="utf-8")
    pair_scale = source.find('"pair_scaled_metrics"')
    classical_aggregate = source.find('"classical_reaggregate"')
    svi_aggregate = source.find('"svi_reaggregate"')
    targeted_aggregate = source.find('"targeted_nn_reaggregate"')
    seed_summary = source.find('"seed_summary"')
    assert pair_scale >= 0
    assert pair_scale < classical_aggregate
    assert pair_scale < svi_aggregate
    assert pair_scale < targeted_aggregate
    assert pair_scale < seed_summary


def test_completed_run_count_ignores_aggregate_directories(tmp_path: Path) -> None:
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "repair_summary.csv").write_text("ok\n1\n", encoding="utf-8")
    (tmp_path / "run_b").mkdir()
    (tmp_path / "_aggregate").mkdir()
    (tmp_path / "_aggregate" / "repair_summary.csv").write_text("ok\n1\n", encoding="utf-8")

    assert load_postprocess_module().completed_run_count(tmp_path) == 1


def test_heldout_audit_can_reuse_matching_complete_output(tmp_path: Path) -> None:
    runs = [tmp_path / "run_a", tmp_path / "run_b"]
    pd.DataFrame(
        [{"run": run.name, "audit_status": "ok"} for run in runs]
    ).to_csv(tmp_path / "nnproj_heldout_fit_by_run.csv", index=False)
    expected = {"eval_source": "selected", "max_eval_rows": 0, "random_state": 7}
    (tmp_path / "nnproj_heldout_fit_audit_meta.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )

    assert load_audit_module().audit_output_complete(tmp_path, runs, expected)

    expected["random_state"] = 8
    assert not load_audit_module().audit_output_complete(tmp_path, runs, expected)
