from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import pandas as pd

from artifact_paths import DERIVED, FIGURES, LOGS, PANEL, RESULTS


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "code" / "analysis"


def completed_run_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        1
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and (path / "repair_summary.csv").exists()
    )


def completed_manifest_run_count(manifest_path: Path) -> int:
    if not manifest_path.exists():
        return 0
    manifest = pd.read_csv(manifest_path)
    return int(
        manifest["output_dir"].astype(str).map(
            lambda path: (Path(path) / "repair_summary.csv").exists()
        ).sum()
    )


def require_inputs() -> None:
    required = [
        RESULTS / "classical_18m" / "_aggregate" / "ALL_arm_pair_summary.csv",
        RESULTS / "svi_18m" / "_aggregate" / "svi_run_level_contrasts.csv",
        RESULTS / "nn_targeted" / "_aggregate" / "ALL_ml_arm_pair_summary.csv",
        RESULTS / "nn_targeted" / "_aggregate" / "ALL_arm_pair_summary.csv",
        RESULTS / "calendar" / "calendar_spread_pair_results.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    expected_counts = {
        "classical": (RESULTS / "classical_18m", 108),
        "svi": (RESULTS / "svi_18m", 108),
        "targeted NN": (RESULTS / "nn_targeted", 10),
    }
    for label, (root, expected) in expected_counts.items():
        found = completed_run_count(root)
        if found != expected:
            missing.append(f"{label} repair summaries: found {found}, expected {expected}")
    seed_manifest = RESULTS / "nn_seed_18m" / "run_manifest.csv"
    seed_found = completed_manifest_run_count(seed_manifest)
    if seed_found != 180:
        missing.append(f"seed NN repair summaries: found {seed_found}, expected 180")
    if missing:
        raise RuntimeError("Postprocessing prerequisites are incomplete:\n" + "\n".join(missing))


def run_step(name: str, command: Sequence[str]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    record = {"name": name, "command": list(command), "cwd": str(ROOT)}
    (LOGS / f"{name}.command.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
        }
    )
    print(f"[postprocess] {name}", flush=True)
    with (LOGS / f"{name}.log").open("w", encoding="utf-8") as handle:
        subprocess.run(
            list(command), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            text=True, check=True,
        )


def main() -> None:
    require_inputs()
    python = sys.executable
    jobs = os.environ.get("REPAIR_RISK_JOBS", "4")
    classical = RESULTS / "classical_18m"
    classical_agg = classical / "_aggregate"
    targeted_agg = RESULTS / "nn_targeted" / "_aggregate"
    seed = RESULTS / "nn_seed_18m"
    persistence = DERIVED / "hedge_persistence"
    var_diagnostics = DERIVED / "var_style_diagnostics"

    steps: list[tuple[str, list[str]]] = [
        (
            "pair_scaled_metrics",
            [python, str(ANALYSIS / "recompute_pair_scaled_metrics.py"), "--jobs", jobs],
        ),
        (
            "classical_reaggregate",
            [python, str(ANALYSIS / "run_classical_18m_full.py"), "aggregate"],
        ),
        (
            "svi_reaggregate",
            [
                python,
                str(ANALYSIS / "diagnostics" / "run_svi_target_18m.py"),
                "--output-root", str(RESULTS / "svi_18m"),
                "--aggregate-only",
            ],
        ),
        (
            "targeted_nn_reaggregate",
            [python, str(ANALYSIS / "run_targeted_nn_full.py"), "aggregate"],
        ),
        (
            "seed_fit_audit",
            [python, str(ANALYSIS / "nnproj" / "nnproj_seed_replication.py"),
             "audit", "--out-root", str(seed)],
        ),
        (
            "seed_summary",
            [python, str(ANALYSIS / "nnproj" / "nnproj_seed_replication.py"),
             "summarize", "--out-root", str(seed)],
        ),
        (
            "lp_parallel_equivalence",
            [python, str(ANALYSIS / "verify_lp_parallel_equivalence.py")],
        ),
        (
            "lp_range",
            [
                python,
                str(ANALYSIS / "diagnostics" / "run_feasible_curvature_range.py"),
                "--panel", str(PANEL),
                "--output-dir", str(RESULTS / "lp_range"),
                "--classical-root", str(classical),
                "--nnproj-root", str(RESULTS / "nn_targeted"),
                "--jobs", jobs,
            ],
        ),
        (
            "lp_range_tolerance_audit",
            [python, str(ANALYSIS / "audit_lp_range_tolerance.py")],
        ),
        (
            "hedge_persistence",
            [
                python,
                str(ANALYSIS / "diagnostics" / "hedge_persistence.py"),
                "--results-root", str(classical),
                "--output-dir", str(persistence),
            ],
        ),
        (
            "var_style_self_check",
            [python, str(ANALYSIS / "diagnostics" / "var_style_diagnostics.py"),
             "--self-check-only"],
        ),
        (
            "var_style_diagnostics",
            [
                python,
                str(ANALYSIS / "diagnostics" / "var_style_diagnostics.py"),
                "--classical-aggregate", str(classical_agg),
                "--nnproj-aggregate", str(targeted_agg),
                "--persistence-dir", str(persistence),
                "--book-pairwise", str(classical_agg / "ALL_book_pairwise_dispersion.csv"),
                "--coverage-file", str(classical_agg / "ALL_run_book_nextday_coverage.csv"),
                "--output-dir", str(var_diagnostics),
            ],
        ),
        (
            "surface_diagnostics",
            [python, str(ANALYSIS / "derive_surface_diagnostics.py")],
        ),
        (
            "targeted_nn_heldout_fit",
            [
                python,
                str(ANALYSIS / "nnproj" / "audit_nnproj_heldout_fit.py"),
                "--results-root", str(RESULTS / "nn_targeted"),
        "--repair-script", str(ANALYSIS / "learned_pipeline" / "repair_analysis.py"),
                "--call-equivalent-panel", str(PANEL),
                "--output-dir", str(DERIVED / "targeted_nn_heldout_fit"),
                "--eval-source", "selected",
                "--reuse-complete",
            ],
        ),
        (
            "reported_summary",
            [python, str(ANALYSIS / "summarize_full_rerun.py")],
        ),
        (
            "figures",
            [
                python,
                str(ANALYSIS / "diagnostics" / "plot_artifact_figures.py"),
                "--results-root", str(RESULTS),
                "--lp-range-root", str(RESULTS / "lp_range"),
                "--output-dir", str(FIGURES),
            ],
        ),
        (
            "panel_identity",
            [python, str(ANALYSIS / "verify_panel_identity.py")],
        ),
        (
            "rerun_integrity",
            [python, str(ANALYSIS / "verify_full_rerun.py")],
        ),
        (
            "tests",
            [python, "-m", "pytest", "-q", str(ROOT / "tests")],
        ),
    ]
    for name, command in steps:
        run_step(name, command)
    (LOGS / "postprocessing_complete.json").write_text(
        json.dumps({"steps": [name for name, _ in steps], "status": "complete"}, indent=2),
        encoding="utf-8",
    )
    print("[postprocess] complete", flush=True)


if __name__ == "__main__":
    main()
