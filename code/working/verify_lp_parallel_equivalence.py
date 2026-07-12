from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORKING = ROOT / "code" / "working"
if str(WORKING) not in sys.path:
    sys.path.insert(0, str(WORKING))
from artifact_paths import PANEL, RESULTS  # noqa: E402

SCRIPT_DIR = ROOT / "code" / "working" / "paper_repo_scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_feasible_curvature_range as lp  # noqa: E402


CLASSICAL = RESULTS / "classical_18m"
OUTPUT = ROOT / "manifests"


def fixed_sample(targets: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    grouped = targets.groupby(["currency", "date", "expiration_date"], sort=True)
    for _, group in list(grouped)[:6]:
        pieces.append(group.sort_values("strike").head(5))
    if not pieces:
        raise RuntimeError("No LP targets available for equivalence check")
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = lp.discover_runs([CLASSICAL], None, max_runs=1)
    targets = fixed_sample(lp.collect_pairwise_targets(runs))
    serial = lp.compute_ranges(PANEL, targets, jobs=1)
    parallel = lp.compute_ranges(PANEL, targets, jobs=2)
    columns = sorted(set(serial.columns) | set(parallel.columns))
    serial = serial.reindex(columns=columns)
    parallel = parallel.reindex(columns=columns)
    pd.testing.assert_frame_equal(
        serial,
        parallel,
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )
    numeric = serial.select_dtypes(include=[np.number]).columns
    max_difference = 0.0
    if len(numeric):
        delta = np.abs(serial[numeric].to_numpy(float) - parallel[numeric].to_numpy(float))
        max_difference = float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.0
    result = {
        "runs_sampled": 1,
        "maturity_slices": int(targets[["currency", "date", "expiration_date"]].drop_duplicates().shape[0]),
        "target_rows": int(len(targets)),
        "serial_rows": int(len(serial)),
        "parallel_rows": int(len(parallel)),
        "max_numeric_difference": max_difference,
        "audit_pass": True,
    }
    (OUTPUT / "lp_parallel_equivalence.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
