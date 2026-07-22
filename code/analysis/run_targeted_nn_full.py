from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from artifact_paths import PANEL, RESULTS


ROOT = Path(__file__).resolve().parents[2]
REPAIR = ROOT / "code" / "analysis" / "learned_pipeline" / "repair_analysis.py"
AGGREGATE = ROOT / "code" / "analysis" / "learned_pipeline" / "aggregate_results.py"
OUT = RESULTS / "nn_targeted"
SEED = 20260605
REQUIRED = (
    "repair_summary.csv",
    "repaired_surfaces.parquet",
    "repair_diagnostics.csv",
    "strike_pairwise_dispersion.csv",
    "book_pairwise_dispersion.csv",
    "nnproj_model_summary.json",
    "slice_feasibility_audit.csv",
)


@dataclass(frozen=True)
class Run:
    currency: str
    month: str
    selection: str

    @property
    def name(self) -> str:
        return f"{self.currency}_{self.month}_{self.selection}50_nnproj_full_constraints"


RUNS = [
    *[Run(currency, "2023-03", selection) for currency in ("BTC", "ETH") for selection in ("top", "random", "bottom")],
    Run("BTC", "2023-06", "top"),
    Run("ETH", "2023-06", "top"),
    Run("BTC", "2026-05", "top"),
    Run("ETH", "2025-01", "top"),
]


def command(run: Run, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(REPAIR),
        "--call-equivalent-panel", str(PANEL),
        "--output-dir", str(out_dir),
        "--currency", run.currency,
        "--snapshot-hour", "12",
        "--month", run.month,
        "--dte-min-repair", "2",
        "--dte-max-repair", "30",
        "--dte-min-marking", "0",
        "--dte-max-marking", "31",
        "--marking-extra-days", "1",
        "--arms", "L1", "L2",
        "--smooth-rhos", "0.01", "0.1", "1", "10",
        "--include-ml-arm",
        "--nnproj-train-mode", "past",
        "--nnproj-train-lookback-months", "12",
        "--nnproj-embargo-days", "7",
        "--nnproj-min-train-months", "2",
        "--nnproj-hidden", "64,64",
        "--nnproj-random-state", str(SEED),
        "--ranking-score-method", "relspread_h2",
        "--ranking-slice-agg", "p90",
        "--selected-slices", "50",
        "--slice-selection", run.selection,
        "--random-seed", str(SEED),
        "--no-plots",
    ]


def complete(out_dir: Path) -> bool:
    return all((out_dir / name).exists() for name in REQUIRED)


def run_one(run: Run, force: bool) -> dict[str, object]:
    out_dir = OUT / run.name
    if complete(out_dir) and not force:
        return {"run": run.name, "status": "skipped_complete", "returncode": 0, "elapsed_sec": 0.0}
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = command(run, out_dir)
    (out_dir / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    started = time.time()
    with (out_dir / "stdout.log").open("w", encoding="utf-8") as stdout:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    status = "complete" if proc.returncode == 0 and complete(out_dir) else "failed"
    return {"run": run.name, "status": status, "returncode": int(proc.returncode), "elapsed_sec": elapsed}


def run_matrix(jobs: int, force: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "run": run.name,
            "currency": run.currency,
            "month": run.month,
            "selection": run.selection,
            "seed": SEED,
            "output_dir": str(OUT / run.name),
            "command": " ".join(command(run, OUT / run.name)),
        }
        for run in RUNS
    ]).to_csv(OUT / "run_manifest.csv", index=False)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(run_one, run, force): run.name for run in RUNS}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            pd.DataFrame(results).to_csv(OUT / "run_results.csv", index=False)
            print(f"[{result['status']}] {result['run']} ({result['elapsed_sec']:.1f}s)", flush=True)
    failures = [row for row in results if row["returncode"] != 0 or row["status"] == "failed"]
    if failures:
        raise RuntimeError(f"{len(failures)} targeted NN runs failed")


def aggregate() -> None:
    subprocess.run(
        [sys.executable, str(AGGREGATE), "--root", str(OUT), "--output-dir", str(OUT / "_aggregate"), "--write-strike-raw"],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "aggregate", "all"])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command in {"run", "all"}:
        run_matrix(args.jobs, args.force)
    if args.command in {"aggregate", "all"}:
        aggregate()


if __name__ == "__main__":
    main()
