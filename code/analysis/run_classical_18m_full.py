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
REPAIR = ROOT / "code" / "analysis" / "classical_pipeline" / "repair_analysis.py"
AGGREGATE = ROOT / "code" / "analysis" / "classical_pipeline" / "aggregate_results.py"
OUT = RESULTS / "classical_18m"
SEED = 20260605
REQUIRED = (
    "repair_summary.csv",
    "repair_arm_summary.csv",
    "strike_pairwise_dispersion.csv",
    "book_pairwise_dispersion.csv",
    "arm_pair_summary.csv",
    "var_style_arm_summary.csv",
    "repair_diagnostics.csv",
    "slice_feasibility_audit.csv",
)


def selected_months() -> list[str]:
    anchors = ["2022-11", "2023-03", "2023-06", "2024-03", "2025-01", "2026-05"]
    available = pd.period_range("2022-11", "2026-05", freq="M").astype(str).tolist()
    index = {month: i for i, month in enumerate(available)}
    selected = list(anchors)
    while len(selected) < 18:
        candidates = []
        for month in available:
            if month in selected:
                continue
            distance = min(abs(index[month] - index[chosen]) for chosen in selected)
            candidates.append((distance, -index[month], month))
        selected.append(max(candidates)[2])
    return sorted(selected, key=index.__getitem__)


@dataclass(frozen=True)
class Run:
    currency: str
    month: str
    selection: str

    @property
    def name(self) -> str:
        return f"{self.currency}_{self.month}_{self.selection}50_full_constraints"


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
    runs = [
        Run(currency, month, selection)
        for currency in ("BTC", "ETH")
        for month in selected_months()
        for selection in ("top", "random", "bottom")
    ]
    manifest = pd.DataFrame([
        {
            "run": run.name,
            "currency": run.currency,
            "month": run.month,
            "selection": run.selection,
            "selected_slices": 50,
            "output_dir": str(OUT / run.name),
            "command": " ".join(command(run, OUT / run.name)),
        }
        for run in runs
    ])
    manifest.to_csv(OUT / "run_manifest.csv", index=False)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(run_one, run, force): run.name for run in runs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            pd.DataFrame(results).to_csv(OUT / "run_results.csv", index=False)
            print(f"[{result['status']}] {result['run']} ({result['elapsed_sec']:.1f}s)", flush=True)
    failures = [row for row in results if row["returncode"] != 0 or row["status"] == "failed"]
    if failures:
        raise RuntimeError(f"{len(failures)} classical runs failed; see {OUT / 'run_results.csv'}")


def aggregate() -> None:
    cmd = [
        sys.executable,
        str(AGGREGATE),
        "--root", str(OUT),
        "--output-dir", str(OUT / "_aggregate"),
        "--write-strike-raw",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "aggregate", "all"])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command in {"run", "all"}:
        run_matrix(args.jobs, args.force)
    if args.command in {"aggregate", "all"}:
        aggregate()


if __name__ == "__main__":
    main()
