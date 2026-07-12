"""Regenerate pairwise gamma metrics from saved Greeks using matched-pair scales."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "code" / "analysis"
REPORTS = ROOT / "manifests"
COMMON = ANALYSIS / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
from pairwise_metrics import apply_pair_specific_gamma_scale  # noqa: E402


MODULES = {
    "classical": (
        ROOT / "results" / "classical_18m",
        ANALYSIS / "classical_pipeline" / "repair_analysis.py",
    ),
    "svi": (
        ROOT / "results" / "svi_18m",
        ANALYSIS / "learned_pipeline" / "repair_analysis.py",
    ),
    "targeted_nn": (
        ROOT / "results" / "nn_targeted",
        ANALYSIS / "learned_pipeline" / "repair_analysis.py",
    ),
    "seed_nn": (
        ROOT / "results" / "nn_seed_18m",
        ANALYSIS / "learned_pipeline" / "repair_analysis.py",
    ),
}
_MODULE_CACHE: dict[str, Any] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_repair_module(path: Path):
    key = str(path.resolve())
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    name = f"pair_scale_repair_{hashlib.sha1(key.encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import repair module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[key] = module
    return module


def repair_config(module: Any, path: Path):
    saved = json.loads(path.read_text(encoding="utf-8"))
    kwargs = {}
    for field in fields(module.RepairConfig):
        if field.name in saved:
            value = saved[field.name]
            if field.name == "arms" and value is not None:
                value = tuple(str(arm).upper() for arm in value)
            kwargs[field.name] = value
    return module.RepairConfig(**kwargs)


def read_frame(run_dir: Path, stem: str) -> pd.DataFrame:
    parquet = run_dir / f"{stem}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(run_dir / f"{stem}.csv")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".pairscale.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".pairscale.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def update_state_summary(run_dir: Path, strike: pd.DataFrame) -> None:
    path = run_dir / "state_summary.csv"
    if not path.exists() or strike.empty:
        return
    state = pd.read_csv(path)
    keys = [key for key in ("high_score_cell", "threshold_screen_cell", "dte_bucket") if key in state and key in strike]
    if not keys:
        return
    core = strike[strike.get("core_repair_slice", pd.Series(True, index=strike.index)).astype(bool)]
    gamma = (
        core.groupby(keys, dropna=False)["gamma_diff_scaled"]
        .agg(
            median_gamma_diff_scaled="median",
            p90_gamma_diff_scaled=lambda values: float(np.nanpercentile(values, 90)),
        )
        .reset_index()
    )
    state = state.drop(columns=["median_gamma_diff_scaled", "p90_gamma_diff_scaled"], errors="ignore")
    state = state.merge(gamma, on=keys, how="left", validate="one_to_one")
    atomic_write_csv(state, path)


def recompute_one(record: dict[str, str]) -> dict[str, object]:
    module_name = record["module"]
    run_dir = Path(record["run_dir"])
    module_path = Path(record["repair_script"])
    module = load_repair_module(module_path)
    cfg = repair_config(module, run_dir / "repair_config.json")

    strike_path = run_dir / "strike_pairwise_dispersion.csv"
    before_hash = sha256(strike_path)
    strike = read_frame(run_dir, "strike_pairwise_dispersion")
    greeks = pd.read_parquet(run_dir / "greeks.parquet")
    corrected = apply_pair_specific_gamma_scale(strike, greeks, cfg)
    atomic_write_csv(corrected, strike_path)
    atomic_write_parquet(corrected, run_dir / "strike_pairwise_dispersion.parquet")

    book = read_frame(run_dir, "book_pairwise_dispersion")
    arm_summary = module.make_arm_pair_summary(corrected, book)
    protocol = module.make_protocol_comparison_summary(corrected)
    atomic_write_csv(arm_summary, run_dir / "arm_pair_summary.csv")
    atomic_write_csv(protocol, run_dir / "protocol_comparison_summary.csv")

    panel_marking = pd.read_parquet(run_dir / "marking_panel.parquet")
    panel_repair = pd.read_parquet(run_dir / "repair_panel.parquet")
    surfaces = pd.read_parquet(run_dir / "repaired_surfaces.parquet")
    diagnostics = pd.read_csv(run_dir / "repair_diagnostics.csv")
    book_outputs = pd.read_parquet(run_dir / "book_outputs.parquet")
    repair_summary = module.summarize(
        panel_marking,
        panel_repair,
        surfaces,
        diagnostics,
        greeks,
        corrected,
        book_outputs,
        book,
        cfg,
    )
    atomic_write_csv(repair_summary, run_dir / "repair_summary.csv")
    update_state_summary(run_dir, corrected)

    metadata = {
        "metric_version": "pair_specific_gamma_scale_v1",
        "source": "saved greeks and repaired outputs; no repair optimization or neural training rerun",
        "pre_correction_strike_sha256": before_hash,
        "post_correction_strike_sha256": sha256(strike_path),
        "repair_script": str(module_path),
        "pair_scale_definition": "median absolute USD spot gamma over strikes and the two matched repair rules",
    }
    (run_dir / "pair_scale_recompute.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "module": module_name,
        "run": run_dir.name,
        "rows": int(len(corrected)),
        **metadata,
    }


def discover_seed_manifest_runs(manifest_path: Path, script: Path) -> list[dict[str, str]]:
    manifest = pd.read_csv(manifest_path)
    records = []
    for run_dir_text in manifest["output_dir"].astype(str):
        run_dir = Path(run_dir_text)
        required = [
            "strike_pairwise_dispersion.csv",
            "greeks.parquet",
            "repair_config.json",
            "repair_summary.csv",
        ]
        if all((run_dir / filename).exists() for filename in required):
            records.append(
                {"module": "seed_nn", "run_dir": str(run_dir), "repair_script": str(script)}
            )
    return records


def discover(selected_modules: list[str]) -> list[dict[str, str]]:
    records = []
    for name in selected_modules:
        root, script = MODULES[name]
        if name == "seed_nn":
            records.extend(discover_seed_manifest_runs(root / "run_manifest.csv", script))
            continue
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")):
            required = [
                "strike_pairwise_dispersion.csv",
                "greeks.parquet",
                "repair_config.json",
                "repair_summary.csv",
            ]
            if all((run_dir / filename).exists() for filename in required):
                records.append({"module": name, "run_dir": str(run_dir), "repair_script": str(script)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", nargs="+", choices=sorted(MODULES), default=sorted(MODULES))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    records = discover(args.modules)
    if not records:
        raise RuntimeError("no completed runs found for pair-scale recomputation")
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(recompute_one, record) for record in records]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"[pair-scale {index}/{len(records)}] {result['module']} {result['run']}", flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).sort_values(["module", "run"]).to_csv(
        REPORTS / "pair_scale_recompute_status.csv", index=False
    )
    print(f"Recomputed pair-specific gamma metrics for {len(results)} runs.")


if __name__ == "__main__":
    main()
