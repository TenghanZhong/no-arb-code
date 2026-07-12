from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORKING = ROOT / "code" / "working"
SEED_ROOT = ROOT / "results" / "nn_seed_18m"
GATE_SCRIPT = WORKING / "project_scripts" / "nnproj_gate_escalation.py"
WRAPPER_SCRIPT = WORKING / "project_scripts" / "nnproj_seed_18m_top_audit.py"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_frames(
    source: pd.DataFrame,
    target: pd.DataFrame,
    key_columns: list[str],
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> dict[str, object]:
    source_sorted = source.sort_values(key_columns).reset_index(drop=True)
    target_sorted = target.sort_values(key_columns).reset_index(drop=True)
    keys_equal = bool(
        len(source_sorted) == len(target_sorted)
        and source_sorted[key_columns].equals(target_sorted[key_columns])
    )
    numeric_columns = sorted(
        set(source_sorted.select_dtypes(include=[np.number]).columns)
        & set(target_sorted.select_dtypes(include=[np.number]).columns)
        - set(key_columns)
    )
    max_difference = np.nan
    values_equal = False
    if keys_equal:
        if numeric_columns:
            left = source_sorted[numeric_columns].to_numpy(dtype=float)
            right = target_sorted[numeric_columns].to_numpy(dtype=float)
            difference = np.abs(left - right)
            finite = difference[np.isfinite(difference)]
            max_difference = float(finite.max()) if finite.size else 0.0
            values_equal = bool(np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True))
        else:
            max_difference = 0.0
            values_equal = True
    return {
        "source_rows": int(len(source_sorted)),
        "target_rows": int(len(target_sorted)),
        "keys_equal": keys_equal,
        "numeric_columns_compared": numeric_columns,
        "numeric_values_equal": values_equal,
        "max_abs_difference": max_difference,
        "atol": atol,
        "rtol": rtol,
        "pass": bool(keys_equal and values_equal),
    }


def choose_source_run(manifest: pd.DataFrame) -> pd.Series:
    candidates = manifest[
        manifest["amendment_action"].astype(str).eq("reuse_ranked_subset")
    ].copy()
    if candidates.empty:
        raise RuntimeError("No ranked-subset source run is available for equivalence testing")
    return candidates.sort_values(["month", "currency", "seed"]).iloc[0]


def run_equivalence_target(source: pd.Series, target_dir: Path, cap: int) -> list[str]:
    gate = load_module(GATE_SCRIPT, "seed_equivalence_gate")
    wrapper = load_module(WRAPPER_SCRIPT, "seed_equivalence_wrapper")
    command = wrapper.strict_phase2_command(
        gate,
        target_dir,
        str(source["currency"]),
        str(source["month"]),
        int(source["seed"]),
        danger_slices=cap,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "equivalence_command.json").write_text(
        json.dumps({"command": command, "source_run": source.to_dict()}, indent=2),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    with (target_dir / "equivalence_stdout.log").open("w", encoding="utf-8") as stdout, (
        target_dir / "equivalence_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, default=SEED_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seed_root = args.seed_root.resolve()
    manifest = pd.read_csv(seed_root / "gate_run_manifest.csv")
    source = choose_source_run(manifest)
    cap = int(source["analysis_slice_cap"])
    target_dir = seed_root / "equivalence_top47" / (
        f"{source['currency']}_{source['month']}_seed{int(source['seed'])}_raw{cap}"
    )
    if args.force and target_dir.exists():
        shutil.rmtree(target_dir)
    if not (target_dir / "phase2_summary.csv").exists():
        run_equivalence_target(source, target_dir, cap)

    gate = load_module(GATE_SCRIPT, "seed_equivalence_gate_compare")
    source_rec = source.to_dict()
    target_rec = {"output_dir": str(target_dir), "analysis_slice_cap": cap}
    source_dir = Path(source["output_dir"])
    files = {
        "input_panel": (
            "phase2_input_panel_repair.parquet",
            ["currency", "date", "expiration_date", "strike"],
        ),
        "surfaces": (
            "repaired_surfaces.parquet",
            ["currency", "date", "expiration_date", "strike", "arm"],
        ),
        "greeks": (
            "greeks.parquet",
            ["currency", "date", "expiration_date", "strike", "arm"],
        ),
        "books": (
            "book_outputs.parquet",
            ["currency", "date", "expiration_date", "book", "arm"],
        ),
    }
    comparisons = {}
    for label, (filename, keys) in files.items():
        source_frame = pd.read_parquet(source_dir / filename)
        target_frame = pd.read_parquet(target_dir / filename)
        source_frame = gate.filter_to_run_analysis_slices(source_frame, source_rec)
        target_frame = gate.filter_to_run_analysis_slices(target_frame, target_rec)
        comparisons[label] = compare_frames(source_frame, target_frame, keys)

    source_model = json.loads(
        (source_dir / "nnproj_oos_model_summary.json").read_text(encoding="utf-8")
    ).get("model_info", {})
    target_model = json.loads(
        (target_dir / "nnproj_oos_model_summary.json").read_text(encoding="utf-8")
    ).get("model_info", {})
    model_fields = [
        "status",
        "n_train",
        "n_train_months",
        "train_start",
        "train_end_exclusive",
        "r2_train",
    ]
    model_equal = all(source_model.get(field) == target_model.get(field) for field in model_fields)
    report = {
        "source_run": str(source["run"]),
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "analysis_slice_cap": cap,
        "source_phase2_sha256": sha256(source_dir / "phase2_summary.csv"),
        "target_phase2_sha256": sha256(target_dir / "phase2_summary.csv"),
        "model_fields_compared": model_fields,
        "model_fields_equal": model_equal,
        "comparisons": comparisons,
        "pass": bool(model_equal and all(item["pass"] for item in comparisons.values())),
    }
    output = seed_root / "seed_top49_subset_vs_native_top47_equivalence.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise RuntimeError(f"Seed slice-subset equivalence failed; inspect {output}")


if __name__ == "__main__":
    main()
