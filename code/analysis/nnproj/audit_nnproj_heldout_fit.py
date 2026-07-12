#!/usr/bin/env python3
"""Held-out fit audit for the NNProj target model.

This script does not rerun the repair solvers. It retrains the same embargoed
NNPROJ model used by a run, then evaluates the raw neural proposal on held-out
evaluation rows saved by that run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from option_repair_risk.nnproj_fit_audit import error_metrics, outside_box_metrics, regression_metrics, safe_qlike


def load_repair_module(path: Path):
    spec = importlib.util.spec_from_file_location("learned_repair_fit_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import repair script: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def infer_panel_from_stdout(run_dir: Path) -> Optional[Path]:
    log = run_dir / "run.stdout.log"
    if not log.exists():
        return None
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\[load\]\s+(.+)$", line.strip())
        if m:
            p = Path(m.group(1).strip())
            return p if p.exists() else None
    return None


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def repair_config_from_run(mod, run_cfg: Dict, nn_args: Dict):
    kwargs = {}
    for f in fields(mod.RepairConfig):
        if f.name in run_cfg:
            val = run_cfg[f.name]
            if f.name == "arms" and val is not None:
                val = tuple(str(x).upper() for x in val)
            kwargs[f.name] = val
    kwargs["currency"] = str(nn_args.get("currency", kwargs.get("currency", run_cfg.get("currency", "")))).upper()
    kwargs["snapshot_hour"] = int(nn_args.get("snapshot_hour", kwargs.get("snapshot_hour", 12)))
    kwargs["dte_min_repair"] = float(nn_args.get("dte_min_repair", kwargs.get("dte_min_repair", 2.0)))
    kwargs["dte_max_repair"] = float(nn_args.get("dte_max_repair", kwargs.get("dte_max_repair", 30.0)))
    arms = tuple(str(a).upper() for a in kwargs.get("arms", ()))
    if "NNPROJ" not in arms:
        kwargs["arms"] = tuple(list(arms) + ["NNPROJ"])
    return mod.RepairConfig(**kwargs)


def selected_eval_panel(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "repair_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected eval panel: {path}")
    return pd.read_parquet(path)


def month_eval_panel(mod, panel_path: Path, cfg, months: Iterable[str]) -> pd.DataFrame:
    return mod.load_synthetic_panel(
        panel_path,
        cfg,
        list(months),
        dte_min=float(cfg.dte_min_repair),
        dte_max=float(cfg.dte_max_repair),
        extra_days=0,
    )


def _date_range(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if "date" not in df.columns or len(df) == 0:
        return None, None
    dt = pd.to_datetime(df["date"].astype("int64"), unit="s", utc=True, errors="coerce")
    dt = dt[pd.notna(dt)]
    if len(dt) == 0:
        return None, None
    return str(dt.min()), str(dt.max())


def nnproj_xy_frame(mod, df: pd.DataFrame, *, max_rows: int, random_state: int):
    cols = ["C_mid", "forward_price", "strike", "tau"]
    g = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    g = g[(g["forward_price"] > 0) & (g["strike"] > 0) & (g["tau"] > 0) & (g["C_mid"].between(0, 1))]
    n_before = int(len(g))
    if max_rows > 0 and len(g) > max_rows:
        g = g.sample(n=max_rows, random_state=random_state)
    sig = np.array(
        [
            mod.implied_vol_from_base_call_scalar(
                float(r.C_mid), float(r.forward_price), float(r.strike), float(r.tau)
            )
            for r in g.itertuples(index=False)
        ],
        dtype=float,
    )
    valid = np.isfinite(sig) & (sig > 0) & (sig < 10)
    g = g.iloc[np.where(valid)[0]].copy()
    sig = sig[valid]
    if len(g) == 0:
        return g, np.empty((0, 7)), np.array([]), {"eval_rows_prefilter": n_before, "eval_rows_iv_valid": 0}
    tau = pd.to_numeric(g["tau"], errors="coerce").to_numpy(dtype=float)
    tv = np.maximum(sig**2 * tau, 1e-10)
    X = mod._nnproj_features_from_df(g)
    y = np.log(tv)
    valid2 = np.isfinite(X).all(axis=1) & np.isfinite(y)
    g = g.iloc[np.where(valid2)[0]].copy()
    return g, X[valid2], y[valid2], {
        "eval_rows_prefilter": n_before,
        "eval_rows_iv_valid": int(np.sum(valid)),
        "eval_rows_xy_valid": int(np.sum(valid2)),
    }


def flatten(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def audit_run(
    mod,
    run_dir: Path,
    panel_path: Path,
    eval_source: str,
    max_eval_rows: int,
    random_state: int,
    train_max_rows: Optional[int],
    max_iter: Optional[int],
    hidden_layers: Optional[str],
) -> Dict:
    run_cfg = load_json(run_dir / "repair_config.json")
    model_summary = load_json(run_dir / "nnproj_model_summary.json")
    nn_section = run_cfg.get("nnproj") or model_summary
    nn_args = dict(nn_section.get("args", {}))
    nn_args["synthetic_call_panel"] = str(panel_path)
    nn_args.setdefault("include_ml_arm", True)
    nn_args.setdefault("currency", run_cfg.get("currency"))
    nn_args.setdefault("snapshot_hour", run_cfg.get("snapshot_hour", 12))
    nn_args.setdefault("dte_min_repair", run_cfg.get("dte_min_repair", 2.0))
    nn_args.setdefault("dte_max_repair", run_cfg.get("dte_max_repair", 30.0))
    nn_args.setdefault("month", model_summary.get("args", {}).get("month", []))
    setting_notes = []
    if train_max_rows is not None:
        nn_args["nnproj_max_train_rows"] = int(train_max_rows)
        setting_notes.append(f"max_train_rows={train_max_rows}")
    if max_iter is not None:
        nn_args["nnproj_max_iter"] = int(max_iter)
        setting_notes.append(f"max_iter={max_iter}")
    if hidden_layers:
        nn_args["nnproj_hidden"] = str(hidden_layers)
        setting_notes.append(f"hidden={hidden_layers}")
    mod._NNPROJ_ARGS = nn_args

    cfg = repair_config_from_run(mod, run_cfg, nn_args)
    info = mod.train_nnproj_model(pd.DataFrame(), cfg) or {}
    row = {
        "run": run_dir.name,
        "currency": run_cfg.get("currency"),
        "month": ",".join(str(x) for x in nn_args.get("month", [])),
        "slice_selection": run_cfg.get("slice_selection"),
        "eval_source": eval_source,
        "train_status": info.get("status"),
        "n_train": info.get("n_train"),
        "train_r2": info.get("r2_train"),
        "train_start": info.get("train_start"),
        "train_end_exclusive": info.get("train_end_exclusive"),
        "training_window_valid": info.get("training_window_valid"),
        "history_panel_ok": info.get("history_panel_ok"),
        "training_settings": ";".join(setting_notes) if setting_notes else "default",
    }
    if info.get("model") is None:
        row["audit_status"] = "no_model"
        row["message"] = info.get("message")
        return row

    if eval_source == "selected":
        eval_df = selected_eval_panel(run_dir)
    elif eval_source == "month":
        eval_df = month_eval_panel(mod, panel_path, cfg, nn_args.get("month", []))
    else:
        raise ValueError(f"Unknown eval_source={eval_source}")

    eval_min, eval_max = _date_range(eval_df)
    row.update({"eval_min_date": eval_min, "eval_max_date": eval_max, "eval_rows_raw": int(len(eval_df))})
    g, X, y, xy_meta = nnproj_xy_frame(mod, eval_df, max_rows=max_eval_rows, random_state=random_state)
    row.update(xy_meta)
    if len(y) == 0:
        row["audit_status"] = "no_eval_rows"
        return row

    pred_log = np.asarray(info["model"].predict(X), dtype=float)
    pred_log = np.clip(pred_log, np.log(1e-10), np.log(25.0))
    true_tv = np.exp(y)
    pred_tv = np.exp(pred_log)
    row.update(flatten("oos_log_tv", regression_metrics(y, pred_log)))
    row["oos_total_variance_qlike"] = safe_qlike(true_tv, pred_tv)

    tau = pd.to_numeric(g["tau"], errors="coerce").to_numpy(dtype=float)
    sigma = np.sqrt(np.maximum(pred_tv / np.maximum(tau, 1e-8), 1e-10))
    sigma = np.clip(sigma, 1e-5, 10.0)
    pred_price = np.asarray(
        mod.deribit_call_base_price_from_iv(
            pd.to_numeric(g["forward_price"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(g["strike"], errors="coerce").to_numpy(dtype=float),
            tau,
            sigma,
        ),
        dtype=float,
    )
    pred_price = np.clip(np.where(np.isfinite(pred_price), pred_price, np.nan), 0.0, 1.0)
    actual_price = pd.to_numeric(g["C_mid"], errors="coerce").to_numpy(dtype=float)
    if {"C_bid", "C_ask"}.issubset(g.columns):
        bid = pd.to_numeric(g["C_bid"], errors="coerce").to_numpy(float)
        ask = pd.to_numeric(g["C_ask"], errors="coerce").to_numpy(float)
        spread = ask - bid
        row.update(flatten("oos_price_box", outside_box_metrics(pred_price, bid, ask)))
    else:
        spread = np.full_like(actual_price, np.nan)
        row.update(flatten("oos_price_box", outside_box_metrics(pred_price, spread, spread)))
    row.update(flatten("oos_price", error_metrics(actual_price, pred_price, spread, floor=float(getattr(cfg, "price_floor", 0.0005)))))
    row["audit_status"] = "ok"
    return row


def discover_runs(results_root: Path) -> list[Path]:
    return sorted(
        p for p in results_root.iterdir()
        if p.is_dir() and (p / "nnproj_model_summary.json").exists() and (p / "repair_config.json").exists()
    )


def write_aggregate(df: pd.DataFrame, out_dir: Path) -> None:
    numeric = df.select_dtypes(include=[np.number])
    rows = []
    for col in numeric.columns:
        vals = pd.to_numeric(numeric[col], errors="coerce").dropna()
        if len(vals):
            rows.append({
                "metric": col,
                "n": int(len(vals)),
                "median": float(vals.median()),
                "min": float(vals.min()),
                "max": float(vals.max()),
            })
    pd.DataFrame(rows).to_csv(out_dir / "nnproj_heldout_fit_aggregate.csv", index=False)


def audit_output_complete(out_dir: Path, runs: list[Path], expected_meta: Dict) -> bool:
    csv_path = out_dir / "nnproj_heldout_fit_by_run.csv"
    meta_path = out_dir / "nnproj_heldout_fit_audit_meta.json"
    if not csv_path.exists() or not meta_path.exists():
        return False
    try:
        existing = pd.read_csv(csv_path)
        metadata = load_json(meta_path)
    except Exception:
        return False
    expected_runs = {run.name for run in runs}
    observed_runs = set(existing.get("run", pd.Series(dtype=str)).astype(str))
    statuses = existing.get("audit_status", pd.Series(dtype=str)).astype(str)
    metadata_matches = all(metadata.get(key) == value for key, value in expected_meta.items())
    return bool(
        len(existing) == len(expected_runs)
        and observed_runs == expected_runs
        and statuses.eq("ok").all()
        and metadata_matches
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Audit held-out fit for NNProj runs without rerunning repair solvers.")
    p.add_argument("--results-root", type=Path, default=ROOT / "results" / "nn_targeted")
    p.add_argument("--repair-script", type=Path, default=ROOT / "code" / "analysis" / "learned_pipeline" / "repair_analysis.py")
    p.add_argument("--synthetic-call-panel", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--eval-source", choices=["selected", "month"], default="selected")
    p.add_argument("--max-eval-rows", type=int, default=0, help="0 means all valid eval rows.")
    p.add_argument("--random-state", type=int, default=20260605)
    p.add_argument("--limit-runs", type=int, default=0, help="Maximum runs to use; 0 uses all runs.")
    p.add_argument("--train-max-rows", type=int, default=None,
                   help="Optional training-row cap for this fit audit.")
    p.add_argument("--nnproj-max-iter", type=int, default=None,
                   help="Optional NNProj iteration limit for this fit audit.")
    p.add_argument("--nnproj-hidden", type=str, default=None,
                   help="Optional NNProj hidden-layer sizes, for example 32,32.")
    p.add_argument("--reuse-complete", action="store_true",
                   help="Reuse an exact, all-success audit with matching inputs and settings.")
    args = p.parse_args(argv)

    results_root = args.results_root.resolve()
    runs = discover_runs(results_root)
    if args.limit_runs > 0:
        runs = runs[: args.limit_runs]
    if not runs:
        raise SystemExit(f"No NNPROJ runs found under {results_root}")

    panel_path = args.synthetic_call_panel
    if panel_path is None:
        panel_path = infer_panel_from_stdout(runs[0])
    if panel_path is None:
        panel_path = ROOT / "results" / "panel_build" / "synthetic_call_panel.parquet"
    panel_path = panel_path.resolve()
    if not panel_path.exists():
        raise SystemExit(f"synthetic_call_panel not found: {panel_path}")

    out_dir = args.output_dir or (results_root / "heldout_fit")
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_meta = {
        "results_root": str(results_root),
        "repair_script": str(args.repair_script.resolve()),
        "synthetic_call_panel": str(panel_path),
        "eval_source": args.eval_source,
        "max_eval_rows": args.max_eval_rows,
        "random_state": args.random_state,
        "train_max_rows": args.train_max_rows,
        "nnproj_max_iter": args.nnproj_max_iter,
        "nnproj_hidden": args.nnproj_hidden,
        "n_runs": len(runs),
    }
    if args.reuse_complete and audit_output_complete(out_dir, runs, expected_meta):
        print(f"[reuse] complete held-out fit audit at {out_dir}", flush=True)
        return 0

    mod = load_repair_module(args.repair_script.resolve())
    rows = []
    for run_dir in runs:
        print(f"[audit] {run_dir.name}", flush=True)
        try:
            rows.append(audit_run(
                mod,
                run_dir,
                panel_path,
                args.eval_source,
                args.max_eval_rows,
                args.random_state,
                args.train_max_rows,
                args.nnproj_max_iter,
                args.nnproj_hidden,
            ))
        except Exception as e:
            rows.append({"run": run_dir.name, "audit_status": "error", "message": str(e)})

    df = pd.DataFrame(rows)
    out_csv = out_dir / "nnproj_heldout_fit_by_run.csv"
    df.to_csv(out_csv, index=False)
    write_aggregate(df, out_dir)
    meta = {
        **expected_meta,
        "output_csv": str(out_csv),
    }
    (out_dir / "nnproj_heldout_fit_audit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
