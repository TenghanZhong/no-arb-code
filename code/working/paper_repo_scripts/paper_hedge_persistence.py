#!/usr/bin/env python
"""Signed hedge-gap persistence diagnostic for the risk-reversal anchor.

This script reuses existing `book_outputs.parquet` artifacts. It does not
rerun repair experiments. The diagnostic asks whether the L2--Smooth10 hedge
gap behind the 21 bps economic anchor is sign-persistent across adjacent audit
observations within each run.

The output is intentionally diagnostic. It should only be promoted into the
paper if the persistence evidence is coherent and coverage is adequate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


NEEDED_COLS = [
    "currency",
    "date",
    "expiration_date",
    "book",
    "arm",
    "hedge_underlying_units",
    "underlyer_spot",
    "nextday_available",
    "leg_qty",
]


def normalize_token(x: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(x).upper())


def arm_matches(value: object, selector: str) -> bool:
    v = normalize_token(value)
    s = normalize_token(selector)
    if v == s:
        return True
    if s in {"SMOOTH10", "SMOOTHRHO10", "SMOOTH100"}:
        return v in {"SMOOTHRHO10", "SMOOTHRHO10P0", "SMOOTH10"} or ("SMOOTH" in v and "10" in v)
    return s in v


def is_target_book(value: object, target: str) -> bool:
    v = normalize_token(value)
    t = normalize_token(target)
    if v == t:
        return True
    if t in {"RISKREVERSAL", "RISKREVERSAL5", "RR", "RR5"}:
        return "RISKREVERSAL" in v or v in {"RR", "RR5"}
    return t in v


def robust_bool(x: object, default: bool = True) -> bool:
    if pd.isna(x):
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return default


def parse_json_list(x: object) -> list[float]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in x]
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        import json as _json

        val = _json.loads(s)
        if isinstance(val, list):
            return [float(v) for v in val]
    except Exception:
        pass
    return []


def parse_run_metadata(run: str) -> dict[str, str]:
    # Examples: BTC_2022-11_top50_strict_v38, ETH_2023-03_random50_strict_v38
    m = re.match(r"(?P<currency>[A-Z]+)_(?P<month>\d{4}-\d{2})_(?P<selection>[A-Za-z]+)\d+_", run)
    if not m:
        return {"run": run}
    return {
        "run": run,
        "currency_from_run": m.group("currency"),
        "month": m.group("month"),
        "slice_selection": m.group("selection").lower(),
    }


def read_book_outputs(root: Path, include_regex: str | None, exclude_regex: str | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for f in sorted(root.rglob("book_outputs.parquet")):
        rel = f.relative_to(root).as_posix()
        rel_parts = f.relative_to(root).parts
        if any(part.startswith("_") for part in rel_parts[:-1]):
            continue
        if include_regex and not re.search(include_regex, rel):
            continue
        if exclude_regex and re.search(exclude_regex, rel):
            continue
        try:
            df = pd.read_parquet(f, columns=NEEDED_COLS)
        except Exception as e:
            print(f"[warn] skip {f}: {e}")
            continue
        if df.empty:
            continue
        meta = parse_run_metadata(f.parent.name)
        for k, v in meta.items():
            if k not in df.columns:
                df.insert(0, k, v)
        df.insert(0, "source_file", str(f))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_signed_pairs(
    df: pd.DataFrame,
    *,
    book: str,
    arm_a: str,
    arm_b: str,
    target_notional_usd: float,
) -> pd.DataFrame:
    d = df[df["book"].map(lambda x: is_target_book(x, book))].copy()
    if d.empty:
        return pd.DataFrame()
    keys = ["run", "currency", "month", "slice_selection", "date", "expiration_date", "book"]
    rows: list[dict] = []
    for _, g in d.groupby(keys, dropna=False, sort=False):
        ga = g[g["arm"].map(lambda x: arm_matches(x, arm_a))]
        gb = g[g["arm"].map(lambda x: arm_matches(x, arm_b))]
        if ga.empty or gb.empty:
            continue
        ra = ga.iloc[0]
        rb = gb.iloc[0]
        spot = float(ra["underlyer_spot"])
        ha = float(ra["hedge_underlying_units"])
        hb = float(rb["hedge_underlying_units"])
        if not all(np.isfinite(x) for x in [spot, ha, hb]) or spot <= 0:
            continue
        qty = parse_json_list(ra.get("leg_qty")) or parse_json_list(rb.get("leg_qty"))
        gross_units = sum(abs(x) for x in qty) if qty else 1.0
        base_notional_usd = gross_units * spot
        scale = target_notional_usd / base_notional_usd if base_notional_usd > 0 else np.nan
        signed_gap_units_raw = ha - hb
        signed_gap_usd_raw = signed_gap_units_raw * spot
        signed_gap_usd_target = signed_gap_usd_raw * scale if np.isfinite(scale) else np.nan
        signed_gap_bps_target = signed_gap_usd_target / target_notional_usd * 1e4 if np.isfinite(signed_gap_usd_target) else np.nan
        rows.append(
            {
                "source_file": ra["source_file"],
                "run": ra["run"],
                "currency": ra["currency"],
                "month": ra["month"],
                "slice_selection": ra["slice_selection"],
                "date": int(ra["date"]),
                "expiration_date": int(ra["expiration_date"]),
                "book": ra["book"],
                "arm_a": arm_a,
                "arm_b": arm_b,
                "spot": spot,
                "gross_units": gross_units,
                "base_notional_usd": base_notional_usd,
                "scale_to_target_notional": scale,
                "hedge_a_underlying_units": ha,
                "hedge_b_underlying_units": hb,
                "signed_gap_underlying_units_raw": signed_gap_units_raw,
                "signed_gap_usd_raw": signed_gap_usd_raw,
                "signed_gap_underlying_units_per_target": signed_gap_units_raw * scale if np.isfinite(scale) else np.nan,
                "signed_gap_usd_per_target": signed_gap_usd_target,
                "signed_gap_bps_per_target": signed_gap_bps_target,
                "abs_gap_bps_per_target": abs(signed_gap_bps_target) if np.isfinite(signed_gap_bps_target) else np.nan,
                "sign_gap": int(np.sign(signed_gap_bps_target)) if np.isfinite(signed_gap_bps_target) and signed_gap_bps_target != 0 else 0,
                "nextday_pair_available": robust_bool(ra.get("nextday_available", True), True)
                and robust_bool(rb.get("nextday_available", True), True),
            }
        )
    return pd.DataFrame(rows)


def p90(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.nanpercentile(y, 90)) if len(y) else np.nan


def run_level_coverage(pairs: pd.DataFrame) -> pd.DataFrame:
    return (
        pairs.groupby(["run", "currency", "month", "slice_selection"], dropna=False)
        .agg(
            n_pairs=("signed_gap_bps_per_target", "size"),
            nextday_pair_rate=("nextday_pair_available", "mean"),
            p90_abs_gap_bps=("abs_gap_bps_per_target", p90),
            median_signed_gap_bps=("signed_gap_bps_per_target", "median"),
            positive_share=("sign_gap", lambda s: float((pd.Series(s) > 0).mean())),
        )
        .reset_index()
    )


def persistence_by_run(pairs: pd.DataFrame) -> pd.DataFrame:
    out: list[dict] = []
    if pairs.empty:
        return pd.DataFrame()
    for key, g in pairs.sort_values(["date", "expiration_date"]).groupby(["run", "currency", "month", "slice_selection"], dropna=False):
        g = g[pd.to_numeric(g["signed_gap_bps_per_target"], errors="coerce").notna()].copy()
        if len(g) < 2:
            continue
        signs = np.sign(g["signed_gap_bps_per_target"].to_numpy(dtype=float))
        nonzero = signs != 0
        signs = signs[nonzero]
        vals = g.loc[nonzero, "signed_gap_bps_per_target"].to_numpy(dtype=float)
        if len(vals) < 2:
            continue
        same = signs[1:] == signs[:-1]
        ar1 = np.nan
        if len(vals) >= 3 and np.nanstd(vals[:-1]) > 0 and np.nanstd(vals[1:]) > 0:
            ar1 = float(np.corrcoef(vals[:-1], vals[1:])[0, 1])
        out.append(
            {
                "run": key[0],
                "currency": key[1],
                "month": key[2],
                "slice_selection": key[3],
                "n_obs": int(len(vals)),
                "n_adjacent": int(len(same)),
                "same_sign_adjacent": int(same.sum()),
                "sign_persistence_rate": float(same.mean()) if len(same) else np.nan,
                "ar1_signed_gap_bps": ar1,
                "median_signed_gap_bps": float(np.nanmedian(vals)),
                "p90_abs_gap_bps": float(np.nanpercentile(np.abs(vals), 90)),
                "positive_share": float((vals > 0).mean()),
            }
        )
    return pd.DataFrame(out)


def summarize(pairs_kept: pd.DataFrame, run_cov: pd.DataFrame, persist: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    if pairs_kept.empty:
        return pd.DataFrame()
    adj_total = int(persist["n_adjacent"].sum()) if len(persist) and "n_adjacent" in persist else 0
    same_total = int(persist["same_sign_adjacent"].sum()) if len(persist) and "same_sign_adjacent" in persist else 0
    ar = pd.to_numeric(persist.get("ar1_signed_gap_bps", pd.Series(dtype=float)), errors="coerce").dropna()
    return pd.DataFrame(
        [
            {
                "coverage_filter": min_coverage,
                "n_runs_total": int(run_cov["run"].nunique()),
                "n_runs_coverage_pass": int(run_cov.loc[run_cov["coverage_filter_pass"], "run"].nunique()),
                "n_pairs_coverage_pass": int(len(pairs_kept)),
                "median_run_p90_abs_gap_bps": float(np.nanmedian(run_cov.loc[run_cov["coverage_filter_pass"], "p90_abs_gap_bps"])),
                "median_run_p90_abs_gap_usd_per_1mm": float(np.nanmedian(run_cov.loc[run_cov["coverage_filter_pass"], "p90_abs_gap_bps"]) * 100.0),
                "overall_adjacent_same_sign_rate": same_total / adj_total if adj_total else np.nan,
                "overall_same_sign_adjacent": same_total,
                "overall_adjacent_count": adj_total,
                "median_run_sign_persistence_rate": float(np.nanmedian(persist["sign_persistence_rate"])) if len(persist) else np.nan,
                "median_run_ar1_signed_gap_bps": float(np.nanmedian(ar)) if len(ar) else np.nan,
                "median_run_positive_share": float(np.nanmedian(persist["positive_share"])) if len(persist) else np.nan,
            }
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--book", default="RISK_REVERSAL_5")
    ap.add_argument("--arm-a", default="L2")
    ap.add_argument("--arm-b", default="SMOOTH10")
    ap.add_argument("--target-notional-usd", type=float, default=1_000_000.0)
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--include-regex", default=None)
    ap.add_argument("--exclude-regex", default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_book_outputs(args.results_root, args.include_regex, args.exclude_regex)
    pairs = make_signed_pairs(
        raw,
        book=args.book,
        arm_a=args.arm_a,
        arm_b=args.arm_b,
        target_notional_usd=args.target_notional_usd,
    )
    if pairs.empty:
        raise SystemExit("No signed risk-reversal pairs found.")
    run_cov = run_level_coverage(pairs)
    run_cov["coverage_filter_pass"] = run_cov["nextday_pair_rate"] >= args.min_coverage
    kept_runs = set(run_cov.loc[run_cov["coverage_filter_pass"], "run"])
    pairs_kept = pairs[pairs["run"].isin(kept_runs)].copy()
    persist = persistence_by_run(pairs_kept[pairs_kept["nextday_pair_available"]].copy())
    summary = summarize(pairs_kept, run_cov, persist, args.min_coverage)

    pairs.to_csv(args.output_dir / "risk_reversal_signed_hedge_pairs_all.csv", index=False)
    pairs_kept.to_csv(args.output_dir / "risk_reversal_signed_hedge_pairs_coverage_filtered.csv", index=False)
    run_cov.to_csv(args.output_dir / "risk_reversal_signed_hedge_run_coverage.csv", index=False)
    persist.to_csv(args.output_dir / "risk_reversal_hedge_persistence_by_run.csv", index=False)
    summary.to_csv(args.output_dir / "risk_reversal_hedge_persistence_summary.csv", index=False)

    row = summary.iloc[0]
    note = (
        "% Hedge persistence diagnostic. Use only if promoted after review.\n"
        f"% Coverage-filtered runs: {int(row['n_runs_coverage_pass'])}; pairs: {int(row['n_pairs_coverage_pass'])}.\n"
        f"% Median run p90 absolute gap: {row['median_run_p90_abs_gap_bps']:.2f} bps "
        f"(USD {row['median_run_p90_abs_gap_usd_per_1mm']:.0f} per USD 1mm).\n"
        f"% Adjacent same-sign rate: {row['overall_adjacent_same_sign_rate']:.3f} "
        f"({int(row['overall_same_sign_adjacent'])}/{int(row['overall_adjacent_count'])}); "
        f"median run AR(1): {row['median_run_ar1_signed_gap_bps']:.3f}.\n"
    )
    (args.output_dir / "latex_note_hedge_persistence.tex").write_text(note, encoding="utf-8")
    cfg = vars(args).copy()
    cfg["results_root"] = str(cfg["results_root"])
    cfg["output_dir"] = str(cfg["output_dir"])
    (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
