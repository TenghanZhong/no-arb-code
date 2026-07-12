from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "derived"
SOURCES = {
    "classical": ROOT / "results" / "classical_18m",
    "svi": ROOT / "results" / "svi_18m",
    "targeted_nn": ROOT / "results" / "nn_targeted",
}
PAIR_PRIORITY = {
    tuple(sorted(pair))
    for pair in [
        ("L1", "L2"),
        ("L2", "SMOOTH_RHO_0P01"),
        ("L2", "SMOOTH_RHO_0P1"),
        ("L2", "SMOOTH_RHO_1"),
        ("L2", "SMOOTH_RHO_10"),
        ("L2", "SVI"),
        ("L2", "NNPROJ"),
        ("NNPROJ", "SMOOTH_RHO_10"),
    ]
}
KEY = ["currency", "date", "expiration_date", "strike"]
SLICE = ["currency", "date", "expiration_date"]
EPS = 1e-12
BOUND_REL_TOL = 1e-6
BOUND_ABS_TOL = 1e-10


def bound_diagnostics(
    observed: np.ndarray, bound: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    bound = np.asarray(bound, dtype=float)
    ratio = np.full(observed.shape, np.nan, dtype=float)
    material = np.isfinite(bound) & (bound > BOUND_ABS_TOL)
    ratio[material] = observed[material] / bound[material]
    exceedances = observed > (bound * (1.0 + BOUND_REL_TOL) + BOUND_ABS_TOL)
    return ratio, exceedances


def d2_matrix(x: np.ndarray) -> np.ndarray:
    rows = []
    for i in range(1, len(x) - 1):
        hm = x[i] - x[i - 1]
        hp = x[i + 1] - x[i]
        row = np.zeros(len(x))
        factor = 2.0 / (hm + hp)
        row[i - 1] = factor / hm
        row[i] = -factor * (1.0 / hm + 1.0 / hp)
        row[i + 1] = factor / hp
        rows.append(row)
    return np.asarray(rows)


def parse_run(run: str) -> tuple[str, str, str]:
    match = re.match(r"^(BTC|ETH)_(\d{4}-\d{2})_(top|random|bottom)", run)
    if not match:
        return "", "", "unknown"
    return match.group(1), match.group(2), match.group(3)


def process_run(source: str, run_dir: Path) -> list[pd.DataFrame]:
    surfaces = pd.read_parquet(run_dir / "repaired_surfaces.parquet")
    greeks = pd.read_parquet(run_dir / "greeks.parquet")
    surfaces = surfaces[surfaces.get("core_repair_slice", True).astype(bool)].copy()
    greeks = greeks[greeks.get("core_repair_slice", True).astype(bool)].copy()
    arms = sorted(set(surfaces["arm"].astype(str)))
    pairs = [pair for pair in combinations(arms, 2) if tuple(sorted(pair)) in PAIR_PRIORITY]
    currency, month, selection = parse_run(run_dir.name)
    greek_groups = {
        tuple(key) if isinstance(key, tuple) else (key,): group
        for key, group in greeks.groupby(SLICE, sort=False)
    }
    pieces = []

    for slice_key, surface_slice in surfaces.groupby(SLICE, sort=False):
        slice_key = tuple(slice_key) if isinstance(slice_key, tuple) else (slice_key,)
        piv = surface_slice.pivot_table(index="strike", columns="arm", values="C_repaired", aggfunc="first")
        meta = surface_slice.sort_values("strike").drop_duplicates("strike").set_index("strike")
        gamma_slice = greek_groups.get(slice_key)
        if gamma_slice is None:
            continue
        gamma_piv = gamma_slice.pivot_table(
            index="strike", columns="arm", values="gamma_usd", aggfunc="first"
        )
        strikes = np.asarray(sorted(piv.index), dtype=float)
        if len(strikes) < 3:
            continue
        forward = float(np.nanmedian(meta.loc[strikes, "forward_price"]))
        spot = float(np.nanmedian(meta.loc[strikes, "underlyer_spot"]))
        x = strikes / forward
        D2 = d2_matrix(x)
        for arm_a, arm_b in pairs:
            if arm_a not in piv or arm_b not in piv:
                continue
            common = np.isfinite(piv.loc[strikes, arm_a]) & np.isfinite(piv.loc[strikes, arm_b])
            if not bool(common.all()):
                continue
            ca = piv.loc[strikes, arm_a].to_numpy(float)
            cb = piv.loc[strikes, arm_b].to_numpy(float)
            gap = np.abs(ca - cb)
            d2_gap = np.abs(D2 @ ca - D2 @ cb)
            eps_local = np.array([np.max(gap[i - 1 : i + 2]) for i in range(1, len(x) - 1)])
            hm = x[1:-1] - x[:-2]
            hp = x[2:] - x[1:-1]
            universal_bound = 4.0 * eps_local / (hm * hp)
            amplification_ratio, bound_exceedances = bound_diagnostics(d2_gap, universal_bound)
            model_free_gamma_gap = (x[1:-1] ** 2 / spot) * d2_gap
            lower = np.maximum(meta.loc[strikes, "C_bid"].to_numpy(float), np.maximum(1.0 - x, 0.0))
            upper = np.minimum(meta.loc[strikes, "C_ask"].to_numpy(float), 1.0)
            box_width = np.maximum(upper - lower, EPS)
            black_gamma_gap = np.abs(
                gamma_piv.loc[strikes[1:-1], arm_a].to_numpy(float)
                - gamma_piv.loc[strikes[1:-1], arm_b].to_numpy(float)
            )
            pieces.append(
                pd.DataFrame(
                    {
                        "source": source,
                        "run": run_dir.name,
                        "currency": currency or slice_key[0],
                        "month": month,
                        "selection": selection,
                        "date": int(slice_key[1]),
                        "expiration_date": int(slice_key[2]),
                        "strike": strikes[1:-1],
                        "arm_a": arm_a,
                        "arm_b": arm_b,
                        "pair": f"{arm_a}__{arm_b}",
                        "price_gap_abs": gap[1:-1],
                        "price_gap_effective_box": gap[1:-1] / box_width[1:-1],
                        "local_price_budget": eps_local,
                        "d2_gap": d2_gap,
                        "d2_universal_bound": universal_bound,
                        "d2_amplification_ratio": amplification_ratio,
                        "d2_bound_exceedance": bound_exceedances,
                        "model_free_curvature_gamma_gap": model_free_gamma_gap,
                        "black_gamma_gap": black_gamma_gap,
                        "h_minus": hm,
                        "h_plus": hp,
                    }
                )
            )
    return pieces


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pieces = []
    run_status = []
    for source, root in SOURCES.items():
        if not root.exists():
            continue
        run_dirs = sorted(
            path for path in root.iterdir()
            if path.is_dir() and not path.name.startswith("_") and (path / "repaired_surfaces.parquet").exists()
        )
        for index, run_dir in enumerate(run_dirs, 1):
            run_pieces = process_run(source, run_dir)
            pieces.extend(run_pieces)
            run_status.append({"source": source, "run": run_dir.name, "pair_slice_frames": len(run_pieces)})
            if index % 10 == 0:
                print(f"[{source}] processed {index}/{len(run_dirs)} runs", flush=True)
    if not pieces:
        raise RuntimeError("No completed surface runs were found")
    raw = pd.concat(pieces, ignore_index=True)
    raw.to_parquet(OUT / "surface_mechanism_diagnostics.parquet", index=False)
    pd.DataFrame(run_status).to_csv(OUT / "surface_mechanism_run_status.csv", index=False)

    rows = []
    group_cols = ["source", "run", "currency", "month", "selection", "pair"]
    for key, group in raw.groupby(group_cols, dropna=False):
        corr = group[["d2_gap", "black_gamma_gap"]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "n_interior_strikes": len(group),
                "median_price_gap_abs": float(group["price_gap_abs"].median()),
                "p90_price_gap_abs": float(group["price_gap_abs"].quantile(0.9)),
                "median_price_gap_effective_box": float(group["price_gap_effective_box"].median()),
                "p90_price_gap_effective_box": float(group["price_gap_effective_box"].quantile(0.9)),
                "median_d2_amplification_ratio": float(group["d2_amplification_ratio"].median()),
                "p90_d2_amplification_ratio": float(group["d2_amplification_ratio"].quantile(0.9)),
                "max_d2_amplification_ratio": float(group["d2_amplification_ratio"].max()),
                "d2_bound_exceedance_count": int(group["d2_bound_exceedance"].astype(bool).sum()),
                "median_model_free_curvature_gamma_gap": float(group["model_free_curvature_gamma_gap"].median()),
                "spearman_d2_vs_black_gamma": float(corr["d2_gap"].corr(corr["black_gamma_gap"], method="spearman")) if len(corr) >= 3 else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "surface_mechanism_by_evaluation.csv", index=False)
    summary_frame = (
        summary.groupby(["source", "pair"], as_index=False)
        .median(numeric_only=True)
    )
    summary_frame.to_csv(OUT / "surface_mechanism_summary.csv", index=False)
    audit = {
        "rows": int(len(raw)),
        "evaluation_pair_rows": int(len(summary)),
        "bound_exceedances": int(raw["d2_bound_exceedance"].astype(bool).sum()),
        "max_amplification_ratio": float(raw["d2_amplification_ratio"].max()),
        "bound_relative_tolerance": BOUND_REL_TOL,
        "bound_absolute_tolerance": BOUND_ABS_TOL,
    }
    (OUT / "surface_mechanism_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
