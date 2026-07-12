"""Matched-pair scaling for strike-level repair diagnostics."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


SLICE_KEYS = ["currency", "date", "expiration_date"]
EPS = 1e-12


def _cfg_value(cfg: Any, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def pair_gamma_scales(greeks: pd.DataFrame, epsilon: float = EPS) -> pd.DataFrame:
    required = set(SLICE_KEYS + ["arm", "gamma_usd"])
    missing = sorted(required - set(greeks.columns))
    if missing:
        raise ValueError(f"greeks missing pair-scale columns: {missing}")
    rows: list[dict[str, object]] = []
    for slice_key, group in greeks.groupby(SLICE_KEYS, sort=False):
        arms = sorted(group["arm"].dropna().astype(str).unique())
        for arm_a, arm_b in combinations(arms, 2):
            values = pd.to_numeric(
                group.loc[group["arm"].astype(str).isin({arm_a, arm_b}), "gamma_usd"],
                errors="coerce",
            ).abs().dropna()
            scale = float(values.median()) if len(values) else float(epsilon)
            rows.append(
                {
                    **dict(zip(SLICE_KEYS, slice_key)),
                    "_pair_a": arm_a,
                    "_pair_b": arm_b,
                    "_pair_gamma_scale": max(scale, float(epsilon)),
                }
            )
    return pd.DataFrame(rows)


def apply_pair_specific_gamma_scale(
    strike_dispersion: pd.DataFrame,
    greeks: pd.DataFrame,
    cfg: Any,
    epsilon: float = EPS,
) -> pd.DataFrame:
    """Replace all-arm gamma scaling with the manuscript's matched-pair scale."""
    if strike_dispersion.empty:
        return strike_dispersion.copy()
    required = set(SLICE_KEYS + ["arm_a", "arm_b", "gamma_diff", "gamma_rr", "price_diff_box"])
    missing = sorted(required - set(strike_dispersion.columns))
    if missing:
        raise ValueError(f"strike dispersion missing pair-scale columns: {missing}")

    output = strike_dispersion.copy()
    output["_pair_a"] = output[["arm_a", "arm_b"]].astype(str).min(axis=1)
    output["_pair_b"] = output[["arm_a", "arm_b"]].astype(str).max(axis=1)
    scales = pair_gamma_scales(greeks, epsilon=epsilon)
    output = output.merge(scales, on=SLICE_KEYS + ["_pair_a", "_pair_b"], how="left", validate="many_to_one")
    if output["_pair_gamma_scale"].isna().any():
        missing_rows = int(output["_pair_gamma_scale"].isna().sum())
        raise ValueError(f"missing matched-pair gamma scale for {missing_rows} strike rows")

    output["slice_gamma_scale"] = output["_pair_gamma_scale"]
    output["gamma_diff_scaled"] = (
        pd.to_numeric(output["gamma_diff"], errors="coerce") / output["_pair_gamma_scale"]
    )
    material = (
        pd.to_numeric(output["price_diff_box"], errors="coerce")
        < _cfg_value(cfg, "price_diff_box_threshold", 0.25)
    ) & (
        pd.to_numeric(output["gamma_rr"], errors="coerce")
        > _cfg_value(cfg, "gamma_rr_threshold", 0.20)
    ) & (
        pd.to_numeric(output["gamma_diff_scaled"], errors="coerce")
        > _cfg_value(
            cfg,
            "gamma_scaled_only_threshold",
            _cfg_value(cfg, "gamma_material_threshold", 0.05),
        )
    )
    output["small_price_large_gamma"] = material.fillna(False).astype(bool)
    output["small_price_large_gamma_scaled"] = material.fillna(False).astype(bool)
    return output.drop(columns=["_pair_a", "_pair_b", "_pair_gamma_scale"])
