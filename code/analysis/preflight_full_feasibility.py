from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from artifact_paths import PANEL


ROOT = Path(__file__).resolve().parents[2]
CLASSICAL_REPAIR = ROOT / "code" / "analysis" / "classical_pipeline" / "repair_analysis.py"
SEED_REPAIR = ROOT / "code" / "analysis" / "learned_pipeline" / "repair_analysis.py"
OUT_CSV = ROOT / "manifests" / "full_feasibility_preflight_by_slice.csv"
OUT_SUMMARY = ROOT / "manifests" / "full_feasibility_preflight_by_cell.csv"

CLASSICAL_ANCHORS = ["2022-11", "2023-03", "2023-06", "2024-03", "2025-01", "2026-05"]
SEED_MONTHS = [
    "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03",
    "2024-05", "2024-08", "2024-10", "2024-12", "2025-02", "2025-04",
    "2025-06", "2025-08", "2025-10", "2025-12", "2026-02", "2026-04",
]


def load_module(path: Path = CLASSICAL_REPAIR, name: str = "repair_full_preflight"):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def selection_config(module, currency: str):
    return module.RepairConfig(
        currency=currency,
        snapshot_hour=12,
        selected_slices=None,
        max_slices=None,
    )


def classical_months() -> list[str]:
    available = pd.period_range("2022-11", "2026-05", freq="M").astype(str).tolist()
    index = {month: i for i, month in enumerate(available)}
    selected = list(CLASSICAL_ANCHORS)
    while len(selected) < 18:
        candidates = []
        for month in available:
            if month in selected:
                continue
            distance = min(abs(index[month] - index[chosen]) for chosen in selected)
            candidates.append((distance, -index[month], month))
        selected.append(max(candidates)[2])
    return sorted(selected, key=index.__getitem__)


def main() -> None:
    modules = {
        "classical": load_module(CLASSICAL_REPAIR, "repair_classical_preflight"),
        "seed": load_module(SEED_REPAIR, "repair_seed_preflight"),
    }
    classical = classical_months()
    all_rows = []
    for design, months in (("classical", classical), ("seed", SEED_MONTHS)):
        mod = modules[design]
        for currency in ("BTC", "ETH"):
            cfg = selection_config(mod, currency)
            panel = mod.load_synthetic_panel(
                PANEL,
                cfg,
                months,
                dte_min=2.0,
                dte_max=30.0,
                extra_days=0,
            )
            state = mod.add_state_flags_base(panel, cfg)
            audit = mod.audit_full_slice_feasibility(state, cfg)
            eligible = mod.select_repair_slices(state, cfg)
            eligible_keys = eligible[["currency", "date", "expiration_date"]].drop_duplicates()
            eligible_keys["rank_eligible"] = True
            audit = audit.merge(
                eligible_keys,
                on=["currency", "date", "expiration_date"],
                how="left",
                validate="one_to_one",
            )
            audit["rank_eligible"] = audit["rank_eligible"].fillna(False).astype(bool)
            audit["design"] = design
            audit["currency"] = currency
            audit["month"] = pd.to_datetime(
                audit["date"].astype("int64"), unit="s", utc=True
            ).dt.strftime("%Y-%m")
            all_rows.append(audit)
            print(
                f"[{design}:{currency}] slices={len(audit):,}; "
                f"feasible={int(audit['full_feasible'].sum()):,}; "
                f"eligible={int(audit['rank_eligible'].sum()):,}",
                flush=True,
            )

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    summary = (
        combined.groupby(["design", "currency", "month"], as_index=False)
        .agg(
            candidate_slices=("full_feasible", "size"),
            feasible_slices=("full_feasible", "sum"),
            eligible_slices=("rank_eligible", "sum"),
            max_forward_dispersion_rel=("forward_dispersion_rel", "max"),
            bad_box_slices=("bad_box_count", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        )
    )
    summary["infeasible_slices"] = summary["candidate_slices"] - summary["feasible_slices"]
    summary["required_slices"] = 50
    seed_common_count = int(summary.loc[summary["design"] == "seed", "eligible_slices"].min())
    summary.loc[summary["design"] == "seed", "required_slices"] = seed_common_count
    summary["meets_design_requirements"] = summary["eligible_slices"] >= summary["required_slices"]
    summary.to_csv(OUT_SUMMARY, index=False)

    failed = summary[~summary["meets_design_requirements"]]
    config = {
        "classical_months": classical,
        "seed_months": SEED_MONTHS,
        "classical_slices_per_cell": 50,
        "seed_replication_slices_per_cell": seed_common_count,
        "seed_replication_eligibility": "full feasibility, minimum strike count, and finite ranking score",
        "classical_repair_script": str(CLASSICAL_REPAIR),
        "seed_repair_script": str(SEED_REPAIR),
        "min_strikes": {
            design: int(modules[design].RepairConfig().min_strikes)
            for design in modules
        },
        "cells": int(len(summary)),
        "failed_cells": int(len(failed)),
    }
    (ROOT / "manifests" / "full_feasibility_preflight_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    if not failed.empty:
        raise RuntimeError("Fixed analysis design check failed:\n" + failed.to_string(index=False))


if __name__ == "__main__":
    main()
