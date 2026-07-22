from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "code" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))
from artifact_paths import PANEL as CANONICAL, RESULTS  # noqa: E402

OUTPUT = ROOT / "manifests"
MODULES = {
    "classical": RESULTS / "classical_18m",
    "svi": RESULTS / "svi_18m",
    "targeted_nn": RESULTS / "nn_targeted",
}
SEED_MANIFEST = RESULTS / "nn_seed_18m" / "run_manifest.csv"
KEY = ["currency", "date", "expiration_date", "strike"]
VALUE = ["forward_price", "C_bid", "C_ask", "C_mid"]


def normalized_panel(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[KEY + VALUE].copy()
    out["date"] = out["date"].astype("int64")
    out["expiration_date"] = out["expiration_date"].astype("int64")
    return out.sort_values(KEY).drop_duplicates(KEY, keep="first").set_index(KEY)


def seed_panel_paths(manifest_path: Path) -> list[Path]:
    manifest = pd.read_csv(manifest_path)
    return [Path(value) / "repair_panel.parquet" for value in manifest["output_dir"]]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    canonical = normalized_panel(pd.read_parquet(CANONICAL, columns=KEY + VALUE))
    rows: list[dict[str, object]] = []
    module_paths = {
        module: sorted(root.glob("*/repair_panel.parquet"))
        for module, root in MODULES.items()
        if root.exists()
    }
    module_paths["seed_nn"] = seed_panel_paths(SEED_MANIFEST)
    for module, panel_paths in module_paths.items():
        for panel_path in panel_paths:
            run = normalized_panel(pd.read_parquet(panel_path, columns=KEY + VALUE))
            common = run.index.intersection(canonical.index)
            record: dict[str, object] = {
                "module": module,
                "run": panel_path.parent.name,
                "input_rows": int(len(run)),
                "missing_canonical_rows": int(len(run) - len(common)),
            }
            for column in VALUE:
                if len(common):
                    delta = np.abs(
                        run.loc[common, column].to_numpy(float)
                        - canonical.loc[common, column].to_numpy(float)
                    )
                    record[f"max_abs_diff_{column}"] = float(np.nanmax(delta))
                else:
                    record[f"max_abs_diff_{column}"] = np.nan
            value_diffs = [record[f"max_abs_diff_{column}"] for column in VALUE]
            record["identity_pass"] = bool(
                record["missing_canonical_rows"] == 0
                and all(np.isfinite(value) and value <= 1e-14 for value in value_diffs)
            )
            rows.append(record)

    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT / "run_input_panel_identity.csv", index=False)
    summary = {
        "canonical_panel": str(CANONICAL),
        "runs_checked": int(len(audit)),
        "runs_passed": int(audit.get("identity_pass", pd.Series(dtype=bool)).sum()),
        "runs_failed": int((~audit.get("identity_pass", pd.Series(dtype=bool))).sum()),
        "max_missing_rows": int(audit.get("missing_canonical_rows", pd.Series([0])).max()),
        "max_value_difference": float(
            audit.filter(like="max_abs_diff_").max(numeric_only=True).max()
        ),
    }
    (OUTPUT / "run_input_panel_identity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if summary["runs_failed"]:
        raise RuntimeError("Run input differs from the canonical call-equivalent quote panel")


if __name__ == "__main__":
    main()
