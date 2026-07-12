from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "code" / "analysis"
PREFLIGHT = ANALYSIS / "preflight_full_feasibility.py"
SENSITIVITY = ANALYSIS / "nnproj" / "nnproj_sensitivity.py"
REPLICATION = ANALYSIS / "nnproj" / "nnproj_seed_replication.py"
PAIR_SCALE = ANALYSIS / "recompute_pair_scaled_metrics.py"
POSTPROCESS = ANALYSIS / "run_postprocessing_full.py"
VERIFY = ANALYSIS / "verify_full_rerun.py"
PANEL_IDENTITY = ANALYSIS / "verify_panel_identity.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_seed_preflight_uses_the_exact_seed_pipeline_selection_defaults() -> None:
    preflight = load_module(PREFLIGHT, "test_seed_preflight")
    analysis = load_module(SENSITIVITY, "test_seed_sensitivity")
    seed_repair = preflight.load_module(preflight.SEED_REPAIR, "test_seed_repair")

    cfg = preflight.selection_config(seed_repair, "ETH")

    assert preflight.SEED_REPAIR.resolve() == analysis.REPAIR_SCRIPT.resolve()
    assert cfg.min_strikes == seed_repair.RepairConfig().min_strikes == 7
    assert cfg.selected_slices is None
    assert cfg.max_slices is None


def write_ranked_run(run_dir: Path, successes: list[bool]) -> None:
    rows = []
    diagnostics = []
    for rank, success in enumerate(successes, start=1):
        expiration = 1_700_000_000 + rank * 86_400
        for strike in (90.0, 100.0, 110.0):
            rows.append(
                {
                    "currency": "BTC",
                    "date": 1_690_000_000,
                    "expiration_date": expiration,
                    "strike": strike,
                    "slice_rank_desc": rank,
                    "core_repair_slice": True,
                }
            )
        diagnostics.append(
            {
                "currency": "BTC",
                "date": 1_690_000_000,
                "expiration_date": expiration,
                "arm": "NNPROJ",
                "success": success,
            }
        )
    run_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(run_dir / "repair_panel.parquet", index=False)
    pd.DataFrame(diagnostics).to_csv(run_dir / "repair_diagnostics.csv", index=False)


def test_seed_support_is_restricted_to_the_fixed_analysis_slice_cap(tmp_path: Path) -> None:
    analysis = load_module(SENSITIVITY, "test_seed_support")
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    write_ranked_run(run_a, [True, True, True])
    write_ranked_run(run_b, [True, False, True])
    group = pd.DataFrame(
        [
            {"output_dir": str(run_a), "seed": 1, "analysis_slice_cap": 2},
            {"output_dir": str(run_b), "seed": 2, "analysis_slice_cap": 2},
        ]
    )

    support = analysis.nnproj_support_sets(group)

    assert len(support["attempt_union"]) == 2
    assert len(support["attempt_intersection"]) == 2
    assert len(support["success_intersection"]) == 1


def test_seed_command_sets_the_analysis_slice_count(tmp_path: Path) -> None:
    analysis = load_module(SENSITIVITY, "test_seed_command")
    replication = load_module(REPLICATION, "test_seed_replication_command")

    cmd = replication.build_repair_command(
        analysis,
        tmp_path / "out",
        "ETH",
        "2026-04",
        20260605,
        selected_slices=47,
    )

    idx = cmd.index("--selected-slices")
    assert cmd[idx + 1] == "47"


def make_manifest(tmp_path: Path) -> Path:
    rows = []
    for name in ("btc_seed_1", "eth_seed_1"):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "repair_summary.csv").write_text("status\ncomplete\n", encoding="utf-8")
        for required in (
            "strike_pairwise_dispersion.csv",
            "greeks.parquet",
            "repair_config.json",
        ):
            (run_dir / required).touch()
        rows.append(
            {
                "run": name,
                "output_dir": str(run_dir),
                "analysis_slice_cap": 47,
            }
        )
    path = tmp_path / "run_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_seed_pair_scaling_discovers_runs_from_manifest(tmp_path: Path) -> None:
    module = load_module(PAIR_SCALE, "test_pair_scale_manifest")
    manifest = make_manifest(tmp_path)

    records = module.discover_seed_manifest_runs(manifest, Path("repair_analysis.py"))

    assert {Path(record["run_dir"]).name for record in records} == {"btc_seed_1", "eth_seed_1"}


def test_postprocessing_counts_completed_seed_runs_from_manifest(tmp_path: Path) -> None:
    module = load_module(POSTPROCESS, "test_postprocess_manifest")
    manifest = make_manifest(tmp_path)

    assert module.completed_manifest_run_count(manifest) == 2


def test_integrity_uses_selected_slice_counts_from_manifest(tmp_path: Path) -> None:
    module = load_module(VERIFY, "test_verify_manifest")
    manifest = make_manifest(tmp_path)

    specs = module.seed_run_specs(manifest)

    assert {(spec["run_dir"].name, spec["selected_slices"]) for spec in specs} == {
        ("btc_seed_1", 47),
        ("eth_seed_1", 47),
    }


def test_panel_identity_discovers_seed_panels_from_manifest(tmp_path: Path) -> None:
    module = load_module(PANEL_IDENTITY, "test_panel_identity_manifest")
    manifest = make_manifest(tmp_path)

    paths = module.seed_panel_paths(manifest)

    assert {path.parent.name for path in paths} == {"btc_seed_1", "eth_seed_1"}
