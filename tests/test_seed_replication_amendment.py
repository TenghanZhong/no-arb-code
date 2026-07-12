from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKING = ROOT / "code" / "working"
PREFLIGHT = WORKING / "preflight_full_feasibility.py"
GATE = WORKING / "project_scripts" / "nnproj_gate_escalation.py"
WRAPPER = WORKING / "project_scripts" / "nnproj_seed_18m_top_audit.py"
PAIR_SCALE = WORKING / "recompute_pair_scaled_metrics.py"
POSTPROCESS = WORKING / "run_postprocessing_full.py"
VERIFY = WORKING / "verify_full_rerun.py"
EQUIVALENCE = WORKING / "verify_seed_slice_subset_equivalence.py"
PANEL_IDENTITY = WORKING / "verify_panel_identity.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_seed_preflight_uses_the_exact_seed_pipeline_selection_defaults() -> None:
    preflight = load_module(PREFLIGHT, "test_seed_preflight")
    gate = load_module(GATE, "test_seed_gate")
    seed_phase2 = preflight.load_module(preflight.SEED_PHASE2, "test_seed_phase2")

    cfg = preflight.selection_config(seed_phase2, "ETH")

    assert preflight.SEED_PHASE2.resolve() == gate.PHASE2.resolve()
    assert cfg.min_strikes == seed_phase2.Phase2Config().min_strikes == 7
    assert cfg.danger_slices is None
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
    pd.DataFrame(rows).to_parquet(run_dir / "phase2_input_panel_repair.parquet", index=False)
    pd.DataFrame(diagnostics).to_csv(run_dir / "repair_diagnostics.csv", index=False)


def test_seed_support_is_restricted_to_the_frozen_analysis_slice_cap(tmp_path: Path) -> None:
    gate = load_module(GATE, "test_seed_gate_support")
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

    support = gate.nnproj_support_sets(group)

    assert len(support["attempt_union"]) == 2
    assert len(support["attempt_intersection"]) == 2
    assert len(support["success_intersection"]) == 1


def test_seed_command_can_override_the_raw_slice_count(tmp_path: Path) -> None:
    gate = load_module(GATE, "test_seed_gate_command")
    wrapper = load_module(WRAPPER, "test_seed_wrapper_command")

    cmd = wrapper.strict_phase2_command(
        gate,
        tmp_path / "out",
        "ETH",
        "2026-04",
        20260605,
        danger_slices=47,
    )

    idx = cmd.index("--danger-slices")
    assert cmd[idx + 1] == "47"


def test_manifest_amendment_changes_only_the_ineligible_cell(tmp_path: Path) -> None:
    gate = load_module(GATE, "test_seed_gate_manifest")
    wrapper = load_module(WRAPPER, "test_seed_wrapper_manifest")
    manifest = pd.DataFrame(
        [
            {
                "currency": currency,
                "month": month,
                "seed": seed,
                "selection": "top",
                "hidden": "64,64",
                "run": f"{currency}_{month}_{seed}",
                "run_name": f"{currency}_{month}_{seed}",
                "output_dir": str(tmp_path / "stage18_top49" / f"{currency}_{month}_{seed}"),
                "stage": "stage18_top49",
                "command": "original",
            }
            for currency, month in (("BTC", "2026-04"), ("ETH", "2026-04"))
            for seed in gate.NN_SEEDS
        ]
    )

    amended = wrapper.build_amended_manifest(
        manifest,
        out_root=tmp_path,
        gate=gate,
        analysis_slice_cap=47,
        rerun_cells={("ETH", "2026-04")},
    )

    assert set(amended["analysis_slice_cap"]) == {47}
    btc = amended[amended["currency"] == "BTC"]
    eth = amended[amended["currency"] == "ETH"]
    assert set(btc["raw_slice_count"]) == {49}
    assert set(eth["raw_slice_count"]) == {47}
    assert btc["output_dir"].str.contains("stage18_top49", regex=False).all()
    assert eth["output_dir"].str.contains("stage18_top47_missing", regex=False).all()
    assert eth["command"].str.contains("--danger-slices 47", regex=False).all()


def make_mixed_manifest(tmp_path: Path) -> Path:
    rows = []
    for name, raw_count in (("old_top49", 49), ("new_top47", 47)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "phase2_summary.csv").write_text("status\ncomplete\n", encoding="utf-8")
        for required in (
            "strike_pairwise_dispersion.csv",
            "greeks.parquet",
            "phase2_config.json",
        ):
            (run_dir / required).touch()
        rows.append(
            {
                "run": name,
                "output_dir": str(run_dir),
                "raw_slice_count": raw_count,
                "analysis_slice_cap": 47,
            }
        )
    path = tmp_path / "gate_run_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_seed_pair_scaling_discovers_runs_from_the_amended_manifest(tmp_path: Path) -> None:
    module = load_module(PAIR_SCALE, "test_pair_scale_manifest")
    manifest = make_mixed_manifest(tmp_path)

    records = module.discover_seed_manifest_runs(manifest, Path("phase2.py"))

    assert {Path(record["run_dir"]).name for record in records} == {"old_top49", "new_top47"}


def test_postprocessing_counts_completed_seed_runs_from_manifest(tmp_path: Path) -> None:
    module = load_module(POSTPROCESS, "test_postprocess_manifest")
    manifest = make_mixed_manifest(tmp_path)

    assert module.completed_manifest_run_count(manifest) == 2


def test_integrity_uses_raw_and_analysis_slice_counts_from_manifest(tmp_path: Path) -> None:
    module = load_module(VERIFY, "test_verify_manifest")
    manifest = make_mixed_manifest(tmp_path)

    specs = module.seed_run_specs(manifest)

    assert {(spec["run_dir"].name, spec["raw_slice_count"], spec["analysis_slice_cap"]) for spec in specs} == {
        ("old_top49", 49, 47),
        ("new_top47", 47, 47),
    }


def test_equivalence_comparison_checks_keys_and_numeric_values() -> None:
    module = load_module(EQUIVALENCE, "test_seed_equivalence")
    source = pd.DataFrame(
        {
            "currency": ["BTC", "BTC"],
            "date": [1, 1],
            "strike": [90.0, 100.0],
            "value": [0.1, 0.2],
        }
    )
    target = source.iloc[::-1].reset_index(drop=True)

    report = module.compare_frames(source, target, ["currency", "date", "strike"])

    assert report["keys_equal"] is True
    assert report["numeric_values_equal"] is True
    assert report["max_abs_difference"] == 0.0


def test_panel_identity_discovers_seed_panels_from_manifest(tmp_path: Path) -> None:
    module = load_module(PANEL_IDENTITY, "test_panel_identity_manifest")
    manifest = make_mixed_manifest(tmp_path)

    paths = module.seed_panel_paths(manifest)

    assert {path.parent.name for path in paths} == {"old_top49", "new_top47"}
