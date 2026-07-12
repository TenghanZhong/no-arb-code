#!/usr/bin/env python3
"""Entry point for compact verification and the complete repair-risk rerun."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def canonical_panel_spec() -> tuple[str, int]:
    fields = (ROOT / "data" / "input_panel.sha256").read_text(encoding="ascii").split()
    return fields[0], int(fields[1])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_panel(path: Path, *, allow_mismatch: bool) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash, expected_size = canonical_panel_spec()
    actual_size = path.stat().st_size
    actual_hash = sha256(path)
    matches = actual_size == expected_size and actual_hash == expected_hash
    if not matches and not allow_mismatch:
        raise RuntimeError(
            "Input panel does not match the canonical artifact input. "
            "Use --allow-panel-mismatch only for an explicit sensitivity run.\n"
            f"expected size/hash: {expected_size} / {expected_hash}\n"
            f"actual size/hash:   {actual_size} / {actual_hash}"
        )
    label = "canonical" if matches else "noncanonical (explicitly allowed)"
    print(f"Panel verified: {label}; {actual_size} bytes; sha256={actual_hash}")


def full_environment(panel: Path, jobs: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "REPAIR_RISK_PANEL": str(panel.resolve()),
            "REPAIR_RISK_RESULTS_ROOT": str((ROOT / "results").resolve()),
            "REPAIR_RISK_DERIVED_ROOT": str((ROOT / "derived").resolve()),
            "REPAIR_RISK_FIGURES_ROOT": str((ROOT / "figures").resolve()),
            "REPAIR_RISK_LOG_ROOT": str((ROOT / "logs").resolve()),
            "REPAIR_RISK_JOBS": str(jobs),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(ROOT / "code" / "working" / "project_src"),
                    str(ROOT / "code" / "working" / "common"),
                    str(ROOT / "code" / "working"),
                    env.get("PYTHONPATH", ""),
                ]
            ),
        }
    )
    return env


def full_commands(panel: Path, jobs: int) -> Iterable[list[str]]:
    working = ROOT / "code" / "working"
    yield [PYTHON, str(working / "preflight_full_feasibility.py")]
    yield [PYTHON, str(working / "run_classical_18m_full.py"), "stage", "--jobs", str(jobs)]
    yield [
        PYTHON,
        str(working / "paper_repo_scripts" / "run_svi_target_18m.py"),
        "--output-root",
        str(ROOT / "results" / "svi_18m"),
        "--jobs",
        str(jobs),
    ]
    yield [PYTHON, str(working / "run_targeted_nn_full.py"), "stage", "--jobs", str(jobs)]
    yield [
        PYTHON,
        str(working / "project_scripts" / "nnproj_seed_18m_top_audit.py"),
        "stage",
        "--out-root",
        str(ROOT / "results" / "nn_seed_18m"),
        "--jobs",
        str(jobs),
    ]
    yield [
        PYTHON,
        str(working / "paper_repo_scripts" / "run_calendar_certified_robustness.py"),
        "--panel",
        str(panel.resolve()),
        "--output-dir",
        str(ROOT / "results" / "calendar"),
    ]
    yield [PYTHON, str(working / "run_postprocessing_full.py")]


def command_quick() -> None:
    run([PYTHON, str(ROOT / "scripts" / "build_review_outputs.py")])


def command_verify() -> None:
    command_quick()
    run([PYTHON, str(ROOT / "scripts" / "verify_artifact.py")])
    run([PYTHON, "-m", "pytest", "-q", str(ROOT / "tests")])


def command_full(args: argparse.Namespace) -> None:
    panel = args.panel.resolve()
    check_panel(panel, allow_mismatch=args.allow_panel_mismatch)
    env = full_environment(panel, args.jobs)
    commands = list(full_commands(panel, args.jobs))
    if args.dry_run:
        print("Resolved full-run environment:")
        for key in (
            "REPAIR_RISK_PANEL",
            "REPAIR_RISK_RESULTS_ROOT",
            "REPAIR_RISK_DERIVED_ROOT",
            "REPAIR_RISK_FIGURES_ROOT",
            "REPAIR_RISK_LOG_ROOT",
            "REPAIR_RISK_JOBS",
        ):
            print(f"  {key}={env[key]}")
        print("Commands:")
        for item in commands:
            print("  + " + " ".join(item))
        return
    for item in commands:
        run(item, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("quick", help="rebuild review outputs from compact data")
    subparsers.add_parser("verify", help="rebuild, audit, and run tests")
    full = subparsers.add_parser("full", help="execute the complete empirical rerun")
    full.add_argument("--panel", type=Path, required=True)
    full.add_argument("--jobs", type=int, default=4)
    full.add_argument("--dry-run", action="store_true")
    full.add_argument("--allow-panel-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "quick":
        command_quick()
    elif args.command == "verify":
        command_verify()
    else:
        command_full(args)


if __name__ == "__main__":
    main()
