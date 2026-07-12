#!/usr/bin/env python3
"""Verify numerical checkpoints, artifact structure, and path portability."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".tex", ".txt", ".yml", ".yaml", ".gitignore"}
EXCLUDED_PARTS = {".git", "generated", "__pycache__", ".pytest_cache"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float = 1e-8) -> None:
    require(abs(actual - expected) <= tolerance, f"expected {expected}, found {actual}")


def verify_reported_results() -> None:
    expected = json.loads((ROOT / "manifests" / "expected_results.json").read_text(encoding="utf-8"))
    quick = json.loads((ROOT / "generated" / "quick_summary.json").read_text(encoding="utf-8"))
    full = json.loads((ROOT / "outputs" / "full_rerun_summary.json").read_text(encoding="utf-8"))
    lp_audit = json.loads((ROOT / "outputs" / "audits" / "lp_sample_audit.json").read_text(encoding="utf-8"))
    require(quick["figure4"]["rows"] == expected["lp_range"]["observations"], "LP row count differs")
    close(quick["figure4"]["spearman"], expected["lp_range"]["spearman_rank_correlation"])
    close(quick["figure4"]["log_log_r2"], expected["lp_range"]["log_log_r2"])
    require(lp_audit["strict_check"]["exceedances"] == expected["lp_range"]["strict_tolerance_exceedances"], "strict LP audit count differs")
    require(lp_audit["numerical_audit"]["remaining_exceedances"] == expected["lp_range"]["absolute_1e_9_exceedances"], "LP tolerance-audit count differs")
    l2_nn = quick["figure5"]["pairs"]["L2__NNPROJ"]["median_p90_gamma_gap"]
    nn_smooth = quick["figure5"]["pairs"]["NNPROJ__SMOOTH_RHO_10"]["median_p90_gamma_gap"]
    close(l2_nn, expected["targeted_nn"]["median_p90_gamma_gap_l2_nnproj"])
    close(nn_smooth, expected["targeted_nn"]["median_p90_gamma_gap_nnproj_smooth10"])
    close(full["classical"]["median_evaluation_p90_smooth10_gamma_gap"], expected["classical"]["median_evaluation_p90_smooth10_gamma_gap"])
    require(full["seed_replication"]["runs_completed"] == expected["seed_replication"]["refits"], "seed refit count differs")


def verify_structure() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "environment.yml",
        ROOT / "config" / "analysis_design.json",
        ROOT / "data" / "input_panel.sha256",
        ROOT / "outputs" / "full_rerun_summary.json",
        ROOT / "generated" / "quick_summary.json",
        ROOT / "generated" / "tables" / "table1_source.csv",
    ]
    required.extend((ROOT / "generated" / "figures" / name) for name in (
        "fig_mechanism_overview.pdf",
        "validation_protocol_tikz_singlecol.pdf",
        "fig_rho_path.pdf",
        "fig_feasible_range_scatter.pdf",
        "fig_nnproj_pairs.pdf",
        "fig_var_relative_difference_classical_common.pdf",
    ))
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    require(not missing, "missing required files: " + ", ".join(missing))
    row_expectations = {
        "figure3_rho_path.csv": 864,
        "figure4_lp_range.csv": 51189,
        "figure5_nnproj_pairs.csv": 360,
        "figure6_var_relative_difference.csv": 12,
    }
    for name, expected_rows in row_expectations.items():
        line_count = sum(1 for _ in (ROOT / "outputs" / "figure_sources" / name).open("r", encoding="utf-8")) - 1
        require(line_count == expected_rows, f"{name}: expected {expected_rows} rows, found {line_count}")
    oversized = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts and path.stat().st_size > 50 * 1024 * 1024]
    require(not oversized, "files exceed 50 MiB: " + ", ".join(str(path.relative_to(ROOT)) for path in oversized))
    symlinks = [path for path in ROOT.rglob("*") if path.is_symlink()]
    require(not symlinks, "artifact contains symlinks")


def verify_output_checksums() -> None:
    manifest = ROOT / "manifests" / "output_checksums.sha256"
    require(manifest.is_file(), "missing output checksum manifest")
    for line in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / Path(relative)
        require(path.is_file(), f"checksum target is missing: {relative}")
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        require(value == expected, f"checksum mismatch: {relative}")


def text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.name == ".gitignore" or path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def verify_artifact_portability() -> None:
    identity_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"Users[\\/]\d+",
            r"Baidu" + r"Syncdisk",
            r"github\.com/[A-Za-z0-9_.-]+/no-arb-code",
            r"D:[\\/]\d+",
        )
    ]
    secret_patterns = [
        re.compile(pattern)
        for pattern in (
            r"AK" + r"IA[0-9A-Z]{16}",
            r"ghp" + r"_[A-Za-z0-9]{30,}",
            r"github" + r"_pat_[A-Za-z0-9_]{40,}",
            r"-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        )
    ]
    findings: list[str] = []
    for path in text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in identity_patterns + secret_patterns:
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    require(not findings, "identity/secret scan findings:\n" + "\n".join(findings))


def main() -> None:
    verify_structure()
    verify_output_checksums()
    verify_reported_results()
    verify_artifact_portability()
    print("Artifact verification passed: structure, numerical checkpoints, and path portability.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Artifact verification failed: {exc}", file=sys.stderr)
        raise
