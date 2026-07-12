"""Portable paths shared by the anonymous replication artifact."""

from __future__ import annotations

import os
from pathlib import Path


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]


def configured_path(env_name: str, default: Path) -> Path:
    """Return an optional environment override as an absolute path."""
    value = os.environ.get(env_name)
    return Path(value).expanduser().resolve() if value else default.resolve()


PANEL = configured_path(
    "REPAIR_RISK_PANEL",
    ARTIFACT_ROOT / "data" / "synthetic_call_panel.parquet",
)
RESULTS = configured_path("REPAIR_RISK_RESULTS_ROOT", ARTIFACT_ROOT / "results")
DERIVED = configured_path("REPAIR_RISK_DERIVED_ROOT", ARTIFACT_ROOT / "derived")
FIGURES = configured_path("REPAIR_RISK_FIGURES_ROOT", ARTIFACT_ROOT / "figures")
LOGS = configured_path("REPAIR_RISK_LOG_ROOT", ARTIFACT_ROOT / "logs")
