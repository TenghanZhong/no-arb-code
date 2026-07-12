from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATHS = json.loads((ROOT / "config" / "paths.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str, cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(
            list(args),
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.stdout.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def main() -> None:
    manifest_dir = ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    snapshot = ROOT / "code" / "original_snapshot"
    rows: list[dict[str, object]] = []
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(snapshot).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with (manifest_dir / "source_snapshot_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    panel = Path(PATHS["synthetic_call_panel"])
    paper_repo = Path(PATHS["paper_repo"])
    environment = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "panel_path": str(panel),
        "panel_bytes": panel.stat().st_size,
        "panel_sha256": sha256(panel),
        "paper_git_head": command("git", "rev-parse", "HEAD", cwd=paper_repo),
        "paper_git_status_short": command("git", "status", "--short", cwd=paper_repo),
        "pip_freeze_file": str(manifest_dir / "pip_freeze.txt"),
    }
    (manifest_dir / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    (manifest_dir / "pip_freeze.txt").write_text(
        command(sys.executable, "-m", "pip", "freeze") + "\n", encoding="utf-8"
    )
    print(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
