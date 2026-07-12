#!/usr/bin/env python3
"""Write SHA-256 checksums for the compact artifact outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
MANIFEST = ROOT / "manifests" / "output_checksums.sha256"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    files = sorted(path for path in OUTPUTS.rglob("*") if path.is_file())
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {len(lines)} checksums to {MANIFEST}")


if __name__ == "__main__":
    main()
