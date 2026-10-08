#!/usr/bin/env python3
"""Package the compact preliminary matrix evidence with SHA-256 metadata."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Any

from matrix_common import sha256, write_json


PATTERNS = (
    "design/matrix-cases.json",
    "design/matrix-cases.csv",
    "baselines/imported-baselines.json",
    "baselines/imported-baselines.csv",
    "y-baseline-screen/screen-matrix-run.json",
    "y-baseline-screen/screen-matrix-run.csv",
    "y-baseline-screen/*/probe-result.json",
    "probes/*/probe-result.json",
    "full/*/case-result.json",
    "full/*/v322-pose-conveyor-validation.json",
    "release/matrix-summary.json",
    "release/matrix-results.csv",
    "release/REPORT.md",
    "release/y-boundaries.json",
)


def collect(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in PATTERNS:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = collect(root)
    entries: list[dict[str, Any]] = [{
        "path": str(Path("v3_docking_error_matrix") / path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    } for path in files]
    manifest = {
        "schema": "alfa.v322_docking_error_preliminary_evidence.v1",
        "scope": "preliminary baselines, screens, complete-case outcomes and validations",
        "data_root": "v3_docking_error_matrix",
        "file_count": len(entries),
        "files": entries,
    }
    manifest_path = output / "PRELIMINARY_MANIFEST.json"
    write_json(manifest_path, manifest)
    archive = output / "v322-docking-error-matrix-preliminary-2026.10.08.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(manifest_path, arcname=manifest_path.name)
        for path in files:
            stream.add(
                path,
                arcname=str(Path("v3_docking_error_matrix") / path.relative_to(root)),
            )
    write_json(output / "ARCHIVE.json", {
        "schema": "alfa.v322_docking_error_preliminary_archive.v1",
        "name": archive.name,
        "size": archive.stat().st_size,
        "sha256": sha256(archive),
        "manifest": manifest_path.name,
        "manifest_sha256": sha256(manifest_path),
    })
    print(
        f"EVIDENCE files={len(files)} size={archive.stat().st_size} "
        f"sha256={sha256(archive)} archive={archive}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
