#!/usr/bin/env python3
"""Package the compact preliminary matrix evidence with SHA-256 metadata."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path
from typing import Any

from matrix_common import is_single_axis_error, read_json, sha256, write_json


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
    "aggressive-screen/**/probe-result.json",
    "aggressive-refine/**/probe-result.json",
    "aggressive-full/**/case-result.json",
    "aggressive-full/**/v322-pose-conveyor-validation.json",
    "release/matrix-summary.json",
    "release/matrix-results.csv",
    "release/REPORT.md",
    "release/y-boundaries.json",
    "release/SINGLE_AXIS_RANGE.json",
    "release/SINGLE_AXIS_RANGE.md",
)


def collect(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in PATTERNS:
        files.update(path for path in root.glob(pattern) if path.is_file())
    output = []
    for path in sorted(files):
        result_path = next(
            (
                candidate for candidate in (
                    path.parent / "case-result.json",
                    path.parent / "probe-result.json",
                ) if candidate.is_file()
            ),
            None,
        )
        if result_path is not None:
            result = read_json(result_path)
            error = result.get("error", {})
            if not is_single_axis_error(
                float(error.get("dx_m", 0.0)),
                float(error.get("dy_m", 0.0)),
                float(error.get("yaw_deg", 0.0)),
            ):
                continue
        output.append(path)
    return output


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
    release_dir = root / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    for name in ("SINGLE_AXIS_RANGE.json", "SINGLE_AXIS_RANGE.md"):
        shutil.copyfile(Path(__file__).resolve().parent / name, release_dir / name)
    files = collect(root)
    entries: list[dict[str, Any]] = [{
        "path": str(Path("v3_docking_error_matrix") / path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    } for path in files]
    manifest = {
        "schema": "alfa.v322_docking_error_single_axis_evidence.v1",
        "scope": "independent X, Y, and Yaw sweeps with other variables fixed at zero",
        "data_root": "v3_docking_error_matrix",
        "file_count": len(entries),
        "files": entries,
    }
    manifest_path = output / "PRELIMINARY_MANIFEST.json"
    write_json(manifest_path, manifest)
    archive = output / "v322-docking-error-single-axis-2026.10.09.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(manifest_path, arcname=manifest_path.name)
        for path in files:
            stream.add(
                path,
                arcname=str(Path("v3_docking_error_matrix") / path.relative_to(root)),
            )
    write_json(output / "ARCHIVE.json", {
        "schema": "alfa.v322_docking_error_single_axis_archive.v1",
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
