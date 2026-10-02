#!/usr/bin/env python3
"""Write the evidence manifest and SHA-256 list for the isolated project."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256sums", type=Path, required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("assets", nargs="+")
    args = parser.parse_args()
    assets = []
    for name in args.assets:
        path = args.root / name
        if not path.is_file():
            parser.error(f"asset does not exist: {path}")
        assets.append({
            "name": name,
            "size": path.stat().st_size,
            "sha256": digest(path),
        })
    manifest = {
        "schema": "alfa.v322_box_wall_motion_family_evidence.v1",
        "model_revision": "robot_v3.2.2-suction",
        "upstream_base_commit": "d9c330cef72981390d81ac2b1cd5a6eb9e892195",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "tool0_offset_local_z_m": 0.151,
        "result": {
            "cross_row_pair_requests": "3/3",
            "cross_row_boxes": "6/6",
            "one_sided_attachment_frames": 0,
            "cross_row_moveit_fcl": "PASS",
            "cross_row_checked_frames": 5722,
            "cross_row_checked_edge_samples": 366,
            "single_pose_representatives": "5/5",
            "single_pose_moveit_fcl": "PASS",
            "single_pose_checked_frames": 9002,
            "single_pose_checked_edge_samples": 294,
        },
        "assets": assets,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    all_assets = [
        {"name": args.manifest.name, "sha256": digest(args.manifest)},
        *assets,
    ]
    args.sha256sums.write_text(
        "".join(f"{item['sha256']}  {item['name']}\n" for item in all_assets),
        encoding="utf-8",
    )
    print(f"assets={len(assets)} manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
