#!/usr/bin/env python3
"""Prepare the certified V3.2.2 model for Foam and Cricket."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_URDF = Path(
    "/home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/"
    "range_certification_v322_tool0151/model/alfa-robot-v322-tool0151.urdf"
)
DEFAULT_SRDF = REPO / "ros2_ws/src/alfa_robot_moveit_config/config/alfa_robot.srdf"
DESCRIPTION = REPO / "ros2_ws/src/alfa_robot_description"
FIXED_JOINTS = {"head_joint", "head_pitch_joint"}
MOVABLE_JOINTS = [
    "updown",
    *[f"left_joint{i}" for i in range(1, 8)],
    *[f"right_joint{i}" for i in range(1, 8)],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_mesh_uri(uri: str) -> Path:
    prefix = "package://alfa_robot_description/"
    if not uri.startswith(prefix):
        raise ValueError(f"unsupported mesh URI: {uri}")
    path = (DESCRIPTION / uri.removeprefix(prefix)).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--srdf", type=Path, default=DEFAULT_SRDF)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/v322_vamp_paper_repro/model",
    )
    args = parser.parse_args()

    source_urdf = args.urdf.resolve()
    source_srdf = args.srdf.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    rewritten_meshes = 0
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str(resolve_mesh_uri(mesh.attrib["filename"])))
        rewritten_meshes += 1

    fixed = []
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        if name not in FIXED_JOINTS:
            continue
        joint.set("type", "fixed")
        for tag in (
            "axis",
            "limit",
            "dynamics",
            "safety_controller",
            "calibration",
            "mimic",
        ):
            for element in list(joint.findall(tag)):
                joint.remove(element)
        fixed.append(name)

    movable = [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") not in {"fixed", "floating"}
    ]
    if movable != MOVABLE_JOINTS:
        raise RuntimeError(f"unexpected 15D joint order: {movable}")
    if set(fixed) != FIXED_JOINTS:
        raise RuntimeError(f"failed to fix head joints: {fixed}")

    ET.indent(tree, space="  ")
    prepared_urdf = output / "alfa-v322-15d-mesh.urdf"
    tree.write(prepared_urdf, encoding="utf-8", xml_declaration=True)
    prepared_srdf = output / "alfa-v322-15d.srdf"
    shutil.copy2(source_srdf, prepared_srdf)

    collision_count = len(root.findall(".//collision"))
    manifest = {
        "schema": "alfa.v322.vamp_paper_model.v1",
        "source_urdf": str(source_urdf),
        "source_urdf_sha256": sha256(source_urdf),
        "source_srdf": str(source_srdf),
        "source_srdf_sha256": sha256(source_srdf),
        "prepared_urdf": str(prepared_urdf),
        "prepared_urdf_sha256": sha256(prepared_urdf),
        "prepared_srdf": str(prepared_srdf),
        "prepared_srdf_sha256": sha256(prepared_srdf),
        "dimension": len(movable),
        "joint_order": movable,
        "fixed_for_benchmark": sorted(FIXED_JOINTS),
        "collision_geometry_count": collision_count,
        "rewritten_mesh_uris": rewritten_meshes,
    }
    manifest_path = output / "model-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
