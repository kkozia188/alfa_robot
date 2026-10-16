#!/usr/bin/env python3
"""Record whether the certified V3.2.2 model can run in stock VAMP."""

from __future__ import annotations

import argparse
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path(
    "/home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/"
    "range_certification_v322_tool0151/model/alfa-robot-v322-tool0151.urdf"
)


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v322_vamp_parallel/v3-vamp-readiness.json"),
    )
    args = parser.parse_args()
    model = args.model.resolve()
    root = ET.fromstring(model.read_text(encoding="utf-8"))
    collisions = root.findall(".//collision/geometry")
    geometry_counts = {
        kind: sum(geometry.find(kind) is not None for geometry in collisions)
        for kind in ("mesh", "sphere", "box", "cylinder")
    }
    movable = [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") not in {"fixed", "floating"}
    ]
    vamp_robots: list[str] = []
    if importlib.util.find_spec("vamp") is not None:
        import vamp

        vamp_robots = list(vamp.robots)
    payload = {
        "schema": "alfa.v322_vamp_readiness.v1",
        "model": str(model),
        "movable_joint_count": len(movable),
        "movable_joints": movable,
        "collision_geometry": geometry_counts,
        "stock_vamp_robots": vamp_robots,
        "stock_vamp_supports_v3": False,
        "production_ready": False,
        "required_before_v3_use": [
            "Generate and visually inspect a sphere-only V3.2.2 collision URDF.",
            "Compile the V3 kinematics/collision kernel with Cricket.",
            "Represent the mirrored dual arms, fixed opposite arm, and attached box.",
            "Compare every VAMP state/edge decision against MoveIt/FCL on certified tasks.",
            "Retain MoveIt/FCL as the final acceptance gate until zero false negatives are shown.",
        ],
        "decision": (
            "Stock VAMP cannot load this mesh-based V3.2.2 model directly; use it only "
            "for framework throughput experiments until the custom kernel is validated."
        ),
    }
    atomic_write(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
