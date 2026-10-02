#!/usr/bin/python3
"""Build validator input from fresh pose-driven single-arm pickup payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "alfa.v3_scoop_5x5_dual_replay.v1"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def attachment(payload: dict[str, Any], side: str) -> dict[str, Any]:
    return {
        "tool_link": f"{side}_tool0",
        "center_in_tool": payload["tool_to_box_center"],
        "rotation_in_tool": payload["tool_to_box_rotation"],
        "size": payload["carried_box_size_tool"],
    }


def obstacles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    values = [
        {
            "id": f"neighbor_box_{index}",
            "center": center,
            "size": payload["box_size"],
        }
        for index, center in enumerate(payload["neighbor_centers"])
    ]
    ground = payload["ground"]
    if ground["enabled"]:
        values.append({
            "id": "ground",
            "center": [
                0.0,
                0.0,
                float(ground["collision_top_z"]) - 0.5 * float(ground["size"][2]),
            ],
            "size": ground["size"],
        })
    if payload["warehouse"]["enabled"]:
        values.extend(payload["warehouse"]["panels"])
    return values


def operation(task: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    side = str(payload["side"])
    frames = []
    for frame in payload["frames"]:
        attached = bool(frame.get("box_attached", False))
        frames.append({
            "stage": frame["stage"],
            "joints": frame["joints"],
            "left_attached": attached and side == "left",
            "right_attached": attached and side == "right",
        })
    return {
        "label": task["request_id"],
        "kind": "pose_driven_pickup_retreat",
        "row": task["row_from_top"],
        "base_pose_map": payload["base_pose_map"],
        "allow_ground_model_base": True,
        "obstacles": obstacles(payload),
        "left_attachment": attachment(payload, "left"),
        "right_attachment": attachment(payload, "right"),
        "frames": frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = read_json(args.fresh_report)
    operations = []
    joint_names = None
    for task in report["tasks"]:
        if not task["success"]:
            parser.error(f"fresh task failed: {task['request_id']}")
        payload = read_json(Path(task["selected"]["payload"]))
        if joint_names is None:
            joint_names = payload["joint_names"]
        elif payload["joint_names"] != joint_names:
            parser.error("fresh payload joint contracts differ")
        operations.append(operation(task, payload))
    replay = {
        "schema": SCHEMA,
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "joint_names": joint_names,
        "operations": operations,
        "scope": "fresh pose-driven pickup and loaded-retreat validation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"operations={len(operations)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
