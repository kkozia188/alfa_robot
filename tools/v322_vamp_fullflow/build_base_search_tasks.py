#!/usr/bin/python3
"""Build loaded delivery and empty return tasks from current-run VAMP payloads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


GROUPS = [
    (1, 5), (2, 4), (3,), (6, 10), (7, 9), (8,),
    (11, 15), (12, 14), (13,), (16, 20), (17, 19), (18,),
    (21, 25), (22, 24), (23,),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def select_payload(box_id: int, roots: list[Path]) -> tuple[dict[str, Any], Path]:
    for root in roots:
        path = root / f"box-{box_id:02d}.json"
        if path.is_file():
            payload = read_json(path)
            if payload.get("success"):
                return payload, path
    raise RuntimeError(f"no successful current-run VAMP payload for box {box_id}")


def last_frame(payload: dict[str, Any], stage: str) -> dict[str, Any]:
    return [frame for frame in payload["frames"] if frame["stage"] == stage][-1]


def attachment(payload: dict[str, Any], side: str) -> dict[str, Any]:
    return {
        "tool_link": f"{side}_tool0",
        "center_in_tool": payload["tool_to_box_center"],
        "rotation_in_tool": payload["tool_to_box_rotation"],
        "size": payload["box_size"],
    }


def default_attachment(side: str) -> dict[str, Any]:
    return {
        "tool_link": f"{side}_tool0", "center_in_tool": [0.0, 0.0, 0.15],
        "rotation_in_tool": [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        "size": [0.3, 0.4, 0.4],
    }


def box_pose(box_id: int) -> list[float]:
    row, column = divmod(box_id - 1, 5)
    return [0.9, (2 - column) * 0.4, (4.5 - row) * 0.4]


def obstacles(remaining: set[int]) -> list[dict[str, Any]]:
    output = [{"id": "ground", "center": [0.0, 0.0, -0.055], "size": [6.0, 6.0, 0.1]}]
    opening_x, length, width, height, thickness = -1.18, 2.38, 2.38, 2.35, 0.05
    center_x, center_z = opening_x + length * 0.5, height * 0.5
    output.extend([
        {"id": "warehouse_left_wall", "center": [center_x, width * 0.5 + thickness * 0.5, center_z], "size": [length, thickness, height]},
        {"id": "warehouse_right_wall", "center": [center_x, -width * 0.5 - thickness * 0.5, center_z], "size": [length, thickness, height]},
        {"id": "warehouse_ceiling", "center": [center_x, 0.0, height + thickness * 0.5], "size": [length, width + 2 * thickness, thickness]},
        {"id": "warehouse_rear_wall", "center": [opening_x + length + thickness * 0.5, 0.0, center_z], "size": [thickness, width + 2 * thickness, height]},
    ])
    for box_id in sorted(remaining):
        output.append({"id": f"wall_box_{box_id}", "center": box_pose(box_id), "size": [0.296, 0.396, 0.396]})
    return output


def phase_operation(
    label: str, phase: str, start_joints: list[float], goal_joints: list[float], start_base: list[float],
    goal_base: list[float], left_attached: bool, right_attached: bool,
    scene: list[dict[str, Any]], left_spec: dict[str, Any], right_spec: dict[str, Any],
) -> dict[str, Any]:
    start = {"stage": "chest_carry_retract", "joints": start_joints,
             "left_attached": left_attached, "right_attached": right_attached,
             "base_pose_map": start_base}
    goal = {"stage": "chest_carry_ready", "joints": goal_joints,
            "left_attached": left_attached, "right_attached": right_attached,
            "base_pose_map": goal_base}
    return {
        "kind": "online_pose_pure_rrt_phase", "label": label,
        "phase": phase, "base_pose_map": start_base,
        "allow_ground_model_base": True, "obstacles": scene,
        "left_attachment": left_spec, "right_attachment": right_spec,
        "frames": [start, goal], "box_planning": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-missing-groups", action="store_true")
    args = parser.parse_args()
    roots = [path.resolve() for path in args.payload_dir]
    selected = {}
    for box_id in range(1, 26):
        try:
            selected[box_id] = select_payload(box_id, roots)
        except RuntimeError:
            if not args.skip_missing_groups:
                raise
    names = [str(value) for value in selected[1][0]["joint_names"]]
    remaining = set(range(1, 26))
    operations = []
    manifest = {}
    for cycle, group in enumerate(GROUPS, 1):
        if any(box_id not in selected for box_id in group):
            continue
        payloads = [selected[box_id][0] for box_id in group]
        for box_id in group:
            manifest[str(box_id)] = str(selected[box_id][1])
        remaining.difference_update(group)
        base_x = float(payloads[0]["base_pose_map"][0])
        pickup = [base_x, 0.0, 0.0]
        safe_exit = [base_x - 2.35, 0.0, math.pi]
        left = next((item for item in payloads if item["side"] == "left"), None)
        right = next((item for item in payloads if item["side"] == "right"), None)
        combined = list(last_frame(payloads[0], "rrt_to_place")["joints"])
        for payload in payloads:
            source = last_frame(payload, "rrt_to_place")["joints"]
            side = str(payload["side"])
            combined[names.index("updown")] = source[names.index("updown")]
            for joint in range(1, 8):
                index = names.index(f"{side}_joint{joint}")
                combined[index] = source[index]
        home = [-0.3, 0.0,
                *[math.radians(v) for v in (155, -105, 20, 90, -90, -40, 0)],
                *[math.radians(v) for v in (-155, 105, -20, -90, 90, 40, 0)]]
        left_spec = attachment(left, "left") if left else default_attachment("left")
        right_spec = attachment(right, "right") if right else default_attachment("right")
        label = " + ".join(f"Box {box_id}" for box_id in group)
        scene = obstacles(remaining)
        operations.append(phase_operation(
            f"Cycle {cycle:02d} {label} / loaded_delivery", "loaded_delivery",
            combined, combined, pickup, safe_exit, left is not None, right is not None,
            scene, left_spec, right_spec))
        operations.append(phase_operation(
            f"Cycle {cycle:02d} {label} / dual_stow", "dual_stow",
            combined, home, safe_exit, safe_exit, False, False,
            scene, left_spec, right_spec))
        operations.append(phase_operation(
            f"Cycle {cycle:02d} {label} / empty_return", "empty_return",
            home, home, safe_exit, pickup, False, False,
            scene, left_spec, right_spec))
    write_json(args.output.resolve(), {
        "schema": "alfa.v3_scoop_5x5_dual_replay.v1",
        "phase_search_only": True,
        "cached_replay_or_joint_endpoint_used": False,
        "joint_names": names, "operations": operations,
        "box_planning": [], "payload_manifest": manifest,
    })
    print(f"RESULT operations={len(operations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
