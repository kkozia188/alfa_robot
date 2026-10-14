#!/usr/bin/python3
"""Validate chest-first, safe-backoff, 180-degree turn, and equal-height release."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
RERUN_DIR = WORKSPACE / "ros2_ws/src/alfa_robot_rerun"
if str(RERUN_DIR) not in sys.path:
    sys.path.insert(0, str(RERUN_DIR))

from alfa_robot_rerun.visualize_rerun import UrdfRobot


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def shortest_angle_delta(left: float, right: float) -> float:
    return math.atan2(math.sin(right - left), math.cos(right - left))


def base_transform(pose: list[float]) -> np.ndarray:
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    transform = np.eye(4)
    transform[:3, :3] = np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])
    transform[:3, 3] = [float(pose[0]), float(pose[1]), 0.0]
    return transform


def carried_center(
    robot: UrdfRobot,
    names: list[str],
    frame: dict[str, Any],
    operation: dict[str, Any],
    side: str,
) -> np.ndarray:
    joints = {
        name: float(value) for name, value in zip(names, frame["joints"])
    }
    attachment = operation[f"{side}_attachment"]
    tool = robot.fk(joints)[str(attachment["tool_link"])]
    center_base = tool @ np.asarray(
        [*attachment["center_in_tool"], 1.0], dtype=float
    )
    pose = [float(value) for value in frame["base_pose_map"]]
    return (base_transform(pose) @ center_base)[:3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--equal-height-tolerance-m", type=float, default=1e-6)
    args = parser.parse_args()

    replay = read_json(args.replay.resolve())
    validation = read_json(args.validation.resolve())
    robot = UrdfRobot(args.model_urdf.resolve().read_text(encoding="utf-8"))
    names = [str(value) for value in replay["joint_names"]]
    cases: list[dict[str, Any]] = []
    failures: list[str] = []
    maximum_height_error = 0.0

    for operation in replay["operations"]:
        if operation["kind"] == "base_transition":
            continue
        frames = list(operation["frames"])
        stages = [str(frame["stage"]) for frame in frames]
        chest_index = stages.index("chest_carry_ready")
        backoff_index = stages.index("base_backoff_after_chest_carry")
        turn_index = stages.index("base_turn_180_loaded_at_safe_exit")
        release_index = stages.index("release_after_turn_at_safe_exit")
        ordered = chest_index < backoff_index < turn_index < release_index
        pickup_pose = [float(value) for value in operation["base_pose_map"]]
        turn_pose = [float(value) for value in frames[turn_index]["base_pose_map"]]
        release_pose = [float(value) for value in frames[release_index]["base_pose_map"]]
        backoff = math.hypot(
            turn_pose[0] - pickup_pose[0], turn_pose[1] - pickup_pose[1]
        )
        lateral = abs(turn_pose[1] - pickup_pose[1])
        turn_deg = abs(math.degrees(shortest_angle_delta(
            pickup_pose[2], release_pose[2]
        )))
        attached_before_release = frames[release_index - 1]
        active_sides = [
            side for side in ("left", "right")
            if int(operation.get(f"{side}_box_id", 0)) > 0
        ]
        centers = {
            side: carried_center(
                robot, names, attached_before_release, operation, side
            ).tolist()
            for side in active_sides
        }
        height_error = (
            abs(centers["left"][2] - centers["right"][2])
            if set(active_sides) == {"left", "right"} else 0.0
        )
        maximum_height_error = max(maximum_height_error, height_error)
        checks = {
            "chest_before_backoff": ordered,
            "backoff_2_35m": math.isclose(backoff, 2.35, abs_tol=1e-9),
            "lateral_shuttle_zero": lateral <= 1e-12,
            "turn_180deg": math.isclose(turn_deg, 180.0, abs_tol=1e-9),
            "release_detached": (
                not bool(frames[release_index].get("left_attached"))
                and not bool(frames[release_index].get("right_attached"))
            ),
            "dual_release_equal_height": (
                height_error <= args.equal_height_tolerance_m
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            failures.append(f"{operation['label']}: {', '.join(failed)}")
        cases.append({
            "label": operation["label"],
            "boxes": [
                int(operation.get(f"{side}_box_id", 0))
                for side in active_sides
            ],
            "checks": checks,
            "backoff_m": backoff,
            "lateral_shuttle_m": lateral,
            "turn_deg": turn_deg,
            "release_centers_map_m": centers,
            "dual_release_height_error_m": height_error,
        })

    maximum_task_ms = max(
        float(item["selected_task_core_ms"]) for item in replay["box_planning"]
    )
    report = {
        "schema": "alfa.v322_chest_safe_turn_contract.v1",
        "success": bool(validation.get("success")) and not failures,
        "model_revision": replay["model_revision"],
        "box_count": len(replay["box_planning"]),
        "cycle_count": len(cases),
        "maximum_task_core_ms": maximum_task_ms,
        "boxes_within_3s": sum(
            float(item["selected_task_core_ms"]) < 3000.0
            for item in replay["box_planning"]
        ),
        "checked_frames": int(validation["checked_frames"]),
        "checked_edge_samples": int(validation["checked_edge_samples"]),
        "maximum_left_tilt_deg": float(validation["maximum_left_tilt_deg"]),
        "maximum_right_tilt_deg": float(validation["maximum_right_tilt_deg"]),
        "maximum_dual_release_height_error_m": maximum_height_error,
        "equal_height_tolerance_m": args.equal_height_tolerance_m,
        "failures": failures,
        "cases": cases,
    }
    write_json(args.output.resolve(), report)
    print(
        f"RESULT success={report['success']} boxes={report['box_count']}/25 "
        f"cycles={report['cycle_count']}/15 max_plan={maximum_task_ms / 1000.0:.3f}s "
        f"max_equal_height_error={1000.0 * maximum_height_error:.6f}mm"
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
