#!/usr/bin/python3
"""Apply the MOTION-258 safe-backoff 180-degree unload route."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


RIGHT_STAGES = {
    "base_right_to_conveyor",
    "edge_interpolation_to_base_right_to_conveyor",
}
LEFT_STAGES = {
    "base_left_from_conveyor",
    "edge_interpolation_to_base_left_from_conveyor",
}
FORWARD_STAGES = {
    "base_forward_to_pickup",
    "edge_interpolation_to_base_forward_to_pickup",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def first_stage_index(frames: list[dict[str, Any]], stages: set[str]) -> int:
    return next(
        index for index, frame in enumerate(frames)
        if str(frame["stage"]) in stages
    )


def turn_frames(
    joints: list[float],
    pose: list[float],
    goal_yaw: float,
    stage: str,
    left_attached: bool,
    right_attached: bool,
    maximum_yaw_step_deg: float,
) -> list[dict[str, Any]]:
    yaw_delta = goal_yaw - float(pose[2])
    steps = max(1, math.ceil(abs(math.degrees(yaw_delta)) / maximum_yaw_step_deg))
    return [{
        "stage": stage,
        "joints": list(joints),
        "left_attached": left_attached,
        "right_attached": right_attached,
        "base_pose_map": [
            float(pose[0]),
            float(pose[1]),
            float(pose[2]) + yaw_delta * index / steps,
        ],
    } for index in range(1, steps + 1)]


def transform_operation(
    operation: dict[str, Any],
    base_speed_m_s: float,
    maximum_yaw_step_deg: float,
) -> dict[str, Any]:
    if operation.get("kind") == "base_transition":
        result = copy.deepcopy(operation)
        result["base_speed_m_s"] = base_speed_m_s
        return result

    frames = list(operation["frames"])
    right_start = first_stage_index(frames, RIGHT_STAGES)
    release_index = first_stage_index(frames, {"release_at_right_conveyor"})
    left_start = first_stage_index(frames, LEFT_STAGES)
    forward_start = first_stage_index(frames, FORWARD_STAGES)
    if not right_start < release_index < left_start < forward_start:
        raise ValueError(f"{operation['label']} has an invalid conveyor route")
    prefix = copy.deepcopy(frames[:right_start])
    stow = copy.deepcopy(frames[release_index + 1:left_start])
    suffix = copy.deepcopy(frames[forward_start:])
    backed = prefix[-1]
    backed_pose = [
        float(value) for value in backed.get("base_pose_map", operation["base_pose_map"])
    ]
    initial_yaw = backed_pose[2]
    direction = -1.0 if initial_yaw > 1e-12 else 1.0
    turned_yaw = initial_yaw + direction * math.pi
    carried_joints = [float(value) for value in backed["joints"]]
    left_attached = bool(backed.get("left_attached"))
    right_attached = bool(backed.get("right_attached"))
    loaded_turn = turn_frames(
        carried_joints, backed_pose, turned_yaw,
        "base_turn_180_loaded_at_safe_exit",
        left_attached, right_attached, maximum_yaw_step_deg,
    )
    release = copy.deepcopy(frames[release_index])
    release.update({
        "stage": "release_after_turn_at_safe_exit",
        "base_pose_map": [backed_pose[0], backed_pose[1], turned_yaw],
    })
    for frame in stow:
        frame["base_pose_map"] = [backed_pose[0], backed_pose[1], turned_yaw]
        frame["stage"] = str(frame["stage"]).replace(
            "stow_outside_warehouse", "stow_at_safe_exit"
        )
    empty_turn = turn_frames(
        [float(value) for value in stow[-1]["joints"]],
        [backed_pose[0], backed_pose[1], turned_yaw],
        initial_yaw,
        "base_turn_180_empty_to_pickup_heading",
        False, False, maximum_yaw_step_deg,
    )
    result = copy.deepcopy(operation)
    result.update({
        "kind": "dual_safe_turn_unload_cycle",
        "route_strategy": "chest_backoff_turn_180_equal_height_release_return",
        "turn_direction": "clockwise" if direction < 0.0 else "counterclockwise",
        "turn_angle_deg": 180.0,
        "base_speed_m_s": base_speed_m_s,
        "right_shuttle_distance_m": 0.0,
        "unload_pose_map": [backed_pose[0], backed_pose[1], turned_yaw],
        "frames": prefix + loaded_turn + [release] + stow + empty_turn + suffix,
    })
    result.pop("conveyor_pose_map", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-speed-m-s", type=float, default=0.25)
    parser.add_argument("--maximum-yaw-step-deg", type=float, default=1.0)
    args = parser.parse_args()
    if args.base_speed_m_s <= 0.0 or args.maximum_yaw_step_deg <= 0.0:
        parser.error("speed and yaw step must be positive")
    source = read_json(args.input.resolve())
    result = copy.deepcopy(source)
    result["route_revision"] = "motion258_safe_exit_turn_unload_v1"
    result["route_contract"] = {
        "chest_ready_before_backoff": True,
        "backoff_before_turn_m": 2.35,
        "turn_angle_deg": 180.0,
        "lateral_shuttle_m": 0.0,
        "base_speed_m_s": args.base_speed_m_s,
        "maximum_yaw_step_deg": args.maximum_yaw_step_deg,
        "dual_release_equal_height_required": True,
    }
    result["operations"] = [
        transform_operation(
            operation, args.base_speed_m_s, args.maximum_yaw_step_deg
        )
        for operation in source["operations"]
    ]
    write_json(args.output.resolve(), result)
    print(
        f"RESULT cycles={sum(item['kind'] != 'base_transition' for item in result['operations'])} "
        f"frames={sum(len(item['frames']) for item in result['operations'])} "
        f"output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
