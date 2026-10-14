#!/usr/bin/env python3
"""Replace the conveyor shuttle with a safe-exit 180 degree unload cycle."""

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


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def first_stage_index(frames: list[dict[str, Any]], stages: set[str]) -> int:
    for index, frame in enumerate(frames):
        if str(frame["stage"]) in stages:
            return index
    raise ValueError(f"operation does not contain any of {sorted(stages)}")


def turn_frames(
    joints: list[float],
    pose: list[float],
    start_yaw: float,
    goal_yaw: float,
    stage: str,
    left_attached: bool,
    right_attached: bool,
    maximum_yaw_step_deg: float,
) -> list[dict[str, Any]]:
    yaw_delta = goal_yaw - start_yaw
    steps = max(1, math.ceil(abs(math.degrees(yaw_delta)) / maximum_yaw_step_deg))
    return [
        {
            "stage": stage,
            "joints": list(joints),
            "left_attached": left_attached,
            "right_attached": right_attached,
            "base_pose_map": [
                float(pose[0]),
                float(pose[1]),
                start_yaw + yaw_delta * index / steps,
            ],
        }
        for index in range(1, steps + 1)
    ]


def transform_operation(
    operation: dict[str, Any],
    turn_direction: str = "auto",
    base_speed_m_s: float = 0.25,
    maximum_yaw_step_deg: float = 1.0,
) -> dict[str, Any]:
    if operation.get("kind") == "base_transition":
        result = copy.deepcopy(operation)
        result["base_speed_m_s"] = base_speed_m_s
        return result
    if base_speed_m_s <= 0.0:
        raise ValueError("base_speed_m_s must be positive")
    if maximum_yaw_step_deg <= 0.0:
        raise ValueError("maximum_yaw_step_deg must be positive")
    if turn_direction not in {"auto", "counterclockwise", "clockwise"}:
        raise ValueError(
            "turn_direction must be auto, clockwise, or counterclockwise"
        )

    frames = list(operation["frames"])
    right_start = first_stage_index(frames, RIGHT_STAGES)
    release_index = first_stage_index(frames, {"release_at_right_conveyor"})
    left_start = first_stage_index(frames, LEFT_STAGES)
    forward_start = first_stage_index(frames, FORWARD_STAGES)
    if not right_start < release_index < left_start < forward_start:
        raise ValueError("conveyor shuttle stages are not in the expected order")

    prefix = copy.deepcopy(frames[:right_start])
    stow = copy.deepcopy(frames[release_index + 1 : left_start])
    suffix = copy.deepcopy(frames[forward_start:])
    if not prefix or not stow or not suffix:
        raise ValueError("operation has an incomplete backoff/stow/return route")

    backed_frame = prefix[-1]
    backed_pose = [
        float(value)
        for value in backed_frame.get("base_pose_map", operation["base_pose_map"])
    ]
    original_yaw = backed_pose[2]
    selected_direction = turn_direction
    if selected_direction == "auto":
        selected_direction = (
            "clockwise" if original_yaw > 1.0e-12 else "counterclockwise"
        )
    direction = 1.0 if selected_direction == "counterclockwise" else -1.0
    turned_yaw = original_yaw + direction * math.pi
    carried_joints = [float(value) for value in backed_frame["joints"]]
    left_attached = bool(backed_frame.get("left_attached", False))
    right_attached = bool(backed_frame.get("right_attached", False))
    if not left_attached and not right_attached:
        raise ValueError("loaded turn must start with at least one attached box")

    loaded_turn = turn_frames(
        carried_joints,
        backed_pose,
        original_yaw,
        turned_yaw,
        "base_turn_180_loaded_at_safe_exit",
        left_attached,
        right_attached,
        maximum_yaw_step_deg,
    )
    release = copy.deepcopy(frames[release_index])
    release.update({
        "stage": "release_after_turn_at_safe_exit",
        "base_pose_map": [backed_pose[0], backed_pose[1], turned_yaw],
    })
    for frame in stow:
        frame["base_pose_map"] = [backed_pose[0], backed_pose[1], turned_yaw]
        if frame["stage"] == "post_release_stow_outside_warehouse":
            frame["stage"] = "post_release_stow_at_safe_exit"
        elif frame["stage"] == "edge_interpolation_to_post_release_stow_outside_warehouse":
            frame["stage"] = "edge_interpolation_to_post_release_stow_at_safe_exit"

    stowed_joints = [float(value) for value in stow[-1]["joints"]]
    empty_turn = turn_frames(
        stowed_joints,
        backed_pose,
        turned_yaw,
        original_yaw,
        "base_turn_180_empty_to_pickup_heading",
        False,
        False,
        maximum_yaw_step_deg,
    )

    result = copy.deepcopy(operation)
    result.update({
        "kind": "dual_safe_turn_unload_cycle",
        "route_strategy": "backoff_turn_180_release_stow_turn_back_return",
        "turn_direction": selected_direction,
        "turn_angle_deg": 180.0,
        "base_speed_m_s": base_speed_m_s,
        "right_shuttle_distance_m": 0.0,
        "unload_pose_map": [backed_pose[0], backed_pose[1], turned_yaw],
        "frames": prefix + loaded_turn + [release] + stow + empty_turn + suffix,
    })
    result.pop("conveyor_pose_map", None)
    return result


def transform_replay(
    replay: dict[str, Any],
    turn_direction: str = "auto",
    base_speed_m_s: float = 0.25,
    maximum_yaw_step_deg: float = 1.0,
) -> dict[str, Any]:
    result = copy.deepcopy(replay)
    result["route_revision"] = "safe_exit_turn_unload_v1"
    result["route_contract"] = {
        "backoff_before_turn_m": 2.35,
        "turn_angle_deg": 180.0,
        "turn_direction_strategy": turn_direction,
        "lateral_shuttle_m": 0.0,
        "base_speed_m_s": base_speed_m_s,
        "maximum_yaw_step_deg": maximum_yaw_step_deg,
    }
    result["operations"] = [
        transform_operation(
            operation,
            turn_direction=turn_direction,
            base_speed_m_s=base_speed_m_s,
            maximum_yaw_step_deg=maximum_yaw_step_deg,
        )
        for operation in replay["operations"]
    ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--turn-direction",
        choices=("auto", "counterclockwise", "clockwise"),
        default="auto",
    )
    parser.add_argument("--base-speed-m-s", type=float, default=0.25)
    parser.add_argument("--maximum-yaw-step-deg", type=float, default=1.0)
    args = parser.parse_args()
    transformed = transform_replay(
        read_json(args.input),
        turn_direction=args.turn_direction,
        base_speed_m_s=args.base_speed_m_s,
        maximum_yaw_step_deg=args.maximum_yaw_step_deg,
    )
    write_json(args.output, transformed)
    cycle_count = sum(
        operation.get("kind") != "base_transition"
        for operation in transformed["operations"]
    )
    frame_count = sum(len(operation["frames"]) for operation in transformed["operations"])
    print(
        f"RESULT cycles={cycle_count} frames={frame_count} "
        f"turn={args.turn_direction} base_speed={args.base_speed_m_s:.3f} "
        f"output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
