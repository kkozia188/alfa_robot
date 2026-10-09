#!/usr/bin/python3
"""Assemble the complete V3.2.2 chest-carry conveyor replay."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


GROUPS: list[tuple[int, ...]] = [
    (1, 5), (2, 4), (3,),
    (6, 10), (7, 9), (8,),
    (11, 15), (12, 14), (13,),
    (16, 20), (17, 19), (18,),
    (21, 25), (22, 24), (23,),
]
PICKUP_STAGES = ("precontact", "cartesian_approach", "attach_box", "cartesian_retreat")
HOME = [
    -0.3,
    0.0,
    *[math.radians(value) for value in (155, -105, 20, 90, -90, -40, 0)],
    *[math.radians(value) for value in (-155, 105, -20, -90, 90, 40, 0)],
]
BASE_SPEED_M_S = 0.35
BACKOFF_M = 2.35
RIGHT_SHUTTLE_M = 1.50
BASE_STEP_M = 0.01
SHARED_UPDOWN_STRATEGIES = {
    (2, 4): "minimum",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def active_indices(names: list[str], side: str) -> list[int]:
    return [names.index(f"{side}_joint{index}") for index in range(1, 8)]


def interpolate(values: list[list[float]], count: int) -> list[list[float]]:
    if not values:
        raise ValueError("cannot interpolate an empty path")
    if count == 1 or len(values) == 1:
        return [list(values[-1]) for _ in range(count)]
    output: list[list[float]] = []
    for sample in range(count):
        coordinate = sample * (len(values) - 1) / (count - 1)
        left = int(math.floor(coordinate))
        right = min(left + 1, len(values) - 1)
        ratio = coordinate - left
        output.append([
            begin + (end - begin) * ratio
            for begin, end in zip(values[left], values[right])
        ])
    return output


def stage_frames(payload: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    return [frame for frame in payload["frames"] if frame["stage"] == stage]


def transfer_frames(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frames = list(payload["frames"])
    retreat_end = max(
        index for index, frame in enumerate(frames)
        if frame["stage"] == "cartesian_retreat"
    )
    release = next(
        index for index, frame in enumerate(frames)
        if frame["stage"] == "release_at_place"
    )
    result = [
        frame for frame in frames[retreat_end:release]
        if bool(frame.get("box_attached"))
    ]
    if len(result) < 2:
        raise ValueError(f"box {payload['target_box_id']} has no chest transfer")
    return result


def pair_stage(
    left: dict[str, Any],
    right: dict[str, Any],
    names: list[str],
    stage: str,
    current: list[float],
) -> list[dict[str, Any]]:
    left_values = [[float(value) for value in frame["joints"]] for frame in stage_frames(left, stage)]
    right_values = [[float(value) for value in frame["joints"]] for frame in stage_frames(right, stage)]
    count = max(len(left_values), len(right_values))
    left_values = interpolate(left_values, count)
    right_values = interpolate(right_values, count)
    left_indices = active_indices(names, "left")
    right_indices = active_indices(names, "right")
    attached = stage in {"attach_box", "cartesian_retreat"}
    output: list[dict[str, Any]] = []
    for left_joints, right_joints in zip(left_values, right_values):
        if abs(left_joints[0] - right_joints[0]) > 1e-6:
            raise ValueError(f"paired pickup stage {stage} has incompatible updown")
        combined = list(current)
        # The shared lift must never lower an arm earlier than its independently
        # validated path. A larger updown value is physically higher.
        combined[0] = max(left_joints[0], right_joints[0])
        for index in left_indices:
            combined[index] = left_joints[index]
        for index in right_indices:
            combined[index] = right_joints[index]
        output.append({
            "stage": stage,
            "joints": combined,
            "left_attached": attached,
            "right_attached": attached,
        })
        current = combined
    return output


def paired_transfer(
    left: dict[str, Any],
    right: dict[str, Any],
    names: list[str],
    current: list[float],
) -> tuple[list[dict[str, Any]], float]:
    left_source = [[float(value) for value in frame["joints"]] for frame in transfer_frames(left)]
    right_source = [[float(value) for value in frame["joints"]] for frame in transfer_frames(right)]
    count = max(len(left_source), len(right_source))
    left_values = interpolate(left_source, count)
    right_values = interpolate(right_source, count)
    left_indices = active_indices(names, "left")
    right_indices = active_indices(names, "right")
    group = tuple(sorted((int(left["target_box_id"]), int(right["target_box_id"]))))
    strategy = SHARED_UPDOWN_STRATEGIES.get(group, "maximum")
    output: list[dict[str, Any]] = []
    maximum_updown_divergence = 0.0
    for left_joints, right_joints in zip(left_values, right_values):
        maximum_updown_divergence = max(
            maximum_updown_divergence, abs(left_joints[0] - right_joints[0])
        )
        combined = list(current)
        # Preserve the higher of the two independently safe lift positions so
        # neither carried box is lowered toward the chassis prematurely.
        combined[0] = (
            min(left_joints[0], right_joints[0])
            if strategy == "minimum" else max(left_joints[0], right_joints[0])
        )
        for index in left_indices:
            combined[index] = left_joints[index]
        for index in right_indices:
            combined[index] = right_joints[index]
        output.append({
            "stage": "chest_carry_retract",
            "joints": combined,
            "left_attached": True,
            "right_attached": True,
        })
        current = combined
    output[-1]["stage"] = "chest_carry_ready"
    return output, maximum_updown_divergence


def single_forward(payload: dict[str, Any], bridge: dict[str, Any]) -> list[dict[str, Any]]:
    side = str(payload["side"])
    output = [{
        "stage": "group_start" if index == 0 else "empty_transition_whole_body_bridge",
        "joints": [float(value) for value in frame["joints"]],
        "left_attached": False,
        "right_attached": False,
    } for index, frame in enumerate(bridge["frames"])]
    for stage in PICKUP_STAGES:
        for frame in stage_frames(payload, stage):
            attached = bool(frame.get("box_attached"))
            output.append({
                "stage": stage,
                "joints": [float(value) for value in frame["joints"]],
                "left_attached": attached and side == "left",
                "right_attached": attached and side == "right",
            })
    for index, frame in enumerate(transfer_frames(payload)):
        output.append({
            "stage": (
                "chest_carry_ready" if index == len(transfer_frames(payload)) - 1
                else "chest_carry_retract"
            ),
            "joints": [float(value) for value in frame["joints"]],
            "left_attached": side == "left",
            "right_attached": side == "right",
        })
    return output


def pair_forward(
    left: dict[str, Any], right: dict[str, Any], bridge: dict[str, Any]
) -> tuple[list[dict[str, Any]], float]:
    names = [str(value) for value in left["joint_names"]]
    output = [{
        "stage": "group_start" if index == 0 else "empty_transition_whole_body_bridge",
        "joints": [float(value) for value in frame["joints"]],
        "left_attached": False,
        "right_attached": False,
    } for index, frame in enumerate(bridge["frames"])]
    current = list(output[-1]["joints"])
    for stage in PICKUP_STAGES:
        combined = pair_stage(left, right, names, stage, current)
        output.extend(combined)
        current = list(combined[-1]["joints"])
    transfer, divergence = paired_transfer(left, right, names, current)
    output.extend(transfer)
    return output, divergence


def base_frames(
    joints: list[float],
    start: list[float],
    goal: list[float],
    stage: str,
    left_attached: bool,
    right_attached: bool,
) -> list[dict[str, Any]]:
    distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
    steps = max(1, math.ceil(distance / BASE_STEP_M))
    return [{
        "stage": stage,
        "joints": list(joints),
        "left_attached": left_attached,
        "right_attached": right_attached,
        "base_pose_map": [
            start[0] + (goal[0] - start[0]) * index / steps,
            start[1] + (goal[1] - start[1]) * index / steps,
            start[2] + (goal[2] - start[2]) * index / steps,
        ],
    } for index in range(1, steps + 1)]


def densify_frames(
    frames: list[dict[str, Any]], default_base_pose: list[float]
) -> list[dict[str, Any]]:
    output = [copy.deepcopy(frames[0])]
    maximum_joint_step = math.radians(3.0)
    for target in frames[1:]:
        source = output[-1]
        source_joints = [float(value) for value in source["joints"]]
        target_joints = [float(value) for value in target["joints"]]
        deltas = [right - left for left, right in zip(source_joints, target_joints)]
        source_pose = [float(value) for value in source.get("base_pose_map", default_base_pose)]
        target_pose = [float(value) for value in target.get("base_pose_map", source_pose)]
        base_distance = math.hypot(
            target_pose[0] - source_pose[0], target_pose[1] - source_pose[1]
        )
        steps = max(
            1,
            math.ceil(max(abs(value) for value in deltas[2:]) / maximum_joint_step),
            math.ceil(abs(deltas[0]) / 0.01),
            math.ceil(base_distance / BASE_STEP_M),
        )
        source_attached = (
            bool(source.get("left_attached")), bool(source.get("right_attached"))
        )
        target_attached = (
            bool(target.get("left_attached")), bool(target.get("right_attached"))
        )
        if source_attached != target_attached and steps > 1:
            raise ValueError(
                f"attachment changes across moving edge to {target['stage']}"
            )
        for index in range(1, steps):
            ratio = index / steps
            output.append({
                "stage": f"edge_interpolation_to_{target['stage']}",
                "joints": [
                    value + delta * ratio
                    for value, delta in zip(source_joints, deltas)
                ],
                "left_attached": source_attached[0],
                "right_attached": source_attached[1],
                "base_pose_map": [
                    source_pose[axis] + (target_pose[axis] - source_pose[axis]) * ratio
                    for axis in range(3)
                ],
            })
        output.append(copy.deepcopy(target))
    return output


def build_cycle(
    group: tuple[int, ...],
    payloads: dict[int, dict[str, Any]],
    bridge: dict[str, Any],
    original: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    selected = [payloads[box_id] for box_id in group]
    if len(group) == 2:
        left = next(payload for payload in selected if payload["side"] == "left")
        right = next(payload for payload in selected if payload["side"] == "right")
        forward, divergence = pair_forward(left, right, bridge)
    else:
        forward = single_forward(selected[0], bridge)
        divergence = 0.0
    pickup_pose = [float(value) for value in original["base_pose_map"]]
    backed_pose = [pickup_pose[0] - BACKOFF_M, pickup_pose[1], pickup_pose[2]]
    conveyor_pose = [backed_pose[0], backed_pose[1] - RIGHT_SHUTTLE_M, backed_pose[2]]
    held_left = any(payload["side"] == "left" for payload in selected)
    held_right = any(payload["side"] == "right" for payload in selected)
    frames = copy.deepcopy(forward)
    chest_joints = list(frames[-1]["joints"])
    frames.extend(base_frames(
        chest_joints, pickup_pose, backed_pose,
        "base_backoff_after_chest_carry", held_left, held_right,
    ))
    frames.extend(base_frames(
        chest_joints, backed_pose, conveyor_pose,
        "base_right_to_conveyor", held_left, held_right,
    ))
    frames.append({
        "stage": "release_at_right_conveyor",
        "joints": chest_joints,
        "left_attached": False,
        "right_attached": False,
        "base_pose_map": conveyor_pose,
    })
    for source in reversed(forward[:-1]):
        frames.append({
            "stage": "post_release_stow_outside_warehouse",
            "joints": list(source["joints"]),
            "left_attached": False,
            "right_attached": False,
            "base_pose_map": conveyor_pose,
        })
    frames.extend(base_frames(
        list(frames[-1]["joints"]), conveyor_pose, backed_pose,
        "base_left_from_conveyor", False, False,
    ))
    frames.extend(base_frames(
        list(frames[-1]["joints"]), backed_pose, pickup_pose,
        "base_forward_to_pickup", False, False,
    ))
    operation = copy.deepcopy(original)
    bridge_ms = float(bridge.get("total_ms", 0.0))
    planning_entries = [{
        "box_id": int(payload["target_box_id"]),
        "row": (int(payload["target_box_id"]) - 1) // 5 + 1,
        "column": (int(payload["target_box_id"]) - 1) % 5 + 1,
        "side": str(payload["side"]),
        "mode": str(payload["grasp_mode"]),
        "source": "fresh_chest_carry_planning",
        "selected_task_core_ms": float(payload.get("total_ms", 0.0)),
        "accumulated_task_search_ms": float(payload.get("total_ms", 0.0)),
        "selected_transition_core_ms": bridge_ms,
        "accumulated_transition_search_ms": bridge_ms,
    } for payload in selected]
    operation.update({
        "kind": "dual_chest_carry_conveyor_cycle" if len(group) == 2
        else "single_chest_carry_conveyor_cycle",
        "pickup_mode": "simultaneous" if len(group) == 2 else "single",
        "pickup_order": list(group),
        "base_speed_m_s": BASE_SPEED_M_S,
        "backoff_distance_m": BACKOFF_M,
        "right_shuttle_distance_m": RIGHT_SHUTTLE_M,
        "conveyor_pose_map": conveyor_pose,
        "chest_carry": {
            "required_before_base_motion": True,
            "base_speed_m_s": BASE_SPEED_M_S,
            "maximum_source_updown_divergence_m": divergence,
        },
        "transition_bridge": {
            "total_ms": bridge_ms,
            "transition_group": "whole_body",
            "transition_strategy": "fresh_home_to_chest_pickup_precontact",
            "frame_count": len(bridge["frames"]),
        },
        "transition_strategy": "fresh_whole_body_bridge",
        "box_planning": planning_entries,
        "frames": densify_frames(frames, pickup_pose),
    })
    return operation, divergence


def build_reposition(original: dict[str, Any]) -> dict[str, Any]:
    operation = copy.deepcopy(original)
    start = [float(value) for value in original["base_pose_map"]]
    goal = [float(value) for value in original["base_pose_goal_map"]]
    operation.update({
        "base_speed_m_s": BASE_SPEED_M_S,
        "frames": densify_frames([{
            "stage": "base_transition_start",
            "joints": list(HOME),
            "left_attached": False,
            "right_attached": False,
            "base_pose_map": start,
        }, *base_frames(
            HOME, start, goal, "base_transition", False, False
        )], start),
    })
    return operation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-replay", type=Path, required=True)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = read_json(args.baseline_replay.resolve())
    payloads = {
        box_id: read_json(args.payload_dir.resolve() / f"box-{box_id:02d}.json")
        for box_id in range(1, 26)
    }
    originals = {operation["label"]: operation for operation in baseline["operations"]}
    reposition = next(
        operation for operation in baseline["operations"]
        if operation["kind"] == "base_transition"
    )
    operations: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    box_planning: list[dict[str, Any]] = []
    for group_index, group in enumerate(GROUPS):
        if group_index == 9:
            operations.append(build_reposition(reposition))
        label = " + ".join(f"Box {box_id}" for box_id in group)
        key = "-".join(f"{box_id:02d}" for box_id in group)
        bridge = read_json(args.bridge_dir.resolve() / f"group-{key}.json")
        operation, divergence = build_cycle(
            group, payloads, bridge, originals[label]
        )
        operations.append(operation)
        box_planning.extend(operation["box_planning"])
        summary.append({
            "group": list(group),
            "frame_count": len(operation["frames"]),
            "maximum_source_updown_divergence_m": divergence,
        })
    replay = {
        "schema": "alfa.v3_scoop_5x5_dual_replay.v1",
        "chest_carry_schema": "alfa.v322_chest_carry_fast_base.v1",
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "source_replay": str(args.baseline_replay.resolve()),
        "joint_names": list(baseline["joint_names"]),
        "group_order": [list(group) for group in GROUPS],
        "planning_time_definition": {
            "selected_task_core_ms": "fresh chest-carry payload planner wall time",
            "base_motion": "geometry sampled at 1 cm; 0.35 m/s affects replay timing only",
        },
        "box_planning": box_planning,
        "chest_carry": {
            "front_box_center_base_m": [0.75, "side_sign*0.32", 0.95],
            "top_box_center_base_m": [0.90, "side_sign*0.40", 0.85],
            "carry_updown_m": -0.30,
            "base_speed_m_s": BASE_SPEED_M_S,
            "base_moves_only_after_chest_ready": True,
        },
        "operations": operations,
        "assembly_summary": summary,
    }
    write_json(args.output.resolve(), replay)
    print(
        f"RESULT groups={len(GROUPS)} operations={len(operations)} "
        f"frames={sum(len(operation['frames']) for operation in operations)} "
        f"output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
