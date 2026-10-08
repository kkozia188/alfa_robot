#!/usr/bin/python3
"""Build dual-arm cycles from an arbitrary planar pickup pose."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = (
    Path(__file__).resolve().parent.parent / "v3_yaw_robustness/baseline"
)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v3_5x5_grasp_sequence_rerun as sequence
import v3_scoop_5x5_dual_sequence as dual
import v3_scoop_5x5_grasp_sequence_rerun as scoop
from v3_scoop_5x5_two_standoff_demo import (
    VEHICLE_FRONT_X_IN_BASE_M,
    base_transition_operation,
)
from matrix_common import local_shuttle_poses


SCHEMA = "alfa.v3_scoop_5x5_conveyor_shuttle_replay.v1"
DEFAULT_RIGHT_SHUTTLE_M = 1.5
DEFAULT_BACKOFF_M = 2.35
DUAL_TRANSITION_STRATEGIES = {
    (16, 20): "synchronized",
}
DUAL_UPDOWN_POLICIES = {
    (22, 24): "right",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def load_path_map(raw: str) -> dict[int, Path]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("path map must be a JSON object")
    return {int(box_id): Path(str(path)).resolve() for box_id, path in value.items()}


def load_named_path_map(raw: str) -> dict[str, Path]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("named path map must be a JSON object")
    return {str(name): Path(str(path)).resolve() for name, path in value.items()}


def chained_dual_transition_bridges(
    direct_paths: dict[str, Path],
    prefix_replay_paths: dict[str, Path],
    suffix_paths: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    bridges = {name: read_json(path) for name, path in direct_paths.items()}
    for name, replay_path in prefix_replay_paths.items():
        left_id, right_id = (int(value) for value in name.split("+", maxsplit=1))
        label = f"Box {left_id} + Box {right_id}"
        replay = read_json(replay_path)
        matches = [
            operation for operation in replay.get("operations", [])
            if operation.get("label") == label
        ]
        if len(matches) != 1:
            raise ValueError(f"prefix replay has no unique operation {label}")
        frames = [
            frame for frame in matches[0]["frames"]
            if frame["stage"] == "group_start"
            or str(frame["stage"]).startswith("empty_transition")
        ]
        suffix_path = suffix_paths.get(name)
        if suffix_path:
            suffix = read_json(suffix_path)
            if not suffix.get("success") or not suffix.get("frames"):
                raise ValueError(f"invalid dual transition suffix for {name}")
            suffix_frames = list(suffix["frames"])
            if max(
                abs(float(left) - float(right))
                for left, right in zip(frames[-1]["joints"], suffix_frames[0]["joints"])
            ) > 1e-6:
                raise ValueError(f"dual transition suffix start mismatch for {name}")
            frames.extend(suffix_frames[1:])
        bridges[name] = {
            "success": True,
            "transition_group": "whole_body",
            "transition_strategy": "validated_prefix_and_local_suffix",
            "frames": frames,
        }
    unknown_suffixes = set(suffix_paths) - set(prefix_replay_paths)
    if unknown_suffixes:
        raise ValueError(f"dual suffix lacks prefix replay: {sorted(unknown_suffixes)}")
    return bridges


def interpolate_base_frames(
    joints: list[float],
    start_pose: list[float],
    goal_pose: list[float],
    step_m: float,
    stage: str,
    left_attached: bool,
    right_attached: bool,
) -> list[dict[str, Any]]:
    distance = math.hypot(
        goal_pose[0] - start_pose[0], goal_pose[1] - start_pose[1]
    )
    yaw_delta = math.atan2(
        math.sin(goal_pose[2] - start_pose[2]),
        math.cos(goal_pose[2] - start_pose[2]),
    )
    steps = max(1, math.ceil(distance / step_m), math.ceil(abs(yaw_delta) / math.radians(1.0)))
    return [{
        "stage": stage,
        "joints": list(joints),
        "left_attached": left_attached,
        "right_attached": right_attached,
        "base_pose_map": [
            start_pose[0] + (goal_pose[0] - start_pose[0]) * index / steps,
            start_pose[1] + (goal_pose[1] - start_pose[1]) * index / steps,
            start_pose[2] + yaw_delta * index / steps,
        ],
    } for index in range(1, steps + 1)]


def densify_operation_frames(
    frames: list[dict[str, Any]],
    maximum_joint_step_deg: float = 3.0,
    maximum_lift_step_m: float = 0.01,
    maximum_base_step_m: float = 0.01,
    default_base_pose: list[float] | None = None,
) -> list[dict[str, Any]]:
    if not frames:
        raise ValueError("cannot densify empty operation")
    output = [copy.deepcopy(frames[0])]
    default_pose = list(default_base_pose or [0.0, 0.0, 0.0])
    joint_step = math.radians(maximum_joint_step_deg)
    for target in frames[1:]:
        source = output[-1]
        source_attached = (
            bool(source.get("left_attached", False)),
            bool(source.get("right_attached", False)),
        )
        target_attached = (
            bool(target.get("left_attached", False)),
            bool(target.get("right_attached", False)),
        )
        joint_deltas = [
            float(right) - float(left)
            for left, right in zip(source["joints"], target["joints"])
        ]
        source_pose = [float(value) for value in source.get("base_pose_map", default_pose)]
        target_pose = [float(value) for value in target.get("base_pose_map", source_pose)]
        base_distance = math.hypot(
            target_pose[0] - source_pose[0], target_pose[1] - source_pose[1]
        )
        yaw_delta = math.atan2(
            math.sin(target_pose[2] - source_pose[2]),
            math.cos(target_pose[2] - source_pose[2]),
        )
        steps = max(
            1,
            math.ceil(max(abs(value) for value in joint_deltas[2:]) / joint_step),
            math.ceil(abs(joint_deltas[0]) / maximum_lift_step_m),
            math.ceil(base_distance / maximum_base_step_m),
            math.ceil(abs(yaw_delta) / math.radians(1.0)),
        )
        if source_attached != target_attached and steps > 1:
            raise ValueError("attachment state changed across a moving edge")
        for index in range(1, steps):
            ratio = index / steps
            output.append({
                "stage": f"edge_interpolation_to_{target['stage']}",
                "joints": [
                    float(value) + delta * ratio
                    for value, delta in zip(source["joints"], joint_deltas)
                ],
                "left_attached": source_attached[0],
                "right_attached": source_attached[1],
                "base_pose_map": [
                    source_pose[0] + (target_pose[0] - source_pose[0]) * ratio,
                    source_pose[1] + (target_pose[1] - source_pose[1]) * ratio,
                    source_pose[2] + yaw_delta * ratio,
                ],
            })
        output.append(copy.deepcopy(target))
    return output


def append_task_until_release(
    frames: list[dict[str, Any]],
    task: dict[str, Any],
    held_sides: set[str],
    joint_names: list[str],
) -> None:
    side = str(task["side"])
    held_indices = {
        held_side: [
            joint_names.index(f"{held_side}_joint{index}") for index in range(1, 8)
        ]
        for held_side in held_sides
    }
    for source in [*task.get("transition_frames", []), *task["payload"]["frames"]]:
        if str(source["stage"]) == "release_at_place":
            break
        source_attached = bool(source.get("box_attached", False))
        if source_attached:
            held_sides.add(side)
        joints = [float(value) for value in source["joints"]]
        if side not in held_indices:
            for held_side, indices in held_indices.items():
                for index in indices:
                    joints[index] = float(frames[-1]["joints"][index])
        frames.append({
            "stage": str(source["stage"]),
            "joints": joints,
            "left_attached": "left" in held_sides,
            "right_attached": "right" in held_sides,
        })
    if side not in held_sides:
        raise ValueError(f"box {task['box_id']} has no attached pickup segment")


def append_conveyor_shuttle(
    frames: list[dict[str, Any]],
    pickup_pose: list[float],
    backoff_m: float,
    right_shuttle_m: float,
    step_m: float,
    held_sides: set[str],
    post_release_frames: list[dict[str, Any]] | None = None,
) -> None:
    if not held_sides or not frames:
        raise ValueError("conveyor shuttle requires at least one attached box")
    joints = list(frames[-1]["joints"])
    backed_pose, conveyor_pose = local_shuttle_poses(
        pickup_pose, backoff_m, right_shuttle_m
    )
    frames.extend(interpolate_base_frames(
        joints,
        pickup_pose,
        backed_pose,
        step_m,
        "base_backoff_before_conveyor",
        "left" in held_sides,
        "right" in held_sides,
    ))
    frames.extend(interpolate_base_frames(
        joints,
        backed_pose,
        conveyor_pose,
        step_m,
        "base_right_to_conveyor",
        "left" in held_sides,
        "right" in held_sides,
    ))
    frames.append({
        "stage": "release_at_right_conveyor",
        "joints": joints,
        "left_attached": False,
        "right_attached": False,
        "base_pose_map": list(conveyor_pose),
    })
    for index, source in enumerate(post_release_frames or []):
        source_joints = [float(value) for value in source["joints"]]
        if index == 0 and max(
            abs(left - right) for left, right in zip(joints, source_joints)
        ) > 1e-6:
            raise ValueError("post-release stow does not start at the carried state")
        frames.append({
            "stage": "post_release_stow_outside_warehouse",
            "joints": source_joints,
            "left_attached": False,
            "right_attached": False,
            "base_pose_map": list(conveyor_pose),
        })
        joints = source_joints
    frames.extend(interpolate_base_frames(
        joints,
        conveyor_pose,
        backed_pose,
        step_m,
        "base_left_from_conveyor",
        False,
        False,
    ))
    frames.extend(interpolate_base_frames(
        joints,
        backed_pose,
        pickup_pose,
        step_m,
        "base_forward_to_pickup",
        False,
        False,
    ))
    held_sides.clear()


def task_base_pose(task: dict[str, Any]) -> list[float]:
    return dual.task_base_pose(task)


def apply_overrides(
    tasks: dict[int, dict[str, Any]],
    payload_paths: dict[int, Path],
    transition_paths: dict[int, Path],
    transition_suffix_paths: dict[int, Path],
    center_transition_prefix_replay_paths: dict[int, Path],
    center_transition_suffix_paths: dict[int, Path],
) -> None:
    for box_id, path in payload_paths.items():
        payload = read_json(path)
        if not payload.get("success") or int(payload.get("target_box_id", 0)) != box_id:
            raise ValueError(f"invalid payload override for box {box_id}: {path}")
        task = tasks[box_id]
        payload_frames = list(payload.get("frames", []))
        if (
            payload_frames
            and not any(frame.get("stage") == "precontact" for frame in payload_frames)
            and payload_frames[0].get("stage") == "cartesian_approach"
        ):
            precontact = copy.deepcopy(payload_frames[0])
            precontact["stage"] = "precontact"
            precontact["box_attached"] = False
            payload["frames"] = [precontact, *payload_frames]
        task["payload"] = payload
        task["side"] = str(payload["side"])
        task["mode"] = str(payload["grasp_mode"])
        task["updown"] = float(payload["initial_updown"])
        task["planning_source"] = "conveyor_pickup_replan"
    for box_id, path in transition_paths.items():
        transition = read_json(path)
        if not transition.get("success") or not transition.get("frames"):
            raise ValueError(f"invalid transition override for box {box_id}: {path}")
        tasks[box_id]["transition_frames"] = list(transition["frames"])
        tasks[box_id]["transition_core_ms"] = float(transition.get("total_ms", 0.0))
    for box_id, path in transition_suffix_paths.items():
        transition = read_json(path)
        if not transition.get("success") or not transition.get("frames"):
            raise ValueError(f"invalid transition suffix for box {box_id}: {path}")
        frames = list(tasks[box_id]["transition_frames"])
        suffix = list(transition["frames"])
        if max(
            abs(float(left) - float(right))
            for left, right in zip(frames[-1]["joints"], suffix[0]["joints"])
        ) > 1e-6:
            raise ValueError(f"transition suffix start mismatch for box {box_id}")
        tasks[box_id]["transition_frames"] = [*frames, *suffix[1:]]
        tasks[box_id]["transition_core_ms"] = (
            float(tasks[box_id].get("transition_core_ms", 0.0))
            + float(transition.get("total_ms", 0.0))
        )
    for box_id, replay_path in center_transition_prefix_replay_paths.items():
        replay = read_json(replay_path)
        matches = [
            operation for operation in replay.get("operations", [])
            if operation.get("label") == f"Box {box_id}"
        ]
        if len(matches) != 1:
            raise ValueError(f"center prefix replay has no unique box {box_id}")
        frames = [
            frame for frame in matches[0]["frames"]
            if frame["stage"] == "group_start"
            or str(frame["stage"]).startswith("center_entry")
            or frame["stage"] in {"between_boxes", "precontact"}
        ]
        suffix_path = center_transition_suffix_paths.get(box_id)
        if suffix_path:
            suffix = read_json(suffix_path)
            if not suffix.get("success") or not suffix.get("frames"):
                raise ValueError(f"invalid center transition suffix for box {box_id}")
            suffix_frames = list(suffix["frames"])
            if max(
                abs(float(left) - float(right))
                for left, right in zip(frames[-1]["joints"], suffix_frames[0]["joints"])
            ) > 1e-6:
                raise ValueError(f"center transition suffix start mismatch for box {box_id}")
            frames.extend(suffix_frames[1:])
        tasks[box_id]["transition_frames"] = frames
        tasks[box_id]["transition_core_ms"] = 0.0
    unknown_center_suffixes = (
        set(center_transition_suffix_paths) - set(center_transition_prefix_replay_paths)
    )
    if unknown_center_suffixes:
        raise ValueError(
            f"center suffix lacks prefix replay: {sorted(unknown_center_suffixes)}"
        )


def operation_for_group(
    group: tuple[int, int | None],
    tasks: dict[int, dict[str, Any]],
    current: list[float],
    removed: set[int],
    fallback_attachments: dict[str, dict[str, Any]],
    backoff_m: float,
    right_shuttle_m: float,
    step_m: float,
    post_release_transitions: dict[int, dict[str, Any]],
    dual_transition_bridges: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[float]]:
    box_ids = [box_id for box_id in group if box_id is not None]
    group_tasks = [tasks[box_id] for box_id in box_ids]
    pickup_pose = task_base_pose(group_tasks[0])
    if len(pickup_pose) != 3 or not all(math.isfinite(value) for value in pickup_pose):
        raise ValueError(f"box {box_ids[0]} has an invalid planar pickup pose")
    for task in group_tasks[1:]:
        if max(
            abs(left - right)
            for left, right in zip(pickup_pose, task_base_pose(task))
        ) > 1e-9:
            raise ValueError(f"group {box_ids} does not share one pickup base pose")

    if (
        len(group_tasks) == 2
        and group_tasks[0]["side"] == "left"
        and group_tasks[1]["side"] == "right"
        and group_tasks[0]["mode"] == group_tasks[1]["mode"]
        and math.isclose(
            float(group_tasks[0]["updown"]),
            float(group_tasks[1]["updown"]),
            abs_tol=1e-9,
        )
    ):
        group_key = f"{box_ids[0]}+{box_ids[1]}"
        transition_bridge = dual_transition_bridges.get(group_key)
        operation, _ = dual.combine_pair(
            group_tasks[0],
            group_tasks[1],
            current,
            removed,
            DUAL_UPDOWN_POLICIES.get(tuple(box_ids), "left"),
            (
                "whole_body_bridge" if transition_bridge
                else DUAL_TRANSITION_STRATEGIES.get(tuple(box_ids), "reference")
            ),
            transition_bridge,
        )
        frames = operation["frames"]
        if not frames:
            raise ValueError(f"dual group {box_ids} has no frames")
        partial_pickup = frames[-1]["stage"] != "release_at_place"
        if frames[-1]["stage"] == "release_at_place":
            frames.pop()
        elif not (
            bool(frames[-1].get("left_attached"))
            and bool(frames[-1].get("right_attached"))
        ):
            raise ValueError(
                f"dual group {box_ids} does not end with both boxes attached"
            )
        held_sides = {"left", "right"}
        post_release = post_release_transitions.get(box_ids[-1])
        automatic_stow = None
        if partial_pickup and not post_release:
            automatic_stow = [
                {
                    "stage": "released_pickup_end",
                    "joints": [float(value) for value in frames[-1]["joints"]],
                },
                *[
                {
                    "stage": "reverse_validated_dual_pickup",
                    "joints": [float(value) for value in source["joints"]],
                }
                for source in reversed(frames[:-1])
                ],
            ]
        append_conveyor_shuttle(
            frames,
            pickup_pose,
            backoff_m,
            right_shuttle_m,
            step_m,
            held_sides,
            (
                list(post_release.get("frames", []))
                if post_release else automatic_stow
            ),
        )
        operation.update({
            "kind": "dual_conveyor_cycle",
            "conveyor_pose_map": [
                pickup_pose[0] - backoff_m,
                pickup_pose[1] - right_shuttle_m,
                pickup_pose[2],
            ],
            "right_shuttle_distance_m": right_shuttle_m,
            "backoff_distance_m": backoff_m,
            "pickup_order": [int(task["box_id"]) for task in group_tasks],
            "pickup_mode": "simultaneous",
            "frames": frames,
        })
        operation["frames"] = densify_operation_frames(
            operation["frames"], default_base_pose=pickup_pose
        )
        return operation, list(frames[-1]["joints"])

    if len(group_tasks) == 2:
        raise ValueError(
            f"group {box_ids} cannot be composed as a simultaneous dual pickup"
        )

    frames: list[dict[str, Any]] = [{
        "stage": "group_start",
        "joints": list(current),
        "left_attached": False,
        "right_attached": False,
        "base_pose_map": list(pickup_pose),
    }]
    held_sides: set[str] = set()
    execution_tasks = sorted(group_tasks, key=lambda task: float(task["updown"]))
    joint_names = [str(value) for value in group_tasks[0]["payload"]["joint_names"]]
    for task in execution_tasks:
        append_task_until_release(frames, task, held_sides, joint_names)
    final_box_id = box_ids[-1]
    post_release = post_release_transitions.get(final_box_id)
    automatic_stow = None
    if not post_release:
        automatic_stow = [
            {
                "stage": "released_pickup_end",
                "joints": [float(value) for value in frames[-1]["joints"]],
            },
            *[
                {
                    "stage": "post_release_reverse_pickup",
                    "joints": [float(value) for value in source["joints"]],
                }
                for source in reversed(frames[:-1])
            ],
        ]
    append_conveyor_shuttle(
        frames,
        pickup_pose,
        backoff_m,
        right_shuttle_m,
        step_m,
        held_sides,
        list(post_release.get("frames", [])) if post_release else automatic_stow,
    )

    if not post_release and automatic_stow is None and len(group_tasks) == 1 and not any(
        str(frame["stage"]) == "release_at_place"
        for frame in group_tasks[-1]["payload"]["frames"]
    ):
        for source in reversed(group_tasks[-1]["payload"]["frames"]):
            frames.append({
                "stage": "post_conveyor_reverse_pickup",
                "joints": [float(value) for value in source["joints"]],
                "left_attached": False,
                "right_attached": False,
                "base_pose_map": list(pickup_pose),
            })
        for source in reversed(group_tasks[-1].get("transition_frames", [])):
            frames.append({
                "stage": "post_conveyor_reverse_entry",
                "joints": [float(value) for value in source["joints"]],
                "left_attached": False,
                "right_attached": False,
                "base_pose_map": list(pickup_pose),
            })

    attachments = dict(fallback_attachments)
    for task in group_tasks:
        attachments[str(task["side"])] = dual.attachment(task)
    left_id = next(
        (int(task["box_id"]) for task in group_tasks if task["side"] == "left"), 0
    )
    right_id = next(
        (int(task["box_id"]) for task in group_tasks if task["side"] == "right"), 0
    )
    kind = "dual_conveyor_cycle" if len(group_tasks) == 2 else "single_conveyor_cycle"
    operation = {
        "label": " + ".join(f"Box {box_id}" for box_id in box_ids),
        "kind": kind,
        "row": int(group_tasks[0]["row"]),
        "left_box_id": left_id,
        "right_box_id": right_id,
        "mode": "+".join(str(task["mode"]) for task in group_tasks),
        "base_pose_map": list(pickup_pose),
        "conveyor_pose_map": [
            pickup_pose[0] - backoff_m,
            pickup_pose[1] - right_shuttle_m,
            pickup_pose[2],
        ],
        "right_shuttle_distance_m": right_shuttle_m,
        "backoff_distance_m": backoff_m,
        "pickup_order": [int(task["box_id"]) for task in execution_tasks],
        "pickup_mode": "sequential_low_to_high" if len(group_tasks) == 2 else "single",
        "removed_before": sorted(removed),
        "left_attachment": attachments["left"],
        "right_attachment": attachments["right"],
        "box_planning": [dual.task_planning_summary(task) for task in group_tasks],
        "obstacles": dual.obstacles(removed, set(box_ids)),
        "allow_ground_model_base": True,
        "frames": frames,
    }
    operation["frames"] = densify_operation_frames(
        operation["frames"], default_base_pose=pickup_pose
    )
    return operation, list(operation["frames"][-1]["joints"])


def build_replay(
    cache_path: Path,
    output_path: Path,
    backoff_m: float,
    right_shuttle_m: float,
    step_m: float,
    payload_paths: dict[int, Path] | None = None,
    transition_paths: dict[int, Path] | None = None,
    transition_suffix_paths: dict[int, Path] | None = None,
    post_release_paths: dict[int, Path] | None = None,
    dual_transition_bridge_paths: dict[str, Path] | None = None,
    dual_transition_prefix_replay_paths: dict[str, Path] | None = None,
    dual_transition_suffix_paths: dict[str, Path] | None = None,
    center_transition_prefix_replay_paths: dict[int, Path] | None = None,
    center_transition_suffix_paths: dict[int, Path] | None = None,
) -> dict[str, Any]:
    cache = read_json(cache_path)
    if cache.get("schema") != scoop.PLAN_CACHE_SCHEMA:
        raise ValueError("input is not a scoop plan cache")
    tasks = {int(task["box_id"]): copy.deepcopy(task) for task in cache["tasks"]}
    if set(tasks) != set(range(1, 26)):
        raise ValueError("plan cache must contain boxes 1..25")
    apply_overrides(
        tasks,
        payload_paths or {},
        transition_paths or {},
        transition_suffix_paths or {},
        center_transition_prefix_replay_paths or {},
        center_transition_suffix_paths or {},
    )
    post_release_transitions = {
        box_id: read_json(path) for box_id, path in (post_release_paths or {}).items()
    }
    dual_transition_bridges = chained_dual_transition_bridges(
        dual_transition_bridge_paths or {},
        dual_transition_prefix_replay_paths or {},
        dual_transition_suffix_paths or {},
    )
    for box_id, transition in post_release_transitions.items():
        if not transition.get("success") or not transition.get("frames"):
            raise ValueError(f"invalid post-release transition for box {box_id}")

    names = [str(value) for value in tasks[1]["payload"]["joint_names"]]
    current = sequence.folded_start_joints(cache["station_config"])
    fallback_attachments = {
        "left": dual.attachment(tasks[1]),
        "right": dual.attachment(tasks[5]),
    }
    operations: list[dict[str, Any]] = []
    removed: set[int] = set()
    previous_pose: list[float] | None = None
    for group in dual.GROUPS:
        first_task = tasks[group[0]]
        pickup_pose = task_base_pose(first_task)
        if previous_pose is not None and max(
            abs(left - right) for left, right in zip(previous_pose, pickup_pose)
        ) > 1e-9:
            reposition = first_task.get("base_reposition", {})
            if not reposition.get("required"):
                raise ValueError(f"box {group[0]} lacks a validated base reposition")
            operations.append(base_transition_operation(
                [float(value) for value in reposition["transport_joints"]],
                list(reposition.get("stow_frames", [])),
                previous_pose,
                pickup_pose,
                sequence.CONTACT_X - (previous_pose[0] + VEHICLE_FRONT_X_IN_BASE_M),
                sequence.CONTACT_X - (pickup_pose[0] + VEHICLE_FRONT_X_IN_BASE_M),
                removed,
                fallback_attachments["left"],
                fallback_attachments["right"],
                step_m,
                0.10,
                f"Chassis X reposition before box {group[0]}",
                int(first_task["row"]),
                list(reposition.get("coupled_frames", [])),
            ))
            current = list(operations[-1]["frames"][-1]["joints"])
        operation, current = operation_for_group(
            group,
            tasks,
            current,
            removed,
            fallback_attachments,
            backoff_m,
            right_shuttle_m,
            step_m,
            post_release_transitions,
            dual_transition_bridges,
        )
        operations.append(operation)
        group_ids = {box_id for box_id in group if box_id is not None}
        removed.update(group_ids)
        for task in (tasks[box_id] for box_id in group_ids):
            fallback_attachments[str(task["side"])] = dual.attachment(task)
        previous_pose = pickup_pose

    replay = {
        "schema": dual.SCHEMA,
        "conveyor_shuttle_schema": SCHEMA,
        "source_plan_cache": str(cache_path.resolve()),
        "joint_names": names,
        "group_order": [
            [left, right] if right is not None else [left]
            for left, right in dual.GROUPS
        ],
        "conveyor_shuttle": {
            "direction": "robot_local_reverse_then_local_right",
            "pickup_pose_policy": "consume each task base_pose_map",
            "backoff_distance_m": backoff_m,
            "right_shuttle_distance_m": right_shuttle_m,
            "translation_step_m": step_m,
            "cycle_count": len(dual.GROUPS),
        },
        "planning_time_definition": {
            "selected_task_core_ms": "source validated task planning",
            "base_conveyor_shuttle": "planar-root interpolation; validated by MoveIt/FCL",
        },
        "box_planning": [
            dual.task_planning_summary(tasks[box_id]) for box_id in range(1, 26)
        ],
        "operations": operations,
    }
    write_json(output_path, replay)
    return replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backoff-m", type=float, default=DEFAULT_BACKOFF_M)
    parser.add_argument("--right-shuttle-m", type=float, default=DEFAULT_RIGHT_SHUTTLE_M)
    parser.add_argument("--base-step-m", type=float, default=0.01)
    parser.add_argument("--task-payload-map", default="")
    parser.add_argument("--task-transition-map", default="")
    parser.add_argument("--task-transition-suffix-map", default="")
    parser.add_argument("--post-release-transition-map", default="")
    parser.add_argument("--dual-transition-bridge-map", default="")
    parser.add_argument("--dual-transition-prefix-replay-map", default="")
    parser.add_argument("--dual-transition-suffix-map", default="")
    parser.add_argument("--center-transition-prefix-replay-map", default="")
    parser.add_argument("--center-transition-suffix-map", default="")
    args = parser.parse_args()
    if args.backoff_m <= 0.0 or args.right_shuttle_m <= 0.0:
        parser.error("backoff-m and right-shuttle-m must be positive")
    if not 0.0 < args.base_step_m <= 0.01:
        parser.error("base-step-m must be in (0, 0.01]")
    replay = build_replay(
        args.plan_cache.resolve(),
        args.output.resolve(),
        args.backoff_m,
        args.right_shuttle_m,
        args.base_step_m,
        load_path_map(args.task_payload_map),
        load_path_map(args.task_transition_map),
        load_path_map(args.task_transition_suffix_map),
        load_path_map(args.post_release_transition_map),
        load_named_path_map(args.dual_transition_bridge_map),
        load_named_path_map(args.dual_transition_prefix_replay_map),
        load_named_path_map(args.dual_transition_suffix_map),
        load_path_map(args.center_transition_prefix_replay_map),
        load_path_map(args.center_transition_suffix_map),
    )
    print(
        f"CONVEYOR_REPLAY groups={len(replay['group_order'])} "
        f"operations={len(replay['operations'])} output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
