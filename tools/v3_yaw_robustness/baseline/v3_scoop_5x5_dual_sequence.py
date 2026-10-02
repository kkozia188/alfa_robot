#!/usr/bin/python3
"""Compose the approved single-arm 5x5 paths into strict row-wise dual-arm groups."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import scan_v3_single_arm_box_wall as scan
import v3_5x5_grasp_sequence_rerun as sequence


SCHEMA = "alfa.v3_scoop_5x5_dual_replay.v1"
GROUPS: list[tuple[int, int | None]] = [
    (1, 5), (2, 4), (3, None),
    (6, 10), (7, 9), (8, None),
    (11, 15), (12, 14), (13, None),
    (16, 20), (17, 19), (18, None),
    (21, 25), (22, 24), (23, None),
]


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


def interpolate_positions(frames: list[dict[str, Any]], count: int) -> list[list[float]]:
    if not frames or count < 1:
        raise ValueError("cannot interpolate an empty frame list")
    values = [[float(value) for value in frame["joints"]] for frame in frames]
    if count == 1:
        return [list(values[-1])]
    if len(values) == 1:
        return [list(values[0]) for _ in range(count)]
    output: list[list[float]] = []
    for sample in range(count):
        coordinate = sample * (len(values) - 1) / (count - 1)
        left = min(int(math.floor(coordinate)), len(values) - 1)
        right = min(left + 1, len(values) - 1)
        ratio = coordinate - left
        output.append([
            begin + (end - begin) * ratio
            for begin, end in zip(values[left], values[right])
        ])
    return output


def active_indices(names: list[str], side: str) -> list[int]:
    return [names.index(f"{side}_joint{index}") for index in range(1, 8)]


def stage_frames(task: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "empty_transition":
        return list(task["transition_frames"])
    return [
        frame for frame in task["payload"]["frames"]
        if str(frame["stage"]) == stage
    ]


def stage_order(task: dict[str, Any]) -> list[str]:
    return ["empty_transition", *dict.fromkeys(
        str(frame["stage"]) for frame in task["payload"]["frames"]
    )]


def attached_for_stage(stage: str) -> bool:
    return stage not in {
        "empty_transition", "precontact", "cartesian_approach", "release_at_place"
    }


def transition_updown(
    policy: str,
    left: float,
    right: float,
    sample: int,
    count: int,
    start: float,
    goal: float,
) -> float:
    if policy == "left":
        return left
    if policy == "right":
        return right
    if policy == "minimum":
        return min(left, right)
    if policy == "maximum":
        return max(left, right)
    if policy == "average":
        return 0.5 * (left + right)
    if policy == "linear":
        return start + (goal - start) * sample / max(1, count - 1)
    raise ValueError(f"unknown transition updown policy: {policy}")


def combine_empty_transition(
    left_frames: list[dict[str, Any]],
    right_frames: list[dict[str, Any]],
    other_bridge_frames: list[dict[str, Any]] | None,
    state: list[float],
    left_indices: list[int],
    right_indices: list[int],
    updown_index: int,
    policy: str,
    strategy: str,
) -> tuple[list[dict[str, Any]], list[float]]:
    left_samples = interpolate_positions(left_frames, len(left_frames))
    right_samples = interpolate_positions(right_frames, len(right_frames))
    target_updown = transition_updown(
        policy,
        float(left_samples[-1][updown_index]),
        float(right_samples[-1][updown_index]),
        1,
        2,
        float(state[updown_index]),
        float(left_samples[-1][updown_index]),
    )
    output: list[dict[str, Any]] = []
    current = list(state)
    if strategy == "synchronized":
        count = max(len(left_samples), len(right_samples))
        left_samples = interpolate_positions(left_frames, count)
        right_samples = interpolate_positions(right_frames, count)
        for sample, (left_joints, right_joints) in enumerate(
            zip(left_samples, right_samples)
        ):
            combined = list(current)
            for index in left_indices:
                combined[index] = left_joints[index]
            for index in right_indices:
                combined[index] = right_joints[index]
            combined[updown_index] = transition_updown(
                policy,
                float(left_joints[updown_index]),
                float(right_joints[updown_index]),
                sample,
                count,
                float(state[updown_index]),
                target_updown,
            )
            output.append({
                "stage": "empty_transition_synchronized",
                "joints": combined,
                "left_attached": False,
                "right_attached": False,
            })
            current = combined
        return output, current
    if strategy == "whole_body_bridge":
        if not other_bridge_frames:
            raise ValueError("whole_body_bridge requires bridge frames")
        bridge = [
            [float(value) for value in frame["joints"]]
            for frame in other_bridge_frames
        ]
        if len(bridge[0]) != len(current):
            raise ValueError("whole-body bridge joint order differs")
        if max(
            abs(value - expected)
            for value, expected in zip(bridge[0], current)
        ) > 1e-6:
            raise ValueError("whole-body bridge start does not match current state")
        expected_goal = list(current)
        for index in left_indices:
            expected_goal[index] = left_samples[-1][index]
        for index in right_indices:
            expected_goal[index] = right_samples[-1][index]
        expected_goal[updown_index] = target_updown
        relevant = [updown_index, *left_indices, *right_indices]
        if max(
            abs(bridge[-1][index] - expected_goal[index])
            for index in relevant
        ) > 1e-6:
            raise ValueError(
                "whole-body bridge goal does not match paired precontact state"
            )
        output = [{
            "stage": "empty_transition_whole_body_bridge",
            "joints": joints,
            "left_attached": False,
            "right_attached": False,
        } for joints in bridge[1:]]
        return output, list(bridge[-1])
    if strategy not in {
        "reference", "reference_bridge", "reference_follow", "reference_reverse",
        "reference_suffix"
    }:
        raise ValueError(f"unknown transition strategy: {strategy}")

    reference_right = policy == "right"
    if policy in {"minimum", "maximum"}:
        left_target = float(left_samples[-1][updown_index])
        right_target = float(right_samples[-1][updown_index])
        reference_right = (
            right_target < left_target if policy == "minimum"
            else right_target > left_target
        )
    reference_side = "right" if reference_right else "left"
    other_side = "left" if reference_right else "right"
    reference_samples = right_samples if reference_right else left_samples
    other_samples = left_samples if reference_right else right_samples
    other_indices = left_indices if reference_right else right_indices
    if strategy in {"reference_bridge", "reference_reverse"}:
        if not other_bridge_frames:
            raise ValueError(f"{strategy} requires bridge frames")
        other_samples = interpolate_positions(
            other_bridge_frames, len(other_bridge_frames)
        )
    for joints in reference_samples:
        combined = list(joints)
        output.append({
            "stage": f"empty_transition_reference_{reference_side}",
            "joints": combined,
            "left_attached": False,
            "right_attached": False,
        })
        current = combined
    if strategy == "reference_suffix":
        nearest = min(
            range(len(other_samples)),
            key=lambda sample: sum(
                (other_samples[sample][index] - current[index]) ** 2
                for index in other_indices
            ),
        )
        other_path = other_samples[nearest:]
    elif strategy in {"reference_bridge", "reference_follow", "reference_reverse"}:
        other_path = other_samples
    else:
        other_path = other_samples[-1:]
    for joints in other_path:
        combined = list(current)
        for index in other_indices:
            combined[index] = joints[index]
        combined[updown_index] = target_updown
        output.append({
            "stage": f"empty_transition_{other_side}_post_lift",
            "joints": combined,
            "left_attached": False,
            "right_attached": False,
        })
        current = combined
    return output, current


def attachment(task: dict[str, Any]) -> dict[str, Any]:
    payload = task["payload"]
    return {
        "tool_link": str(payload["tool_link"]),
        "center_in_tool": [float(value) for value in payload["tool_to_box_center"]],
        "rotation_in_tool": [
            [float(value) for value in row]
            for row in payload["tool_to_box_rotation"]
        ],
        "size": [float(value) for value in payload["carried_box_size_tool"]],
    }


def task_base_pose(task: dict[str, Any]) -> list[float]:
    values = task.get("payload", {}).get("base_pose_map", [0.0, 0.0, 0.0])
    if len(values) != 3:
        raise ValueError(f"box {task.get('box_id')} has invalid base_pose_map")
    pose = [float(value) for value in values]
    if not all(math.isfinite(value) for value in pose):
        raise ValueError(f"box {task.get('box_id')} has non-finite base_pose_map")
    return pose


def paired_base_pose(
    left_task: dict[str, Any], right_task: dict[str, Any]
) -> list[float]:
    left = task_base_pose(left_task)
    right = task_base_pose(right_task)
    if max(abs(a - b) for a, b in zip(left, right)) > 1e-9:
        raise ValueError(
            f"paired boxes {left_task['box_id']}+{right_task['box_id']} "
            "use different base poses"
        )
    return left


def task_planning_summary(task: dict[str, Any]) -> dict[str, Any]:
    planning_search = task.get("planning_search", {})
    task_search = planning_search.get("task", {})
    transition_search = planning_search.get("transition", {})
    transition_core_ms = task.get("transition_core_ms")
    if transition_core_ms is None:
        transition_core_ms = sum(
            float(leg.get("total_ms", 0.0))
            for leg in task.get("transition_motion", {}).get("search_legs", [])
        )
    task_core_ms = float(task["payload"].get("total_ms", 0.0))
    payload_metrics = task["payload"].get("metrics", {})
    failed_segments = [
        str(item.get("diagnostic", ""))
        for item in task["payload"].get("segment_search", [])
        if not bool(item.get("success")) and item.get("diagnostic")
    ]
    return {
        "box_id": int(task["box_id"]),
        "row": int(task["row"]),
        "column": int(task["column"]),
        "side": str(task["side"]),
        "mode": str(task["mode"]),
        "source": str(task.get("planning_source", "golden_cache")),
        "selected_task_core_ms": task_core_ms,
        "accumulated_task_search_ms": float(
            task.get(
                "task_accumulated_search_ms",
                task_search.get("process_wall_ms", task_core_ms),
            )
        ),
        "selected_transition_core_ms": float(transition_core_ms),
        "accumulated_transition_search_ms": float(
            task.get(
                "transition_accumulated_search_ms",
                transition_search.get("process_wall_ms", transition_core_ms),
            )
        ),
        "ik_calls": int(payload_metrics.get("ik_calls", 0)),
        "ik_ms": float(payload_metrics.get("ik_ms", 0.0)),
        "cartesian_transfer_attempts": int(
            payload_metrics.get("cartesian_transfer_attempts", 0)
        ),
        "tcp_shortcut_attempts": int(
            payload_metrics.get("tcp_shortcut_attempts", 0)
        ),
        "tcp_shortcut_successes": int(
            payload_metrics.get("tcp_shortcut_successes", 0)
        ),
        "joint_rrt_fallbacks": int(
            payload_metrics.get("joint_rrt_fallbacks", 0)
        ),
        "joint_rrt_ms": float(payload_metrics.get("joint_rrt_ms", 0.0)),
        "failed_segment_diagnostics": failed_segments,
    }


def obstacles(
    removed: set[int],
    active: set[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = [
        {"id": "ground", "center": [0.0, 0.0, -0.055], "size": [6.0, 6.0, 0.1]},
    ]
    for panel in sequence.warehouse_panels():
        output.append({
            "id": str(panel["id"]),
            "center": [float(value) for value in panel["center"]],
            "size": [float(value) for value in panel["size"]],
        })
    inset_size = [
        sequence.BOX_DEPTH - 0.004,
        sequence.BOX_WIDTH - 0.004,
        sequence.BOX_HEIGHT - 0.004,
    ]
    for box_id, _, _, x, y, z in scan.box_specs(
        sequence.CONTACT_X,
        sequence.BOX_DEPTH,
        sequence.BOX_WIDTH,
        sequence.BOX_HEIGHT,
    ):
        if box_id in removed or box_id in active:
            continue
        output.append({
            "id": f"wall_box_{box_id}",
            "center": [float(x), float(y), float(z)],
            "size": inset_size,
        })
    return output


def combine_pair(
    left_task: dict[str, Any],
    right_task: dict[str, Any],
    current: list[float],
    removed: set[int],
    updown_policy: str,
    transition_strategy: str,
    transition_bridge: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[float]]:
    if left_task["side"] != "left" or right_task["side"] != "right":
        raise ValueError("dual group must contain a left task followed by a right task")
    if left_task["mode"] != right_task["mode"]:
        raise ValueError("mixed front/top pairs are not supported")
    left_names = [str(value) for value in left_task["payload"]["joint_names"]]
    right_names = [str(value) for value in right_task["payload"]["joint_names"]]
    if left_names != right_names:
        raise ValueError("paired task joint order differs")
    names = left_names
    left_indices = active_indices(names, "left")
    right_indices = active_indices(names, "right")
    updown_index = names.index("updown")
    left_order = stage_order(left_task)
    right_order = stage_order(right_task)
    if left_order != right_order:
        raise ValueError(
            f"stage order differs for {left_task['box_id']}+{right_task['box_id']}: "
            f"{left_order} != {right_order}"
        )

    frames: list[dict[str, Any]] = [{
        "stage": "group_start",
        "joints": list(current),
        "left_attached": False,
        "right_attached": False,
    }]
    state = list(current)
    for stage in left_order:
        left_source = stage_frames(left_task, stage)
        right_source = stage_frames(right_task, stage)
        if not left_source or not right_source:
            raise ValueError(f"paired stage {stage} is empty")
        if stage == "empty_transition":
            other_bridge_frames = (
                list(transition_bridge["frames"]) if transition_bridge else None
            )
            if transition_strategy == "reference_reverse":
                reference_right = updown_policy == "right"
                other_task = left_task if reference_right else right_task
                other_bridge_frames = list(reversed(other_task["payload"]["frames"]))
            transition_left_source = left_source
            transition_right_source = right_source
            if transition_strategy == "whole_body_bridge":
                transition_left_source = [
                    left_source[0], stage_frames(left_task, "precontact")[-1]
                ]
                transition_right_source = [
                    right_source[0], stage_frames(right_task, "precontact")[-1]
                ]
            transition, state = combine_empty_transition(
                transition_left_source,
                transition_right_source,
                other_bridge_frames,
                state,
                left_indices,
                right_indices,
                updown_index,
                updown_policy,
                transition_strategy,
            )
            frames.extend(transition)
            continue
        count = max(len(left_source), len(right_source))
        left_samples = interpolate_positions(left_source, count)
        right_samples = interpolate_positions(right_source, count)
        for sample, (left_joints, right_joints) in enumerate(
            zip(left_samples, right_samples)
        ):
            combined = list(state)
            for index in left_indices:
                combined[index] = left_joints[index]
            for index in right_indices:
                combined[index] = right_joints[index]
            left_updown = left_joints[updown_index]
            right_updown = right_joints[updown_index]
            if abs(left_updown - right_updown) > 1e-6:
                raise ValueError(
                    f"stage {stage} has incompatible updown values "
                    f"{left_updown} / {right_updown}"
                )
            combined[updown_index] = 0.5 * (left_updown + right_updown)
            attached = attached_for_stage(stage)
            frames.append({
                "stage": stage,
                "joints": combined,
                "left_attached": attached,
                "right_attached": attached,
            })
            state = combined

    left_id = int(left_task["box_id"])
    right_id = int(right_task["box_id"])
    operation = {
        "label": f"Box {left_id} + Box {right_id}",
        "kind": "dual",
        "row": int(left_task["row"]),
        "left_box_id": left_id,
        "right_box_id": right_id,
        "mode": str(left_task["mode"]),
        "base_pose_map": paired_base_pose(left_task, right_task),
        "updown_policy": updown_policy,
        "transition_strategy": transition_strategy,
        "removed_before": sorted(removed),
        "left_attachment": attachment(left_task),
        "right_attachment": attachment(right_task),
        "box_planning": [
            task_planning_summary(left_task), task_planning_summary(right_task)
        ],
        "obstacles": obstacles(removed, {left_id, right_id}),
        "allow_ground_model_base": True,
        "frames": frames,
    }
    if transition_bridge:
        operation["transition_bridge"] = {
            "total_ms": float(transition_bridge.get("total_ms", 0.0)),
            "transition_group": str(transition_bridge.get("transition_group", "")),
            "transition_strategy": str(
                transition_bridge.get("transition_strategy", "")
            ),
            "frame_count": len(transition_bridge.get("frames", [])),
        }
    return operation, state


def center_operation(
    task: dict[str, Any],
    current: list[float],
    removed: set[int],
    fallback_attachment: dict[str, dict[str, Any]],
    entry_bridge: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[float]]:
    names = [str(value) for value in task["payload"]["joint_names"]]
    frames: list[dict[str, Any]] = [{
        "stage": "group_start",
        "joints": list(current),
        "left_attached": False,
        "right_attached": False,
    }]
    transition_frames = list(task["transition_frames"])
    if entry_bridge:
        bridge_frames = list(entry_bridge["frames"])
        if not bridge_frames or not transition_frames:
            raise ValueError("center entry bridge and task transition must have frames")
        if any(
            abs(float(left) - float(right)) > 1e-6
            for left, right in zip(bridge_frames[0]["joints"], current)
        ):
            raise ValueError("center entry bridge does not start at the group state")
        if any(
            abs(float(left) - float(right)) > 1e-6
            for left, right in zip(
                bridge_frames[-1]["joints"], transition_frames[0]["joints"]
            )
        ):
            raise ValueError("center entry bridge does not reach the golden transition")
        for source in bridge_frames[1:]:
            frames.append({
                "stage": "center_entry_bridge",
                "joints": [float(value) for value in source["joints"]],
                "left_attached": False,
                "right_attached": False,
            })
        transition_frames = transition_frames[1:]
    for source in [*transition_frames, *task["payload"]["frames"]]:
        attached = bool(source.get("box_attached", False))
        frames.append({
            "stage": str(source["stage"]),
            "joints": [float(value) for value in source["joints"]],
            "left_attached": attached and task["side"] == "left",
            "right_attached": attached and task["side"] == "right",
        })
    box_id = int(task["box_id"])
    active_attachment = attachment(task)
    attachments = dict(fallback_attachment)
    attachments[str(task["side"])] = active_attachment
    operation = {
        "label": f"Box {box_id}",
        "kind": "single_center",
        "row": int(task["row"]),
        "left_box_id": box_id if task["side"] == "left" else 0,
        "right_box_id": box_id if task["side"] == "right" else 0,
        "mode": str(task["mode"]),
        "base_pose_map": task_base_pose(task),
        "removed_before": sorted(removed),
        "left_attachment": attachments["left"],
        "right_attachment": attachments["right"],
        "box_planning": [task_planning_summary(task)],
        "obstacles": obstacles(removed, {box_id}),
        "allow_ground_model_base": True,
        "frames": frames,
    }
    if entry_bridge:
        operation["transition_strategy"] = "center_entry_bridge"
        operation["center_entry_bridge"] = {
            "total_ms": float(entry_bridge.get("total_ms", 0.0)),
            "transition_group": str(entry_bridge.get("transition_group", "")),
            "transition_strategy": str(entry_bridge.get("transition_strategy", "")),
            "frame_count": len(entry_bridge.get("frames", [])),
        }
    return operation, list(frames[-1]["joints"])


def load_policy_map(value: str) -> dict[str, str]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("updown policy map must be a JSON object")
    return {str(key): str(item) for key, item in parsed.items()}


def load_edge_repair_map(value: str) -> dict[str, list[dict[str, Any]]]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("edge repair map must be a JSON object")
    repairs: dict[str, list[dict[str, Any]]] = {}
    for key, items in parsed.items():
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise ValueError(f"edge repairs for {key} must be a list of objects")
        repairs[str(key)] = items
    return repairs


def apply_edge_repairs(
    operation: dict[str, Any],
    key: str,
    joint_names: list[str],
    repair_map: dict[str, list[dict[str, Any]]],
) -> None:
    applied: list[dict[str, Any]] = []
    for repair in sorted(
        repair_map.get(key, []),
        key=lambda item: int(item["before_frame"]),
        reverse=True,
    ):
        before_index = int(repair["before_frame"])
        frames = operation["frames"]
        if before_index < 0 or before_index + 1 >= len(frames):
            raise ValueError(f"edge repair {key}:{before_index} is outside the operation")
        before = frames[before_index]
        after = frames[before_index + 1]
        if (
            bool(before["left_attached"]) != bool(after["left_attached"])
            or bool(before["right_attached"]) != bool(after["right_attached"])
        ):
            raise ValueError(f"edge repair {key}:{before_index} crosses an attachment change")
        joint = str(repair["joint"])
        joint_index = joint_names.index(joint)
        offset_deg = float(repair["offset_deg"])
        waypoint = copy.deepcopy(before)
        waypoint["stage"] = "edge_repair"
        waypoint["joints"] = [
            0.5 * (float(left) + float(right))
            for left, right in zip(before["joints"], after["joints"])
        ]
        waypoint["joints"][joint_index] += math.radians(offset_deg)
        frames.insert(before_index + 1, waypoint)
        applied.append({
            "before_frame": before_index,
            "joint": joint,
            "offset_deg": offset_deg,
        })
    if applied:
        operation["edge_repairs"] = list(reversed(applied))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box5-payload", type=Path)
    parser.add_argument("--box5-transition", type=Path)
    parser.add_argument(
        "--task-payload-map",
        default='',
        help='JSON map from box ID to a replacement full-task result JSON path',
    )
    parser.add_argument(
        "--task-transition-map",
        default='',
        help='JSON map from box ID to a replacement state-transition result JSON path',
    )
    parser.add_argument(
        "--ignored-source-transition-box-ids",
        default='',
        help='Comma-separated box IDs whose source transitions are replaced by group bridges',
    )
    parser.add_argument(
        "--default-updown-policy",
        choices=("left", "right", "minimum", "maximum", "average", "linear"),
        default="left",
    )
    parser.add_argument(
        "--updown-policy-map",
        default="",
        help='JSON map such as {"6+10":"right"}',
    )
    parser.add_argument(
        "--default-transition-strategy",
        choices=(
            "reference", "reference_bridge", "reference_follow", "reference_reverse",
            "reference_suffix", "synchronized", "whole_body_bridge"
        ),
        default="reference",
    )
    parser.add_argument(
        "--transition-strategy-map",
        default='',
        help='JSON map such as {"1+5":"synchronized"}',
    )
    parser.add_argument(
        "--transition-bridge-map",
        default='',
        help='JSON map from pair key to a state-transition result JSON path',
    )
    parser.add_argument(
        "--center-entry-bridge-map",
        default='',
        help='JSON map from center box ID to a state-transition result JSON path',
    )
    parser.add_argument(
        "--edge-repair-map",
        default='',
        help='JSON map from group key to local one-joint edge repair specifications',
    )
    args = parser.parse_args()

    cache = read_json(args.plan_cache)
    tasks = {int(task["box_id"]): copy.deepcopy(task) for task in cache["tasks"]}
    for task in tasks.values():
        task["planning_source"] = "golden_cache"
    if bool(args.box5_payload) != bool(args.box5_transition):
        parser.error("box5 payload and transition must be supplied together")
    if args.box5_payload:
        box5_payload = read_json(args.box5_payload)
        box5_transition = read_json(args.box5_transition)
        tasks[5]["updown"] = -0.25
        tasks[5]["payload"] = box5_payload
        tasks[5]["transition_frames"] = box5_transition["frames"]
        tasks[5]["planning_source"] = "dual_height_replan"
        tasks[5]["task_accumulated_search_ms"] = float(
            box5_payload.get("total_ms", 0.0)
        )
        tasks[5]["transition_core_ms"] = float(
            box5_transition.get("total_ms", 0.0)
        )
        tasks[5]["transition_accumulated_search_ms"] = float(
            box5_transition.get("total_ms", 0.0)
        )
    task_payload_paths = load_policy_map(args.task_payload_map)
    for key, path in task_payload_paths.items():
        box_id = int(key)
        if box_id not in tasks:
            raise ValueError(f"task payload override has invalid box ID: {box_id}")
        payload = read_json(Path(path))
        task = tasks[box_id]
        if (
            not bool(payload.get("success"))
            or int(payload.get("target_box_id", 0)) != box_id
            or str(payload.get("side")) != str(task["side"])
            or str(payload.get("grasp_mode")) != str(task["mode"])
        ):
            raise ValueError(f"task payload override does not match box {box_id}")
        task["payload"] = payload
        task["planning_source"] = "dual_task_replan"
        task["task_accumulated_search_ms"] = float(payload.get("total_ms", 0.0))
    task_transition_paths = load_policy_map(args.task_transition_map)
    for key, path in task_transition_paths.items():
        box_id = int(key)
        if box_id not in tasks:
            raise ValueError(f"task transition override has invalid box ID: {box_id}")
        transition = read_json(Path(path))
        if not bool(transition.get("success")) or not transition.get("frames"):
            raise ValueError(f"task transition override failed for box {box_id}")
        tasks[box_id]["transition_frames"] = list(transition["frames"])
        tasks[box_id]["transition_core_ms"] = float(transition.get("total_ms", 0.0))
        tasks[box_id]["transition_accumulated_search_ms"] = float(
            transition.get("total_ms", 0.0)
        )
    ignored_source_transitions = {
        int(value.strip())
        for value in args.ignored_source_transition_box_ids.split(",")
        if value.strip()
    }
    for box_id in ignored_source_transitions:
        if box_id not in tasks:
            raise ValueError(f"ignored source transition has invalid box ID: {box_id}")
        if str(box_id) in task_transition_paths:
            raise ValueError(
                f"box {box_id} cannot ignore and replace its source transition"
            )
        tasks[box_id]["transition_core_ms"] = 0.0
        tasks[box_id]["transition_accumulated_search_ms"] = 0.0

    names = [str(value) for value in tasks[1]["payload"]["joint_names"]]
    current = sequence.folded_start_joints(cache["station_config"])
    policies = load_policy_map(args.updown_policy_map)
    transition_strategies = load_policy_map(args.transition_strategy_map)
    transition_bridge_paths = load_policy_map(args.transition_bridge_map)
    transition_bridges = {
        key: read_json(Path(path))
        for key, path in transition_bridge_paths.items()
    }
    center_entry_bridge_paths = load_policy_map(args.center_entry_bridge_map)
    center_entry_bridges = {
        key: read_json(Path(path))
        for key, path in center_entry_bridge_paths.items()
    }
    edge_repairs = load_edge_repair_map(args.edge_repair_map)
    removed: set[int] = set()
    operations: list[dict[str, Any]] = []
    fallback_attachment = {
        "left": attachment(tasks[1]),
        "right": attachment(tasks[5]),
    }

    for left_id, right_id in GROUPS:
        if right_id is None:
            operation, current = center_operation(
                tasks[left_id], current, removed, fallback_attachment,
                center_entry_bridges.get(str(left_id)),
            )
            removed.add(left_id)
        else:
            key = f"{left_id}+{right_id}"
            operation, current = combine_pair(
                tasks[left_id],
                tasks[right_id],
                current,
                removed,
                policies.get(key, args.default_updown_policy),
                transition_strategies.get(key, args.default_transition_strategy),
                transition_bridges.get(key),
            )
            fallback_attachment = {
                "left": attachment(tasks[left_id]),
                "right": attachment(tasks[right_id]),
            }
            removed.update({left_id, right_id})
        group_key = str(left_id) if right_id is None else f"{left_id}+{right_id}"
        apply_edge_repairs(operation, group_key, names, edge_repairs)
        operations.append(operation)

    if removed != set(range(1, 26)):
        raise RuntimeError(f"dual sequence did not consume all boxes: {sorted(removed)}")
    output = {
        "schema": SCHEMA,
        "source_plan_cache": str(args.plan_cache.resolve()),
        "source_plan_cache_schema": cache["schema"],
        "joint_names": names,
        "group_order": [
            [left, right] if right is not None else [left]
            for left, right in GROUPS
        ],
        "planning_time_definition": {
            "selected_task_core_ms": "planner wall time for the selected task path",
            "accumulated_task_search_ms": "task search wall time including retries",
            "selected_transition_core_ms": "selected source transition planning time",
            "accumulated_transition_search_ms": (
                "source transition search wall time including retries"
            ),
            "dual_bridge_total_ms": "additional dual-composition bridge search",
        },
        "box_planning": [
            task_planning_summary(tasks[box_id]) for box_id in range(1, 26)
        ],
        "operations": operations,
    }
    write_json(args.output, output)
    print(f"DUAL_REPLAY groups={len(operations)} frames={sum(len(op['frames']) for op in operations)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
