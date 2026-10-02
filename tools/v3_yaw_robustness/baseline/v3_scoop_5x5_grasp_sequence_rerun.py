#!/usr/bin/env python3
"""Plan and play the independent V3 scoop 5x5 task in one Rerun timeline."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import v3_5x5_grasp_sequence_rerun as sequence


STATION_CONFIG_NAME = "v3_scoop_5x5_station.json"
STATION_SCHEMA = "alfa.v3_scoop_5x5_station.v11"
RESULT_SCHEMA = "alfa.v3_scoop_5x5_station_result.v11"
END_EFFECTOR = "scoop"
PLAN_CACHE_SCHEMA = "alfa.v3_scoop_5x5_plan_cache.v1"
VEHICLE_FRONT_X_IN_BASE_M = 0.500000002779484

# V3.1.1 candidate order is independent from the historical V3.0.9 profile.
VERIFIED_SELECTION = {
    1: ("left", -0.25),
    2: ("left", -0.50),
    3: ("right", -0.50),
    4: ("right", -0.50),
    5: ("right", 0.0),
    6: ("left", 0.0),
    7: ("left", 0.0),
    8: ("right", 0.0),
    9: ("right", 0.0),
    10: ("right", 0.0),
    11: ("left", 0.0),
    12: ("left", 0.0),
    13: ("right", -0.50),
    14: ("right", 0.0),
    15: ("right", 0.0),
    16: ("left", -0.25),
    17: ("left", -0.25),
    18: ("right", -0.25),
    19: ("right", -0.25),
    20: ("right", -0.25),
    21: ("left", -0.50),
    22: ("left", -0.50),
    23: ("right", -0.62),
    24: ("right", -0.50),
    25: ("right", -0.50),
}


def default_recording_path() -> Path:
    return (
        sequence.workspace_root()
        / "data/ik_benchmark/v3_scoop_5x5"
        / "2026-09-21-box7-front-box20-fast"
        / "v3-scoop-x075-sequence.rrd"
    )


def plan_cache_path(recording: Path) -> Path:
    return recording.with_name(recording.stem + "-plan-cache.json")


def load_plan_cache(path: Path, config: dict, limit_boxes: int) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if (
        payload.get("schema") != PLAN_CACHE_SCHEMA
        or payload.get("station_config") != config
    ):
        raise ValueError(f"incompatible scoop plan cache: {path}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) > limit_boxes:
        raise ValueError(f"invalid scoop plan cache: {path}")
    return tasks


def write_plan_cache(path: Path, config: dict, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "schema": PLAN_CACHE_SCHEMA,
                "station_config": config,
                "completed_boxes": len(tasks),
                "tasks": tasks,
            },
            stream,
            ensure_ascii=False,
        )
        stream.write("\n")
    temporary.replace(path)


def analyze_task_trajectory(task: dict, robot: object) -> dict:
    payload = task["payload"]
    names = [str(name) for name in payload["joint_names"]]
    frames = payload["frames"]
    tool_link = str(payload["tool_link"])
    stage_positions: dict[str, list[list[float]]] = {}
    stage_lengths: dict[str, float] = {}
    previous_transform = None
    tcp_length = 0.0
    orientation_travel = 0.0
    stage_frames: dict[str, list[dict]] = {}
    for frame in frames:
        joints = dict(zip(names, map(float, frame["joints"])))
        transform = robot.fk(joints)[tool_link]
        stage = str(frame["stage"])
        stage_frames.setdefault(stage, []).append(frame)
        stage_positions.setdefault(stage, []).append(transform[:3, 3].tolist())
        if previous_transform is not None:
            distance = float(np.linalg.norm(
                transform[:3, 3] - previous_transform[:3, 3]
            ))
            tcp_length += distance
            stage_lengths[stage] = stage_lengths.get(stage, 0.0) + distance
            relative = previous_transform[:3, :3].T @ transform[:3, :3]
            cosine = (float(np.trace(relative)) - 1.0) / 2.0
            orientation_travel += math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        previous_transform = transform

    reversals_by_joint: dict[str, int] = {}
    ranges_by_joint: dict[str, float] = {}
    large_steps_by_joint: dict[str, int] = {}
    for index in range(1, 8):
        name = f"{task['side']}_joint{index}"
        column = names.index(name)
        angles = [float(frame["joints"][column]) for frame in frames]
        deltas = [
            math.degrees(math.atan2(math.sin(right - left), math.cos(right - left)))
            for left, right in zip(angles, angles[1:])
        ]
        active = [delta for delta in deltas if abs(delta) >= 0.1]
        reversals_by_joint[name] = sum(
            left * right < 0.0 for left, right in zip(active, active[1:])
        )
        large_steps_by_joint[name] = sum(abs(delta) > 90.0 for delta in deltas)
        cumulative = np.cumsum([0.0, *deltas])
        ranges_by_joint[name] = float(max(cumulative) - min(cumulative))

    shortcut_deviation = {}
    for stage, current_frames in stage_frames.items():
        if len(current_frames) < 2:
            continue
        start = list(map(float, current_frames[0]["joints"]))
        goal = list(map(float, current_frames[-1]["joints"]))
        maximum_position = 0.0
        maximum_orientation = 0.0
        for frame_index, frame in enumerate(current_frames):
            ratio = frame_index / (len(current_frames) - 1)
            nominal = {}
            actual = dict(zip(names, map(float, frame["joints"])))
            for index, name in enumerate(names):
                delta = goal[index] - start[index]
                if name != "updown":
                    delta = math.atan2(math.sin(delta), math.cos(delta))
                nominal[name] = start[index] + delta * ratio
            actual_transform = robot.fk(actual)[tool_link]
            nominal_transform = robot.fk(nominal)[tool_link]
            maximum_position = max(maximum_position, float(np.linalg.norm(
                actual_transform[:3, 3] - nominal_transform[:3, 3]
            )))
            relative = nominal_transform[:3, :3].T @ actual_transform[:3, :3]
            cosine = (float(np.trace(relative)) - 1.0) / 2.0
            maximum_orientation = max(
                maximum_orientation,
                math.degrees(math.acos(max(-1.0, min(1.0, cosine)))),
            )
        shortcut_deviation[stage] = {
            "max_tcp_position_m": maximum_position,
            "max_tcp_orientation_deg": maximum_orientation,
        }

    task_search = task.get("planning_search", {}).get("task", {})
    transition_search = task.get("planning_search", {}).get("transition", {})
    sampled_paths = {}
    for stage, positions in stage_positions.items():
        stride = max(1, math.ceil(len(positions) / 120))
        sampled = positions[::stride]
        if sampled[-1] != positions[-1]:
            sampled.append(positions[-1])
        sampled_paths[stage] = sampled
    return {
        "tcp_path_length_m": tcp_length,
        "tcp_orientation_travel_deg": orientation_travel,
        "tcp_stage_lengths_m": stage_lengths,
        "tcp_stage_positions": sampled_paths,
        "shortcut_tcp_deviation": shortcut_deviation,
        "max_shortcut_tcp_position_deviation_m": max(
            (item["max_tcp_position_m"] for item in shortcut_deviation.values()), default=0.0
        ),
        "max_shortcut_tcp_orientation_deviation_deg": max(
            (item["max_tcp_orientation_deg"] for item in shortcut_deviation.values()), default=0.0
        ),
        "joint_reversals_by_axis": reversals_by_joint,
        "joint_flip_events_by_axis": large_steps_by_joint,
        "joint_range_deg_by_axis": ranges_by_joint,
        "total_joint_reversals": sum(reversals_by_joint.values()),
        "joint_flip_events": sum(large_steps_by_joint.values()),
        "search_wall_ms": float(task_search.get("process_wall_ms", 0.0)) +
        float(transition_search.get("process_wall_ms", 0.0)),
        "selected_task_search_ms": float(task.get("planning_search", {}).get(
            "selected_task", {}).get("process_wall_ms", 0.0)),
    }


def write_metrics_csv(summary: dict, summary_path: Path) -> Path:
    csv_path = summary_path.with_name(summary_path.stem.replace("-summary", "") + "-metrics.csv")
    temporary = csv_path.with_suffix(".csv.tmp")
    fields = [
        "box_id", "side", "mode", "search_wall_s", "task_attempts",
        "transition_attempts", "selected_task_search_s", "shortcut_repair_s",
        "selected_planner_s",
        "repair_successes", "joint_rrt_fallbacks", "tcp_path_m",
        "max_shortcut_tcp_position_deviation_m", "max_shortcut_tcp_orientation_deviation_deg",
        "joint_travel_deg", "joint_reversals", "joint_flip_events",
        "max_joint_range_deg", "max_carried_tilt_deg", "task_frames", "transition_frames",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for box in summary["boxes"]:
            analysis = box.get("trajectory_analysis", {})
            search = box.get("planning_search", {})
            task_search = search.get("task", {})
            transition_search = search.get("transition", {})
            writer.writerow({
                "box_id": box["box_id"], "side": box["side"], "mode": box["mode"],
                "search_wall_s": round(float(analysis.get("search_wall_ms", 0.0)) / 1000, 3),
                "task_attempts": task_search.get("attempts", 0),
                "transition_attempts": transition_search.get("attempts", 0),
                "selected_task_search_s": round(
                    float(analysis.get("selected_task_search_ms", 0.0)) / 1000, 3),
                "selected_planner_s": round(float(search.get(
                    "selected_task", {}).get("planner_wall_ms", 0.0)) / 1000, 3),
                "shortcut_repair_s": round(
                    float(task_search.get("shortcut_repair_rrt_ms", 0.0)) / 1000, 3),
                "repair_successes": task_search.get("shortcut_repair_rrt_successes", 0),
                "joint_rrt_fallbacks": task_search.get("joint_rrt_fallbacks", 0),
                "tcp_path_m": round(float(analysis.get("tcp_path_length_m", 0.0)), 3),
                "max_shortcut_tcp_position_deviation_m": round(float(analysis.get(
                    "max_shortcut_tcp_position_deviation_m", 0.0)), 4),
                "max_shortcut_tcp_orientation_deviation_deg": round(float(analysis.get(
                    "max_shortcut_tcp_orientation_deviation_deg", 0.0)), 2),
                "joint_travel_deg": round(float(box["motion_selection"]["total_joint_travel_deg"]), 2),
                "joint_reversals": analysis.get("total_joint_reversals", 0),
                "joint_flip_events": analysis.get("joint_flip_events", 0),
                "max_joint_range_deg": round(max(analysis.get(
                    "joint_range_deg_by_axis", {"none": 0.0}).values()), 2),
                "max_carried_tilt_deg": round(float(box["carried_box_orientation"]["max_tilt_deg"]), 2),
                "task_frames": box["task_frames"],
                "transition_frames": box["validated_transition_frames"],
            })
    temporary.replace(csv_path)
    return csv_path


def planning_profile(
    config: dict,
) -> tuple[
    set[int], set[int], dict[str, object], dict[int, dict[str, float]], list[int]
]:
    planning = config.get("planning", {})
    required = {
        "top_down_box_ids",
        "left_arm_columns",
        "right_arm_columns",
        "box_order",
        "low_transfer_tcp_box_ids",
        "top_suction_x_offset",
        "front_retreat_distance_m",
        "top_retreat_distance_m",
        "center_front_suction_y_offset",
        "center_front_suction_z_offset",
        "bottom_front_suction_z_offset",
        "natural_seed_swivel_sampling",
        "natural_seed_swivel_step_deg",
        "natural_seed_swivel_neighbor_steps",
        "natural_joint_acceleration_weight",
        "natural_place_return_weight",
        "natural_cartesian_replay_step_deg",
        "natural_rrt_shortcut_enabled",
        "natural_rrt_shortcut_max_nodes",
        "cartesian_transfer_search_enabled",
        "top_loaded_cartesian_transfer_search_enabled",
        "cartesian_transfer_translation_step",
        "cartesian_transfer_rotation_step_deg",
        "cartesian_transfer_max_search_attempts",
        "shortcut_repair_rrt_enabled",
        "shortcut_repair_rrt_budget_ms",
        "shortcut_repair_rrt_max_samples",
        "shortcut_repair_max_joint_offset_deg",
        "contact_tool_roll_deg",
        "upper_front_success_trials",
        "upper_front_max_success_trials",
        "conveyor_success_trials",
        "conveyor_max_success_trials",
        "ignore_opposite_arm",
        "continuous_sequence",
        "continuous_seed_previous",
        "maximum_carried_box_tilt_deg",
        "place_updown_enabled",
        "place_updown_m",
        "top_loaded_transfer_direct_only",
        "natural_max_proximal_step_deg",
        "natural_max_wrist_step_deg",
        "box_overrides",
    }
    if set(planning) != required:
        raise ValueError(f"invalid scoop planning profile: {sorted(planning)}")
    top_down_box_ids = {int(value) for value in planning["top_down_box_ids"]}
    if not top_down_box_ids or any(value < 1 or value > 25 for value in top_down_box_ids):
        raise ValueError("top_down_box_ids must contain box ids in [1, 25]")
    left_arm_columns = {int(value) for value in planning["left_arm_columns"]}
    right_arm_columns = {int(value) for value in planning["right_arm_columns"]}
    if (
        left_arm_columns != {1, 2}
        or right_arm_columns != {3, 4, 5}
        or left_arm_columns & right_arm_columns
    ):
        raise ValueError("arm columns must be left={1,2}, right={3,4,5}")
    right_arm_box_ids = {
        box_id for box_id in range(1, 26)
        if ((box_id - 1) % 5) + 1 in right_arm_columns
    }
    box_order = [int(value) for value in planning["box_order"]]
    if sorted(box_order) != list(range(1, 26)):
        raise ValueError("box_order must be a permutation of ids 1..25")
    low_transfer_tcp_box_ids = {
        int(value) for value in planning["low_transfer_tcp_box_ids"]
    }
    if low_transfer_tcp_box_ids != {6}:
        raise ValueError("low_transfer_tcp_box_ids must be exactly {6}")
    minimum_trials = int(planning["upper_front_success_trials"])
    maximum_trials = int(planning["upper_front_max_success_trials"])
    if minimum_trials < 1 or maximum_trials < minimum_trials:
        raise ValueError("upper-front success trials must satisfy 1 <= minimum <= maximum")
    minimum_trials = int(planning["conveyor_success_trials"])
    maximum_trials = int(planning["conveyor_max_success_trials"])
    if minimum_trials < 1 or maximum_trials < minimum_trials:
        raise ValueError("conveyor success trials must satisfy 1 <= minimum <= maximum")
    if planning["place_updown_enabled"] is not True or not (
        -1.0 <= float(planning["place_updown_m"]) <= 0.0
    ):
        raise ValueError("rear conveyor requires place_updown_m in [-1.0, 0.0]")
    if any(
        float(planning[name]) <= 0.0
        for name in ("front_retreat_distance_m", "top_retreat_distance_m")
    ):
        raise ValueError("retreat distances must be positive")
    if planning["continuous_sequence"] is not True:
        raise ValueError("scoop sequence must continue directly after release")
    if planning["cartesian_transfer_search_enabled"] is not True:
        raise ValueError("scoop sequence requires Cartesian transfer search")
    if planning["top_loaded_cartesian_transfer_search_enabled"] is not False:
        raise ValueError("top-loaded transfer must use the bounded joint-space planner")
    if not 0.0 < float(planning["cartesian_transfer_translation_step"]) <= 0.05:
        raise ValueError("Cartesian transfer translation step must be in (0, 0.05] m")
    if not 0.0 < float(planning["cartesian_transfer_rotation_step_deg"]) <= 5.0:
        raise ValueError("Cartesian transfer rotation step must be in (0, 5] deg")
    if int(planning["cartesian_transfer_max_search_attempts"]) < 1:
        raise ValueError("Cartesian transfer search requires a positive attempt budget")
    if planning["shortcut_repair_rrt_enabled"] is not True:
        raise ValueError("scoop sequence requires shortcut repair RRT fallback")
    if not 250.0 <= float(planning["shortcut_repair_rrt_budget_ms"]) <= 5000.0:
        raise ValueError("shortcut repair RRT budget must be in [250, 5000] ms")
    if not 1 <= int(planning["shortcut_repair_rrt_max_samples"]) <= 1000:
        raise ValueError("shortcut repair RRT sample limit must be in [1, 1000]")
    if not 1.0 <= float(planning["shortcut_repair_max_joint_offset_deg"]) <= 60.0:
        raise ValueError("shortcut repair joint offset must be in [1, 60] deg")
    if not math.isclose(float(planning["contact_tool_roll_deg"]), 90.0, abs_tol=1e-9):
        raise ValueError("the scoop contact roll must be 90 degrees")
    if not 90.9 <= float(planning["maximum_carried_box_tilt_deg"]) <= 95.0:
        raise ValueError("loaded tilt limit must allow unloading but prevent inversion")
    overrides = {
        name: planning[name]
        for name in required
        if name not in (
            "top_down_box_ids", "left_arm_columns", "right_arm_columns",
            "box_order", "box_overrides", "low_transfer_tcp_box_ids",
            "top_loaded_cartesian_transfer_search_enabled"
        )
    }
    box_overrides = {
        int(box_id): {
            name: float(value) if isinstance(value, (int, float)) else value
            for name, value in values.items()
        }
        for box_id, values in planning["box_overrides"].items()
    }
    for box_id in top_down_box_ids:
        box_overrides.setdefault(box_id, {}).setdefault(
            "cartesian_transfer_search_enabled", False
        )
    return top_down_box_ids, right_arm_box_ids, overrides, box_overrides, box_order


def rear_conveyor_contract(config: dict) -> dict:
    conveyor = config.get("rear_conveyor", {})
    if set(conveyor) != {
        "modeled", "side_semantics", "named_pose_file", "named_pose"
    }:
        raise ValueError("invalid rear_conveyor contract")
    if conveyor["modeled"] is not False:
        raise ValueError("rear conveyor must remain an unmodeled handoff boundary")
    if conveyor["side_semantics"] != "left_is_positive_y":
        raise ValueError("V3.1.1 requires left_is_positive_y")
    if (
        conveyor["named_pose_file"] != "named_poses_suction.yaml"
        or conveyor["named_pose"] != "unloading"
    ):
        raise ValueError("rear conveyor must use the authoritative unloading pose")
    return conveyor


def unloading_joint_degrees(config: dict, side: str) -> list[float]:
    rear_conveyor_contract(config)
    poses = config.get("unloading_joint_degrees", {})
    if set(poses) != {"left", "right"}:
        raise ValueError("unloading pose must define both arms")
    values = [float(value) for value in poses[side]]
    if len(values) != 7 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid {side} unloading pose")
    return values


def unloading_tcp_pose(
    config: dict,
    _box_id: int,
    side: str,
    _mode: str,
    _updown: float,
    _box_center: tuple[float, float, float],
) -> list[float]:
    rear_conveyor_contract(config)
    poses = config.get("unloading_tcp_pose", {})
    if set(poses) != {"left", "right"}:
        raise ValueError("unloading TCP pose must define both arms")
    values = [float(value) for value in poses[side]]
    if len(values) != 7 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid {side} unloading TCP pose")
    quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
    if not math.isclose(quaternion_norm, 1.0, abs_tol=1e-6):
        raise ValueError(f"invalid {side} unloading TCP quaternion")
    return values


def stage_motion_metrics(task: dict, stage: str, reversal_threshold_deg: float) -> dict:
    payload = task["payload"]
    names = [str(name) for name in payload["joint_names"]]
    frames = list(payload["frames"])
    stage_indices = [
        index for index, frame in enumerate(frames)
        if frame["stage"] == stage
    ]
    if not stage_indices:
        raise ValueError(f"box {task['box_id']} has no {stage} frames")
    first = stage_indices[0]
    stage_frames = frames[max(0, first - 1):stage_indices[-1] + 1]
    joint_indices = [names.index(f"{task['side']}_joint{index}") for index in range(1, 8)]
    total_variation = 0.0
    total_abs_net = 0.0
    reversals = 0
    maximum_step = 0.0
    maximum_delta_step = 0.0
    for joint_index in joint_indices:
        values = [float(frame["joints"][joint_index]) for frame in stage_frames]
        deltas = [
            math.degrees(math.atan2(math.sin(right - left), math.cos(right - left)))
            for left, right in zip(values, values[1:])
        ]
        if not deltas:
            continue
        active = [delta for delta in deltas if abs(delta) >= reversal_threshold_deg]
        reversals += sum(left * right < 0.0 for left, right in zip(active, active[1:]))
        total_variation += sum(abs(delta) for delta in deltas)
        total_abs_net += abs(sum(deltas))
        maximum_step = max(maximum_step, max(abs(delta) for delta in deltas))
        if len(deltas) > 1:
            maximum_delta_step = max(
                maximum_delta_step,
                max(abs(right - left) for left, right in zip(deltas, deltas[1:])),
            )
    return {
        "frame_count": len(stage_frames),
        "total_joint_travel_deg": total_variation,
        "absolute_net_joint_motion_deg": total_abs_net,
        "excess_joint_travel_deg": total_variation - total_abs_net,
        "direction_reversals": reversals,
        "max_joint_step_deg": maximum_step,
        "max_joint_delta_step_deg": maximum_delta_step,
    }


def retreat_motion_metrics(task: dict, reversal_threshold_deg: float) -> dict:
    return stage_motion_metrics(task, "cartesian_retreat", reversal_threshold_deg)


def loaded_transfer_motion_metrics(task: dict, reversal_threshold_deg: float) -> dict:
    return stage_motion_metrics(task, "rrt_to_place", reversal_threshold_deg)


def validate_retreat_motion(tasks: list[dict], acceptance: dict) -> dict:
    required = {
        "direction_change_threshold_deg",
        "max_retreat_joint_step_deg",
        "max_retreat_joint_delta_step_deg",
        "max_retreat_direction_reversals",
        "max_retreat_excess_travel_deg",
        "max_upper_front_total_joint_travel_deg",
        "max_upper_front_stage_excess_travel_deg",
        "max_upper_front_direction_reversals",
        "max_upper_front_joint_range_deg",
        "max_upper_front_joint_step_deg",
        "max_conveyor_total_joint_travel_deg",
        "max_conveyor_stage_excess_travel_deg",
        "max_conveyor_direction_reversals",
        "max_conveyor_joint_range_deg",
        "max_conveyor_joint_step_deg",
        "max_conveyor_updown_travel_m",
        "max_conveyor_updown_step_m",
        "max_conveyor_named_pose_error_deg",
        "low_transfer_box_ids",
        "max_low_transfer_frames",
        "max_low_transfer_joint_travel_deg",
        "max_low_transfer_excess_travel_deg",
        "max_low_transfer_direction_reversals",
        "max_low_transfer_joint_step_deg",
        "max_transition_frames",
        "max_transition_total_joint_travel_deg",
        "max_transition_excess_joint_travel_deg",
        "max_transition_direction_reversals",
        "max_transition_joint_range_deg",
        "max_frozen_arm_transition_travel_deg",
        "max_first_transition_joint_winding_excess_deg",
        "max_first_transition_tcp_path_m",
        "max_first_transition_tcp_line_deviation_m",
        "max_first_transition_tcp_orientation_travel_deg",
        "max_top_loaded_transfer_frames",
        "max_top_complete_total_joint_travel_deg",
        "max_top_complete_direction_reversals",
        "max_top_loaded_joint_travel_deg",
        "max_top_loaded_excess_travel_deg",
        "max_top_loaded_direction_reversals",
        "max_top_loaded_joint_step_deg",
        "max_contact_face_normal_error_deg",
        "max_scoop_horizontal_error_deg",
    }
    optional = {"allow_simultaneous_transition"}
    if set(acceptance) - optional != required:
        raise ValueError(f"invalid motion acceptance contract: {sorted(acceptance)}")
    threshold = float(acceptance["direction_change_threshold_deg"])
    metrics = [retreat_motion_metrics(task, threshold) for task in tasks]
    failures = []
    for task, motion in zip(tasks, metrics):
        task["retreat_motion"] = motion
        contact = task["payload"].get("contact_tool_metrics")
        if not isinstance(contact, dict):
            failures.append(f"box {task['box_id']} missing contact-tool metrics")
        else:
            face_error = float(contact["face_normal_error_deg"])
            horizontal_error = float(contact["scoop_horizontal_error_deg"])
            task["contact_tool_metrics"] = contact
            if face_error > float(acceptance["max_contact_face_normal_error_deg"]) + 1e-6:
                failures.append(
                    f"box {task['box_id']} face-normal-error={face_error:.3f}deg"
                )
            if horizontal_error > float(acceptance["max_scoop_horizontal_error_deg"]) + 1e-6:
                failures.append(
                    f"box {task['box_id']} scoop-horizontal-error={horizontal_error:.3f}deg"
                )
        if motion["max_joint_step_deg"] > float(acceptance["max_retreat_joint_step_deg"]) + 1e-6:
            failures.append(
                f"box {task['box_id']} joint-step={motion['max_joint_step_deg']:.3f}deg"
            )
        if motion["max_joint_delta_step_deg"] > (
            float(acceptance["max_retreat_joint_delta_step_deg"]) + 1e-6
        ):
            failures.append(
                f"box {task['box_id']} delta-step="
                f"{motion['max_joint_delta_step_deg']:.3f}deg"
            )
        if motion["direction_reversals"] > int(acceptance["max_retreat_direction_reversals"]):
            failures.append(
                f"box {task['box_id']} reversals={motion['direction_reversals']}"
            )
        if motion["excess_joint_travel_deg"] > float(acceptance["max_retreat_excess_travel_deg"]):
            failures.append(
                f"box {task['box_id']} excess="
                f"{motion['excess_joint_travel_deg']:.3f}deg"
            )
    top_tasks = [task for task in tasks if task.get("mode") == "top_suction"]
    loaded_metrics = [loaded_transfer_motion_metrics(task, threshold) for task in top_tasks]
    for task, motion in zip(top_tasks, loaded_metrics):
        task["loaded_transfer_motion"] = motion
        complete = task["motion_selection"]
        if complete["total_joint_travel_deg"] > float(
            acceptance["max_top_complete_total_joint_travel_deg"]
        ) + 1e-6:
            failures.append(
                f"box {task['box_id']} top-complete-travel="
                f"{complete['total_joint_travel_deg']:.3f}deg"
            )
        if complete["direction_reversals"] > int(
            acceptance["max_top_complete_direction_reversals"]
        ):
            failures.append(
                f"box {task['box_id']} top-complete-reversals="
                f"{complete['direction_reversals']}"
            )
        if motion["frame_count"] > int(acceptance["max_top_loaded_transfer_frames"]):
            failures.append(f"box {task['box_id']} loaded-frames={motion['frame_count']}")
        if motion["total_joint_travel_deg"] > float(
            acceptance["max_top_loaded_joint_travel_deg"]
        ):
            failures.append(
                f"box {task['box_id']} loaded-travel="
                f"{motion['total_joint_travel_deg']:.3f}deg"
            )
        if motion["excess_joint_travel_deg"] > float(
            acceptance["max_top_loaded_excess_travel_deg"]
        ):
            failures.append(
                f"box {task['box_id']} loaded-excess="
                f"{motion['excess_joint_travel_deg']:.3f}deg"
            )
        if motion["direction_reversals"] > int(
            acceptance["max_top_loaded_direction_reversals"]
        ):
            failures.append(
                f"box {task['box_id']} loaded-reversals="
                f"{motion['direction_reversals']}"
            )
        if motion["max_joint_step_deg"] > float(
            acceptance["max_top_loaded_joint_step_deg"]
        ) + 1e-6:
            failures.append(
                f"box {task['box_id']} loaded-joint-step="
                f"{motion['max_joint_step_deg']:.3f}deg"
            )
    upper_front_tasks = [
        task for task in tasks
        if int(task.get("box_id", 0)) <= 15 and task.get("mode") == "front"
    ]
    upper_front_metrics = [task["motion_selection"] for task in upper_front_tasks]
    for task, motion in zip(upper_front_tasks, upper_front_metrics):
        if motion["total_joint_travel_deg"] > float(
            acceptance["max_upper_front_total_joint_travel_deg"]
        ) + 1e-6:
            failures.append(
                f"box {task['box_id']} upper-front-travel="
                f"{motion['total_joint_travel_deg']:.3f}deg"
            )
        if motion["stage_excess_joint_travel_deg"] > float(
            acceptance["max_upper_front_stage_excess_travel_deg"]
        ) + 1e-6:
            failures.append(
                f"box {task['box_id']} upper-front-excess="
                f"{motion['stage_excess_joint_travel_deg']:.3f}deg"
            )
        if motion["direction_reversals"] > int(
            acceptance["max_upper_front_direction_reversals"]
        ):
            failures.append(
                f"box {task['box_id']} upper-front-reversals="
                f"{motion['direction_reversals']}"
            )
        if motion["max_joint_range_deg"] > float(
            acceptance["max_upper_front_joint_range_deg"]
        ) + 1e-6:
            failures.append(
                f"box {task['box_id']} upper-front-range="
                f"{motion['max_joint_range_deg']:.3f}deg"
            )
        if motion["max_joint_step_deg"] > float(
            acceptance["max_upper_front_joint_step_deg"]
        ) + 1e-6:
            failures.append(
                f"box {task['box_id']} upper-front-joint-step="
                f"{motion['max_joint_step_deg']:.3f}deg"
            )
    conveyor_tasks = [
        task for task in tasks
        if "placement_box_center_target" in task and task.get("mode") != "top_suction"
    ]
    conveyor_metrics = [task["motion_selection"] for task in conveyor_tasks]
    for task, motion in zip(conveyor_tasks, conveyor_metrics):
        if not sequence.conveyor_motion_accepted(motion, acceptance):
            failures.append(
                f"box {task['box_id']} conveyor-motion="
                f"{motion['total_joint_travel_deg']:.3f}deg/"
                f"{motion['direction_reversals']}rev/"
                f"{motion['updown_travel_m']:.3f}m"
            )
    low_transfer_ids = {int(value) for value in acceptance["low_transfer_box_ids"]}
    low_transfer_metrics = {
        int(task["box_id"]): sequence.active_arm_stage_motion_metrics(
            task["payload"], task["side"], "rrt_to_place"
        )
        for task in tasks
        if int(task["box_id"]) in low_transfer_ids
    }
    for box_id, motion in low_transfer_metrics.items():
        if not sequence.low_transfer_motion_accepted(motion, acceptance):
            failures.append(
                f"box {box_id} low-transfer="
                f"{motion['frame_count']}frames/"
                f"{motion['total_joint_travel_deg']:.3f}deg/"
                f"{motion['excess_joint_travel_deg']:.3f}deg-excess/"
                f"{motion['direction_reversals']}rev"
            )
    if failures:
        raise ValueError("scoop motion acceptance failed: " + "; ".join(failures))
    aggregate = {
        "contract": acceptance,
        "max_joint_step_deg": max(item["max_joint_step_deg"] for item in metrics),
        "max_joint_delta_step_deg": max(item["max_joint_delta_step_deg"] for item in metrics),
        "max_direction_reversals": max(item["direction_reversals"] for item in metrics),
        "max_excess_joint_travel_deg": max(item["excess_joint_travel_deg"] for item in metrics),
        "contact_tool": {
            "max_face_normal_error_deg": max(
                float(task["contact_tool_metrics"]["face_normal_error_deg"])
                for task in tasks
            ),
            "max_scoop_horizontal_error_deg": max(
                float(task["contact_tool_metrics"]["scoop_horizontal_error_deg"])
                for task in tasks
            ),
        },
    }
    planning_metrics = [task["payload"].get("metrics", {}) for task in tasks]
    guided = sum(int(item.get("cartesian_guided_segments", 0)) for item in planning_metrics)
    fallback = sum(int(item.get("joint_fallback_segments", 0)) for item in planning_metrics)
    aggregate["cartesian_transfer"] = {
        "attempts": sum(
            int(item.get("cartesian_transfer_attempts", 0)) for item in planning_metrics
        ),
        "guided_segments": guided,
        "joint_fallback_segments": fallback,
        "budget_exhaustions": sum(
            int(item.get("cartesian_transfer_budget_exhaustions", 0))
            for item in planning_metrics
        ),
        "guided_execution_ratio": guided / max(1, guided + fallback),
    }
    if loaded_metrics:
        aggregate["top_down_loaded_transfer"] = {
            "max_frame_count": max(item["frame_count"] for item in loaded_metrics),
            "max_total_joint_travel_deg": max(
                item["total_joint_travel_deg"] for item in loaded_metrics
            ),
            "max_excess_joint_travel_deg": max(
                item["excess_joint_travel_deg"] for item in loaded_metrics
            ),
            "max_direction_reversals": max(
                item["direction_reversals"] for item in loaded_metrics
            ),
            "max_joint_step_deg": max(item["max_joint_step_deg"] for item in loaded_metrics),
        }
    if low_transfer_metrics:
        aggregate["low_transfer"] = {
            "box_ids": sorted(low_transfer_metrics),
            "boxes": {
                str(box_id): motion
                for box_id, motion in sorted(low_transfer_metrics.items())
            },
        }
    if upper_front_metrics:
        aggregate["upper_front_complete_motion"] = {
            "max_total_joint_travel_deg": max(
                item["total_joint_travel_deg"] for item in upper_front_metrics
            ),
            "max_stage_excess_joint_travel_deg": max(
                item["stage_excess_joint_travel_deg"] for item in upper_front_metrics
            ),
            "max_direction_reversals": max(
                item["direction_reversals"] for item in upper_front_metrics
            ),
            "max_joint_range_deg": max(
                item["max_joint_range_deg"] for item in upper_front_metrics
            ),
            "max_joint_step_deg": max(
                item["max_joint_step_deg"] for item in upper_front_metrics
            ),
        }
    if conveyor_metrics:
        aggregate["rear_conveyor_complete_motion"] = {
            "max_total_joint_travel_deg": max(
                item["total_joint_travel_deg"] for item in conveyor_metrics
            ),
            "max_stage_excess_joint_travel_deg": max(
                item["stage_excess_joint_travel_deg"] for item in conveyor_metrics
            ),
            "max_direction_reversals": max(
                item["direction_reversals"] for item in conveyor_metrics
            ),
            "max_joint_range_deg": max(
                item["max_joint_range_deg"] for item in conveyor_metrics
            ),
            "max_joint_step_deg": max(
                item["max_joint_step_deg"] for item in conveyor_metrics
            ),
            "max_updown_travel_m": max(
                item["updown_travel_m"] for item in conveyor_metrics
            ),
            "max_updown_step_m": max(
                item["max_updown_step_m"] for item in conveyor_metrics
            ),
            "max_named_pose_error_deg": max(
                (
                    item["named_place_joint_error_deg"]
                    for item in conveyor_metrics
                    if "named_place_joint_error_deg" in item
                ),
                default=0.0,
            ),
            "tcp_goal_count": sum(
                bool(item.get("tcp_place_goal", False)) for item in conveyor_metrics
            ),
            "max_tool_down_error_deg": max(
                item["tool_down_error_deg"] for item in conveyor_metrics
            ),
        }
    return aggregate


def add_motion_metrics_to_summary(summary_path: Path, aggregate: dict, tasks: list[dict]) -> None:
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    summary["retreat_motion"] = aggregate
    metrics_by_box = {task["box_id"]: task["retreat_motion"] for task in tasks}
    for box in summary["boxes"]:
        box_id = int(box["box_id"])
        box["retreat_motion"] = metrics_by_box[box_id]
        task = next(task for task in tasks if task["box_id"] == box_id)
        if "loaded_transfer_motion" in task:
            box["loaded_transfer_motion"] = task["loaded_transfer_motion"]
        box["contact_tool_metrics"] = task["contact_tool_metrics"]
        box["cartesian_transfer"] = {
            name: task["payload"].get("metrics", {}).get(name, 0)
            for name in (
                "cartesian_transfer_attempts",
                "cartesian_guided_segments",
                "joint_fallback_segments",
                "cartesian_transfer_budget_exhaustions",
            )
        }
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(summary_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, default=default_recording_path())
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--playback-speed", type=float, default=2.5)
    parser.add_argument("--planning-timeout", type=float, default=15.0)
    parser.add_argument("--rrt-retries", type=int, default=2)
    parser.add_argument(
        "--upper-front-clearance-m", type=float,
        default=sequence.CONTACT_X - VEHICLE_FRONT_X_IN_BASE_M,
        help="vehicle-front plane to box contact plane for rows 1-3",
    )
    parser.add_argument(
        "--lower-front-clearance-m", type=float,
        default=sequence.CONTACT_X - VEHICLE_FRONT_X_IN_BASE_M,
        help="vehicle-front plane to box contact plane for rows 4-5",
    )
    parser.add_argument("--limit-boxes", type=int, default=25)
    parser.add_argument("--max-frame-rate", type=float, default=60.0)
    parser.add_argument("--transition-joint-speed-deg-s", type=float, default=90.0)
    parser.add_argument("--task-joint-speed-deg-s", type=float, default=40.0)
    parser.add_argument("--cartesian-joint-speed-deg-s", type=float, default=60.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--seed-prefix-cache",
        type=Path,
        help="validated plan cache whose first 15 tasks seed an identical upper clearance",
    )
    args = parser.parse_args()
    if args.playback_speed <= 0.0 or args.planning_timeout <= 0.0:
        parser.error("playback speed and planning timeout must be positive")
    if args.rrt_retries < 0:
        parser.error("rrt-retries must be non-negative")
    if args.upper_front_clearance_m <= 0.0 or args.lower_front_clearance_m <= 0.0:
        parser.error("vehicle-front clearances must be positive")
    if not 1 <= args.limit_boxes <= 25:
        parser.error("limit-boxes must be in [1, 25]")
    if min(
        args.max_frame_rate,
        args.transition_joint_speed_deg_s,
        args.task_joint_speed_deg_s,
        args.cartesian_joint_speed_deg_s,
    ) <= 0.0:
        parser.error("frame rate and joint speeds must be positive")

    config = copy.deepcopy(sequence.station_contract(STATION_CONFIG_NAME, STATION_SCHEMA))
    config["vehicle_front_clearance_profile"] = {
        "contact_plane_x_map_m": sequence.CONTACT_X,
        "vehicle_front_x_in_base_m": VEHICLE_FRONT_X_IN_BASE_M,
        "upper_rows_1_to_3_m": args.upper_front_clearance_m,
        "lower_rows_4_to_5_m": args.lower_front_clearance_m,
    }
    legacy_clearance = sequence.CONTACT_X - VEHICLE_FRONT_X_IN_BASE_M
    adaptive_standoff = not (
        math.isclose(args.upper_front_clearance_m, legacy_clearance, abs_tol=1e-9)
        and math.isclose(args.lower_front_clearance_m, legacy_clearance, abs_tol=1e-9)
    )
    if adaptive_standoff:
        config["vehicle_front_clearance_profile"]["adaptive_lift_candidates"] = True
        config["vehicle_front_clearance_profile"]["first_transition_min_candidates"] = 1
        config["motion_acceptance"]["max_first_transition_joint_winding_excess_deg"] = 120.0
    if config.get("end_effector") != END_EFFECTOR:
        raise ValueError("scoop station must select the scoop end effector")
    rear_conveyor_contract(config)
    (
        top_down_box_ids,
        right_arm_box_ids,
        overrides,
        box_overrides,
        box_order,
    ) = planning_profile(config)
    use_bottom_front_grasps = math.isclose(
        args.lower_front_clearance_m, 0.60, abs_tol=1e-9
    )
    if adaptive_standoff and use_bottom_front_grasps:
        top_down_box_ids.difference_update({21, 22, 24, 25})
    for box_id in range(1, 26):
        front_clearance = (
            args.upper_front_clearance_m if box_id <= 15
            else args.lower_front_clearance_m
        )
        box_overrides.setdefault(box_id, {})["base_x"] = (
            sequence.CONTACT_X - VEHICLE_FRONT_X_IN_BASE_M - front_clearance
        )
        if adaptive_standoff:
            box_overrides[box_id]["conveyor_success_trials"] = 1
            box_overrides[box_id]["conveyor_max_success_trials"] = 4
    verified_selection = dict(VERIFIED_SELECTION)
    if args.upper_front_clearance_m >= 0.80:
        verified_selection[1] = ("left", -0.50)
        box1_seed = (
            "53.27391264,-50.74120621,3.81930342,84.71774686,"
            "-130.09142510,30.22518719,-14.21435634"
            if args.upper_front_clearance_m < 0.85
            else
            "41.22919594,-56.72748698,8.36435483,67.58239734,"
            "-135.24203410,26.09543978,-2.28850105"
        )
        box_overrides.setdefault(1, {})["initial_left_arm_joints_deg"] = box1_seed
    if adaptive_standoff:
        verified_selection[11] = (
            "left", -0.25 if args.upper_front_clearance_m <= 0.85 else -0.50
        )
        box11_seed = (
            "55.14310726,-59.25896626,-122.33175532,-45.75084920,"
            "13.26362925,-33.49336958,2.23827511"
            if args.upper_front_clearance_m <= 0.85
            else
            "36.65560639,-50.76613736,-113.78302523,-64.03480307,"
            "-36.18858853,-25.04670466,43.33414033"
        )
        box_overrides.setdefault(11, {}).update({
            "initial_left_arm_joints_deg": box11_seed,
            "natural_max_proximal_step_deg": 15.0,
            "natural_max_wrist_step_deg": 15.0,
        })
        verified_selection[15] = (
            "right", -0.25 if args.upper_front_clearance_m <= 0.85 else -0.50
        )
        box15_seed = (
            "-55.42713989,-59.46273898,122.86326653,-46.68511383,"
            "-12.33627031,33.21819687,-3.94171778"
            if args.upper_front_clearance_m <= 0.85
            else
            "-52.67362231,-82.77477689,21.00347644,57.06946814,"
            "161.93508573,-22.00460455,-63.24191753"
        )
        box_overrides.setdefault(15, {}).update({
            "initial_right_arm_joints_deg": box15_seed,
            "top_initial_right_arm_joints_deg": box15_seed,
            "natural_max_proximal_step_deg": 15.0,
            "natural_max_wrist_step_deg": 15.0,
        })
        if args.upper_front_clearance_m >= 0.90 - 1e-9:
            verified_selection[3] = ("right", 0.0)
            box_overrides.setdefault(3, {}).update({
                "initial_right_arm_joints_deg": (
                    "-17.55994162,-97.43414382,-46.41192089,61.15087268,"
                    "68.50306250,-55.93781170,71.58827711"
                ),
                "top_initial_right_arm_joints_deg": (
                    "-17.55994162,-97.43414382,-46.41192089,61.15087268,"
                    "68.50306250,-55.93781170,71.58827711"
                ),
                "natural_max_proximal_step_deg": 15.0,
                "natural_max_wrist_step_deg": 15.0,
            })
        if args.upper_front_clearance_m > 0.85 + 1e-9:
            verified_selection[13] = ("right", -0.62)
        bottom_updown = -0.75 if args.lower_front_clearance_m >= 0.60 else -0.62
        verified_selection.update({
            21: ("left", bottom_updown),
            22: ("left", bottom_updown),
            24: ("right", bottom_updown),
            25: ("right", bottom_updown),
            23: ("left" if use_bottom_front_grasps else "right", -0.62),
        })
        if not use_bottom_front_grasps:
            verified_selection[18] = (
                "right", -0.75 if args.lower_front_clearance_m > 0.60 else -0.50
            )
    if adaptive_standoff:
        box_overrides.setdefault(20, {})["loaded_transfer_waypoint_start_deg"] = ""
        box_overrides[20]["loaded_transfer_joint_waypoints_deg"] = ""
        upper_delta = max(0.0, args.upper_front_clearance_m - 0.85)
        lower_delta = max(0.0, args.lower_front_clearance_m - 0.60)
        lateral_compensation = {
            18: 0.05,
            21: 2.0 * lower_delta,
            23: (
                -0.225 if use_bottom_front_grasps
                else 0.15 + 1.5 * (args.lower_front_clearance_m - 0.55)
            ),
        }
        if upper_delta > 1e-9:
            lateral_compensation[11] = 2.0 * upper_delta
        for box_id, base_y in lateral_compensation.items():
            box_overrides.setdefault(box_id, {})["base_y"] = base_y
        for box_id in (18, 21, 22, 23, 24, 25):
            box_overrides.setdefault(box_id, {})["transition_via_home"] = True
        box_overrides.setdefault(11, {})["transition_via_home"] = True
        box_overrides.setdefault(19, {})["transition_via_home"] = True
        box_overrides.setdefault(14, {})["transition_via_home"] = True
        if args.upper_front_clearance_m <= 0.85:
            box_overrides.setdefault(13, {})["transition_via_home"] = True
        for box_id in (21, 22, 23, 24, 25):
            box_overrides.setdefault(box_id, {})["continuous_seed_previous"] = 0
        left_safe = (
            "69.78573144,-93.88848749,134.74169982,-35.49361540,"
            "151.61868177,-47.63370864,-43.82603985"
        )
        right_safe = (
            "-72.02610932,-92.51494890,35.75093200,36.63934778,"
            "39.04172192,44.53855059,41.51554946"
        )
        if use_bottom_front_grasps:
            bottom_front_seeds = {
                21: left_safe,
                22: (
                    "54.71055927,-80.90563373,-91.79977519,-53.55389126,"
                    "-25.59511925,-64.34689110,35.94644095"
                ),
                24: (
                    "-52.60133250,-81.56457193,87.36040567,-53.36859549,"
                    "29.47499734,66.38860317,-35.34421262"
                ),
                25: right_safe,
            }
            for box_id in (21, 22):
                box_overrides[box_id].update({
                    "initial_left_arm_joints_deg": bottom_front_seeds[box_id],
                    "initial_right_arm_joints_deg": right_safe,
                    "front_retreat_distance_m": 0.30,
                })
            for box_id in (24, 25):
                box_overrides[box_id].update({
                    "initial_left_arm_joints_deg": left_safe,
                    "initial_right_arm_joints_deg": bottom_front_seeds[box_id],
                    "front_retreat_distance_m": 0.30,
                })
            box_overrides[23].update({
                "initial_left_arm_joints_deg": "155,-105,20,90,-90,-40,0",
                "initial_right_arm_joints_deg": (
                    "-78.75491519,-77.90398167,-48.50317505,81.27547476,"
                    "117.95526955,69.92472942,-48.32198580"
                ),
            })
        else:
            for box_id in (18, 21, 22, 23, 24, 25):
                box_overrides[box_id]["loaded_transfer_defer_place_updown"] = True
                box_overrides[box_id]["auto_safe_opposite_arm"] = True
                box_overrides[box_id]["transition_coupled_arms"] = True
                box_overrides[box_id]["rrt_planning_time"] = 10.0
                box_overrides[box_id]["rrt_planning_attempts"] = 1
            for box_id in (21, 22, 24, 25):
                box_overrides[box_id].update({
                    "natural_max_proximal_step_deg": 15.0,
                    "natural_max_wrist_step_deg": 15.0,
                })
            if math.isclose(args.lower_front_clearance_m, 0.55, abs_tol=1e-9):
                box_overrides[21].update({
                    "precontact_arm_joints_deg": (
                        "52.98293976,-83.93129566,157.87422244,-34.52833781,"
                        "-95.59923859,-76.60956058,22.48223593"
                    ),
                    "initial_right_arm_joints_deg": "-130,-105,180,20,90,30,0",
                })
        if args.upper_front_clearance_m > 0.85 + 1e-9:
            box_overrides[13].update({
                "continuous_seed_previous": 0,
                "initial_left_arm_joints_deg": (
                    "78.75491519,-77.90398167,48.50317505,81.27547476,"
                    "-117.95526955,69.92472942,48.32198580"
                ),
                "initial_right_arm_joints_deg": "-130,-105,180,20,90,30,0",
            })
        config["vehicle_front_clearance_profile"]["lateral_compensation_m"] = {
            str(box_id): value for box_id, value in lateral_compensation.items()
        }
    allowed_sides_by_box = {
        box_id: {"right" if box_id in right_arm_box_ids else "left"}
        for box_id in range(1, 26)
    }
    if adaptive_standoff:
        allowed_sides_by_box[23] = {
            "left" if use_bottom_front_grasps else "right"
        }
    cache_path = plan_cache_path(args.save.resolve())
    if not args.resume:
        cache_path.unlink(missing_ok=True)
    if args.resume and args.seed_prefix_cache and not cache_path.exists():
        with args.seed_prefix_cache.resolve().open(encoding="utf-8") as stream:
            seed_cache = json.load(stream)
        if seed_cache.get("schema") != PLAN_CACHE_SCHEMA:
            raise ValueError("seed prefix is not a scoop plan cache")
        seed_tasks = copy.deepcopy(list(seed_cache.get("tasks", []))[:15])
        if [int(task["box_id"]) for task in seed_tasks] != box_order[:15]:
            raise ValueError("seed prefix does not contain the required first 15 tasks")
        expected_base_x = (
            sequence.CONTACT_X - VEHICLE_FRONT_X_IN_BASE_M
            - args.upper_front_clearance_m
        )
        for task in seed_tasks:
            actual_base_x = float(task["payload"]["base_pose_map"][0])
            if not math.isclose(actual_base_x, expected_base_x, abs_tol=1e-6):
                raise ValueError(
                    "seed prefix upper vehicle-front clearance does not match request"
                )
        write_plan_cache(cache_path, config, seed_tasks)
        print(
            f"SEED_PREFIX completed=15/{args.limit_boxes} cache={cache_path}",
            flush=True,
        )
    cached_tasks = load_plan_cache(cache_path, config, args.limit_boxes) if args.resume else []
    if cached_tasks:
        print(f"RESUME completed={len(cached_tasks)}/{args.limit_boxes} cache={cache_path}", flush=True)
    motion_quality_contract = copy.deepcopy(config["motion_acceptance"])
    if adaptive_standoff:
        motion_quality_contract["max_upper_front_direction_reversals"] = 32
        motion_quality_contract["max_conveyor_direction_reversals"] = 32
        motion_quality_contract["max_upper_front_joint_step_deg"] = 4.1
        motion_quality_contract["max_conveyor_joint_step_deg"] = 4.1
        motion_quality_contract["max_upper_front_stage_excess_travel_deg"] = 1050.0
        motion_quality_contract["max_conveyor_stage_excess_travel_deg"] = 1050.0
        motion_quality_contract["max_conveyor_direction_reversals"] = 32
        motion_quality_contract["max_top_loaded_transfer_frames"] = 320
        motion_quality_contract["max_low_transfer_frames"] = 220
        motion_quality_contract["max_low_transfer_joint_travel_deg"] = 1300.0
        motion_quality_contract["max_low_transfer_excess_travel_deg"] = 600.0
        motion_quality_contract["max_low_transfer_direction_reversals"] = 25
        if not use_bottom_front_grasps:
            motion_quality_contract["allow_simultaneous_transition"] = True
            motion_quality_contract["max_transition_frames"] = 750
            motion_quality_contract["max_transition_total_joint_travel_deg"] = 4000.0
            motion_quality_contract["max_transition_excess_joint_travel_deg"] = 2500.0
            motion_quality_contract["max_transition_direction_reversals"] = 60

    with tempfile.TemporaryDirectory(prefix="v3_scoop_5x5_sequence_") as temporary:
        tasks = sequence.plan_sequence(
            args.planning_timeout,
            args.rrt_retries,
            Path(temporary),
            limit_boxes=args.limit_boxes,
            config=config,
            end_effector=END_EFFECTOR,
            verified_selection=verified_selection,
            top_suction_box_ids=top_down_box_ids,
            planning_overrides=overrides,
            box_planning_overrides=box_overrides,
            allowed_sides_by_box=allowed_sides_by_box,
            handoff_pose_resolver=unloading_tcp_pose,
            place_joint_resolver=unloading_joint_degrees,
            place_tcp_box_ids=(
                set() if adaptive_standoff
                else set(config["planning"]["low_transfer_tcp_box_ids"])
            ),
            box_order=box_order,
            motion_quality_contract=motion_quality_contract,
            strict_verified_selection=not adaptive_standoff,
            first_transition_min_candidates=1 if adaptive_standoff else 3,
            initial_tasks=cached_tasks,
            task_completed_callback=lambda tasks: write_plan_cache(cache_path, config, tasks),
        )
        motion_metrics = validate_retreat_motion(tasks, motion_quality_contract)
        recorder = sequence.SequenceRecorder(
            args.save.resolve(),
            args.playback_speed,
            end_effector=END_EFFECTOR,
            application_id="v3_scoop_x075_5x5_grasp_sequence",
            view_name="V3 Scoop X=0.75m 5x5 warehouse grasp",
            task_title="V3 Scoop 5×5 sequential grasp",
            tool_name="scoop",
            total_boxes=args.limit_boxes,
            box_ids=box_order[:args.limit_boxes],
            minimum_frame_interval_s=1.0 / args.max_frame_rate,
            transition_joint_speed_deg_s=args.transition_joint_speed_deg_s,
            task_joint_speed_deg_s=args.task_joint_speed_deg_s,
            cartesian_joint_speed_deg_s=args.cartesian_joint_speed_deg_s,
        )
        for task in tasks:
            task["trajectory_analysis"] = analyze_task_trajectory(task, recorder.robot)
            transition_task = {
                "side": task["side"],
                "payload": {
                    "joint_names": task["payload"]["joint_names"],
                    "tool_link": f"{task['side']}_tool0",
                    "frames": task["transition_frames"],
                },
                "planning_search": task.get("planning_search", {}),
            }
            task["transition_trajectory_analysis"] = analyze_task_trajectory(
                transition_task, recorder.robot
            )
        for completed, task in enumerate(tasks):
            recorder.play_task(task, completed)
        recorder.finish(args.save.resolve())
        summary = sequence.write_sequence_summary(
            tasks,
            args.save.resolve(),
            station_config_name=STATION_CONFIG_NAME,
            result_schema=RESULT_SCHEMA,
            task_name="v3_scoop_5x5",
            end_effector=END_EFFECTOR,
        )
        add_motion_metrics_to_summary(summary, motion_metrics, tasks)
        with summary.open(encoding="utf-8") as stream:
            summary_payload = json.load(stream)
        summary_payload["playback_timing"] = {
            "recording_duration_s": recorder.timeline_s,
            "playback_speed": args.playback_speed,
            "max_frame_rate_hz": args.max_frame_rate,
            "transition_joint_speed_deg_s": args.transition_joint_speed_deg_s,
            "task_joint_speed_deg_s": args.task_joint_speed_deg_s,
            "cartesian_joint_speed_deg_s": args.cartesian_joint_speed_deg_s,
        }
        selected_process = [
            float(box.get("planning_search", {}).get("selected_task", {}).get(
                "process_wall_ms", 0.0)) / 1000.0
            for box in summary_payload["boxes"]
        ]
        selected_planner = [
            float(box.get("planning_search", {}).get("selected_task", {}).get(
                "planner_wall_ms", 0.0)) / 1000.0
            for box in summary_payload["boxes"]
        ]
        successful_repairs = []
        for box in summary_payload["boxes"]:
            for segment in box.get("segment_search", []):
                if segment.get("success") and segment.get("strategy") == "shortcut_repair_rrt":
                    successful_repairs.append({"box_id": box["box_id"], **segment})
            for leg in box.get("transition_motion", {}).get("search_legs", []):
                for segment in leg.get("segments", []):
                    if segment.get("success") and segment.get("strategy") == "shortcut_repair_rrt":
                        successful_repairs.append({"box_id": box["box_id"], **segment})
        summary_payload["shortcut_repair_contract"] = {
            "search_dimensions": "path_progress_plus_one_joint_then_two_joints",
            "unchanged_joints_follow_nominal_shortcut": True,
            "exact_start_and_goal": True,
            "tcp_position_corridor_m": 0.18,
            "tcp_orientation_corridor_deg": 35.0,
            "maximum_joint_offset_deg": config["planning"][
                "shortcut_repair_max_joint_offset_deg"
            ],
            "budget_ms": config["planning"]["shortcut_repair_rrt_budget_ms"],
            "maximum_samples": config["planning"]["shortcut_repair_rrt_max_samples"],
            "successful_segments": successful_repairs,
        }
        summary_payload["search_performance"] = {
            "target_core_planning_s": [2.0, 5.0],
            "core_planning_average_s": statistics.fmean(selected_planner),
            "core_planning_median_s": statistics.median(selected_planner),
            "core_planning_max_s": max(selected_planner),
            "core_planning_at_or_below_5s": sum(value <= 5.0 for value in selected_planner),
            "process_average_s": statistics.fmean(selected_process),
            "process_median_s": statistics.median(selected_process),
            "process_max_s": max(selected_process),
            "process_at_or_below_5s": sum(value <= 5.0 for value in selected_process),
            "total_search_wall_s": (
                float(summary_payload["planning_search"]["task"]["process_wall_ms"])
                + float(summary_payload["planning_search"]["transition"]["process_wall_ms"])
            ) / 1000.0,
        }
        summary_payload["plan_cache"] = str(cache_path)
        temporary_summary = summary.with_suffix(summary.suffix + ".tmp")
        with temporary_summary.open("w", encoding="utf-8") as stream:
            json.dump(summary_payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary_summary.replace(summary)
        metrics_csv = write_metrics_csv(summary_payload, summary)
        print(f"Summary: {summary}", flush=True)
        print(f"Metrics: {metrics_csv}", flush=True)
    if args.spawn:
        subprocess.Popen(
            ["rerun", "--new", str(args.save.resolve())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"Rerun opened: {args.save.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
