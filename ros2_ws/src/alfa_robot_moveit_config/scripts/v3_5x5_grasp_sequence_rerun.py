#!/usr/bin/env python3
"""Plan and play the complete X=0.75m V3 5x5 grasp sequence in one Rerun."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import scan_v3_single_arm_box_wall as scan

from alfa_robot_rerun.visualize_rerun import (
    UrdfRobot,
    log_robot_state,
    log_robot_static_model,
    prefer_matching_rerun_cli,
    render_current_urdf,
)


CONTACT_X = 0.75
BOX_DEPTH = 0.30
BOX_WIDTH = 0.40
BOX_HEIGHT = 0.40
STATION_CONFIG_NAME = "v3_5x5_station.json"
WAREHOUSE_OPENING_X = -1.18
WAREHOUSE_CENTER_Y = 0.0
WAREHOUSE_FLOOR_Z = 0.0
WAREHOUSE_LENGTH = 2.38
WAREHOUSE_WIDTH = 2.38
WAREHOUSE_HEIGHT = 2.35
WAREHOUSE_WALL_THICKNESS = 0.05
VERIFIED_SELECTION = {
    1: ("right", 0.0),
    2: ("right", 0.0),
    3: ("left", -0.25),
    4: ("left", 0.0),
    5: ("left", 0.0),
    6: ("right", 0.0),
    7: ("right", -0.50),
    8: ("left", -0.50),
    9: ("left", -0.50),
    10: ("left", 0.0),
    11: ("right", 0.0),
    12: ("right", 0.0),
    13: ("left", 0.0),
    14: ("left", 0.0),
    15: ("left", 0.0),
    16: ("right", -0.75),
    17: ("right", -0.25),
    18: ("left", -0.25),
    19: ("left", -0.25),
    20: ("left", -0.75),
    21: ("right", -0.75),
    22: ("right", -0.75),
    23: ("left", -0.75),
    24: ("left", -0.75),
    25: ("left", -0.75),
}


def workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".ai_teamwork").is_dir():
            return parent
    return Path.cwd()


def default_recording_path() -> Path:
    return (
        workspace_root()
        / "data/ik_benchmark/v3_single_arm_5x5_box_wall"
        / "x075_robot_inside_warehouse_238"
        / "v3-x075-folded-tcp-handoff-sequence.rrd"
    )


def placed_box_center(payload: dict[str, Any]) -> list[float]:
    pose = [float(value) for value in payload["achieved_place_tcp_pose"]]
    offset = [float(value) for value in payload["tool_to_box_center"]]
    if len(pose) != 7 or len(offset) != 3:
        raise ValueError("placement center requires a 7D TCP and 3D tool offset")
    x, y, z, w = pose[3:]
    vx, vy, vz = offset
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rotated = [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]
    return [pose[index] + rotated[index] for index in range(3)]


def placed_tool_down_error_deg(payload: dict[str, Any]) -> float:
    pose = [float(value) for value in payload["achieved_place_tcp_pose"]]
    if len(pose) != 7:
        raise ValueError("placement tool direction requires a 7D TCP")
    x, y, z, w = pose[3:]
    vx, vy, vz = 0.0, 0.0, 1.0
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    tool_z = [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]
    norm = math.sqrt(sum(value * value for value in tool_z))
    cosine = max(-1.0, min(1.0, -tool_z[2] / norm))
    return math.degrees(math.acos(cosine))


def named_place_joint_error_deg(
    payload: dict[str, Any], side: str, target_degrees: list[float]
) -> float:
    names = [str(name) for name in payload["joint_names"]]
    release = next(
        frame for frame in payload["frames"] if frame["stage"] == "release_at_place"
    )
    actual = [
        float(release["joints"][names.index(f"{side}_joint{index}")])
        for index in range(1, 8)
    ]
    target = [math.radians(float(value)) for value in target_degrees]
    return max(
        abs(math.degrees(math.atan2(math.sin(left - right), math.cos(left - right))))
        for left, right in zip(actual, target)
    )


def write_sequence_summary(
    tasks: list[dict[str, Any]],
    recording: Path,
    *,
    station_config_name: str = STATION_CONFIG_NAME,
    result_schema: str = "alfa.v3_5x5_station_result.v1",
    task_name: str = "single_arm_suction_5x5",
    end_effector: str = "legacy_suction",
) -> Path:
    summary_path = recording.with_name(recording.stem + "-summary.json")
    payload = {
        "schema": result_schema,
        "task_name": task_name,
        "end_effector": end_effector,
        "recording": str(recording),
        "completed_boxes": len(tasks),
        "validated_transitions": sum(bool(task["transition_frames"]) for task in tasks),
        "station_config": station_config_name,
        "return_to_ready": bool(tasks[0]["payload"].get("return_to_ready", True)) if tasks else True,
        "total_active_arm_joint_travel_deg": sum(
            task.get("motion_selection", {}).get("total_joint_travel_deg", 0.0)
            for task in tasks
        ),
        "total_sequence_joint_travel_deg": complete_sequence_joint_travel_deg(tasks),
        "planning_search": sequence_search_metrics(tasks),
        "boxes": [
            {
                "box_id": task["box_id"],
                "side": task["side"],
                "mode": task["mode"],
                "updown_m": task["updown"],
                "handoff_tcp_target": task["handoff_tcp_pose"],
                "handoff_tcp_actual": task["payload"]["achieved_place_tcp_pose"],
                **(
                    {
                        "place_arm_joints_deg": task["place_arm_joints_deg"],
                        "place_arm_joint_error_deg": named_place_joint_error_deg(
                            task["payload"], task["side"], task["place_arm_joints_deg"]
                        ),
                    }
                    if "place_arm_joints_deg" in task
                    else {}
                ),
                "ignore_opposite_arm": bool(
                    task["payload"].get("ignore_opposite_arm", False)
                ),
                **(
                    {
                        "placement_box_center_target": task["placement_box_center_target"],
                        "placement_box_center_actual": placed_box_center(task["payload"]),
                        "placement_tool_down_error_deg": placed_tool_down_error_deg(
                            task["payload"]
                        ),
                    }
                    if "placement_box_center_target" in task
                    else {}
                ),
                "validated_transition_frames": len(task["transition_frames"]),
                "task_frames": len(task["payload"]["frames"]),
                "carried_box_orientation": task.get("carried_box_orientation", {}),
                "transition_motion": task.get("transition_motion", {}),
                "planning_search": task.get("planning_search", {}),
                "segment_search": task["payload"].get("segment_search", []),
                "trajectory_analysis": task.get("trajectory_analysis", {}),
                "transition_trajectory_analysis": task.get(
                    "transition_trajectory_analysis", {}
                ),
                **(
                    {"motion_selection": task["motion_selection"]}
                    if "motion_selection" in task
                    else {}
                ),
            }
            for task in tasks
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(summary_path)
    return summary_path


def station_contract(
    config_name: str = STATION_CONFIG_NAME,
    expected_schema: str = "alfa.v3_5x5_station.v1",
) -> dict[str, Any]:
    source = SCRIPT_DIR.parent / "config" / config_name
    if not source.is_file():
        from ament_index_python.packages import get_package_share_directory

        source = Path(get_package_share_directory("alfa_robot_moveit_config")) / "config" / config_name
    with source.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema") != expected_schema:
        raise ValueError(f"unsupported station contract: {source}")
    for group, expected in (("ready_joint_degrees", 7),):
        entries = config[group]
        if set(entries) != {"left", "right_front", "right_top_suction"}:
            raise ValueError(f"incomplete station {group}: {source}")
        if any(len(values) != expected or not all(math.isfinite(v) for v in values)
               for values in entries.values()):
            raise ValueError(f"invalid station {group}: {source}")
    if expected_schema == "alfa.v3_5x5_station.v1":
        entries = config["handoff_tcp_at_updown_zero"]
        if set(entries) != {"left", "right_front", "right_top_suction"} or any(
            len(values) != 7 or not all(math.isfinite(value) for value in values)
            for values in entries.values()
        ):
            raise ValueError(f"incomplete station handoff_tcp_at_updown_zero: {source}")
    folded = config["folded_start_joint_degrees"]
    if set(folded) != {"left", "right"} or any(
        len(values) != 7 or not all(math.isfinite(v) for v in values)
        for values in folded.values()
    ):
        raise ValueError(f"invalid folded_start_joint_degrees: {source}")
    return config


def station_key(side: str, mode: str) -> str:
    return "left" if side == "left" else (
        "right_top_suction" if mode == "top_suction" else "right_front"
    )


def handoff_tcp_pose(
    config: dict[str, Any], side: str, mode: str, updown: float
) -> list[float]:
    pose = list(config["handoff_tcp_at_updown_zero"][station_key(side, mode)])
    pose[2] += updown
    return pose


def folded_start_joints(config: dict[str, Any]) -> list[float]:
    degrees = config["folded_start_joint_degrees"]
    updown = float(config.get("folded_start_updown_m", 0.0))
    if not -1.0 <= updown <= 0.0:
        raise ValueError("folded_start_updown_m must be in [-1.0, 0.0]")
    return [updown, 0.0] + [
        math.radians(value) for side in ("left", "right") for value in degrees[side]
    ]


def planning_args(
    timeout: float,
    config: dict[str, Any] | None = None,
    *,
    end_effector: str = "legacy_suction",
) -> SimpleNamespace:
    config = config or station_contract()
    ready = config["ready_joint_degrees"]
    degrees = lambda values: ",".join(str(value) for value in values)
    return SimpleNamespace(
        timeout=timeout,
        end_effector=end_effector,
        initial_arm_pose="zero",
        initial_left_arm_joints_deg=degrees(ready["left"]),
        initial_right_arm_joints_deg=degrees(ready["right_front"]),
        top_initial_right_arm_joints_deg=degrees(ready["right_top_suction"]),
        ignore_opposite_arm=True,
        continuous_sequence=False,
        continuous_plan_approach=False,
        continuous_seed_previous=True,
        maximum_carried_box_tilt_deg=180.0,
        top_suction_x_offset=-0.10,
        front_retreat_distance_m=0.35,
        top_retreat_distance_m=0.35,
        center_front_suction_y_offset=0.08,
        center_front_suction_z_offset=-0.05,
        bottom_front_suction_z_offset=0.12,
        contact_x=CONTACT_X,
        box_depth=BOX_DEPTH,
        box_width=BOX_WIDTH,
        box_height=BOX_HEIGHT,
        box_grid_rows=5,
        box_grid_columns=5,
        box_grid_center_y=0.0,
        box_grid_bottom_z=0.0,
        ground_enabled=True,
        ground_surface_z=0.0,
        ground_clearance=0.005,
        ground_size_x=6.0,
        ground_size_y=6.0,
        ground_thickness=0.10,
        warehouse_enabled=True,
        warehouse_opening_x=WAREHOUSE_OPENING_X,
        warehouse_center_y=WAREHOUSE_CENTER_Y,
        warehouse_floor_z=WAREHOUSE_FLOOR_Z,
        warehouse_length=WAREHOUSE_LENGTH,
        warehouse_width=WAREHOUSE_WIDTH,
        warehouse_height=WAREHOUSE_HEIGHT,
        warehouse_wall_thickness=WAREHOUSE_WALL_THICKNESS,
        psi_step_deg=5.0,
        maximum_cartesian_joint_step_deg=15.0,
        precontact_candidate_limit=8,
        rrt_planning_time=2.0,
        rrt_planning_attempts=3,
        planning_seed=0,
        natural_seed_swivel_sampling=False,
        natural_seed_swivel_step_deg=1.0,
        natural_seed_swivel_neighbor_steps=2,
        natural_joint_acceleration_weight=0.0,
        natural_joint_wrap_weight=0.0,
        natural_place_return_weight=0.0,
        natural_cartesian_replay_step_deg=0.0,
        natural_rrt_shortcut_enabled=False,
        natural_rrt_shortcut_max_nodes=0,
        cartesian_transfer_search_enabled=False,
        cartesian_transfer_translation_step=0.02,
        cartesian_transfer_rotation_step_deg=2.0,
        cartesian_transfer_max_search_attempts=0,
        shortcut_repair_rrt_enabled=False,
        shortcut_repair_rrt_budget_ms=1800.0,
        shortcut_repair_rrt_max_samples=240,
        shortcut_repair_max_joint_offset_deg=35.0,
        contact_tool_roll_deg=0.0,
        place_arm_joints_deg="",
        loaded_transfer_joint_waypoints_deg="",
        loaded_transfer_waypoint_start_deg="",
        loaded_transfer_joint_waypoints_alt_deg="",
        loaded_transfer_waypoint_alt_start_deg="",
        transition_waypoint_start_joints="",
        transition_waypoint_goal_joints="",
        transition_joint_waypoints="",
        validated_waypoint_profiles_path="",
        upper_front_success_trials=1,
        upper_front_max_success_trials=1,
        conveyor_success_trials=1,
        conveyor_max_success_trials=1,
        place_updown_enabled=False,
        place_updown_m=0.0,
        top_loaded_transfer_direct_only=False,
        natural_max_proximal_step_deg=12.0,
        natural_max_wrist_step_deg=8.0,
        analytic_path_only=False,
    )


def ordered_candidates(
    box_id: int,
    row: int,
    y: float,
    z: float,
    mode: str,
    verified_selection: dict[int, tuple[str, float]] | None = None,
    rows: int = 5,
) -> list[tuple[str, float]]:
    fallback = scan.candidate_order(row, y, z, 0.25, mode, rows)
    selection = VERIFIED_SELECTION if verified_selection is None else verified_selection
    preferred = selection.get(box_id)
    return fallback if preferred is None else [
        preferred, *[candidate for candidate in fallback if candidate != preferred]
    ]


def retryable_failure(result: dict[str, Any]) -> bool:
    return str(result.get("failure_stage", "")) in {
        "rrt_to_precontact",
        "rrt_return",
        "rrt_to_place",
        "rrt_to_ready",
        "rrt_to_updown_safe",
        "transition_ompl",
        "transition_path_collision",
        "timeout",
        "process_exit",
    }


def planner_search_metrics(results: list[dict[str, Any]]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "attempts": len(results),
        "successes": sum(bool(result.get("success")) for result in results),
        "process_wall_ms": 0.0,
        "planner_wall_ms": 0.0,
        "ik_calls": 0,
        "ik_ms": 0.0,
        "collision_checks": 0,
        "collision_ms": 0.0,
        "analytic_path_ms": 0.0,
        "rrt_approach_ms": 0.0,
        "rrt_return_ms": 0.0,
        "rrt_shortcut_edges_checked": 0,
        "rrt_shortcut_ms": 0.0,
        "cartesian_transfer_attempts": 0,
        "cartesian_transfer_budget_exhaustions": 0,
        "tcp_shortcut_attempts": 0,
        "tcp_shortcut_successes": 0,
        "tcp_shortcut_ms": 0.0,
        "shortcut_repair_rrt_attempts": 0,
        "shortcut_repair_rrt_successes": 0,
        "shortcut_repair_rrt_samples": 0,
        "shortcut_repair_rrt_ms": 0.0,
        "joint_rrt_fallbacks": 0,
        "joint_rrt_ms": 0.0,
    }
    integer_fields = (
        "ik_calls",
        "collision_checks",
        "cartesian_transfer_attempts",
        "cartesian_transfer_budget_exhaustions",
        "rrt_shortcut_edges_checked",
        "tcp_shortcut_attempts",
        "tcp_shortcut_successes",
        "shortcut_repair_rrt_attempts",
        "shortcut_repair_rrt_successes",
        "shortcut_repair_rrt_samples",
        "joint_rrt_fallbacks",
    )
    timing_fields = (
        "ik_ms",
        "collision_ms",
        "analytic_path_ms",
        "rrt_approach_ms",
        "rrt_return_ms",
        "rrt_shortcut_ms",
        "tcp_shortcut_ms",
        "shortcut_repair_rrt_ms",
        "joint_rrt_ms",
    )
    for result in results:
        metrics["process_wall_ms"] += float(result.get("wall_ms", 0.0))
        payload = result.get("trajectory_result")
        if not isinstance(payload, dict):
            continue
        metrics["planner_wall_ms"] += float(payload.get("total_ms", 0.0))
        payload_metrics = payload.get("metrics", {})
        for name in integer_fields:
            metrics[name] += int(payload_metrics.get(name, 0))
        for name in timing_fields:
            metrics[name] += float(payload_metrics.get(name, 0.0))
    return metrics


def sequence_search_metrics(tasks: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    fields = (
        "attempts",
        "successes",
        "process_wall_ms",
        "planner_wall_ms",
        "ik_calls",
        "ik_ms",
        "collision_checks",
        "collision_ms",
        "analytic_path_ms",
        "rrt_approach_ms",
        "rrt_return_ms",
        "rrt_shortcut_edges_checked",
        "rrt_shortcut_ms",
        "cartesian_transfer_attempts",
        "cartesian_transfer_budget_exhaustions",
        "tcp_shortcut_attempts",
        "tcp_shortcut_successes",
        "tcp_shortcut_ms",
        "shortcut_repair_rrt_attempts",
        "shortcut_repair_rrt_successes",
        "shortcut_repair_rrt_samples",
        "shortcut_repair_rrt_ms",
        "joint_rrt_fallbacks",
        "joint_rrt_ms",
    )
    return {
        phase: {
            name: sum(
                task.get("planning_search", {}).get(phase, {}).get(name, 0)
                for task in tasks
            )
            for name in fields
        }
        for phase in ("task", "transition")
    }


def active_arm_motion_metrics(
    payload: dict[str, Any], side: str, reversal_threshold_deg: float = 0.10
) -> dict[str, float | int]:
    names = [str(name) for name in payload.get("joint_names", [])]
    frames = list(payload.get("frames", []))
    joint_indices = [names.index(f"{side}_joint{index}") for index in range(1, 8)]
    if len(frames) < 2:
        return {
            "selection_score": 0.0,
            "total_joint_travel_deg": 0.0,
            "stage_excess_joint_travel_deg": 0.0,
            "direction_reversals": 0,
            "max_joint_range_deg": 0.0,
            "max_joint_step_deg": 0.0,
            "updown_travel_m": 0.0,
            "max_updown_step_m": 0.0,
        }

    total_travel = 0.0
    maximum_range = 0.0
    maximum_step = 0.0
    for joint_index in joint_indices:
        cumulative = 0.0
        unwrapped = [0.0]
        for left, right in zip(frames, frames[1:]):
            delta = math.degrees(
                math.atan2(
                    math.sin(float(right["joints"][joint_index]) - float(left["joints"][joint_index])),
                    math.cos(float(right["joints"][joint_index]) - float(left["joints"][joint_index])),
                )
            )
            total_travel += abs(delta)
            maximum_step = max(maximum_step, abs(delta))
            cumulative += delta
            unwrapped.append(cumulative)
        maximum_range = max(maximum_range, max(unwrapped) - min(unwrapped))

    updown_travel = 0.0
    maximum_updown_step = 0.0
    if "updown" in names:
        updown_index = names.index("updown")
        updown_deltas = [
            float(right["joints"][updown_index]) - float(left["joints"][updown_index])
            for left, right in zip(frames, frames[1:])
        ]
        updown_travel = sum(abs(delta) for delta in updown_deltas)
        maximum_updown_step = max((abs(delta) for delta in updown_deltas), default=0.0)

    stage_excess = 0.0
    reversals = 0
    stage_start = 0
    while stage_start < len(frames):
        stage_end = stage_start + 1
        while (
            stage_end < len(frames)
            and frames[stage_end]["stage"] == frames[stage_start]["stage"]
        ):
            stage_end += 1
        stage_frames = frames[stage_start:stage_end]
        for joint_index in joint_indices:
            deltas = [
                math.degrees(
                    math.atan2(
                        math.sin(float(right["joints"][joint_index]) - float(left["joints"][joint_index])),
                        math.cos(float(right["joints"][joint_index]) - float(left["joints"][joint_index])),
                    )
                )
                for left, right in zip(stage_frames, stage_frames[1:])
            ]
            stage_excess += sum(abs(delta) for delta in deltas) - abs(sum(deltas))
            active = [delta for delta in deltas if abs(delta) >= reversal_threshold_deg]
            reversals += sum(left * right < 0.0 for left, right in zip(active, active[1:]))
        stage_start = stage_end

    selection_score = (
        total_travel + 2.0 * stage_excess + 80.0 * reversals + 2.0 * maximum_range
    )
    return {
        "selection_score": selection_score,
        "total_joint_travel_deg": total_travel,
        "stage_excess_joint_travel_deg": stage_excess,
        "direction_reversals": reversals,
        "max_joint_range_deg": maximum_range,
        "max_joint_step_deg": maximum_step,
        "updown_travel_m": updown_travel,
        "max_updown_step_m": maximum_updown_step,
    }


def active_arm_stage_motion_metrics(
    payload: dict[str, Any],
    side: str,
    stage: str,
    reversal_threshold_deg: float = 0.10,
) -> dict[str, float | int]:
    names = [str(name) for name in payload.get("joint_names", [])]
    frames = list(payload.get("frames", []))
    stage_indices = [
        index for index, frame in enumerate(frames) if frame["stage"] == stage
    ]
    if not stage_indices:
        raise ValueError(f"trajectory has no {stage} frames")
    first = stage_indices[0]
    stage_frames = frames[max(0, first - 1):stage_indices[-1] + 1]
    joint_indices = [names.index(f"{side}_joint{index}") for index in range(1, 8)]
    total_travel = 0.0
    total_net = 0.0
    reversals = 0
    maximum_step = 0.0
    for joint_index in joint_indices:
        deltas = [
            math.degrees(
                math.atan2(
                    math.sin(
                        float(right["joints"][joint_index])
                        - float(left["joints"][joint_index])
                    ),
                    math.cos(
                        float(right["joints"][joint_index])
                        - float(left["joints"][joint_index])
                    ),
                )
            )
            for left, right in zip(stage_frames, stage_frames[1:])
        ]
        total_travel += sum(abs(delta) for delta in deltas)
        total_net += abs(sum(deltas))
        active = [delta for delta in deltas if abs(delta) >= reversal_threshold_deg]
        reversals += sum(left * right < 0.0 for left, right in zip(active, active[1:]))
        maximum_step = max(maximum_step, max((abs(delta) for delta in deltas), default=0.0))
    return {
        "frame_count": len(stage_frames),
        "total_joint_travel_deg": total_travel,
        "excess_joint_travel_deg": total_travel - total_net,
        "direction_reversals": reversals,
        "max_joint_step_deg": maximum_step,
    }


def dual_arm_motion_metrics(payload: dict[str, Any]) -> dict[str, float | int]:
    left = active_arm_motion_metrics(payload, "left")
    right = active_arm_motion_metrics(payload, "right")
    names = [str(name) for name in payload.get("joint_names", [])]
    frames = list(payload.get("frames", []))
    winding_by_joint: dict[str, float] = {}
    if len(frames) >= 2:
        for side in ("left", "right"):
            for joint_index in range(1, 8):
                name = f"{side}_joint{joint_index}"
                column = names.index(name)
                values = [float(frame["joints"][column]) for frame in frames]
                deltas = [
                    math.degrees(math.atan2(math.sin(right - left), math.cos(right - left)))
                    for left, right in zip(values, values[1:])
                ]
                travel = sum(abs(delta) for delta in deltas)
                shortest = abs(math.degrees(math.atan2(
                    math.sin(values[-1] - values[0]),
                    math.cos(values[-1] - values[0]),
                )))
                winding_by_joint[name] = max(0.0, travel - shortest)
    return {
        "frame_count": len(payload.get("frames", [])),
        "left_arm_joint_travel_deg": float(left["total_joint_travel_deg"]),
        "right_arm_joint_travel_deg": float(right["total_joint_travel_deg"]),
        "total_joint_travel_deg": float(left["total_joint_travel_deg"]) +
        float(right["total_joint_travel_deg"]),
        "stage_excess_joint_travel_deg": float(left["stage_excess_joint_travel_deg"]) +
        float(right["stage_excess_joint_travel_deg"]),
        "direction_reversals": int(left["direction_reversals"]) +
        int(right["direction_reversals"]),
        "max_joint_range_deg": max(
            float(left["max_joint_range_deg"]), float(right["max_joint_range_deg"])
        ),
        "max_joint_step_deg": max(
            float(left["max_joint_step_deg"]), float(right["max_joint_step_deg"])
        ),
        "updown_travel_m": float(left["updown_travel_m"]),
        "max_updown_step_m": float(left["max_updown_step_m"]),
        "max_joint_winding_excess_deg": max(winding_by_joint.values(), default=0.0),
        "joint_winding_excess_deg": winding_by_joint,
    }


def transition_motion_accepted(
    motion: dict[str, float | int], contract: dict[str, Any]
) -> bool:
    return (
        int(motion["frame_count"]) <= int(contract["max_transition_frames"])
        and float(motion["total_joint_travel_deg"])
        <= float(contract["max_transition_total_joint_travel_deg"]) + 1e-6
        and float(motion["stage_excess_joint_travel_deg"])
        <= float(contract["max_transition_excess_joint_travel_deg"]) + 1e-6
        and int(motion["direction_reversals"])
        <= int(contract["max_transition_direction_reversals"])
        and float(motion["max_joint_range_deg"])
        <= float(contract["max_transition_joint_range_deg"]) + 1e-6
        and not bool(motion.get("simultaneous_dual_arm_motion", True))
    )


def upper_front_motion_accepted(
    motion: dict[str, float | int], contract: dict[str, Any]
) -> bool:
    return (
        float(motion["total_joint_travel_deg"])
        <= float(contract["max_upper_front_total_joint_travel_deg"]) + 1e-6
        and float(motion["stage_excess_joint_travel_deg"])
        <= float(contract["max_upper_front_stage_excess_travel_deg"]) + 1e-6
        and int(motion["direction_reversals"])
        <= int(contract["max_upper_front_direction_reversals"])
        and float(motion["max_joint_range_deg"])
        <= float(contract["max_upper_front_joint_range_deg"]) + 1e-6
        and float(motion["max_joint_step_deg"])
        <= float(contract["max_upper_front_joint_step_deg"]) + 1e-6
    )


def conveyor_motion_accepted(
    motion: dict[str, float | int], contract: dict[str, Any]
) -> bool:
    terminal_pose_ok = (
        bool(motion.get("tcp_place_goal", False))
        or float(motion.get("named_place_joint_error_deg", math.inf))
        <= float(contract["max_conveyor_named_pose_error_deg"]) + 1e-6
        if "max_conveyor_named_pose_error_deg" in contract
        else float(motion["tool_down_error_deg"])
        <= float(contract["max_conveyor_tool_down_error_deg"]) + 1e-6
    )
    return (
        float(motion["total_joint_travel_deg"])
        <= float(contract["max_conveyor_total_joint_travel_deg"]) + 1e-6
        and float(motion["stage_excess_joint_travel_deg"])
        <= float(contract["max_conveyor_stage_excess_travel_deg"]) + 1e-6
        and int(motion["direction_reversals"])
        <= int(contract["max_conveyor_direction_reversals"])
        and float(motion["max_joint_range_deg"])
        <= float(contract["max_conveyor_joint_range_deg"]) + 1e-6
        and float(motion["max_joint_step_deg"])
        <= float(contract["max_conveyor_joint_step_deg"]) + 1e-6
        and float(motion["updown_travel_m"])
        <= float(contract["max_conveyor_updown_travel_m"]) + 1e-6
        and float(motion["max_updown_step_m"])
        <= float(contract["max_conveyor_updown_step_m"]) + 1e-6
        and terminal_pose_ok
    )


def low_transfer_motion_accepted(
    motion: dict[str, float | int], contract: dict[str, Any]
) -> bool:
    return (
        int(motion["frame_count"]) <= int(contract["max_low_transfer_frames"])
        and float(motion["total_joint_travel_deg"])
        <= float(contract["max_low_transfer_joint_travel_deg"]) + 1e-6
        and float(motion["excess_joint_travel_deg"])
        <= float(contract["max_low_transfer_excess_travel_deg"]) + 1e-6
        and int(motion["direction_reversals"])
        <= int(contract["max_low_transfer_direction_reversals"])
        and float(motion["max_joint_step_deg"])
        <= float(contract["max_low_transfer_joint_step_deg"]) + 1e-6
    )


def top_loaded_motion_accepted(
    motion: dict[str, float | int], contract: dict[str, Any]
) -> bool:
    return (
        int(motion["frame_count"]) <= int(contract["max_top_loaded_transfer_frames"])
        and float(motion["total_joint_travel_deg"])
        <= float(contract["max_top_loaded_joint_travel_deg"]) + 1e-6
        and float(motion["excess_joint_travel_deg"])
        <= float(contract["max_top_loaded_excess_travel_deg"]) + 1e-6
        and int(motion["direction_reversals"])
        <= int(contract["max_top_loaded_direction_reversals"])
        and float(motion["max_joint_step_deg"])
        <= float(contract["max_top_loaded_joint_step_deg"]) + 1e-6
    )


def check_handoff_result(payload: dict[str, Any], expected_pose: list[float]) -> None:
    if not payload.get("success") or payload.get("task_mode") != "full_extract":
        raise ValueError("handoff requires a successful full_extract result")
    stages = [frame["stage"] for frame in payload.get("frames", [])]
    return_to_ready = bool(payload.get("return_to_ready", True))
    for stage in ("rrt_to_place", "release_at_place") + (("rrt_to_ready",) if return_to_ready else ()):
        if stage not in stages:
            raise ValueError(f"handoff result is missing {stage}")
    if stages.index("rrt_to_place") > stages.index("release_at_place") or (
        return_to_ready and stages.index("release_at_place") >= stages.index("rrt_to_ready")
    ):
        raise ValueError("handoff frames are out of order")
    if not return_to_ready and (
        stages[-1] != "release_at_place" or "rrt_to_ready" in stages or "updown_to_task" in stages
    ):
        raise ValueError("continuous handoff must end at release without a home return")
    actual = payload.get("achieved_place_tcp_pose", [])
    if len(actual) != 7 or any(
        abs(float(actual[index]) - expected_pose[index]) > 0.001 for index in range(3)
    ):
        raise ValueError("handoff TCP position missed its configured target")
    rotation_dot = abs(sum(float(a) * b for a, b in zip(actual[3:], expected_pose[3:])))
    if rotation_dot < math.cos(math.radians(0.5) / 2.0):
        raise ValueError("handoff TCP orientation missed its configured target")


def carried_box_orientation_metrics(payload: dict[str, Any]) -> dict[str, float | int]:
    tilts = [
        float(frame["carried_box_tilt_deg"])
        for frame in payload["frames"] if frame.get("box_attached")
    ]
    if not tilts or not all(math.isfinite(value) for value in tilts):
        raise ValueError("loaded frames require finite carried_box_tilt_deg")
    return {
        "max_tilt_deg": max(tilts),
        "mean_tilt_deg": sum(tilts) / len(tilts),
        "inverted_frames": sum(value > 95.0 + 1e-6 for value in tilts),
        "loaded_frames": len(tilts),
    }


def complete_sequence_joint_travel_deg(tasks: list[dict[str, Any]]) -> float:
    travel = 0.0
    previous = None
    previous_names = None
    for task in tasks:
        names = task["payload"]["joint_names"]
        if previous_names is not None and names != previous_names:
            raise ValueError("sequence joint order changed")
        indices = [index for index, name in enumerate(names) if "_joint" in name]
        for frame in [*task["transition_frames"], *task["payload"]["frames"]]:
            if previous is not None:
                travel += sum(
                    abs(math.degrees(float(frame["joints"][index]) - float(previous[index])))
                    for index in indices
                )
            previous = frame["joints"]
        previous_names = names
    return travel


def warehouse_panels() -> list[dict[str, Any]]:
    center_x = WAREHOUSE_OPENING_X + WAREHOUSE_LENGTH * 0.5
    center_z = WAREHOUSE_FLOOR_Z + WAREHOUSE_HEIGHT * 0.5
    half_thickness = WAREHOUSE_WALL_THICKNESS * 0.5
    return [
        {
            "id": "warehouse_left_wall",
            "center": [
                center_x,
                WAREHOUSE_CENTER_Y + WAREHOUSE_WIDTH * 0.5 + half_thickness,
                center_z,
            ],
            "size": [WAREHOUSE_LENGTH, WAREHOUSE_WALL_THICKNESS, WAREHOUSE_HEIGHT],
        },
        {
            "id": "warehouse_right_wall",
            "center": [
                center_x,
                WAREHOUSE_CENTER_Y - WAREHOUSE_WIDTH * 0.5 - half_thickness,
                center_z,
            ],
            "size": [WAREHOUSE_LENGTH, WAREHOUSE_WALL_THICKNESS, WAREHOUSE_HEIGHT],
        },
        {
            "id": "warehouse_ceiling",
            "center": [
                center_x,
                WAREHOUSE_CENTER_Y,
                WAREHOUSE_FLOOR_Z + WAREHOUSE_HEIGHT + half_thickness,
            ],
            "size": [
                WAREHOUSE_LENGTH,
                WAREHOUSE_WIDTH + 2.0 * WAREHOUSE_WALL_THICKNESS,
                WAREHOUSE_WALL_THICKNESS,
            ],
        },
        {
            "id": "warehouse_rear_wall",
            "center": [
                WAREHOUSE_OPENING_X + WAREHOUSE_LENGTH + half_thickness,
                WAREHOUSE_CENTER_Y,
                center_z,
            ],
            "size": [
                WAREHOUSE_WALL_THICKNESS,
                WAREHOUSE_WIDTH + 2.0 * WAREHOUSE_WALL_THICKNESS,
                WAREHOUSE_HEIGHT,
            ],
        },
    ]


def plan_sequence(
    timeout: float,
    rrt_retries: int,
    scratch: Path,
    *,
    limit_boxes: int = 25,
    config: dict[str, Any] | None = None,
    end_effector: str = "legacy_suction",
    verified_selection: dict[int, tuple[str, float]] | None = None,
    top_suction_box_ids: set[int] | None = None,
    planning_overrides: dict[str, Any] | None = None,
    box_planning_overrides: dict[int, dict[str, Any]] | None = None,
    allowed_sides_by_box: dict[int, set[str]] | None = None,
    handoff_pose_resolver: Callable[
        [dict[str, Any], int, str, str, float, tuple[float, float, float]],
        list[float],
    ] | None = None,
    placement_box_center_resolver: Callable[
        [dict[str, Any], str, str], list[float]
    ] | None = None,
    place_joint_resolver: Callable[
        [dict[str, Any], str], list[float]
    ] | None = None,
    place_tcp_box_ids: set[int] | None = None,
    box_order: list[int] | None = None,
    active_box_ids: list[int] | set[int] | None = None,
    wall_specs: list[tuple[int, int, int, float, float, float]] | None = None,
    motion_quality_contract: dict[str, Any] | None = None,
    strict_verified_selection: bool = False,
    initial_tasks: list[dict[str, Any]] | None = None,
    task_completed_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    config = config or station_contract()
    wall_specs = list(wall_specs or scan.box_specs(
        CONTACT_X, BOX_DEPTH, BOX_WIDTH, BOX_HEIGHT
    ))
    specs_by_id = {spec[0]: spec for spec in wall_specs}
    if len(specs_by_id) != len(wall_specs) or not specs_by_id:
        raise ValueError("wall_specs must contain unique positive box ids")
    all_ids = set(specs_by_id)
    if min(all_ids) < 1:
        raise ValueError("wall_specs must contain unique positive box ids")
    active_ids = list(active_box_ids) if active_box_ids is not None else list(specs_by_id)
    if len(set(active_ids)) != len(active_ids) or not set(active_ids) <= all_ids:
        raise ValueError("active_box_ids must be unique ids present in wall_specs")
    order = list(box_order) if box_order is not None else active_ids
    if len(order) != len(active_ids) or set(order) != set(active_ids):
        raise ValueError("box_order must be a permutation of active_box_ids")
    if not 1 <= limit_boxes <= len(order):
        raise ValueError(f"limit_boxes must be in [1, {len(order)}]")
    args = planning_args(timeout, config, end_effector=end_effector)
    for name, value in (planning_overrides or {}).items():
        if not hasattr(args, name):
            raise ValueError(f"unknown planning override: {name}")
        setattr(args, name, value)
    for box_id, overrides in (box_planning_overrides or {}).items():
        if box_id not in all_ids:
            raise ValueError(f"invalid box planning override id: {box_id}")
        for name in overrides:
            if not hasattr(args, name):
                raise ValueError(f"unknown box planning override: {name}")
    for box_id, allowed_sides in (allowed_sides_by_box or {}).items():
        if box_id not in all_ids or not allowed_sides or not allowed_sides <= {"left", "right"}:
            raise ValueError(f"invalid allowed sides for box {box_id}: {allowed_sides}")
    place_tcp_box_ids = set(place_tcp_box_ids or ())
    if not place_tcp_box_ids <= all_ids:
        raise ValueError("place_tcp_box_ids must contain ids present in wall_specs")
    if place_tcp_box_ids and handoff_pose_resolver is None:
        raise ValueError("place_tcp_box_ids requires a handoff_pose_resolver")
    base_planning_values = vars(args).copy()
    verified_selection = VERIFIED_SELECTION if verified_selection is None else verified_selection
    top_suction_box_ids = (
        scan.DEFAULT_TOP_SUCTION_BOX_IDS
        if top_suction_box_ids is None
        else top_suction_box_ids
    )
    tasks: list[dict[str, Any]] = list(initial_tasks or [])
    removed: set[int] = all_ids - set(active_ids)
    removed.update(int(task["box_id"]) for task in tasks)
    cached_ids = [int(task["box_id"]) for task in tasks]
    if cached_ids != order[:len(cached_ids)] or len(tasks) > limit_boxes:
        raise ValueError("initial tasks must be a prefix of the requested box order")
    specs = [specs_by_id[box_id] for box_id in order[len(tasks):limit_boxes]]
    for box_id, row, column, x, y, z in specs:
        planning_attempt_results: list[dict[str, Any]] = []
        transition_attempt_results: list[dict[str, Any]] = []
        for name, value in base_planning_values.items():
            setattr(args, name, value)
        for name, value in (box_planning_overrides or {}).get(box_id, {}).items():
            setattr(args, name, value)
        args.continuous_plan_approach = False
        box_planning_values = vars(args).copy()
        mode = scan.grasp_mode_for_box(box_id, top_suction_box_ids)
        selected: dict[str, Any] | None = None
        attempt_serial = 0
        candidates = ordered_candidates(
            box_id, row, y, z, mode, verified_selection, args.box_grid_rows
        )
        if strict_verified_selection:
            candidates = candidates[:1]
        allowed_sides = (allowed_sides_by_box or {}).get(box_id)
        if allowed_sides:
            candidates = [candidate for candidate in candidates if candidate[0] in allowed_sides]
        for side, updown in candidates:
            for name in ("initial_left_arm_joints_deg", "initial_right_arm_joints_deg",
                         "top_initial_right_arm_joints_deg"):
                setattr(args, name, box_planning_values[name])
            if args.continuous_sequence and tasks:
                previous = tasks[-1]["payload"]
                previous_names = previous["joint_names"]
                positions = previous["frames"][-1]["joints"]
                if bool(args.continuous_seed_previous):
                    active_degrees = ",".join(
                        f"{math.degrees(positions[previous_names.index(f'{side}_joint{index}')]):.8f}"
                        for index in range(1, 8)
                    )
                    setattr(args, f"initial_{side}_arm_joints_deg", active_degrees)
                    if side == "right":
                        args.top_initial_right_arm_joints_deg = active_degrees
                opposite = "right" if side == "left" else "left"
                if updown >= float(args.place_updown_m):
                    opposite_degrees = ",".join(
                        f"{math.degrees(positions[previous_names.index(f'{opposite}_joint{index}')]):.8f}"
                        for index in range(1, 8)
                    )
                    setattr(args, f"initial_{opposite}_arm_joints_deg", opposite_degrees)
                    if opposite == "right":
                        args.top_initial_right_arm_joints_deg = opposite_degrees
            use_place_tcp_goal = box_id in place_tcp_box_ids
            place_joint_degrees = (
                [float(value) for value in place_joint_resolver(config, side)]
                if place_joint_resolver and not use_place_tcp_goal
                else []
            )
            if place_joint_degrees and len(place_joint_degrees) != 7:
                raise ValueError("named place pose must contain seven arm joints")
            args.place_arm_joints_deg = ",".join(
                f"{value:.8f}" for value in place_joint_degrees
            )
            pose = [] if place_joint_degrees else (
                handoff_pose_resolver(config, box_id, side, mode, updown, (x, y, z))
                if handoff_pose_resolver
                else handoff_tcp_pose(config, side, mode, updown)
            )
            args.place_tcp_pose = ",".join(f"{value:.8f}" for value in pose)
            conveyor_task = (
                placement_box_center_resolver is not None
                or bool(place_joint_degrees)
                or use_place_tcp_goal
            )
            low_transfer_task = box_id in {
                int(value)
                for value in (motion_quality_contract or {}).get("low_transfer_box_ids", [])
            }
            upper_front_task = (
                mode == "front" and row <= max(0, args.box_grid_rows - 2)
            )
            if conveyor_task:
                success_target = max(1, int(args.conveyor_success_trials))
                max_success_target = max(
                    success_target, int(args.conveyor_max_success_trials)
                )
            elif upper_front_task:
                success_target = max(1, int(args.upper_front_success_trials))
                max_success_target = max(
                    success_target, int(args.upper_front_max_success_trials)
                )
            else:
                success_target = 1
                max_success_target = 1
            required_success_target = 1 if strict_verified_selection else success_target

            def motion_accepted(motion: dict[str, Any]) -> bool:
                if motion_quality_contract is None:
                    return True
                if args.continuous_sequence and float(motion["carried_box_max_tilt_deg"]) > (
                    float(args.maximum_carried_box_tilt_deg) + 1e-6
                ):
                    return False
                if conveyor_task and mode != "top_suction" and not conveyor_motion_accepted(
                    motion, motion_quality_contract
                ):
                    return False
                if low_transfer_task and not low_transfer_motion_accepted(
                    motion["loaded_transfer"], motion_quality_contract
                ):
                    return False
                if mode == "top_suction" and not top_loaded_motion_accepted(
                    motion["loaded_transfer"], motion_quality_contract
                ):
                    return False
                if mode == "top_suction" and (
                    float(motion["total_joint_travel_deg"])
                    > float(motion_quality_contract["max_top_complete_total_joint_travel_deg"])
                    + 1e-6
                    or int(motion["direction_reversals"])
                    > int(motion_quality_contract["max_top_complete_direction_reversals"])
                ):
                    return False
                return not upper_front_task or upper_front_motion_accepted(
                    motion, motion_quality_contract
                )

            successful_results: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
            for attempt_index in range(max_success_target + rrt_retries):
                attempt_serial += 1
                base_seed = int(box_planning_values.get("planning_seed", 0))
                args.planning_seed = (
                    ((base_seed + (attempt_serial - 1) * 0x9E3779B9) & 0x7FFFFFFF) or 1
                    if base_seed > 0 else 0
                )
                result_path = scratch / f"box_{box_id:02d}_attempt_{attempt_serial:02d}.json"
                result = scan.run_attempt(
                    args,
                    box_id,
                    mode,
                    removed,
                    side,
                    updown,
                    (x, y, z),
                    result_json_path=result_path,
                )
                planning_attempt_results.append(result)
                print(
                    f"PLAN {box_id:02d}/{len(order)} {mode} {side} updown={updown:.2f} "
                    f"try={attempt_index + 1} "
                    f"{'SUCCESS' if result['success'] else result['failure_stage']}"
                    f"{'' if result['success'] else ': ' + str(result.get('failure_reason', ''))}",
                    flush=True,
                )
                payload = result.get("trajectory_result")
                if result["success"] and isinstance(payload, dict):
                    achieved_pose = [
                        float(value) for value in payload["achieved_place_tcp_pose"]
                    ]
                    target_pose = achieved_pose if place_joint_degrees else pose
                    check_handoff_result(payload, target_pose)
                    motion = active_arm_motion_metrics(payload, side)
                    if args.continuous_sequence:
                        orientation = carried_box_orientation_metrics(payload)
                        motion["carried_box_max_tilt_deg"] = orientation["max_tilt_deg"]
                        motion["selection_score"] += 10.0 * float(orientation["mean_tilt_deg"])
                    if conveyor_task:
                        motion["tool_down_error_deg"] = placed_tool_down_error_deg(payload)
                    if place_joint_degrees:
                        motion["named_place_joint_error_deg"] = named_place_joint_error_deg(
                            payload, side, place_joint_degrees
                        )
                    if use_place_tcp_goal:
                        motion["tcp_place_goal"] = True
                    if mode == "top_suction" or use_place_tcp_goal or low_transfer_task:
                        motion["loaded_transfer"] = active_arm_stage_motion_metrics(
                            payload, side, "rrt_to_place"
                        )
                    successful_results.append((result, payload, motion))
                    quality_reached = any(
                        motion_accepted(item[2]) for item in successful_results
                    )
                    if len(successful_results) >= success_target and quality_reached:
                        break
                    if len(successful_results) >= max_success_target:
                        break
                    continue
                if not retryable_failure(result) and not strict_verified_selection:
                    break

            eligible_results = successful_results
            if motion_quality_contract is not None and (conveyor_task or upper_front_task):
                if len(successful_results) < required_success_target:
                    print(
                        f"QUALITY_REJECT {box_id:02d}/{len(order)} {side} updown={updown:.2f} "
                        f"successes={len(successful_results)}<{required_success_target}",
                        flush=True,
                    )
                    continue
                eligible_results = [
                    item for item in successful_results
                    if motion_accepted(item[2])
                ]
                if not eligible_results:
                    best_motion = min(
                        (item[2] for item in successful_results),
                        key=lambda item: float(item["selection_score"]),
                    )
                    print(
                        f"QUALITY_REJECT {box_id:02d}/{len(order)} {side} updown={updown:.2f} "
                        f"successes={len(successful_results)} "
                        f"travel={best_motion['total_joint_travel_deg']:.1f}deg "
                        f"excess={best_motion['stage_excess_joint_travel_deg']:.1f}deg "
                        f"reversals={best_motion['direction_reversals']} "
                        f"range={best_motion['max_joint_range_deg']:.1f}deg "
                        f"tilt={best_motion.get('carried_box_max_tilt_deg', 0.0):.1f}deg "
                        f"loaded={best_motion.get('loaded_transfer', {}).get('total_joint_travel_deg', 0.0):.1f}deg",
                        flush=True,
                    )
                    continue
            for result, payload, motion in sorted(
                eligible_results, key=lambda item: float(item[2]["selection_score"])
            ):
                transition_frames: list[dict[str, Any]] = []
                transition_motion: dict[str, Any] = {}
                transition_ok = True
                if tasks or not initial_tasks:
                    # Per-box transfer overrides apply to the loaded task only.
                    # Empty inter-box transitions retain the station's Cartesian search policy.
                    args.cartesian_transfer_search_enabled = base_planning_values[
                        "cartesian_transfer_search_enabled"
                    ]
                    previous = tasks[-1]["payload"] if tasks else None
                    if previous and previous["joint_names"] != payload["joint_names"]:
                        raise RuntimeError("joint contract changed between box tasks")
                    from_joints = (
                        previous["frames"][-1]["joints"] if previous else folded_start_joints(config)
                    )
                    from_label = (
                        "folded" if previous is None else f"{tasks[-1]['box_id']:02d}"
                    )
                    transition_ok = False
                    target_joints = list(payload["frames"][0]["joints"])
                    opposite_side = "right" if side == "left" else "left"
                    names = payload["joint_names"]
                    intermediate = list(from_joints)
                    for joint_index in range(1, 8):
                        index = names.index(f"{opposite_side}_joint{joint_index}")
                        intermediate[index] = target_joints[index]
                    inactive_changes = any(
                        abs(float(left) - float(right)) > 1e-6
                        for left, right in zip(from_joints, intermediate)
                    )
                    legs = []
                    if inactive_changes:
                        legs.append((opposite_side, list(from_joints), intermediate))
                    legs.append((side, intermediate if inactive_changes else list(from_joints), target_joints))
                    transition_groups: list[str] = []
                    transition_strategies: list[str] = []
                    transition_diagnostics: list[str] = []
                    transition_tcp_legs: list[dict[str, Any]] = []
                    transition_search_legs: list[dict[str, Any]] = []
                    leg_frames: list[dict[str, Any]] = []
                    legs_ok = True
                    for leg_index, (leg_side, leg_from, leg_to) in enumerate(legs):
                        leg_ok = False
                        transition_retries = rrt_retries + (3 if strict_verified_selection else 0)
                        first_transition_search = (
                            previous is None
                            and leg_index == len(legs) - 1
                            and motion_quality_contract is not None
                            and {
                                "max_first_transition_tcp_path_m",
                                "max_first_transition_tcp_line_deviation_m",
                                "max_first_transition_tcp_orientation_travel_deg",
                            } <= set(motion_quality_contract)
                        )
                        if first_transition_search:
                            args.shortcut_repair_rrt_enabled = False
                            transition_retries = max(transition_retries, 23)
                        leg_candidates: list[
                            tuple[float, list[dict[str, Any]], dict[str, Any], dict[str, Any]]
                        ] = []
                        for transition_retry in range(transition_retries + 1):
                            attempt_serial += 1
                            base_seed = int(box_planning_values.get("planning_seed", 0))
                            args.planning_seed = (
                                ((base_seed + (attempt_serial - 1) * 0x9E3779B9) & 0x7FFFFFFF) or 1
                                if base_seed > 0 else 0
                            )
                            transition = scan.run_attempt(
                                args,
                                box_id,
                                mode,
                                removed,
                                leg_side,
                                updown,
                                (x, y, z),
                                result_json_path=scratch /
                                f"box_{box_id:02d}_transition_{attempt_serial:02d}.json",
                                transition_from_joints=leg_from,
                                transition_to_joints=leg_to,
                            )
                            transition_attempt_results.append(transition)
                            if not transition["success"]:
                                print(
                                    f"TRANSITION {from_label}->{box_id:02d} "
                                    f"leg={leg_index + 1}/{len(legs)} {leg_side} "
                                    f"{transition['failure_stage']} {transition['failure_reason']}",
                                    flush=True,
                                )
                                if not retryable_failure(transition):
                                    break
                                continue
                            transition_payload = transition.get("trajectory_result")
                            if not isinstance(transition_payload, dict):
                                raise RuntimeError("transition did not produce replay frames")
                            expected_group = f"{leg_side}_arm_with_updown"
                            actual_group = str(transition_payload.get("transition_group", ""))
                            if actual_group != expected_group:
                                raise RuntimeError(
                                    f"transition expected {expected_group}, got {actual_group}"
                                )
                            frames = list(transition_payload["frames"])
                            leg_motion = dual_arm_motion_metrics({
                                "joint_names": names,
                                "frames": frames,
                            })
                            frozen_side = "right" if leg_side == "left" else "left"
                            frozen_tolerance = float(
                                (motion_quality_contract or {}).get(
                                    "max_frozen_arm_transition_travel_deg", 0.01
                                )
                            )
                            if float(leg_motion[f"{frozen_side}_arm_joint_travel_deg"]) > (
                                frozen_tolerance + 1e-6
                            ):
                                raise RuntimeError("non-working arm moved during a frozen transition leg")
                            tcp_motion = transition_payload.get("transition_tcp_motion", {})
                            if first_transition_search and not isinstance(tcp_motion, dict):
                                continue
                            tcp_path = float(tcp_motion.get("path_length_m", 0.0))
                            tcp_deviation = float(tcp_motion.get("max_line_deviation_m", 0.0))
                            tcp_orientation = float(
                                tcp_motion.get("orientation_travel_deg", 0.0)
                            )
                            tcp_score = (
                                tcp_path
                                + 0.15 * math.radians(tcp_orientation)
                                + 0.5 * tcp_deviation
                                + 0.005 * float(leg_motion["max_joint_winding_excess_deg"])
                            )
                            leg_candidates.append((tcp_score, frames, transition_payload, leg_motion))
                            if first_transition_search:
                                print(
                                    f"TRANSITION_TCP folded->{box_id:02d} "
                                    f"path={tcp_path:.3f}m deviation={tcp_deviation:.3f}m "
                                    f"orientation={tcp_orientation:.1f}deg "
                                    f"winding={float(leg_motion['max_joint_winding_excess_deg']):.1f}deg",
                                    flush=True,
                                )
                                contract = motion_quality_contract or {}
                                eligible_first = [
                                    item for item in leg_candidates
                                    if float(item[2]["transition_tcp_motion"]["path_length_m"])
                                    <= float(contract.get(
                                        "max_first_transition_tcp_path_m", 3.6
                                    )) + 1e-6
                                    and float(item[2]["transition_tcp_motion"][
                                        "max_line_deviation_m"
                                    ]) <= float(contract.get(
                                        "max_first_transition_tcp_line_deviation_m", 0.90
                                    )) + 1e-6
                                    and float(item[2]["transition_tcp_motion"][
                                        "orientation_travel_deg"
                                    ]) <= float(contract.get(
                                        "max_first_transition_tcp_orientation_travel_deg", 550.0
                                    )) + 1e-6
                                    and float(item[3]["max_joint_winding_excess_deg"])
                                    <= float(contract.get(
                                        "max_first_transition_joint_winding_excess_deg", 100.0
                                    )) + 1e-6
                                ]
                                if len(leg_candidates) < 3 or not eligible_first:
                                    continue
                                _, frames, transition_payload, leg_motion = min(
                                    eligible_first, key=lambda item: item[0]
                                )
                            transition_groups.append(actual_group)
                            transition_strategies.append(
                                str(transition_payload.get("transition_strategy", "unknown"))
                            )
                            transition_diagnostics.append(
                                str(transition_payload.get("transition_diagnostic", ""))
                            )
                            if isinstance(transition_payload.get("transition_tcp_motion"), dict):
                                transition_tcp_legs.append({
                                    "side": leg_side,
                                    "strategy": transition_strategies[-1],
                                    **transition_payload["transition_tcp_motion"],
                                })
                            transition_search_legs.append({
                                "side": leg_side,
                                "segments": transition_payload.get("segment_search", []),
                                "total_ms": transition_payload.get("total_ms", 0.0),
                            })
                            leg_frames.extend(frames if not leg_frames else frames[1:])
                            leg_ok = True
                            break
                        if first_transition_search and not leg_ok and leg_candidates:
                            eligible = [
                                item for item in leg_candidates
                                if float(item[2]["transition_tcp_motion"]["path_length_m"])
                                <= float((motion_quality_contract or {})[
                                    "max_first_transition_tcp_path_m"
                                ]) + 1e-6
                                and float(item[2]["transition_tcp_motion"][
                                    "max_line_deviation_m"
                                ]) <= float((motion_quality_contract or {})[
                                    "max_first_transition_tcp_line_deviation_m"
                                ]) + 1e-6
                                and float(item[2]["transition_tcp_motion"][
                                    "orientation_travel_deg"
                                ]) <= float((motion_quality_contract or {})[
                                    "max_first_transition_tcp_orientation_travel_deg"
                                ]) + 1e-6
                                and float(item[3]["max_joint_winding_excess_deg"])
                                <= float((motion_quality_contract or {})[
                                    "max_first_transition_joint_winding_excess_deg"
                                ]) + 1e-6
                            ]
                            if eligible:
                                _, frames, transition_payload, leg_motion = min(
                                    eligible, key=lambda item: item[0]
                                )
                                actual_group = str(transition_payload["transition_group"])
                                transition_groups.append(actual_group)
                                transition_strategies.append(str(
                                    transition_payload.get("transition_strategy", "unknown")
                                ))
                                transition_diagnostics.append(str(
                                    transition_payload.get("transition_diagnostic", "")
                                ))
                                transition_tcp_legs.append({
                                    "side": leg_side,
                                    "strategy": transition_strategies[-1],
                                    **transition_payload["transition_tcp_motion"],
                                })
                                transition_search_legs.append({
                                    "side": leg_side,
                                    "segments": transition_payload.get("segment_search", []),
                                    "total_ms": transition_payload.get("total_ms", 0.0),
                                })
                                leg_frames.extend(frames if not leg_frames else frames[1:])
                                leg_ok = True
                        if not leg_ok:
                            legs_ok = False
                            break
                    if legs_ok:
                        transition_frames = leg_frames
                        transition_motion = dual_arm_motion_metrics({
                            "joint_names": names,
                            "frames": transition_frames,
                        })
                        transition_motion["planning_groups"] = transition_groups
                        transition_motion["planning_strategies"] = transition_strategies
                        transition_motion["planning_diagnostics"] = transition_diagnostics
                        transition_motion["tcp_legs"] = transition_tcp_legs
                        transition_motion["search_legs"] = transition_search_legs
                        transition_motion["active_arm_joint_travel_deg"] = transition_motion[
                            f"{side}_arm_joint_travel_deg"
                        ]
                        transition_motion["inactive_arm_joint_travel_deg"] = transition_motion[
                            f"{'right' if side == 'left' else 'left'}_arm_joint_travel_deg"
                        ]
                        transition_motion["simultaneous_dual_arm_motion"] = False
                        if motion_quality_contract is not None and not transition_motion_accepted(
                            transition_motion, motion_quality_contract
                        ):
                            print(
                                f"TRANSITION_QUALITY {from_label}->{box_id:02d} "
                                f"frames={transition_motion['frame_count']} "
                                f"travel={transition_motion['total_joint_travel_deg']:.1f}deg "
                                f"excess={transition_motion['stage_excess_joint_travel_deg']:.1f}deg "
                                f"reversals={transition_motion['direction_reversals']} "
                                f"range={transition_motion['max_joint_range_deg']:.1f}deg",
                                flush=True,
                            )
                        else:
                            transition_ok = True
                if not transition_ok:
                    continue
                if len(transition_frames[0]["joints"]) != len(from_joints) or any(
                    abs(float(left) - float(right)) > 1e-5
                    for left, right in zip(transition_frames[0]["joints"], from_joints)
                ):
                    raise RuntimeError("transition must start at the previous release state")
                selected = {
                    "box_id": box_id,
                    "row": row,
                    "column": column,
                    "center": [x, y, z],
                    "mode": mode,
                    "side": side,
                    "updown": updown,
                    "handoff_tcp_pose": [
                        float(value) for value in payload["achieved_place_tcp_pose"]
                    ] if place_joint_degrees else pose,
                    **(
                        {"place_arm_joints_deg": place_joint_degrees}
                        if place_joint_degrees
                        else {}
                    ),
                    **(
                        {
                            "placement_box_center_target": placement_box_center_resolver(
                                config, side, mode
                            )
                        }
                        if placement_box_center_resolver
                        else (
                            {"placement_box_center_target": placed_box_center(payload)}
                            if conveyor_task
                            else {}
                        )
                    ),
                    "transition_frames": transition_frames,
                    "payload": payload,
                    "motion_selection": motion,
                    "carried_box_orientation": (
                        carried_box_orientation_metrics(payload) if args.continuous_sequence else {}
                    ),
                    "transition_motion": transition_motion,
                    "transition_planning_groups": transition_motion.get("planning_groups", []),
                    "planning_search": {
                        "task": planner_search_metrics(planning_attempt_results),
                        "transition": planner_search_metrics(transition_attempt_results),
                        "selected_task": planner_search_metrics([result]),
                    },
                }
                print(
                    f"SELECT {box_id:02d}/{len(order)} score={motion['selection_score']:.1f} "
                    f"travel={motion['total_joint_travel_deg']:.1f}deg "
                    f"reversals={motion['direction_reversals']}",
                    flush=True,
                )
                break
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError(f"box {box_id} has no successful complete trajectory")
        tasks.append(selected)
        removed.add(box_id)
        if task_completed_callback is not None:
            task_completed_callback(tasks)
    return tasks


def matrix_to_quaternion(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            values = (
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            values = (
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            values = (
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
    return [float(value) for value in values]


class SequenceRecorder:
    def __init__(
        self,
        save: Path,
        playback_speed: float,
        *,
        end_effector: str = "legacy_suction",
        application_id: str = "v3_x075_5x5_grasp_sequence",
        view_name: str = "V3 X=0.75m 5x5 warehouse grasp",
        task_title: str = "5×5 sequential grasp",
        tool_name: str = "suction",
        total_boxes: int = 25,
        box_ids: list[int] | None = None,
        wall_specs: list[tuple[int, int, int, float, float, float]] | None = None,
        box_size: tuple[float, float, float] = (BOX_DEPTH, BOX_WIDTH, BOX_HEIGHT),
        minimum_frame_interval_s: float = 1.0 / 30.0,
        transition_joint_speed_deg_s: float = math.degrees(1.0),
        task_joint_speed_deg_s: float = 40.0,
        cartesian_joint_speed_deg_s: float = 40.0,
    ) -> None:
        prefer_matching_rerun_cli()
        save.parent.mkdir(parents=True, exist_ok=True)
        save.unlink(missing_ok=True)
        rr.init(application_id, spawn=False)
        rr.save(str(save))
        self.robot = UrdfRobot(render_current_urdf({"end_effector": end_effector}))
        log_robot_static_model(self.robot, "world/robot", log_meshes=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            "world/ground",
            rr.Boxes3D(
                centers=[[0.0, 0.0, -0.05]],
                half_sizes=[[3.0, 3.0, 0.05]],
                colors=[[82, 88, 96, 210]],
                show_labels=False,
            ),
            static=True,
        )
        panels = warehouse_panels()
        rr.log(
            "world/warehouse",
            rr.Boxes3D(
                centers=[panel["center"] for panel in panels],
                half_sizes=[
                    (np.asarray(panel["size"], dtype=float) * 0.5).tolist()
                    for panel in panels
                ],
                colors=[
                    [64, 135, 188, 52],
                    [64, 135, 188, 52],
                    [95, 150, 190, 40],
                    [64, 135, 188, 52],
                ],
                show_labels=False,
            ),
            static=True,
        )
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="/world",
                        contents=["/world/**"],
                        name=view_name,
                    ),
                    rrb.Vertical(
                        rrb.TextDocumentView(
                            origin="/transition_status", name="空载过渡"
                        ),
                        rrb.TimeSeriesView(
                            origin="/metrics/transition", name="空载过渡时间 (s)"
                        ),
                        rrb.TextDocumentView(origin="/status", name="Task status"),
                        row_shares=[0.34, 0.28, 0.38],
                    ),
                    column_shares=[0.73, 0.27],
                ),
                collapse_panels=True,
            )
        )
        self.total_boxes = total_boxes
        selected_ids = set(box_ids) if box_ids is not None else set(range(1, total_boxes + 1))
        self.centers = {
            box_id: [x, y, z]
            for box_id, _, _, x, y, z in (
                wall_specs or scan.box_specs(CONTACT_X, BOX_DEPTH, BOX_WIDTH, BOX_HEIGHT)
            )
            if box_id in selected_ids
        }
        self.box_size = box_size
        self.remaining = set(self.centers)
        self.playback_speed = max(0.1, playback_speed)
        self.timeline_s = 0.0
        self.last_joints: dict[str, float] | None = None
        self.task_title = task_title
        self.tool_name = tool_name
        self.minimum_frame_interval_s = minimum_frame_interval_s
        self.transition_joint_speed_rad_s = math.radians(transition_joint_speed_deg_s)
        self.task_joint_speed_deg_s = task_joint_speed_deg_s
        self.cartesian_joint_speed_deg_s = cartesian_joint_speed_deg_s
        self.last_box_id: int | None = None
        for name, label, color in (
            ("total_search_s", "累计搜索（含失败重试）", [194, 91, 48]),
            ("selected_core_s", "最终采用路径核心规划", [8, 121, 112]),
        ):
            rr.log(
                f"metrics/transition/{name}",
                rr.SeriesLines(colors=[color], names=[label]),
                static=True,
            )

    def set_time(self) -> None:
        rr.set_time("sequence_time", duration=self.timeline_s)

    def advance(self, seconds: float) -> None:
        duration = max(0.0, seconds) / self.playback_speed
        self.timeline_s += duration

    def log_boxes(self, target_id: int | None = None) -> None:
        regular_ids = sorted(self.remaining - ({target_id} if target_id else set()))
        if regular_ids:
            rr.log(
                "world/boxes/remaining",
                rr.Boxes3D(
                    centers=[self.centers[box_id] for box_id in regular_ids],
                    half_sizes=[[value / 2.0 for value in self.box_size]],
                    colors=[[238, 142, 48, 180]],
                    show_labels=False,
                ),
            )
        else:
            rr.log("world/boxes/remaining", rr.Clear(recursive=True))
        if target_id is not None and target_id in self.remaining:
            rr.log(
                "world/boxes/target",
                rr.Boxes3D(
                    centers=[self.centers[target_id]],
                    half_sizes=[[value / 2.0 for value in self.box_size]],
                    colors=[[50, 220, 90, 230]],
                    labels=[f"target {target_id}"],
                    show_labels=True,
                ),
            )
        else:
            rr.log("world/boxes/target", rr.Clear(recursive=True))

    def log_status(self, task: dict[str, Any], completed: int, stage: str) -> None:
        direction = "top-down" if task["mode"] == "top_suction" else "front"
        mode = f"{self.tool_name} {direction}"
        analysis = task.get("trajectory_analysis", {})
        search = task.get("planning_search", {})
        task_search = search.get("task", {})
        transition_search = search.get("transition", {})
        strategies = [
            str(item.get("strategy", ""))
            for item in task.get("payload", {}).get("segment_search", [])
            if item.get("success")
        ]
        repaired_joints = sorted({
            str(name)
            for item in task.get("payload", {}).get("segment_search", [])
            for name in item.get("repaired_joints", [])
            if item.get("success") and item.get("strategy") == "shortcut_repair_rrt"
        })
        maximum_tilt = task.get("carried_box_orientation", {}).get("max_tilt_deg", 0.0)
        rr.log(
            "status",
            rr.TextDocument(
                f"# {self.task_title}\n\n"
                f"- Completed: **{completed}/{self.total_boxes}**\n"
                f"- Current box: **{task['box_id']}**\n"
                f"- Mode: `{mode}`\n"
                f"- Arm: `{task['side']}`\n"
                f"- Updown: `{task['updown']:.2f} m`\n"
                f"- Stage: `{stage}`\n"
                f"- Search: **{float(analysis.get('search_wall_ms', 0.0)) / 1000:.2f}s** "
                f"({task_search.get('attempts', 0)} task, "
                f"{transition_search.get('attempts', 0)} transition)\n"
                f"- Method: `{', '.join(strategies) or 'direct'}`\n"
                f"- Repair joints: `{', '.join(repaired_joints) or 'none'}`\n"
                f"- TCP path: **{float(analysis.get('tcp_path_length_m', 0.0)):.2f}m**\n"
                f"- Joint travel / reversals: **{task.get('motion_selection', {}).get('total_joint_travel_deg', 0.0):.0f}°** "
                f"/ **{analysis.get('total_joint_reversals', 0)}**\n"
                f"- Joint flip events: **{analysis.get('joint_flip_events', 0)}**\n"
                f"- Carried tilt: **{maximum_tilt:.1f}°**\n"
                "- Warehouse: `2.38 × 2.38 × 2.35 m`",
                media_type=rr.MediaType.MARKDOWN,
            ),
        )

    def log_transition_status(self, task: dict[str, Any]) -> None:
        transition = task.get("transition_motion", {})
        search = task.get("planning_search", {}).get("transition", {})
        from_label = "折叠初始位" if self.last_box_id is None else f"{self.last_box_id}号箱"
        to_label = f"{task['box_id']}号箱"
        attempts = int(search.get("attempts", 0))
        successes = int(search.get("successes", 0))
        total_seconds = float(search.get("process_wall_ms", 0.0)) / 1000.0
        search_legs = transition.get("search_legs", [])
        selected_seconds = sum(
            float(leg.get("total_ms", 0.0)) for leg in search_legs
        ) / 1000.0
        strategies = [str(value) for value in transition.get("planning_strategies", [])]
        repairs = [
            segment
            for leg in search_legs
            for segment in leg.get("segments", [])
            if segment.get("success") and segment.get("strategy") == "shortcut_repair_rrt"
        ]
        repair_text = "无"
        if repairs:
            repair_text = "；".join(
                f"{','.join(map(str, item.get('repaired_joints', [])))} "
                f"{float(item.get('shortcut_repair_rrt_ms', 0.0)):.1f}ms"
                for item in repairs
            )
        tcp_legs = transition.get("tcp_legs", [])
        tcp_text = " / ".join(
            f"{float(leg.get('path_length_m', 0.0)):.2f}m"
            for leg in tcp_legs
        ) or "未记录"
        rr.log(
            "transition_status",
            rr.TextDocument(
                f"# {from_label} → {to_label}\n\n"
                f"- 累计搜索：**{total_seconds:.2f}s**\n"
                f"- 最终路径核心规划：**{selected_seconds:.2f}s**\n"
                f"- 尝试：**{attempts}**（成功 {successes}）\n"
                f"- 采用策略：`{' → '.join(strategies) or 'direct'}`\n"
                f"- 局部关节修补：`{repair_text}`\n"
                f"- TCP 路径：**{tcp_text}**\n"
                f"- 过渡帧：**{int(transition.get('frame_count', 0))}**\n"
                f"- 双臂总行程：**{float(transition.get('total_joint_travel_deg', 0.0)):.1f}°**\n"
                f"- 换向：**{int(transition.get('direction_reversals', 0))}**",
                media_type=rr.MediaType.MARKDOWN,
            ),
        )
        rr.log("metrics/transition/total_search_s", rr.Scalars([total_seconds]))
        rr.log("metrics/transition/selected_core_s", rr.Scalars([selected_seconds]))

    def log_carried_box(self, task: dict[str, Any], joints: dict[str, float]) -> None:
        payload = task["payload"]
        transforms = self.robot.fk(joints)
        tool_transform = transforms.get(str(payload["tool_link"]))
        if tool_transform is None:
            raise RuntimeError(f"missing FK for {payload['tool_link']}")
        offset = np.asarray(payload["tool_to_box_center"], dtype=float)
        transform = tool_transform.copy()
        transform[:3, 3] = tool_transform[:3, 3] + tool_transform[:3, :3] @ offset
        tool_to_box_rotation = np.asarray(
            payload.get("tool_to_box_rotation", np.eye(3)), dtype=float
        )
        if tool_to_box_rotation.shape != (3, 3):
            raise ValueError("tool_to_box_rotation must be a 3x3 matrix")
        transform[:3, :3] = tool_transform[:3, :3] @ tool_to_box_rotation
        size = np.asarray(payload["carried_box_size_tool"], dtype=float)
        rr.log(
            "world/boxes/carried",
            rr.Boxes3D(
                centers=[transform[:3, 3].tolist()],
                half_sizes=[(size * 0.5).tolist()],
                quaternions=[matrix_to_quaternion(transform[:3, :3])],
                colors=[[35, 185, 250, 235]],
                labels=[f"carried box {task['box_id']}"],
            ),
        )

    def transition_to(self, target: dict[str, float], task: dict[str, Any], completed: int) -> None:
        names = task["payload"]["joint_names"]
        frames = task.get("transition_frames", [])
        if not frames:
            raise RuntimeError(f"missing validated transition into box {task['box_id']}")
        if self.last_joints is None:
            self.last_joints = {
                name: float(value) for name, value in zip(names, frames[0]["joints"])
            }
            self.set_time()
            log_robot_state(self.robot, self.last_joints, "world/robot")
            self.log_boxes(task["box_id"])
            self.log_status(task, completed, "folded_ready")
            self.advance(0.40)
        for frame in frames:
            joints = {name: float(value) for name, value in zip(names, frame["joints"])}
            if frame["box_attached"]:
                raise RuntimeError("a task transition must not carry a box")
            self.set_time()
            log_robot_state(self.robot, joints, "world/robot")
            self.log_boxes(task["box_id"])
            self.log_status(task, completed, "between_boxes")
            delta = max(abs(joints[name] - self.last_joints[name]) for name in names)
            self.advance(max(
                self.minimum_frame_interval_s,
                min(0.15, delta / self.transition_joint_speed_rad_s),
            ))
            self.last_joints = joints
        if max(abs(self.last_joints[name] - target[name]) for name in names) > 1e-5:
            raise RuntimeError(f"transition into box {task['box_id']} missed its goal")

    def play_task(self, task: dict[str, Any], completed: int) -> None:
        payload = task["payload"]
        names = [str(name) for name in payload["joint_names"]]
        frames = list(payload["frames"])
        if not frames:
            raise RuntimeError(f"box {task['box_id']} has no replay frames")
        first = {name: float(value) for name, value in zip(names, frames[0]["joints"])}
        self.set_time()
        self.log_transition_status(task)
        analysis = task.get("trajectory_analysis", {})
        search = task.get("planning_search", {})
        task_seconds = float(search.get("task", {}).get("process_wall_ms", 0.0)) / 1000.0
        transition_seconds = float(search.get("transition", {}).get("process_wall_ms", 0.0)) / 1000.0
        rr.log(
            "metrics/motion/reversals",
            rr.Scalars([float(analysis.get("total_joint_reversals", 0))]),
        )
        rr.log(
            "metrics/motion/carried_tilt_deg",
            rr.Scalars([float(task.get("carried_box_orientation", {}).get("max_tilt_deg", 0.0))]),
        )
        stage_colors = {
            "cartesian_approach": [35, 170, 114],
            "cartesian_retreat": [235, 164, 51],
            "rrt_to_place": [205, 76, 66],
        }
        stage_paths = analysis.get("tcp_stage_positions", {})
        visible_paths = [
            (stage, positions) for stage, positions in stage_paths.items()
            if len(positions) >= 2
        ]
        if visible_paths:
            rr.log(
                "world/tcp_paths",
                rr.LineStrips3D(
                    strips=[positions for _, positions in visible_paths],
                    colors=[stage_colors.get(stage, [55, 148, 190]) for stage, _ in visible_paths],
                    radii=[0.003] * len(visible_paths),
                    labels=[stage for stage, _ in visible_paths],
                    show_labels=False,
                ),
            )
        rr.log(
            "world/handoff_tcp",
            rr.Points3D(
                positions=[task["handoff_tcp_pose"][:3]],
                colors=[[35, 185, 250, 240]],
                radii=[0.025],
                labels=["handoff TCP"],
            ),
        )
        self.transition_to(first, task, completed)
        target_detached = False
        previous_stage = ""
        previous = self.last_joints
        for frame in frames:
            joints = {name: float(value) for name, value in zip(names, frame["joints"])}
            stage = str(frame["stage"])
            attached = bool(frame["box_attached"])
            self.set_time()
            log_robot_state(self.robot, joints, "world/robot")
            if attached:
                if not target_detached:
                    self.remaining.remove(task["box_id"])
                    target_detached = True
                    self.log_boxes()
                self.log_carried_box(task, joints)
            else:
                self.log_boxes(task["box_id"])
                rr.log("world/boxes/carried", rr.Clear(recursive=True))
            self.log_status(task, completed, stage)
            revolute_delta = 0.0
            if previous is not None:
                revolute_delta = max(
                    (
                        abs(math.degrees(math.atan2(math.sin(value - previous.get(name, value)), math.cos(value - previous.get(name, value)))))
                        for name, value in joints.items()
                        if name != "updown"
                    ),
                    default=0.0,
                )
            joint_speed = (
                self.cartesian_joint_speed_deg_s
                if stage in {"cartesian_approach", "cartesian_retreat"}
                else self.task_joint_speed_deg_s
            )
            delay = max(self.minimum_frame_interval_s, revolute_delta / joint_speed)
            if previous_stage and stage != previous_stage:
                delay = max(delay, 0.25)
            self.advance(min(delay, 0.20))
            previous = joints
            previous_stage = stage
        self.last_joints = previous
        self.last_box_id = int(task["box_id"])
        self.set_time()
        rr.log("world/boxes/carried", rr.Clear(recursive=True))
        self.log_boxes()
        self.log_status(task, completed + 1, "box_removed")
        self.advance(0.35)

    def finish(self, save: Path) -> None:
        self.set_time()
        rr.log("world/boxes/carried", rr.Clear(recursive=True))
        rr.log("world/boxes/target", rr.Clear(recursive=True))
        self.log_boxes()
        rr.log(
            "status",
            rr.TextDocument(
                f"# {self.task_title}\n\n"
                f"- Completed: **{self.total_boxes}/{self.total_boxes}**\n"
                "- Remaining boxes: **0**\n"
                "- Result: **SUCCESS**",
                media_type=rr.MediaType.MARKDOWN,
            ),
        )
        self.advance(0.5)
        rr.disconnect()
        print(
            f"RESULT success={self.total_boxes}/{self.total_boxes} recording={save}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, default=default_recording_path())
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--playback-speed", type=float, default=2.0)
    parser.add_argument("--planning-timeout", type=float, default=35.0)
    parser.add_argument("--rrt-retries", type=int, default=2)
    args = parser.parse_args()
    if args.playback_speed <= 0.0 or args.planning_timeout <= 0.0:
        parser.error("playback speed and planning timeout must be positive")
    if args.rrt_retries < 0:
        parser.error("rrt-retries must be non-negative")

    with tempfile.TemporaryDirectory(prefix="v3_5x5_sequence_") as temporary:
        tasks = plan_sequence(args.planning_timeout, args.rrt_retries, Path(temporary))
        recorder = SequenceRecorder(args.save.resolve(), args.playback_speed)
        for completed, task in enumerate(tasks):
            recorder.play_task(task, completed)
        recorder.finish(args.save.resolve())
        print(f"Summary: {write_sequence_summary(tasks, args.save.resolve())}", flush=True)
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
