#!/usr/bin/env python3
"""Validate the sequential mixed-grasp V3 task on a configurable box wall."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import selectors
import signal
import subprocess
import time
from pathlib import Path


SUCCESS_RE = re.compile(r"calculation completed: SUCCESS total=([0-9.]+)ms")
FAIL_RE = re.compile(
    r"calculation completed: FAILED total=([0-9.]+)ms stage=([^ ]+) reason=(.*)"
)
DEFAULT_TOP_SUCTION_BOX_IDS = {18, 21, 22, 23, 24, 25}


def box_specs(
    contact_x: float,
    depth: float,
    width: float,
    height: float,
    rows: int = 5,
    columns: int = 5,
    center_y: float = 0.0,
    bottom_z: float = 0.0,
):
    center_x = contact_x + 0.5 * depth
    box_id = 0
    for row_from_top in range(rows):
        z = bottom_z + (rows - row_from_top - 0.5) * height
        for column_from_left in range(columns):
            box_id += 1
            y = center_y + ((columns - 1) / 2.0 - column_from_left) * width
            yield box_id, row_from_top + 1, column_from_left + 1, center_x, y, z


def parse_box_ids(value: str) -> set[int]:
    ids = {int(item.strip()) for item in value.split(",") if item.strip()}
    invalid = sorted(item for item in ids if item < 1)
    if invalid:
        raise argparse.ArgumentTypeError(f"box ids must be positive: {invalid}")
    return ids


def grasp_mode_for_box(box_id: int, top_suction_box_ids: set[int]) -> str:
    return "top_suction" if box_id in top_suction_box_ids else "front"


def front_suction_offsets(
    box_id: int,
    center_y: float,
    side: str,
    grasp_mode: str,
    center_y_offset: float,
    center_z_offset: float,
    bottom_z_offset: float,
    rows: int = 5,
    columns: int = 5,
    grid_center_y: float = 0.0,
) -> tuple[float, float]:
    if grasp_mode != "front":
        return 0.0, 0.0
    y_offset = 0.0
    z_offset = 0.0
    if abs(center_y - grid_center_y) <= 1.0e-9:
        y_offset = center_y_offset if side == "left" else -center_y_offset
        z_offset = center_z_offset
    if (box_id - 1) // columns + 1 == rows:
        z_offset = bottom_z_offset
    return y_offset, z_offset


def candidate_order(
    row_from_top: int,
    y: float,
    z: float,
    updown_step: float,
    grasp_mode: str,
    rows: int = 5,
) -> list[tuple[str, float]]:
    preferred_side = "left" if y > 1.0e-9 else "right"
    sides = [preferred_side, "right" if preferred_side == "left" else "left"]
    if abs(y) > 1.0e-9:
        sides = [
            preferred_side,
            "left" if preferred_side == "right" else "right",
        ]
    heights = [-index * updown_step for index in range(int(round(1.0 / updown_step)) + 1)]
    ideal = min(0.0, max(-1.0, z - 1.3))
    heights.sort(key=lambda value: (abs(value - ideal), abs(value)))
    if row_from_top <= max(0, rows - 2) and grasp_mode == "front":
        heights.remove(0.0)
        heights.insert(0, 0.0)
        return [(side, height_value) for height_value in heights for side in sides]
    return [(side, height_value) for side in sides for height_value in heights]


def box_geometry_launch_arguments(args) -> list[str]:
    values = (float(args.box_depth), float(args.box_width), float(args.box_height))
    if values == (0.30, 0.40, 0.40):
        return []
    return [
        f"box_depth:={values[0]:.6f}",
        f"box_width:={values[1]:.6f}",
        f"box_height:={values[2]:.6f}",
    ]


def run_attempt(
    args,
    box_id: int,
    grasp_mode: str,
    removed_box_ids: set[int],
    side: str,
    updown: float,
    center: tuple[float, float, float],
    result_json_path: Path | None = None,
    *,
    transition_from_joints: list[float] | None = None,
    transition_to_joints: list[float] | None = None,
):
    is_transition = transition_from_joints is not None or transition_to_joints is not None
    if is_transition and (transition_from_joints is None or transition_to_joints is None):
        raise ValueError("transition requires both joint states")
    front_y_offset, front_z_offset = front_suction_offsets(
        box_id,
        center[1],
        side,
        grasp_mode,
        args.center_front_suction_y_offset,
        args.center_front_suction_z_offset,
        args.bottom_front_suction_z_offset,
        args.box_grid_rows,
        args.box_grid_columns,
        args.box_grid_center_y,
    )
    initial_right_arm_joints_deg = args.initial_right_arm_joints_deg
    if grasp_mode == "top_suction" and side == "right":
        initial_right_arm_joints_deg = args.top_initial_right_arm_joints_deg
    command = [
        "ros2", "launch", "alfa_robot_moveit_config",
        "v3_single_arm_box_extract_demo.launch.py",
        "start_rviz:=false", "start_rerun:=false", "auto_run_once:=true",
        f"end_effector:={getattr(args, 'end_effector', 'legacy_suction')}",
        f"side:={side}",
        f"ignore_opposite_arm:={'false' if is_transition else ('true' if args.ignore_opposite_arm else 'false')}",
        f"task_mode:={'state_transition' if is_transition else 'full_extract'}",
        f"continuous_sequence:={'true' if getattr(args, 'continuous_sequence', False) else 'false'}",
        "continuous_plan_approach:=" + (
            "true" if getattr(args, "continuous_plan_approach", False) else "false"
        ),
        f"maximum_carried_box_tilt_deg:={getattr(args, 'maximum_carried_box_tilt_deg', 180.0):.6f}",
        "full_box_wall_scene:=true",
        f"box_grid_columns:={args.box_grid_columns}",
        f"box_grid_rows:={args.box_grid_rows}",
        f"box_grid_center_y:={args.box_grid_center_y}",
        f"box_grid_bottom_z:={args.box_grid_bottom_z}",
        f"target_box_id:={box_id}",
        f"grasp_mode:={grasp_mode}",
        f"front_suction_y_offset:={front_y_offset:.6f}",
        f"front_suction_z_offset:={front_z_offset:.6f}",
        f"top_suction_x_offset:={args.top_suction_x_offset:.6f}",
        f"contact_tool_roll_deg:={getattr(args, 'contact_tool_roll_deg', 0.0):.6f}",
        "retreat_distance:=" + (
            f"{getattr(args, 'top_retreat_distance_m', 0.35):.6f}"
            if grasp_mode == "top_suction"
            else f"{getattr(args, 'front_retreat_distance_m', 0.35):.6f}"
        ),
        f"ground_enabled:={'true' if args.ground_enabled else 'false'}",
        f"ground_surface_z:={args.ground_surface_z:.6f}",
        f"ground_clearance:={args.ground_clearance:.6f}",
        f"ground_size_x:={args.ground_size_x:.6f}",
        f"ground_size_y:={args.ground_size_y:.6f}",
        f"ground_thickness:={args.ground_thickness:.6f}",
        f"warehouse_enabled:={'true' if args.warehouse_enabled else 'false'}",
        f"warehouse_opening_x:={args.warehouse_opening_x:.6f}",
        f"warehouse_center_y:={args.warehouse_center_y:.6f}",
        f"warehouse_floor_z:={args.warehouse_floor_z:.6f}",
        f"warehouse_length:={args.warehouse_length:.6f}",
        f"warehouse_width:={args.warehouse_width:.6f}",
        f"warehouse_height:={args.warehouse_height:.6f}",
        f"warehouse_wall_thickness:={args.warehouse_wall_thickness:.6f}",
        f"initial_box_x:={center[0]:.6f}",
        f"initial_box_y:={center[1]:.6f}",
        f"initial_box_z:={center[2]:.6f}",
        f"initial_updown:={updown:.6f}",
        f"initial_arm_pose:={args.initial_arm_pose}",
        f"initial_left_arm_joints_deg:={args.initial_left_arm_joints_deg}",
        f"initial_right_arm_joints_deg:={initial_right_arm_joints_deg}",
        f"psi_step_deg:={args.psi_step_deg:.6f}",
        f"maximum_cartesian_joint_step_deg:={args.maximum_cartesian_joint_step_deg:.6f}",
        f"precontact_candidate_limit:={args.precontact_candidate_limit}",
        f"rrt_planning_time:={args.rrt_planning_time:.6f}",
        f"rrt_planning_attempts:={args.rrt_planning_attempts}",
        "natural_seed_swivel_sampling:=" + (
            "true" if getattr(args, "natural_seed_swivel_sampling", False) else "false"
        ),
        f"natural_seed_swivel_step_deg:={getattr(args, 'natural_seed_swivel_step_deg', 1.0):.6f}",
        f"natural_seed_swivel_neighbor_steps:={getattr(args, 'natural_seed_swivel_neighbor_steps', 2)}",
        f"natural_joint_acceleration_weight:={getattr(args, 'natural_joint_acceleration_weight', 0.0):.6f}",
        f"natural_joint_wrap_weight:={getattr(args, 'natural_joint_wrap_weight', 0.0):.6f}",
        f"natural_place_return_weight:={getattr(args, 'natural_place_return_weight', 0.0):.6f}",
        f"natural_cartesian_replay_step_deg:={getattr(args, 'natural_cartesian_replay_step_deg', 0.0):.6f}",
        "natural_rrt_shortcut_enabled:=" + (
            "true" if getattr(args, "natural_rrt_shortcut_enabled", False) else "false"
        ),
        "natural_rrt_shortcut_max_nodes:=" + str(
            getattr(args, "natural_rrt_shortcut_max_nodes", 0)
        ),
        "cartesian_transfer_search_enabled:=" + (
            "true" if getattr(args, "cartesian_transfer_search_enabled", False) else "false"
        ),
        f"cartesian_transfer_translation_step:={getattr(args, 'cartesian_transfer_translation_step', 0.02):.6f}",
        f"cartesian_transfer_rotation_step_deg:={getattr(args, 'cartesian_transfer_rotation_step_deg', 2.0):.6f}",
        "cartesian_transfer_max_search_attempts:=" + str(
            getattr(args, "cartesian_transfer_max_search_attempts", 0)
        ),
        "shortcut_repair_rrt_enabled:=" + (
            "true" if getattr(args, "shortcut_repair_rrt_enabled", False) else "false"
        ),
        f"shortcut_repair_rrt_budget_ms:={getattr(args, 'shortcut_repair_rrt_budget_ms', 1800.0):.3f}",
        "shortcut_repair_rrt_max_samples:=" + str(
            getattr(args, "shortcut_repair_rrt_max_samples", 240)
        ),
        f"shortcut_repair_max_joint_offset_deg:={getattr(args, 'shortcut_repair_max_joint_offset_deg', 35.0):.3f}",
        "place_updown_enabled:=" + (
            "true" if getattr(args, "place_updown_enabled", False) else "false"
        ),
        f"place_updown:={getattr(args, 'place_updown_m', updown):.6f}",
        "top_loaded_transfer_direct_only:=" + (
            "true" if getattr(args, "top_loaded_transfer_direct_only", False) else "false"
        ),
        f"natural_max_proximal_step_deg:={getattr(args, 'natural_max_proximal_step_deg', 12.0):.6f}",
        f"natural_max_wrist_step_deg:={getattr(args, 'natural_max_wrist_step_deg', 8.0):.6f}",
        f"analytic_path_only:={'true' if args.analytic_path_only else 'false'}",
    ]
    command.extend(box_geometry_launch_arguments(args))
    place_tcp_pose = getattr(args, "place_tcp_pose", "")
    if place_tcp_pose and not is_transition:
        command.append(f"place_tcp_pose:={place_tcp_pose}")
    place_arm_joints_deg = getattr(args, "place_arm_joints_deg", "")
    if place_arm_joints_deg and not is_transition:
        command.append(f"place_arm_joints_deg:={place_arm_joints_deg}")
    loaded_transfer_waypoints = getattr(args, "loaded_transfer_joint_waypoints_deg", "")
    if loaded_transfer_waypoints and not is_transition:
        command.append(
            f"loaded_transfer_joint_waypoints_deg:={loaded_transfer_waypoints}"
        )
    waypoint_start = getattr(args, "loaded_transfer_waypoint_start_deg", "")
    if waypoint_start and not is_transition:
        command.append(f"loaded_transfer_waypoint_start_deg:={waypoint_start}")
    if is_transition:
        command.extend((
            "transition_from_joints:=" + ",".join(f"{value:.10f}" for value in transition_from_joints),
            "transition_to_joints:=" + ",".join(f"{value:.10f}" for value in transition_to_joints),
        ))
    if removed_box_ids:
        command.append(
            "removed_box_ids:=" + ",".join(str(value) for value in sorted(removed_box_ids))
        )
    if result_json_path is not None:
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        result_json_path.unlink(missing_ok=True)
        command.append(f"result_json_path:={result_json_path}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    started = time.monotonic()
    result = {
        "side": side,
        "updown_m": updown,
        "grasp_mode": grasp_mode,
        "end_effector": getattr(args, "end_effector", "legacy_suction"),
        "front_suction_y_offset": front_y_offset,
        "front_suction_z_offset": front_z_offset,
        "place_tcp_pose": place_tcp_pose,
        "place_arm_joints_deg": place_arm_joints_deg,
        "task_mode": "state_transition" if is_transition else "full_extract",
        "removed_box_ids": sorted(removed_box_ids),
        "success": False,
        "duration_ms": "",
        "failure_stage": "timeout",
        "failure_reason": "no complete task result before timeout",
    }
    output_tail: list[str] = []
    try:
        assert process.stdout is not None
        deadline = started + args.timeout
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        while time.monotonic() < deadline:
            events = selector.select(timeout=0.1)
            if not events:
                if process.poll() is not None:
                    result["failure_stage"] = "process_exit"
                    break
                continue
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    result["failure_stage"] = "process_exit"
                    break
                continue
            output_tail.append(line.strip())
            output_tail = output_tail[-10:]
            success = SUCCESS_RE.search(line)
            failure = FAIL_RE.search(line)
            if success:
                result.update(
                    success=True,
                    duration_ms=float(success.group(1)),
                    failure_stage="",
                    failure_reason="",
                )
                break
            if failure:
                result.update(
                    duration_ms=float(failure.group(1)),
                    failure_stage=failure.group(2),
                    failure_reason=failure.group(3).strip(),
                )
                break
            if "v3_single_arm_box_extract_demo" in line and "process has died" in line:
                result["failure_stage"] = "process_exit"
                result["failure_reason"] = line.strip()
                break
        selector.close()
        if not result["success"] and output_tail and not result["failure_reason"]:
            result["failure_reason"] = output_tail[-1]
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    result["wall_ms"] = round((time.monotonic() - started) * 1000.0, 1)
    if result_json_path is not None and result_json_path.is_file():
        with result_json_path.open(encoding="utf-8") as stream:
            result["trajectory_result"] = json.load(stream)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-x", type=float, default=0.75)
    parser.add_argument("--box-depth", type=float, default=0.30)
    parser.add_argument("--box-width", type=float, default=0.40)
    parser.add_argument("--box-height", type=float, default=0.40)
    parser.add_argument("--box-grid-rows", type=int, default=5)
    parser.add_argument("--box-grid-columns", type=int, default=5)
    parser.add_argument("--box-grid-center-y", type=float, default=0.0)
    parser.add_argument("--box-grid-bottom-z", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--end-effector",
        choices=("legacy_suction", "scoop"),
        default="legacy_suction",
    )
    parser.add_argument("--limit-boxes", type=int, default=0)
    parser.add_argument("--box-ids", type=parse_box_ids, default=set())
    parser.add_argument(
        "--top-suction-box-ids",
        type=parse_box_ids,
        default=DEFAULT_TOP_SUCTION_BOX_IDS,
        help="mixed strategy boxes; defaults to box 18 and the complete bottom row",
    )
    parser.add_argument(
        "--grasp-strategy",
        choices=("mixed", "front"),
        default="mixed",
        help="mixed uses top suction for --top-suction-box-ids",
    )
    parser.add_argument(
        "--initial-removed-box-ids",
        type=parse_box_ids,
        default=set(),
        help="boxes already absent before this run, useful for lower-row focused tests",
    )
    parser.add_argument(
        "--sequential-scene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="remove each successful box from subsequent planning scenes",
    )
    parser.add_argument("--updown-step", type=float, default=0.25)
    parser.add_argument(
        "--initial-arm-pose", choices=("zero", "v3_home"), default="zero"
    )
    parser.add_argument(
        "--initial-left-arm-joints-deg", default="0,90,-120,75,0,0,0"
    )
    parser.add_argument(
        "--initial-right-arm-joints-deg", default="-45,-90,120,-75,0,0,0"
    )
    parser.add_argument(
        "--top-initial-right-arm-joints-deg", default="0,-45,120,-75,0,0,0"
    )
    parser.add_argument(
        "--ignore-opposite-arm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--psi-step-deg", type=float, default=5.0)
    parser.add_argument("--maximum-cartesian-joint-step-deg", type=float, default=15.0)
    parser.add_argument("--precontact-candidate-limit", type=int, default=8)
    parser.add_argument("--rrt-planning-time", type=float, default=1.0)
    parser.add_argument("--rrt-planning-attempts", type=int, default=1)
    parser.add_argument(
        "--natural-seed-swivel-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--natural-seed-swivel-step-deg", type=float, default=1.0)
    parser.add_argument("--natural-seed-swivel-neighbor-steps", type=int, default=2)
    parser.add_argument("--natural-joint-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--natural-place-return-weight", type=float, default=0.0)
    parser.add_argument("--natural-cartesian-replay-step-deg", type=float, default=0.0)
    parser.add_argument(
        "--natural-rrt-shortcut-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--place-updown-enabled", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--place-updown-m", type=float, default=0.0)
    parser.add_argument(
        "--top-loaded-transfer-direct-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--natural-max-proximal-step-deg", type=float, default=12.0)
    parser.add_argument("--natural-max-wrist-step-deg", type=float, default=8.0)
    parser.add_argument("--analytic-path-only", action="store_true")
    parser.add_argument("--place-tcp-pose", default="")
    parser.add_argument("--top-suction-x-offset", type=float, default=-0.10)
    parser.add_argument("--center-front-suction-y-offset", type=float, default=0.08)
    parser.add_argument("--center-front-suction-z-offset", type=float, default=-0.05)
    parser.add_argument("--bottom-front-suction-z-offset", type=float, default=0.12)
    parser.add_argument(
        "--ground-enabled", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ground-surface-z", type=float, default=0.0)
    parser.add_argument("--ground-clearance", type=float, default=0.005)
    parser.add_argument("--ground-size-x", type=float, default=6.0)
    parser.add_argument("--ground-size-y", type=float, default=6.0)
    parser.add_argument("--ground-thickness", type=float, default=0.10)
    parser.add_argument(
        "--warehouse-enabled", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--warehouse-opening-x", type=float, default=-1.18)
    parser.add_argument("--warehouse-center-y", type=float, default=0.0)
    parser.add_argument("--warehouse-floor-z", type=float, default=0.0)
    parser.add_argument("--warehouse-length", type=float, default=2.38)
    parser.add_argument("--warehouse-width", type=float, default=2.38)
    parser.add_argument("--warehouse-height", type=float, default=2.35)
    parser.add_argument("--warehouse-wall-thickness", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/v3_single_arm_5x5_box_wall.csv"),
    )
    args = parser.parse_args()

    if args.box_grid_rows < 1 or args.box_grid_columns < 1:
        parser.error("box grid rows and columns must be positive")
    box_count = args.box_grid_rows * args.box_grid_columns
    invalid_ids = sorted(
        box_id
        for box_id in args.box_ids | args.top_suction_box_ids | args.initial_removed_box_ids
        if box_id > box_count
    )
    if invalid_ids:
        parser.error(f"box ids must be in [1, {box_count}]: {invalid_ids}")
    specs = list(box_specs(
        args.contact_x,
        args.box_depth,
        args.box_width,
        args.box_height,
        args.box_grid_rows,
        args.box_grid_columns,
        args.box_grid_center_y,
        args.box_grid_bottom_z,
    ))
    if args.updown_step <= 0.0 or args.updown_step > 1.0:
        parser.error("updown-step must be in (0, 1]")
    selected_ids = args.box_ids
    if selected_ids:
        specs = [spec for spec in specs if spec[0] in selected_ids]
    if args.limit_boxes > 0:
        specs = specs[: args.limit_boxes]

    if min(args.box_depth, args.box_width, args.box_height) <= 0.0:
        parser.error("box dimensions must be positive")
    if args.ground_clearance < 0.0:
        parser.error("ground-clearance must be non-negative")
    if not 0.0 <= args.center_front_suction_y_offset < args.box_width * 0.5:
        parser.error("center-front-suction-y-offset must stay inside the front face")
    if abs(args.center_front_suction_z_offset) >= args.box_height * 0.5:
        parser.error("center-front-suction-z-offset must stay inside the front face")
    if not 0.0 <= args.bottom_front_suction_z_offset < args.box_height * 0.5:
        parser.error("bottom-front-suction-z-offset must stay inside the front face")
    if min(args.ground_size_x, args.ground_size_y, args.ground_thickness) <= 0.0:
        parser.error("ground dimensions must be positive")
    if min(
        args.warehouse_length,
        args.warehouse_width,
        args.warehouse_height,
        args.warehouse_wall_thickness,
    ) <= 0.0:
        parser.error("warehouse dimensions must be positive")

    rows = []
    removed_box_ids = set(args.initial_removed_box_ids)
    for box_id, row, column, x, y, z in specs:
        attempts = []
        grasp_mode = (
            grasp_mode_for_box(box_id, args.top_suction_box_ids)
            if args.grasp_strategy == "mixed"
            else "front"
        )
        scene_removed_box_ids = set(removed_box_ids) if args.sequential_scene else set(
            args.initial_removed_box_ids
        )
        print(
            f"box={box_id:02d} row={row} column={column} "
            f"center=[{x:.2f},{y:.2f},{z:.2f}] grasp={grasp_mode} "
            f"removed={sorted(scene_removed_box_ids)}",
            flush=True,
        )
        selected = None
        for side, updown in candidate_order(
            row, y, z, args.updown_step, grasp_mode, args.box_grid_rows
        ):
            attempt = run_attempt(
                args,
                box_id,
                grasp_mode,
                scene_removed_box_ids,
                side,
                updown,
                (x, y, z),
            )
            attempts.append(attempt)
            print(
                f"  {side} updown={updown:.2f}: "
                f"{'SUCCESS' if attempt['success'] else attempt['failure_stage']}",
                flush=True,
            )
            if attempt["success"]:
                selected = attempt
                break
        if selected and args.sequential_scene:
            removed_box_ids.add(box_id)
        final = selected or attempts[-1]
        rows.append({
            "box_id": box_id,
            "row_from_top": row,
            "column_from_left": column,
            "contact_x_m": args.contact_x,
            "center_x_m": x,
            "center_y_m": y,
            "center_z_m": z,
            "grasp_mode": grasp_mode,
            "scene_removed_box_ids": json.dumps(sorted(scene_removed_box_ids)),
            "ground_enabled": args.ground_enabled,
            "warehouse_enabled": args.warehouse_enabled,
            "ignore_opposite_arm": args.ignore_opposite_arm,
            "success": bool(selected),
            "selected_side": selected["side"] if selected else "",
            "selected_updown_m": selected["updown_m"] if selected else "",
            "duration_ms": selected["duration_ms"] if selected else "",
            "failure_stage": "" if selected else final["failure_stage"],
            "failure_reason": "" if selected else final["failure_reason"],
            "attempt_count": len(attempts),
            "attempts_json": json.dumps(attempts, ensure_ascii=False),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    success_count = sum(int(row["success"]) for row in rows)
    print(f"RESULT success={success_count}/{len(rows)} output={args.output}", flush=True)
    return 0 if success_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
