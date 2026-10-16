#!/usr/bin/python3
"""Plan V3.2.2 pickups that finish at a close, upright chest carry pose."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT_DIR = WORKSPACE / "ros2_ws/src/alfa_robot_moveit_config/scripts"
RERUN_DIR = WORKSPACE / "ros2_ws/src/alfa_robot_rerun"
for path in (SCRIPT_DIR, RERUN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scan_chest_carry_box_wall as scan
from alfa_robot_rerun.visualize_rerun import UrdfRobot


DEFAULT_BASELINE = Path(
    "/home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/"
    "releases/2026-10-01-v322-tool0151-mobile-base-conveyor/v322-plan-cache.json"
)
DEFAULT_MODEL = Path(
    "/home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/"
    "range_certification_v322_tool0151/model/alfa-robot-v322-tool0151.urdf"
)
LEFT_HOME = "155,-105,20,90,-90,-40,0"
RIGHT_HOME = "-155,105,-20,-90,90,40,0"
CHEST_BOX_X_BASE_M = 1.00
CHEST_BOX_Y_BASE_M = 0.60
CHEST_BOX_Z_BASE_M = 1.20
TOP_CENTER_CHEST_BOX_Y_BASE_M = 0.45
BOTTOM_CHEST_BOX_X_BASE_M = 1.05
BOTTOM_CHEST_BOX_Y_BASE_M = 0.45
BOTTOM_CHEST_BOX_Z_BASE_M = 1.10
TOP_CHEST_BOX_X_BASE_M = 0.90
TOP_CHEST_BOX_Y_BASE_M = 0.40
TOP_CHEST_BOX_Z_BASE_M = 0.85
CHEST_UPDOWN_M = -0.30
GUIDE_BOXES = {18: 13, 20: 19}
FIXED_OPPOSITE_ARM_BOXES = {23}


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


def rotation_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = [
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        ]
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = [
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            ]
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = [
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            ]
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = [
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
    norm = math.sqrt(sum(value * value for value in quaternion))
    return [value / norm for value in quaternion]


def frame_joint_map(payload: dict[str, Any], frame: dict[str, Any]) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in zip(payload["joint_names"], frame["joints"])
    }


def contact_tool_transform(robot: UrdfRobot, payload: dict[str, Any]) -> np.ndarray:
    attach = next(
        frame for frame in payload["frames"] if frame["stage"] == "attach_box"
    )
    transforms = robot.fk(frame_joint_map(payload, attach))
    return transforms[str(payload["tool_link"])]


def chest_box_center_base(payload: dict[str, Any]) -> np.ndarray:
    side_sign = 1.0 if payload["side"] == "left" else -1.0
    box_id = int(payload["target_box_id"])
    if payload["grasp_mode"] == "top_suction":
        return np.asarray([
            TOP_CHEST_BOX_X_BASE_M,
            side_sign * TOP_CHEST_BOX_Y_BASE_M,
            TOP_CHEST_BOX_Z_BASE_M,
        ])
    if box_id > 20:
        return np.asarray([
            BOTTOM_CHEST_BOX_X_BASE_M,
            side_sign * BOTTOM_CHEST_BOX_Y_BASE_M,
            BOTTOM_CHEST_BOX_Z_BASE_M,
        ])
    return np.asarray([
        CHEST_BOX_X_BASE_M,
        side_sign * (
            TOP_CENTER_CHEST_BOX_Y_BASE_M if box_id == 3
            else CHEST_BOX_Y_BASE_M
        ),
        CHEST_BOX_Z_BASE_M,
    ])


def chest_target(
    robot: UrdfRobot, payload: dict[str, Any]
) -> tuple[list[float], list[float]]:
    base_x, base_y, base_yaw = [float(value) for value in payload["base_pose_map"]]
    if abs(base_yaw) > 1e-9:
        raise ValueError("the initial chest-carry project supports zero-yaw pickup poses")
    desired_center = chest_box_center_base(payload) + np.asarray(
        [base_x, base_y, 0.0]
    )
    contact = contact_tool_transform(robot, payload)
    rotation = contact[:3, :3]
    offset = np.asarray(payload["tool_to_box_center"], dtype=float)
    tcp_position = desired_center - rotation @ offset
    pose = [
        *tcp_position.tolist(),
        *rotation_to_quaternion_xyzw(rotation),
    ]
    return pose, desired_center.tolist()


def release_geometry(
    robot: UrdfRobot, payload: dict[str, Any]
) -> dict[str, Any]:
    release = next(
        frame for frame in payload["frames"] if frame["stage"] == "release_at_place"
    )
    transform = robot.fk(frame_joint_map(payload, release))[str(payload["tool_link"])]
    center = (
        transform
        @ np.asarray([*payload["tool_to_box_center"], 1.0], dtype=float)
    )[:3]
    rotation = transform[:3, :3] @ np.asarray(
        payload["tool_to_box_rotation"], dtype=float
    )
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rotation[2, 2])))))
    target = chest_box_center_base(payload)
    error = float(np.linalg.norm(center - target))
    return {
        "box_center_base": center.tolist(),
        "target_box_center_base": target.tolist(),
        "box_center_error_m": error,
        "box_tilt_deg": tilt,
        "success": error <= 1e-4 and tilt <= 8.0,
    }


def active_arm_degrees(payload: dict[str, Any], frame: dict[str, Any]) -> str:
    joints = frame_joint_map(payload, frame)
    side = str(payload["side"])
    return ",".join(
        f"{math.degrees(joints[f'{side}_joint{index}']):.12f}"
        for index in range(1, 8)
    )


def arm_degrees(
    payload: dict[str, Any], frame: dict[str, Any], side: str
) -> str:
    joints = frame_joint_map(payload, frame)
    return ",".join(
        f"{math.degrees(joints[f'{side}_joint{index}']):.12f}"
        for index in range(1, 8)
    )


def profile_key(payload: dict[str, Any]) -> str:
    contract = {
        "side": payload["side"],
        "mode": payload["grasp_mode"],
        "center_in_tool": payload["tool_to_box_center"],
        "rotation_in_tool": payload["tool_to_box_rotation"],
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{payload['side']}-{payload['grasp_mode']}-{digest}"


def planner_args(
    task: dict[str, Any],
    payload: dict[str, Any],
    place_pose: list[float],
    profile_degrees: list[float] | None,
    guide_payload: dict[str, Any] | None,
    fast_probe: bool = False,
    parallel_rrt_workers: int = 1,
) -> SimpleNamespace:
    if not task.get("transition_frames"):
        raise ValueError(f"box {task['box_id']} has no validated entry bridge")
    box_id = int(payload["target_box_id"])
    source_precontact = next(
        frame for frame in payload["frames"] if frame["stage"] == "precontact"
    )
    fixed_opposite = box_id in FIXED_OPPOSITE_ARM_BOXES
    initial_left = LEFT_HOME
    initial_right = RIGHT_HOME
    if fixed_opposite:
        if payload["side"] == "left":
            initial_right = arm_degrees(payload, source_precontact, "right")
        else:
            initial_left = arm_degrees(payload, source_precontact, "left")
    return SimpleNamespace(
        base_x=float(payload["base_pose_map"][0]),
        base_y=float(payload["base_pose_map"][1]),
        base_yaw=float(payload["base_pose_map"][2]),
        end_effector="scoop",
        ignore_opposite_arm=False,
        continuous_sequence=True,
        continuous_plan_approach=False,
        maximum_carried_box_tilt_deg=8.0,
        maximum_carried_box_center_z=2.20,
        center_front_suction_y_offset=0.08,
        center_front_suction_z_offset=-0.05,
        bottom_front_suction_z_offset=0.12,
        top_suction_x_offset=-0.10,
        contact_tool_roll_deg=0.0,
        front_retreat_distance_m=(
            0.30 if int(payload["target_box_id"]) > 20 else 0.35
        ),
        top_retreat_distance_m=0.20,
        ground_enabled=True,
        ground_surface_z=0.0,
        ground_clearance=0.005,
        ground_size_x=6.0,
        ground_size_y=6.0,
        ground_thickness=0.10,
        warehouse_enabled=True,
        warehouse_opening_x=-1.18,
        warehouse_center_y=0.0,
        warehouse_floor_z=0.0,
        warehouse_length=2.38,
        warehouse_width=2.38,
        warehouse_height=2.35,
        warehouse_wall_thickness=0.05,
        initial_arm_pose="zero",
        initial_left_arm_joints_deg=initial_left,
        initial_right_arm_joints_deg=initial_right,
        top_initial_right_arm_joints_deg=initial_right,
        psi_step_deg=5.0,
        maximum_cartesian_joint_step_deg=180.0,
        precontact_candidate_limit=1 if fast_probe else 2,
        rrt_planning_time=0.25 if fast_probe else 1.0,
        rrt_planning_attempts=1,
        edge_joint_resolution_deg=1.0,
        natural_seed_swivel_sampling=False,
        natural_seed_swivel_step_deg=1.0,
        natural_seed_swivel_neighbor_steps=2,
        natural_joint_acceleration_weight=0.0,
        natural_joint_wrap_weight=1.0,
        natural_place_return_weight=0.0,
        natural_cartesian_replay_step_deg=0.0,
        natural_rrt_shortcut_enabled=True,
        natural_rrt_shortcut_tcp_cost=True,
        natural_rrt_shortcut_max_nodes=64,
        cartesian_transfer_search_enabled=True,
        cartesian_transfer_translation_step=0.02,
        cartesian_transfer_rotation_step_deg=2.0,
        cartesian_transfer_max_search_attempts=2 if fast_probe else 12,
        shortcut_repair_rrt_enabled=True,
        shortcut_repair_rrt_budget_ms=300.0 if fast_probe else 1000.0,
        shortcut_repair_rrt_max_samples=120 if fast_probe else 400,
        parallel_rrt_workers=parallel_rrt_workers,
        shortcut_repair_max_joint_offset_deg=35.0,
        place_updown_enabled=True,
        place_updown_m=CHEST_UPDOWN_M,
        top_loaded_transfer_direct_only=False,
        loaded_transfer_defer_place_updown=True,
        auto_safe_opposite_arm=(
            not fixed_opposite and (
                box_id in {3, 8, 13, 18, 23}
                or float(payload["initial_updown"]) < CHEST_UPDOWN_M - 1e-9
            )
        ),
        natural_max_proximal_step_deg=180.0,
        natural_max_wrist_step_deg=180.0,
        analytic_path_only=False,
        place_tcp_pose=(
            "" if profile_degrees is not None
            else ",".join(f"{value:.12f}" for value in place_pose)
        ),
        place_arm_joints_deg=(
            "" if profile_degrees is None
            else ",".join(f"{value:.12f}" for value in profile_degrees)
        ),
        precontact_arm_joints_deg="",
        loaded_transfer_joint_waypoints_deg="",
        loaded_transfer_waypoint_start_deg="",
        loaded_transfer_arm_updown_waypoints=(
            coupled_guide_string(guide_payload, str(payload["side"]))
            if guide_payload else ""
        ),
        loaded_transfer_arm_updown_waypoint_start="",
        transition_joint_waypoints="",
        transition_waypoint_start="",
        transition_whole_body_waypoints="",
        transition_whole_body_waypoint_start="",
        timeout=20.0 if fast_probe else 90.0,
    )


def box_center(box_id: int) -> tuple[float, float, float]:
    row = (box_id - 1) // 5
    column = (box_id - 1) % 5
    return (0.9, (2 - column) * 0.4, (4.5 - row) * 0.4)


def parse_box_ids(raw: str) -> list[int]:
    if not raw:
        return list(range(1, 26))
    result = sorted({int(value) for value in raw.split(",") if value.strip()})
    if not result or any(value < 1 or value > 25 for value in result):
        raise ValueError("box IDs must be in [1, 25]")
    return result


def coupled_guide_string(payload: dict[str, Any], side: str) -> str:
    samples: list[str] = []
    names = [str(name) for name in payload["joint_names"]]
    indices = [names.index(f"{side}_joint{index}") for index in range(1, 8)]
    updown_index = names.index("updown")
    for frame in payload["frames"]:
        if not bool(frame.get("box_attached")) or frame["stage"] not in {
            "rrt_to_updown_safe", "updown_to_place", "rrt_to_place"
        }:
            continue
        joints = [float(value) for value in frame["joints"]]
        values = [joints[updown_index], *[
            math.degrees(joints[index]) for index in indices
        ]]
        encoded = ",".join(f"{value:.12f}" for value in values)
        if not samples or samples[-1] != encoded:
            samples.append(encoded)
    if not samples:
        raise ValueError("guide payload contains no attached chest-transfer frames")
    stride = 4
    selected = samples[::stride]
    if selected[-1] != samples[-1]:
        selected.append(samples[-1])
    return ",".join(selected)


def release_arm_degrees(payload: dict[str, Any]) -> list[float]:
    release = next(
        frame for frame in payload["frames"] if frame["stage"] == "release_at_place"
    )
    joints = frame_joint_map(payload, release)
    side = str(payload["side"])
    return [
        math.degrees(joints[f"{side}_joint{index}"])
        for index in range(1, 8)
    ]


def ordered_box_ids(selected: list[int], tasks: dict[int, dict[str, Any]]) -> list[int]:
    first_for_profile: dict[str, int] = {}
    for box_id in selected:
        key = profile_key(tasks[box_id]["payload"])
        first_for_profile.setdefault(key, box_id)
    seeds = set(first_for_profile.values())
    return [box_id for box_id in selected if box_id in seeds] + [
        box_id for box_id in selected if box_id not in seeds
    ]


def main() -> int:
    global CHEST_BOX_X_BASE_M, CHEST_BOX_Y_BASE_M, CHEST_BOX_Z_BASE_M
    global TOP_CENTER_CHEST_BOX_Y_BASE_M
    global BOTTOM_CHEST_BOX_X_BASE_M, BOTTOM_CHEST_BOX_Y_BASE_M
    global BOTTOM_CHEST_BOX_Z_BASE_M
    global TOP_CHEST_BOX_X_BASE_M, TOP_CHEST_BOX_Y_BASE_M, TOP_CHEST_BOX_Z_BASE_M
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cache", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--model-urdf", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--box-ids", default="")
    parser.add_argument("--front-chest-x", type=float, default=CHEST_BOX_X_BASE_M)
    parser.add_argument("--front-chest-y", type=float, default=CHEST_BOX_Y_BASE_M)
    parser.add_argument("--front-chest-z", type=float, default=CHEST_BOX_Z_BASE_M)
    parser.add_argument(
        "--top-center-front-chest-y",
        type=float,
        default=TOP_CENTER_CHEST_BOX_Y_BASE_M,
    )
    parser.add_argument(
        "--bottom-front-chest-x", type=float, default=BOTTOM_CHEST_BOX_X_BASE_M
    )
    parser.add_argument(
        "--bottom-front-chest-y", type=float, default=BOTTOM_CHEST_BOX_Y_BASE_M
    )
    parser.add_argument(
        "--bottom-front-chest-z", type=float, default=BOTTOM_CHEST_BOX_Z_BASE_M
    )
    parser.add_argument("--top-chest-x", type=float, default=TOP_CHEST_BOX_X_BASE_M)
    parser.add_argument("--top-chest-y", type=float, default=TOP_CHEST_BOX_Y_BASE_M)
    parser.add_argument("--top-chest-z", type=float, default=TOP_CHEST_BOX_Z_BASE_M)
    parser.add_argument("--fast-probe", action="store_true")
    parser.add_argument("--parallel-rrt-workers", type=int, default=1)
    parser.add_argument(
        "--fresh-pose-goals",
        action="store_true",
        help="solve each chest TCP pose from its own retreat state; do not reuse joint profiles",
    )
    args = parser.parse_args()
    if not 1 <= args.parallel_rrt_workers <= 32:
        parser.error("--parallel-rrt-workers must be in [1, 32]")

    CHEST_BOX_X_BASE_M = args.front_chest_x
    CHEST_BOX_Y_BASE_M = args.front_chest_y
    CHEST_BOX_Z_BASE_M = args.front_chest_z
    TOP_CENTER_CHEST_BOX_Y_BASE_M = args.top_center_front_chest_y
    BOTTOM_CHEST_BOX_X_BASE_M = args.bottom_front_chest_x
    BOTTOM_CHEST_BOX_Y_BASE_M = args.bottom_front_chest_y
    BOTTOM_CHEST_BOX_Z_BASE_M = args.bottom_front_chest_z
    TOP_CHEST_BOX_X_BASE_M = args.top_chest_x
    TOP_CHEST_BOX_Y_BASE_M = args.top_chest_y
    TOP_CHEST_BOX_Z_BASE_M = args.top_chest_z

    cache = read_json(args.baseline_cache.resolve())
    tasks = {int(task["box_id"]): task for task in cache["tasks"]}
    robot = UrdfRobot(args.model_urdf.resolve().read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    summary: list[dict[str, Any]] = []

    selected = parse_box_ids(args.box_ids)
    for box_id in ordered_box_ids(selected, tasks):
        source = copy.deepcopy(tasks[box_id]["payload"])
        place_pose, desired_center = chest_target(robot, source)
        key = profile_key(source)
        profile_path = profiles_dir / f"{key}.json"
        profile = (
            None if args.fresh_pose_goals
            else read_json(profile_path) if profile_path.is_file() else None
        )
        if profile is None and not args.fresh_pose_goals:
            mirrored_source = copy.deepcopy(source)
            mirrored_source["side"] = (
                "right" if source["side"] == "left" else "left"
            )
            mirrored_key = profile_key(mirrored_source)
            mirrored_path = profiles_dir / f"{mirrored_key}.json"
            if mirrored_path.is_file():
                mirrored = read_json(mirrored_path)
                profile = {
                    **mirrored,
                    "profile_key": key,
                    "source_profile_key": mirrored_key,
                    "side": source["side"],
                    "arm_joints_deg": [
                        -float(value) for value in mirrored["arm_joints_deg"]
                    ],
                }
                write_json(profile_path, profile)
        profile_degrees = (
            [float(value) for value in profile["arm_joints_deg"]]
            if profile else None
        )
        result_path = output_dir / f"box-{box_id:02d}.json"
        guide_box_id = GUIDE_BOXES.get(box_id)
        guide_path = output_dir / f"box-{guide_box_id:02d}.json" if guide_box_id else None
        guide_payload = (
            read_json(guide_path)
            if guide_path is not None and guide_path.is_file() else None
        )
        planned = read_json(result_path) if result_path.is_file() else {}
        geometry = (
            release_geometry(robot, planned)
            if planned.get("success") else {"success": False}
        )
        stages = [str(frame["stage"]) for frame in planned.get("frames", [])]
        resumed = (
            bool(planned.get("success"))
            and bool(geometry["success"])
            and "precontact" in stages
            and "rrt_to_precontact" not in stages
            and "rrt_to_ready" not in stages
        )
        attempt: dict[str, Any] = {}
        if not resumed:
            attempt = scan.run_attempt(
                planner_args(
                    tasks[box_id], source, place_pose, profile_degrees,
                    guide_payload, args.fast_probe, args.parallel_rrt_workers
                ),
                box_id,
                str(source["grasp_mode"]),
                set(int(value) for value in source["removed_box_ids"]),
                str(source["side"]),
                float(source["initial_updown"]),
                box_center(box_id),
                result_path,
            )
            planned = read_json(result_path) if result_path.is_file() else {}
            geometry = (
                release_geometry(robot, planned)
                if planned.get("success") else {"success": False}
            )
        success = bool(planned.get("success")) and bool(geometry["success"])
        desired_center_base = chest_box_center_base(source).tolist()
        if success:
            planned["desired_chest_box_center_base"] = desired_center_base
            planned["desired_chest_box_center_map"] = desired_center
            planned["chest_pullback_contract"] = {
                "strategy": "fresh_tcp_goal_cartesian_first",
                "fallback": "inward_tcp_waypoints_then_local_shortcut",
                "shared_joint_profile": False,
            }
            write_json(result_path, planned)
        if success and profile is None and not args.fresh_pose_goals:
            profile_degrees = release_arm_degrees(planned)
            write_json(profile_path, {
                "schema": "alfa.v322_chest_carry_profile.v1",
                "profile_key": key,
                "source_box_id": box_id,
                "side": source["side"],
                "mode": source["grasp_mode"],
                "center_in_tool": source["tool_to_box_center"],
                "rotation_in_tool": source["tool_to_box_rotation"],
                "arm_joints_deg": profile_degrees,
                "desired_box_center_map": desired_center,
            })
        entry = {
            "box_id": box_id,
            "side": source["side"],
            "mode": source["grasp_mode"],
            "desired_box_center_map": desired_center,
            "desired_box_center_base": desired_center_base,
            "place_tcp_pose_map": place_pose,
            "profile_key": key,
            "profile_seeded": profile is None and not args.fresh_pose_goals,
            "resumed": resumed,
            "success": success,
            "failure_stage": planned.get(
                "failure_stage", attempt.get("failure_stage", "")
            ),
            "failure_reason": planned.get(
                "failure_reason", attempt.get("failure_reason", "")
            ),
            "total_ms": float(planned.get("total_ms", 0.0) or 0.0),
            "release_geometry": geometry,
            "output": str(result_path),
        }
        summary.append(entry)
        write_json(output_dir / "summary.json", summary)
        print(
            f"CHEST_PLAN box={box_id:02d} side={source['side']} "
            f"mode={source['grasp_mode']} profile={key} "
            f"seeded={entry['profile_seeded']} resumed={resumed} "
            f"success={entry['success']} "
            f"stage={entry['failure_stage']} time_ms={entry['total_ms']:.3f}",
            flush=True,
        )
        if not entry["success"]:
            return 1

    print(f"RESULT success={sum(item['success'] for item in summary)}/{len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
