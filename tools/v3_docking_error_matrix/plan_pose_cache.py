#!/usr/bin/env python3
"""Plan base-pose-specific whole-body entry bridges and assemble a plan cache."""

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
sys.path.insert(0, str(SCRIPT_DIR))
import scan_v3_single_arm_box_wall as scan  # noqa: E402


GROUPS = [
    (1, 5), (2, 4), (3,), (6, 10), (7, 9), (8,), (11, 15),
    (12, 14), (13,), (16, 20), (17, 19), (18,), (21, 25),
    (22, 24), (23,),
]
LEFT_HOME_DEG = [155, -105, 20, 90, -90, -40, 0]
RIGHT_HOME_DEG = [-155, 105, -20, -90, 90, 40, 0]
HOME = [
    -0.3, 0.0,
    *[math.radians(value) for value in LEFT_HOME_DEG],
    *[math.radians(value) for value in RIGHT_HOME_DEG],
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def center(box_id: int) -> tuple[float, float, float]:
    row = (box_id - 1) // 5
    column = (box_id - 1) % 5
    return (0.9, (2 - column) * 0.4, (4.5 - row) * 0.4)


def active_indices(side: str) -> range:
    return range(2, 9) if side == "left" else range(9, 16)


def transition_goal(payloads: list[dict[str, Any]]) -> list[float]:
    if len(payloads) == 1:
        return [float(value) for value in payloads[0]["frames"][0]["joints"]]
    goal = list(HOME)
    heights = {round(float(payload["initial_updown"]), 8) for payload in payloads}
    if len(heights) != 1:
        raise RuntimeError("group payloads do not share one updown value")
    goal[0] = heights.pop()
    for payload in payloads:
        first = [float(value) for value in payload["frames"][0]["joints"]]
        for index in active_indices(str(payload["side"])):
            goal[index] = first[index]
    return goal


def args_for(
    base_x: float, base_y: float, yaw_rad: float, timeout_s: float
) -> argparse.Namespace:
    return argparse.Namespace(
        base_x=base_x, base_y=base_y, base_yaw=yaw_rad,
        end_effector="scoop", ignore_opposite_arm=False,
        task_mode="state_transition",
        continuous_sequence=False, continuous_plan_approach=False,
        maximum_carried_box_tilt_deg=95.0,
        maximum_carried_box_center_z=1_000_000.0,
        center_front_suction_y_offset=0.08,
        center_front_suction_z_offset=-0.05,
        bottom_front_suction_z_offset=0.12,
        top_suction_x_offset=-0.10,
        contact_tool_roll_deg=0.0,
        front_retreat_distance_m=0.35,
        top_retreat_distance_m=0.20,
        ground_enabled=True, ground_surface_z=0.0, ground_clearance=0.005,
        ground_size_x=6.0, ground_size_y=6.0, ground_thickness=0.10,
        warehouse_enabled=True, warehouse_opening_x=-1.18,
        warehouse_center_y=0.0, warehouse_floor_z=0.0,
        warehouse_length=2.38, warehouse_width=2.38,
        warehouse_height=2.35, warehouse_wall_thickness=0.05,
        initial_arm_pose="zero",
        initial_left_arm_joints_deg=",".join(map(str, LEFT_HOME_DEG)),
        initial_right_arm_joints_deg=",".join(map(str, RIGHT_HOME_DEG)),
        top_initial_right_arm_joints_deg=",".join(map(str, RIGHT_HOME_DEG)),
        psi_step_deg=5.0, maximum_cartesian_joint_step_deg=180.0,
        precontact_candidate_limit=8, rrt_planning_time=15.0,
        rrt_planning_attempts=8, edge_joint_resolution_deg=1.0,
        natural_seed_swivel_sampling=False, natural_seed_swivel_step_deg=1.0,
        natural_seed_swivel_neighbor_steps=2,
        natural_joint_acceleration_weight=0.0,
        natural_joint_wrap_weight=1.0,
        natural_place_return_weight=0.0,
        natural_cartesian_replay_step_deg=0.0,
        natural_rrt_shortcut_enabled=True,
        natural_rrt_shortcut_tcp_cost=True,
        natural_rrt_shortcut_max_nodes=64,
        cartesian_transfer_search_enabled=False,
        cartesian_transfer_translation_step=0.02,
        cartesian_transfer_rotation_step_deg=2.0,
        cartesian_transfer_max_search_attempts=0,
        shortcut_repair_rrt_enabled=True,
        shortcut_repair_rrt_budget_ms=3000.0,
        shortcut_repair_rrt_max_samples=800,
        shortcut_repair_max_joint_offset_deg=35.0,
        place_updown_enabled=False, place_updown_m=-0.3,
        top_loaded_transfer_direct_only=False,
        loaded_transfer_defer_place_updown=False,
        auto_safe_opposite_arm=False,
        natural_max_proximal_step_deg=180.0,
        natural_max_wrist_step_deg=180.0,
        analytic_path_only=False,
        place_tcp_pose="", place_arm_joints_deg="",
        precontact_arm_joints_deg="", ik_seed_arm_joints_deg="",
        loaded_transfer_joint_waypoints_deg="",
        loaded_transfer_waypoint_start_deg="",
        loaded_transfer_arm_updown_waypoints="",
        loaded_transfer_arm_updown_waypoint_start="",
        transition_joint_waypoints="", transition_waypoint_start="",
        transition_whole_body_waypoints="",
        transition_whole_body_waypoint_start="",
        timeout=timeout_s,
    )


def ensure_precontact(payload: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(payload)
    frames = list(payload["frames"])
    if frames and frames[0]["stage"] == "cartesian_approach":
        first = copy.deepcopy(frames[0])
        first["stage"] = "precontact"
        first["box_attached"] = False
        payload["frames"] = [first, *frames]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaw-deg", type=float, required=True)
    parser.add_argument("--base-dx-m", type=float, default=0.0)
    parser.add_argument("--base-dy-m", type=float, default=0.0)
    parser.add_argument("--pickups-dir", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.yaw_deg) or not -15.0 <= args.yaw_deg <= 15.0:
        raise SystemExit("--yaw-deg must be finite and in [-15, 15]")
    if not math.isfinite(args.base_dx_m) or not -0.40 <= args.base_dx_m <= 0.40:
        raise SystemExit("--base-dx-m must be finite and in [-0.40, 0.40]")
    if not math.isfinite(args.base_dy_m) or not -0.50 <= args.base_dy_m <= 0.50:
        raise SystemExit("--base-dy-m must be finite and in [-0.50, 0.50]")
    if args.timeout_s <= 0.0:
        raise SystemExit("--timeout-s must be positive")
    yaw_rad = math.radians(args.yaw_deg)
    pickups_dir = args.pickups_dir.resolve()
    output_dir = args.output_dir.resolve()
    bridges_dir = output_dir / "bridges"
    bridges_dir.mkdir(parents=True, exist_ok=True)
    baseline_cache = read_json(args.baseline_cache.resolve())

    removed: set[int] = set()
    tasks: list[dict[str, Any]] = []
    pair_bridge_map: dict[str, str] = {}
    bridge_history: list[tuple[tuple[int, ...], list[float], dict[str, Any]]] = []
    audit: list[dict[str, Any]] = []

    for group in GROUPS:
        payloads = [
            read_json(pickups_dir / f"box-{box_id:02d}.json") for box_id in group
        ]
        goal = transition_goal(payloads)
        bridge_path = bridges_dir / (
            "group-" + "-".join(f"{box_id:02d}" for box_id in group) + ".json"
        )
        bridge = read_json(bridge_path) if args.resume and bridge_path.is_file() else None
        if not bridge or not bridge.get("success"):
            first_box = group[0]
            planner_args = args_for(
                (-0.60 if first_box <= 15 else -0.35) + args.base_dx_m,
                args.base_dy_m,
                yaw_rad,
                args.timeout_s,
            )
            direct = scan.run_attempt(
                planner_args,
                first_box,
                str(payloads[0]["grasp_mode"]),
                removed | set(group),
                str(payloads[0]["side"]),
                float(payloads[0]["initial_updown"]),
                center(first_box),
                bridge_path,
                transition_from_joints=HOME,
                transition_to_joints=goal,
            )
            audit.append({
                "group": list(group), "kind": "direct",
                "success": bool(direct.get("success")),
                "failure_stage": direct.get("failure_stage", ""),
                "failure_reason": direct.get("failure_reason", ""),
            })
            if direct.get("success"):
                bridge = read_json(bridge_path)
            else:
                bridge = None
                for prefix_group, prefix_goal, prefix_bridge in reversed(bridge_history):
                    suffix_path = bridge_path.with_name(
                        bridge_path.stem + "-via-" +
                        "-".join(f"{box_id:02d}" for box_id in prefix_group) + ".json"
                    )
                    suffix_attempt = scan.run_attempt(
                        planner_args,
                        first_box,
                        str(payloads[0]["grasp_mode"]),
                        removed | set(group),
                        str(payloads[0]["side"]),
                        float(payloads[0]["initial_updown"]),
                        center(first_box),
                        suffix_path,
                        transition_from_joints=prefix_goal,
                        transition_to_joints=goal,
                    )
                    audit.append({
                        "group": list(group), "kind": "prefix_suffix",
                        "prefix_group": list(prefix_group),
                        "success": bool(suffix_attempt.get("success")),
                        "failure_stage": suffix_attempt.get("failure_stage", ""),
                        "failure_reason": suffix_attempt.get("failure_reason", ""),
                    })
                    if not suffix_attempt.get("success"):
                        continue
                    suffix = read_json(suffix_path)
                    bridge = {
                        **suffix,
                        "success": True,
                        "transition_strategy": "validated_prior_prefix_and_local_suffix",
                        "prefix_group": list(prefix_group),
                        "total_ms": float(prefix_bridge.get("total_ms", 0.0)) +
                        float(suffix.get("total_ms", 0.0)),
                        "frames": [
                            *list(prefix_bridge["frames"]),
                            *list(suffix["frames"])[1:],
                        ],
                    }
                    write_json(bridge_path, bridge)
                    break
                if bridge is None:
                    write_json(output_dir / "bridge-attempts.json", audit)
                    raise RuntimeError(
                        "no whole-body bridge at "
                        f"dx={args.base_dx_m:+.3f}, dy={args.base_dy_m:+.3f}, "
                        f"yaw={args.yaw_deg:+.1f} for group {group}"
                    )
        print(
            f"BRIDGE dx={args.base_dx_m:+.3f} dy={args.base_dy_m:+.3f} "
            f"yaw={args.yaw_deg:+.1f} group={group} "
            f"frames={len(bridge['frames'])} time_ms={float(bridge.get('total_ms', 0.0)):.3f}",
            flush=True,
        )

        if len(group) == 2:
            pair_bridge_map[f"{group[0]}+{group[1]}"] = str(bridge_path)
        for box_id, payload in zip(group, payloads):
            payload = ensure_precontact(payload)
            row = (box_id - 1) // 5 + 1
            column = (box_id - 1) % 5 + 1
            tasks.append({
                "box_id": box_id,
                "row": row,
                "column": column,
                "side": str(payload["side"]),
                "mode": str(payload["grasp_mode"]),
                "updown": float(payload["initial_updown"]),
                "payload": payload,
                "transition_frames": list(bridge["frames"]),
                "transition_core_ms": float(bridge.get("total_ms", 0.0)),
                "transition_accumulated_search_ms": float(bridge.get("total_ms", 0.0)),
                "task_accumulated_search_ms": float(payload.get("total_ms", 0.0)),
                "planning_source": "v322_yaw_fresh_pickup_and_whole_body_bridge",
            })
        removed.update(group)
        bridge_history.append((group, goal, bridge))

    tasks.sort(key=lambda item: int(item["box_id"]))
    next(task for task in tasks if int(task["box_id"]) == 16)["base_reposition"] = {
        "required": True,
        "transport_joints": HOME,
        "stow_frames": [],
        "coupled_frames": [],
    }
    station_config = copy.deepcopy(baseline_cache["station_config"])
    station_config["model_revision"] = "robot_v3.2.2-suction"
    cache = {
        "schema": "alfa.v3_scoop_5x5_plan_cache.v1",
        "model_revision": "robot_v3.2.2-suction",
        "upstream_base_commit": "d9c330cef72981390d81ac2b1cd5a6eb9e892195",
        "yaw_error_deg": args.yaw_deg,
        "base_error_map": {
            "dx_m": args.base_dx_m,
            "dy_m": args.base_dy_m,
            "yaw_deg": args.yaw_deg,
        },
        "station_config": station_config,
        "completed_boxes": 25,
        "tasks": tasks,
    }
    cache_path = output_dir / "v322-pose-plan-cache.json"
    write_json(cache_path, cache)
    map_path = output_dir / "dual-bridge-map.json"
    write_json(map_path, pair_bridge_map)
    write_json(output_dir / "bridge-attempts.json", audit)
    print(
        f"RESULT dx={args.base_dx_m:+.3f} dy={args.base_dy_m:+.3f} "
        f"yaw={args.yaw_deg:+.1f} cache={cache_path} dual_bridge_map={map_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
