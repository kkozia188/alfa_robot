#!/usr/bin/python3
"""Run fresh V3.2.2 MoveIt/FCL pickup planning for compiled pose requests."""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


REPORT_SCHEMA = "alfa.v322_box_wall_motion_family_fresh_pickup_report.v1"
LEFT_HOME = [155, -105, 20, 90, -90, -40, 0]
RIGHT_HOME = [-155, 105, -20, -90, 90, 40, 0]


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_backend(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("v322_family_scan_backend", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load backend script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def backend_args(base_x: float, family: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        base_x=base_x,
        base_y=0.0,
        base_yaw=0.0,
        end_effector="scoop",
        ignore_opposite_arm=False,
        task_mode="full_extract",
        continuous_sequence=True,
        continuous_plan_approach=False,
        maximum_carried_box_tilt_deg=95.0,
        maximum_carried_box_center_z=1_000_000.0,
        center_front_suction_y_offset=float(family["center_front_y_offset_m"]),
        center_front_suction_z_offset=float(family["center_front_z_offset_m"]),
        bottom_front_suction_z_offset=float(family["bottom_front_z_offset_m"]),
        top_suction_x_offset=float(family["top_suction_x_offset_m"]),
        contact_tool_roll_deg=0.0,
        front_retreat_distance_m=0.35,
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
        initial_left_arm_joints_deg=",".join(map(str, LEFT_HOME)),
        initial_right_arm_joints_deg=",".join(map(str, RIGHT_HOME)),
        top_initial_right_arm_joints_deg=",".join(map(str, RIGHT_HOME)),
        psi_step_deg=5.0,
        maximum_cartesian_joint_step_deg=180.0,
        precontact_candidate_limit=8,
        rrt_planning_time=15.0,
        rrt_planning_attempts=8,
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
        cartesian_transfer_search_enabled=False,
        cartesian_transfer_translation_step=0.02,
        cartesian_transfer_rotation_step_deg=2.0,
        cartesian_transfer_max_search_attempts=0,
        shortcut_repair_rrt_enabled=True,
        shortcut_repair_rrt_budget_ms=3000.0,
        shortcut_repair_rrt_max_samples=800,
        shortcut_repair_max_joint_offset_deg=35.0,
        place_updown_enabled=False,
        place_updown_m=-0.3,
        top_loaded_transfer_direct_only=False,
        loaded_transfer_defer_place_updown=False,
        auto_safe_opposite_arm=False,
        natural_max_proximal_step_deg=180.0,
        natural_max_wrist_step_deg=180.0,
        analytic_path_only=True,
        place_tcp_pose="",
        place_arm_joints_deg="",
        precontact_arm_joints_deg="",
        loaded_transfer_joint_waypoints_deg="",
        loaded_transfer_waypoint_start_deg="",
        loaded_transfer_arm_updown_waypoints="",
        loaded_transfer_arm_updown_waypoint_start="",
        transition_joint_waypoints="",
        transition_waypoint_start="",
        transition_whole_body_waypoints="",
        transition_whole_body_waypoint_start="",
        timeout=120.0,
    )


def apply_current_arm_seed(planner_args: SimpleNamespace, task: dict[str, Any]) -> None:
    positions = [float(value) for value in task["current_state"]["joint_positions"]]
    planner_args.initial_left_arm_joints_deg = ",".join(
        f"{math.degrees(value):.10f}" for value in positions[1:8]
    )
    planner_args.initial_right_arm_joints_deg = ",".join(
        f"{math.degrees(value):.10f}" for value in positions[8:15]
    )
    planner_args.top_initial_right_arm_joints_deg = planner_args.initial_right_arm_joints_deg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--backend-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-attempt-limit", type=int, default=6)
    args = parser.parse_args()
    if args.candidate_attempt_limit <= 0:
        parser.error("candidate-attempt-limit must be positive")

    plan = read_json(args.plan)
    family = read_json(args.family)
    backend = load_backend(args.backend_script)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    removed_slots: set[int] = set()
    results: list[dict[str, Any]] = []
    failure_counts: collections.Counter[str] = collections.Counter()

    for task_index, task in enumerate(plan["tasks"], start=1):
        request_id = str(task["request_id"])
        geometry = task["derived_geometry"]
        row = int(geometry["row_from_top"])
        slot = int(geometry["scene_slot"])
        clearance_slots = {int(value) for value in geometry.get("clearance_scene_slots", [])}
        scene_removed_slots = removed_slots | clearance_slots
        center = tuple(float(value) for value in task["box_pose"]["position_m"])
        attempts: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        seen: set[tuple[str, str, float]] = set()
        for candidate in task["candidates"]:
            key = (
                str(candidate["grasp_mode"]),
                str(candidate["arm"]),
                float(candidate["updown_m"]),
            )
            if key in seen:
                continue
            seen.add(key)
            if len(attempts) >= args.candidate_attempt_limit:
                break
            payload_path = args.output_dir / (
                f"{task_index:02d}-{request_id}-attempt-{len(attempts) + 1}.json"
            )
            planner_args = backend_args(-0.60 if row <= 3 else -0.35, family)
            apply_current_arm_seed(planner_args, task)
            planner_args.auto_safe_opposite_arm = (
                candidate.get("opposite_arm_policy") == "auto_safe"
            )
            attempt = backend.run_attempt(
                planner_args,
                slot,
                key[0],
                set(scene_removed_slots),
                key[1],
                key[2],
                center,
                payload_path,
            )
            record = {
                "candidate_rank": candidate["rank"],
                "template": candidate["template"],
                "grasp_mode": key[0],
                "arm": key[1],
                "updown_m": key[2],
                "opposite_arm_policy": candidate.get(
                    "opposite_arm_policy", "hold_current"
                ),
                "success": bool(attempt.get("success")),
                "duration_ms": attempt.get("duration_ms"),
                "wall_ms": attempt.get("wall_ms"),
                "failure_stage": attempt.get("failure_stage", ""),
                "failure_reason": attempt.get("failure_reason", ""),
                "payload": str(payload_path) if payload_path.is_file() else "",
            }
            attempts.append(record)
            if record["success"]:
                selected = record
                removed_slots.add(slot)
                break
        if selected is None:
            final = attempts[-1] if attempts else {
                "failure_stage": "candidate_selection",
                "failure_reason": "no backend candidate attempted",
            }
            reason = f"{final['failure_stage']}: {final['failure_reason']}"
            failure_counts[reason] += 1
        else:
            reason = ""
        results.append({
            "request_id": request_id,
            "row_from_top": row,
            "column_from_left": geometry["column_from_left"],
            "pose_derived_scene_slot": slot,
            "removed_slots_before": sorted(removed_slots - ({slot} if selected else set())),
            "clearance_channels": geometry.get("clearance_channels", []),
            "clearance_scene_slots": sorted(clearance_slots),
            "backend_removed_slots": sorted(scene_removed_slots),
            "success": selected is not None,
            "selected": selected,
            "failure_reason": reason,
            "attempts": attempts,
        })
        print(
            f"[{task_index}/{len(plan['tasks'])}] {request_id}: "
            f"{'SUCCESS' if selected else 'FAILED'} attempts={len(attempts)}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    for row in sorted({int(item["row_from_top"]) for item in results}):
        row_results = [item for item in results if int(item["row_from_top"]) == row]
        successes = [item for item in row_results if item["success"]]
        times = [float(item["selected"]["duration_ms"]) for item in successes]
        rows.append({
            "row_from_top": row,
            "successes": len(successes),
            "attempts": len(row_results),
            "success_rate": len(successes) / len(row_results),
            "planning_time_ms": {
                "mean": sum(times) / len(times) if times else None,
                "maximum": max(times) if times else None,
            },
        })
    report = {
        "schema": REPORT_SCHEMA,
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "evidence_level": "fresh_moveit_fcl_pickup_and_retreat",
        "scope_limit": (
            "Fresh checks cover pregrasp, Cartesian approach, attachment and loaded retreat. "
            "Continuous inter-task bridges and placement replay remain a separate milestone."
        ),
        "successes": sum(item["success"] for item in results),
        "attempts": len(results),
        "row_statistics": rows,
        "failure_reasons": dict(sorted(failure_counts.items())),
        "tasks": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"RESULT success={report['successes']}/{report['attempts']} report={args.report}")
    return 0 if report["successes"] == report["attempts"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
