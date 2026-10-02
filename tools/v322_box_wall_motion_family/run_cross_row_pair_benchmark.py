#!/usr/bin/python3
"""Fresh-plan and compose synchronized dual-arm cycles from two box IDs."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from run_fresh_pickup_benchmark import apply_current_arm_seed, backend_args, load_backend


REPLAY_SCHEMA = "alfa.v3_scoop_5x5_dual_replay.v1"
REPORT_SCHEMA = "alfa.v322_cross_row_pair_benchmark.v1"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_conveyor(script_dir: Path) -> ModuleType:
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    path = script_dir / "v3_scoop_5x5_conveyor_shuttle.py"
    spec = importlib.util.spec_from_file_location("v322_cross_row_conveyor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load conveyor module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def planner_joints(control_state: list[float]) -> list[float]:
    return [
        float(control_state[0]),
        float(control_state[15]),
        *[float(value) for value in control_state[1:8]],
        *[float(value) for value in control_state[8:15]],
    ]


def active_indices(names: list[str], side: str) -> list[int]:
    return [names.index(f"{side}_joint{index}") for index in range(1, 8)]


def ensure_precontact(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    frames = list(result["frames"])
    if frames and frames[0]["stage"] == "cartesian_approach":
        precontact = copy.deepcopy(frames[0])
        precontact["stage"] = "precontact"
        precontact["box_attached"] = False
        result["frames"] = [precontact, *frames]
    return result


def plan_single(
    backend: ModuleType,
    family: dict[str, Any],
    seed_task: dict[str, Any],
    box_id: int,
    side: str,
    mode: str,
    updown: float,
    base_x: float,
    removed: set[int],
    center: tuple[float, float, float],
    output: Path,
    retreat_distance_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    args = backend_args(base_x, family)
    apply_current_arm_seed(args, seed_task)
    args.ignore_opposite_arm = True
    args.auto_safe_opposite_arm = False
    args.front_retreat_distance_m = retreat_distance_m
    args.top_retreat_distance_m = retreat_distance_m
    result = backend.run_attempt(
        args, box_id, mode, removed, side, updown, center, output
    )
    payload = read_json(output) if output.is_file() else {}
    return result, payload


def plan_bridge(
    backend: ModuleType,
    family: dict[str, Any],
    seed_task: dict[str, Any],
    candidate: dict[str, Any],
    removed: set[int],
    centers: dict[int, tuple[float, float, float]],
    start: list[float],
    goal: list[float],
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    args = backend_args(float(candidate["base_pose_map"][0]), family)
    apply_current_arm_seed(args, seed_task)
    args.analytic_path_only = False
    args.ignore_opposite_arm = False
    left_id = int(candidate["left_box_id"])
    result = backend.run_attempt(
        args,
        left_id,
        str(candidate["grasp_mode"]),
        removed | {int(candidate["right_box_id"])},
        "left",
        float(candidate["common_updown_m"]),
        centers[left_id],
        output,
        transition_from_joints=start,
        transition_to_joints=goal,
    )
    payload = read_json(output) if output.is_file() else {}
    return result, payload


def task_from_payload(
    box_id: int,
    row: int,
    column: int,
    payload: dict[str, Any],
    bridge: dict[str, Any],
) -> dict[str, Any]:
    payload = ensure_precontact(payload)
    return {
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
        "planning_source": "cross_row_dual_id_fresh_moveit_fcl",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-plan", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--backend-script", type=Path, required=True)
    parser.add_argument("--script-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--candidate-attempt-limit", type=int, default=12)
    args = parser.parse_args()
    if args.candidate_attempt_limit <= 0:
        parser.error("candidate-attempt-limit must be positive")
    pair_plan = read_json(args.pair_plan)
    family = read_json(args.family)
    backend = load_backend(args.backend_script)
    conveyor = load_conveyor(args.script_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    control_state = [float(value) for value in pair_plan["current_state"]["joint_positions"]]
    start = planner_joints(control_state)
    seed_task = {"current_state": {"joint_positions": control_state}}
    operations: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    box_planning: list[dict[str, Any]] = []
    previous_pose: list[float] | None = None
    previous_obstacles: list[dict[str, Any]] | None = None
    fallback: dict[str, dict[str, Any]] | None = None
    replay_joint_names: list[str] = []

    for request_index, request in enumerate(pair_plan["requests"], start=1):
        targets = {int(item["box_id"]): item for item in request["targets"]}
        centers = {
            box_id: tuple(float(value) for value in target["box_pose"]["position_m"])
            for box_id, target in targets.items()
        }
        base_removed = {int(value) for value in request["removed_box_ids"]}
        attempts = []
        selected = None
        for candidate in request["candidates"][: args.candidate_attempt_limit]:
            rank = int(candidate["rank"])
            candidate_dir = args.output_dir / request["request_id"] / f"candidate-{rank:02d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            left_id = int(candidate["left_box_id"])
            right_id = int(candidate["right_box_id"])
            mode = str(candidate["grasp_mode"])
            updown = float(candidate["common_updown_m"])
            base_x = float(candidate["base_pose_map"][0])
            retreat_distance = float(candidate["retreat_distance_m"])
            left_result, left_payload = plan_single(
                backend, family, seed_task, left_id, "left", mode, updown, base_x,
                base_removed | {right_id}, centers[left_id], candidate_dir / "left.json",
                retreat_distance,
            )
            right_result, right_payload = plan_single(
                backend, family, seed_task, right_id, "right", mode, updown, base_x,
                base_removed | {left_id}, centers[right_id], candidate_dir / "right.json",
                retreat_distance,
            )
            attempt = {
                "candidate_rank": rank,
                "left_box_id": left_id,
                "right_box_id": right_id,
                "grasp_mode": mode,
                "common_updown_m": updown,
                "retreat_distance_m": retreat_distance,
                "base_pose_map": candidate["base_pose_map"],
                "left_success": bool(left_result.get("success")),
                "right_success": bool(right_result.get("success")),
                "left_failure": (
                    f"{left_result.get('failure_stage')}: {left_result.get('failure_reason')}"
                    if not left_result.get("success") else ""
                ),
                "right_failure": (
                    f"{right_result.get('failure_stage')}: {right_result.get('failure_reason')}"
                    if not right_result.get("success") else ""
                ),
            }
            if not left_result.get("success") or not right_result.get("success"):
                attempts.append(attempt)
                continue
            names = [str(value) for value in left_payload["joint_names"]]
            if names != [str(value) for value in right_payload["joint_names"]]:
                raise RuntimeError("paired payload joint contracts differ")
            goal = list(start)
            goal[names.index("updown")] = updown
            for index in active_indices(names, "left"):
                goal[index] = float(left_payload["frames"][0]["joints"][index])
            for index in active_indices(names, "right"):
                goal[index] = float(right_payload["frames"][0]["joints"][index])
            bridge_result, bridge = plan_bridge(
                backend, family, seed_task, candidate, base_removed,
                centers, start, goal, candidate_dir / "bridge.json",
            )
            attempt["bridge_success"] = bool(bridge_result.get("success"))
            attempt["bridge_failure"] = (
                f"{bridge_result.get('failure_stage')}: {bridge_result.get('failure_reason')}"
                if not bridge_result.get("success") else ""
            )
            if not bridge_result.get("success"):
                attempts.append(attempt)
                continue
            left_target = targets[left_id]
            right_target = targets[right_id]
            left_task = task_from_payload(
                left_id, int(left_target["row_from_top"]),
                int(left_target["column_from_left"]), left_payload, bridge,
            )
            right_task = task_from_payload(
                right_id, int(right_target["row_from_top"]),
                int(right_target["column_from_left"]), right_payload, bridge,
            )
            current_fallback = fallback or {
                "left": conveyor.dual.attachment(left_task),
                "right": conveyor.dual.attachment(right_task),
            }
            try:
                operation, terminal = conveyor.operation_for_group(
                    (left_id, right_id),
                    {left_id: left_task, right_id: right_task},
                    list(start),
                    set(base_removed),
                    current_fallback,
                    2.35,
                    1.50,
                    0.01,
                    {},
                    {f"{left_id}+{right_id}": bridge},
                )
            except ValueError as error:
                attempt["composition_failure"] = str(error)
                attempts.append(attempt)
                continue
            if max(abs(a - b) for a, b in zip(terminal, start)) > 1.0e-8:
                attempt["composition_failure"] = "cycle does not return to start posture"
                attempts.append(attempt)
                continue
            one_sided = sum(
                bool(frame.get("left_attached")) != bool(frame.get("right_attached"))
                for frame in operation["frames"]
            )
            if one_sided:
                attempt["composition_failure"] = f"one-sided attachment frames={one_sided}"
                attempts.append(attempt)
                continue
            operation["label"] = request["request_id"]
            operation["pose_driven_request_id"] = request["request_id"]
            operation["row"] = (
                f"{left_target['row_from_top']}+{right_target['row_from_top']}"
            )
            operation["transition_strategy"] = "fresh_cross_row_whole_body_bridge"
            operation["updown_policy"] = "common_pose_selected"
            attempt.update({
                "success": True,
                "left_planning_ms": float(left_payload.get("total_ms", 0.0)),
                "right_planning_ms": float(right_payload.get("total_ms", 0.0)),
                "bridge_planning_ms": float(bridge.get("total_ms", 0.0)),
                "frame_count": len(operation["frames"]),
                "one_sided_attachment_frames": one_sided,
            })
            attempts.append(attempt)
            selected = {
                "attempt": attempt,
                "operation": operation,
                "left_task": left_task,
                "right_task": right_task,
            }
            break
        if selected is None:
            results.append({
                "request_id": request["request_id"],
                "box_ids": request["box_ids"],
                "different_rows": request["different_rows"],
                "success": False,
                "attempts": attempts,
            })
            print(
                f"[{request_index}/{len(pair_plan['requests'])}] "
                f"{request['request_id']}: FAILED attempts={len(attempts)}",
                flush=True,
            )
            continue
        operation = selected["operation"]
        pickup_pose = [float(value) for value in operation["base_pose_map"]]
        if previous_pose is not None and max(
            abs(a - b) for a, b in zip(previous_pose, pickup_pose)
        ) > 1.0e-9:
            transition_frames = [{
                "stage": "base_reposition_start",
                "joints": list(start),
                "left_attached": False,
                "right_attached": False,
                "base_pose_map": list(previous_pose),
            }]
            transition_frames.extend(conveyor.interpolate_base_frames(
                start, previous_pose, pickup_pose, 0.01,
                "base_reposition_between_pair_requests", False, False,
            ))
            operations.append({
                "label": "base reposition between pair requests",
                "kind": "base_transition",
                "row": 0,
                "after_row": 0,
                "left_box_id": 0,
                "right_box_id": 0,
                "mode": "base_reposition",
                "base_pose_map": list(previous_pose),
                "base_pose_goal_map": list(pickup_pose),
                "from_front_clearance_m": 0.85 if previous_pose[0] < -0.5 else 0.60,
                "to_front_clearance_m": 0.85 if pickup_pose[0] < -0.5 else 0.60,
                "base_speed_m_s": 0.10,
                "left_attachment": operation["left_attachment"],
                "right_attachment": operation["right_attachment"],
                "obstacles": previous_obstacles or operation["obstacles"],
                "allow_ground_model_base": True,
                "box_planning": [],
                "frames": transition_frames,
            })
        operations.append(operation)
        previous_pose = pickup_pose
        previous_obstacles = operation["obstacles"]
        fallback = {
            "left": operation["left_attachment"],
            "right": operation["right_attachment"],
        }
        box_planning.extend(operation["box_planning"])
        replay_joint_names = [
            str(value) for value in selected["left_task"]["payload"]["joint_names"]
        ]
        results.append({
            "request_id": request["request_id"],
            "box_ids": request["box_ids"],
            "different_rows": request["different_rows"],
            "success": True,
            "selected": selected["attempt"],
            "attempts": attempts,
        })
        print(
            f"[{request_index}/{len(pair_plan['requests'])}] "
            f"{request['request_id']}: SUCCESS candidate={selected['attempt']['candidate_rank']} "
            f"attempts={len(attempts)}",
            flush=True,
        )

    replay = {
        "schema": REPLAY_SCHEMA,
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "joint_names": replay_joint_names,
        "group_order": [result["box_ids"] for result in results if result["success"]],
        "box_planning": box_planning,
        "operations": operations,
        "scope": "cross-row synchronized dual-arm full cycles from two box IDs",
    }
    report = {
        "schema": REPORT_SCHEMA,
        "model_revision": "robot_v3.2.2-suction",
        "successes": sum(result["success"] for result in results),
        "requests": len(results),
        "different_row_successes": sum(
            result["success"] and result["different_rows"] for result in results
        ),
        "one_sided_attachment_frames": sum(
            int(result.get("selected", {}).get("one_sided_attachment_frames", 0))
            for result in results
        ),
        "results": results,
    }
    write_json(args.replay, replay)
    write_json(args.report, report)
    print(
        f"RESULT success={report['successes']}/{report['requests']} "
        f"one_sided={report['one_sided_attachment_frames']} replay={args.replay}"
    )
    return 0 if report["successes"] == report["requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
