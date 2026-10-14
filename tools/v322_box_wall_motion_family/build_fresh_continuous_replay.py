#!/usr/bin/python3
"""Plan entry bridges and compose continuous full cycles for pose-driven tasks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from run_fresh_pickup_benchmark import apply_current_arm_seed, backend_args, load_backend


SCHEMA = "alfa.v3_scoop_5x5_dual_replay.v1"


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


def load_conveyor_module(script_dir: Path) -> ModuleType:
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    module_path = script_dir / "v3_scoop_5x5_conveyor_shuttle.py"
    spec = importlib.util.spec_from_file_location("v322_family_conveyor", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load conveyor composer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attachment(payload: dict[str, Any], side: str) -> dict[str, Any]:
    return {
        "tool_link": f"{side}_tool0",
        "center_in_tool": payload["tool_to_box_center"],
        "rotation_in_tool": payload["tool_to_box_rotation"],
        "size": payload["carried_box_size_tool"],
    }


def obstacles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    values = [
        {
            "id": f"neighbor_box_{index}",
            "center": center,
            "size": payload["box_size"],
        }
        for index, center in enumerate(payload["neighbor_centers"])
    ]
    ground = payload["ground"]
    if ground["enabled"]:
        values.append({
            "id": "ground",
            "center": [0.0, 0.0, float(ground["collision_top_z"]) - 0.5 * float(ground["size"][2])],
            "size": ground["size"],
        })
    if payload["warehouse"]["enabled"]:
        values.extend(payload["warehouse"]["panels"])
    return values


def plan_bridge(
    backend: ModuleType,
    family: dict[str, Any],
    task: dict[str, Any],
    fresh: dict[str, Any],
    payload: dict[str, Any],
    output: Path,
    start_joints: list[float],
) -> dict[str, Any]:
    if output.is_file():
        existing = read_json(output)
        if existing.get("success") and existing.get("frames"):
            start = existing["frames"][0]["joints"]
            goal = existing["frames"][-1]["joints"]
            if max(abs(float(a) - float(b)) for a, b in zip(start, start_joints)) <= 1.0e-8 and max(
                abs(float(a) - float(b))
                for a, b in zip(goal, payload["frames"][0]["joints"])
            ) <= 1.0e-8:
                return existing
    row = int(task["derived_geometry"]["row_from_top"])
    candidate = task["selected_candidate"]
    planner_args = backend_args(-0.60 if row <= 3 else -0.35, family)
    apply_current_arm_seed(planner_args, task)
    planner_args.analytic_path_only = False
    planner_args.auto_safe_opposite_arm = candidate["opposite_arm_policy"] == "auto_safe"
    center = tuple(float(value) for value in task["box_pose"]["position_m"])
    result = backend.run_attempt(
        planner_args,
        int(task["derived_geometry"]["scene_slot"]),
        str(candidate["grasp_mode"]),
        {int(value) for value in fresh["backend_removed_slots"]},
        str(candidate["arm"]),
        float(candidate["updown_m"]),
        center,
        output,
        transition_from_joints=start_joints,
        transition_to_joints=[float(value) for value in payload["frames"][0]["joints"]],
    )
    if not result.get("success") or not output.is_file():
        raise RuntimeError(
            f"entry bridge failed for {task['request_id']}: "
            f"{result.get('failure_stage')} {result.get('failure_reason')}"
        )
    bridge = read_json(output)
    if not bridge.get("success") or not bridge.get("frames"):
        raise RuntimeError(f"entry bridge payload failed for {task['request_id']}")
    return bridge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--backend-script", type=Path, required=True)
    parser.add_argument("--script-dir", type=Path, required=True)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    fresh_report = read_json(args.fresh_report)
    family = read_json(args.family)
    backend = load_backend(args.backend_script)
    conveyor = load_conveyor_module(args.script_dir)
    fresh_by_request = {task["request_id"]: task for task in fresh_report["tasks"]}
    args.bridge_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    contract_state = [
        float(value) for value in plan["tasks"][0]["current_state"]["joint_positions"]
    ]
    start_joints = [
        contract_state[0],
        contract_state[15],
        *contract_state[1:8],
        *contract_state[8:15],
    ]
    for index, task in enumerate(plan["tasks"], start=1):
        fresh = fresh_by_request[task["request_id"]]
        if not fresh["success"]:
            raise RuntimeError(f"fresh pickup failed for {task['request_id']}")
        payload = read_json(Path(fresh["selected"]["payload"]))
        bridge = plan_bridge(
            backend,
            family,
            task,
            fresh,
            payload,
            args.bridge_dir / f"{index:02d}-{task['request_id']}-entry.json",
            start_joints,
        )
        prepared.append((task, fresh, {
            "box_id": int(task["derived_geometry"]["scene_slot"]),
            "row": int(task["derived_geometry"]["row_from_top"]),
            "column": int(task["derived_geometry"]["column_from_left"]),
            "side": str(payload["side"]),
            "mode": str(payload["grasp_mode"]),
            "updown": float(payload["initial_updown"]),
            "payload": payload,
            "transition_frames": bridge["frames"],
            "transition_core_ms": float(bridge.get("total_ms", 0.0)),
            "transition_accumulated_search_ms": float(bridge.get("total_ms", 0.0)),
            "task_accumulated_search_ms": float(payload.get("total_ms", 0.0)),
            "planning_source": "pose_driven_fresh_moveit_fcl",
        }))
        print(
            f"BRIDGE {index}/{len(plan['tasks'])} {task['request_id']} "
            f"frames={len(bridge['frames'])} ms={float(bridge.get('total_ms', 0.0)):.3f}",
            flush=True,
        )

    payloads = [item[2]["payload"] for item in prepared]
    left_payload = next(payload for payload in payloads if payload["side"] == "left")
    right_payload = next(payload for payload in payloads if payload["side"] == "right")
    fallback = {
        "left": attachment(left_payload, "left"),
        "right": attachment(right_payload, "right"),
    }
    operations: list[dict[str, Any]] = []
    previous_pose: list[float] | None = None
    previous_payload: dict[str, Any] | None = None
    for task, fresh, composed in prepared:
        pickup_pose = [float(value) for value in composed["payload"]["base_pose_map"]]
        if previous_pose is not None and max(
            abs(a - b) for a, b in zip(previous_pose, pickup_pose)
        ) > 1.0e-9:
            frames = [{
                "stage": "base_reposition_start",
                "joints": list(start_joints),
                "left_attached": False,
                "right_attached": False,
                "base_pose_map": list(previous_pose),
            }]
            frames.extend(conveyor.interpolate_base_frames(
                start_joints,
                previous_pose,
                pickup_pose,
                0.01,
                "base_reposition_between_height_bands",
                False,
                False,
            ))
            operations.append({
                "label": "base reposition between representative height bands",
                "kind": "base_transition",
                "row": composed["row"],
                "after_row": 3,
                "left_box_id": 0,
                "right_box_id": 0,
                "mode": "base_reposition",
                "base_pose_map": list(previous_pose),
                "base_pose_goal_map": list(pickup_pose),
                "from_front_clearance_m": 0.85,
                "to_front_clearance_m": 0.60,
                "base_speed_m_s": 0.10,
                "transition_strategy": "validated_planar_linear",
                "updown_policy": "stationary",
                "box_planning": [],
                "left_attachment": fallback["left"],
                "right_attachment": fallback["right"],
                "obstacles": obstacles(previous_payload or composed["payload"]),
                "allow_ground_model_base": True,
                "frames": frames,
            })
        removed = {int(value) for value in fresh["backend_removed_slots"]}
        operation, terminal = conveyor.operation_for_group(
            (composed["box_id"], None),
            {composed["box_id"]: composed},
            list(start_joints),
            removed,
            fallback,
            2.35,
            1.50,
            0.01,
            {},
            {},
        )
        if max(abs(a - b) for a, b in zip(terminal, start_joints)) > 1.0e-8:
            raise RuntimeError(
                f"cycle {task['request_id']} does not return to the caller start posture"
            )
        operation["label"] = str(task["request_id"])
        operation["row"] = composed["row"]
        operation["obstacles"] = obstacles(composed["payload"])
        operation["pose_driven_request_id"] = task["request_id"]
        operation["transition_strategy"] = "fresh_pose_driven_bridge"
        operation["updown_policy"] = "pose_selected"
        operations.append(operation)
        fallback[composed["side"]] = conveyor.dual.attachment(composed)
        previous_pose = pickup_pose
        previous_payload = composed["payload"]

    replay = {
        "schema": SCHEMA,
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "joint_names": payloads[0]["joint_names"],
        "group_order": [[item[2]["box_id"]] for item in prepared],
        "box_planning": [conveyor.dual.task_planning_summary(item[2]) for item in prepared],
        "operations": operations,
        "scope": (
            "five pose-driven representative cycles: entry, pickup, loaded retreat, "
            "base backoff, right conveyor release, outside stow, and return"
        ),
    }
    write_json(args.output, replay)
    print(f"RESULT operations={len(operations)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
