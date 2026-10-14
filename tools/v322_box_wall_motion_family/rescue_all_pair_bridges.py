#!/usr/bin/python3
"""Retry the non-kinematic all-pair failures with a larger bridge budget."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from run_all_pair_matrix import validate_candidate
from run_cross_row_pair_benchmark import (
    active_indices,
    compose_pair_cycle,
    load_conveyor,
    plan_bridge,
    planner_joints,
    read_json,
    task_from_payload,
    write_json,
)
from run_fresh_pickup_benchmark import load_backend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-plan", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--backend-script", type=Path, required=True)
    parser.add_argument("--script-dir", type=Path, required=True)
    parser.add_argument("--planning-time", type=float, default=60.0)
    parser.add_argument("--planning-attempts", type=int, default=32)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    plan = read_json(args.matrix_plan)
    family = read_json(args.family)
    backend = load_backend(args.backend_script)
    conveyor = load_conveyor(args.script_dir)
    control_state = [float(value) for value in plan["current_state"]["joint_positions"]]
    start = planner_joints(control_state)
    seed_task = {"current_state": {"joint_positions": control_state}}
    rescued = []
    unresolved = []
    os.environ.setdefault("ROS_DOMAIN_ID", "211")

    for request in plan["requests"]:
        pair_key = "-".join(f"{int(value):02d}" for value in request["box_ids"])
        pair_dir = args.matrix_root / f"pair-{pair_key}"
        result_path = pair_dir / "result.json"
        if not result_path.is_file():
            continue
        result = read_json(result_path)
        if result.get("success"):
            continue
        bridge_attempts = [
            attempt for attempt in result.get("attempts", [])
            if attempt.get("outcome") == "bridge_failure"
        ]
        if not bridge_attempts:
            continue
        request_candidates = {
            int(candidate["rank"]): candidate for candidate in request["candidates"]
        }
        targets = {int(item["box_id"]): item for item in request["targets"]}
        centers = {
            box_id: tuple(float(value) for value in target["box_pose"]["position_m"])
            for box_id, target in targets.items()
        }
        pair_rescued = False
        for failed in bridge_attempts:
            rank = int(failed["candidate_rank"])
            candidate = request_candidates[rank]
            candidate_dir = pair_dir / f"candidate-{rank:03d}"
            left_payload = read_json(candidate_dir / "left.json")
            right_payload = read_json(candidate_dir / "right.json")
            if not left_payload.get("success") or not right_payload.get("success"):
                continue
            names = [str(value) for value in left_payload["joint_names"]]
            goal = list(start)
            goal[names.index("updown")] = float(candidate["common_updown_m"])
            for index in active_indices(names, "left"):
                goal[index] = float(left_payload["frames"][0]["joints"][index])
            for index in active_indices(names, "right"):
                goal[index] = float(right_payload["frames"][0]["joints"][index])
            for retry in range(1, args.retries + 1):
                bridge_path = candidate_dir / f"bridge-rescue-{retry}.json"
                bridge_result, bridge = plan_bridge(
                    backend, family, seed_task, candidate, set(), centers,
                    start, goal, bridge_path,
                    rrt_planning_time_s=args.planning_time,
                    rrt_planning_attempts=args.planning_attempts,
                )
                if not bridge_result.get("success"):
                    continue
                left_id = int(candidate["left_box_id"])
                right_id = int(candidate["right_box_id"])
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
                operation, terminal = compose_pair_cycle(
                    conveyor, left_task, right_task, list(start), set(), bridge,
                )
                if max(abs(a - b) for a, b in zip(terminal, start)) > 1.0e-8:
                    continue
                one_sided = sum(
                    bool(frame.get("left_attached")) != bool(frame.get("right_attached"))
                    for frame in operation["frames"]
                )
                if one_sided:
                    continue
                operation.update({
                    "label": request["request_id"],
                    "row": f"{left_target['row_from_top']}+{right_target['row_from_top']}",
                    "transition_strategy": "extended_budget_whole_body_bridge",
                    "updown_policy": "common_pose_selected",
                })
                replay = {
                    "schema": "alfa.v3_scoop_5x5_dual_replay.v1",
                    "model_revision": "robot_v3.2.2-suction",
                    "tool0_offset_local_z_m": 0.151,
                    "joint_names": names,
                    "group_order": [[left_id, right_id]],
                    "box_planning": operation["box_planning"],
                    "operations": [operation],
                }
                replay_path = candidate_dir / f"replay-rescue-{retry}.json"
                validation_path = candidate_dir / f"validation-rescue-{retry}.json"
                write_json(replay_path, replay)
                valid, validation, reason = validate_candidate(
                    replay_path, validation_path
                )
                if not valid:
                    continue
                selected = {
                    **failed,
                    "outcome": "success",
                    "bridge_rescue_retry": retry,
                    "bridge_planning_ms": float(bridge.get("total_ms", 0.0)),
                    "left_planning_ms": float(left_payload.get("total_ms", 0.0)),
                    "right_planning_ms": float(right_payload.get("total_ms", 0.0)),
                    "frame_count": len(operation["frames"]),
                    "checked_frames": int(validation.get("checked_frames", 0)),
                    "checked_edge_samples": int(validation.get("checked_edge_samples", 0)),
                    "one_sided_attachment_frames": 0,
                }
                result.update({
                    "status": "success",
                    "success": True,
                    "selected": selected,
                    "rescued_from_exhausted": True,
                })
                result["attempts"].append(selected)
                write_json(result_path, result)
                rescued.append(request["box_ids"])
                pair_rescued = True
                print(
                    f"RESCUED pair={request['box_ids'][0]}+{request['box_ids'][1]} "
                    f"candidate={rank} retry={retry}",
                    flush=True,
                )
                break
            if pair_rescued:
                break
        if not pair_rescued:
            unresolved.append(request["box_ids"])
            print(
                f"UNRESOLVED pair={request['box_ids'][0]}+{request['box_ids'][1]}",
                flush=True,
            )
    print(f"RESULT rescued={len(rescued)} unresolved={len(unresolved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
