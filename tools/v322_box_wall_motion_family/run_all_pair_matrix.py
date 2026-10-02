#!/usr/bin/python3
"""Exhaustively fresh-plan and validate all 25-choose-2 box-ID pairs."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import json
import multiprocessing
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from pair_family import PAIR_SCHEMA, compile_pair_contract
from run_cross_row_pair_benchmark import (
    active_indices,
    compose_pair_cycle,
    ensure_precontact,
    load_conveyor,
    plan_bridge,
    plan_single,
    planner_joints,
    read_json,
    task_from_payload,
    write_json,
)
from run_fresh_pickup_benchmark import load_backend


SCHEMA = "alfa.v322_all_pair_matrix.v1"
_BACKEND = None
_CONVEYOR = None
_FAMILY: dict[str, Any] = {}
_CURRENT_STATE: list[float] = []
_OUTPUT_ROOT = Path(".")
_VALIDATOR_TIMEOUT_S = 90.0
_BACKEND_SCRIPT = ""
_SCRIPT_DIR = ""


def worker_init() -> None:
    global _BACKEND, _CONVEYOR
    identity = multiprocessing.current_process()._identity
    worker_index = identity[0] if identity else (os.getpid() % 40)
    os.environ["ROS_DOMAIN_ID"] = str(120 + worker_index % 80)
    _BACKEND = load_backend(Path(_BACKEND_SCRIPT))
    _CONVEYOR = load_conveyor(Path(_SCRIPT_DIR))


def prefix_marker(prefix: tuple[Any, ...]) -> Path:
    box_id, side, mode, updown, base_x, base_y, base_yaw = prefix
    name = (
        f"box-{int(box_id):02d}_{side}_{mode}_"
        f"u{float(updown):+.2f}_x{float(base_x):+.3f}_y{float(base_y):+.2f}_"
        f"yaw{float(base_yaw):+.3f}.json"
    ).replace("+", "p").replace("-", "m")
    return _OUTPUT_ROOT / "_global_kinematic_failures" / name


def global_failure(prefix: tuple[Any, ...]) -> str:
    marker = prefix_marker(prefix)
    if not marker.is_file():
        return ""
    return str(read_json(marker).get("failure", "precontact_ik"))


def mark_global_failure(
    prefix: tuple[Any, ...], stage: str, reason: str
) -> None:
    if "collision=0" not in reason or stage not in {
        "precontact_ik", "cartesian_approach"
    }:
        return
    marker = prefix_marker(prefix)
    marker.parent.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        value = {
            "prefix": list(prefix),
            "failure": normalize_failure(stage, reason),
            "reason": reason,
        }
        temporary = marker.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, marker)


def payload_result(path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    return ({
        "success": bool(payload.get("success")),
        "failure_stage": str(payload.get("failure_stage", "")),
        "failure_reason": str(payload.get("failure_reason", "")),
    }, payload)


def plan_single_cached(
    seed_task: dict[str, Any],
    box_id: int,
    side: str,
    mode: str,
    updown: float,
    base_x: float,
    removed: set[int],
    center: tuple[float, float, float],
    output: Path,
    retreat_distance: float,
    base_y: float,
    base_yaw: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cached = payload_result(output)
    if cached is not None:
        return cached
    assert _BACKEND is not None
    return plan_single(
        _BACKEND, _FAMILY, seed_task, box_id, side, mode, updown, base_x,
        removed, center, output, retreat_distance,
        base_y,
        base_yaw,
    )


def plan_bridge_cached(
    seed_task: dict[str, Any],
    candidate: dict[str, Any],
    removed: set[int],
    centers: dict[int, tuple[float, float, float]],
    start: list[float],
    goal: list[float],
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cached = payload_result(output)
    if cached is not None:
        return cached
    assert _BACKEND is not None
    return plan_bridge(
        _BACKEND, _FAMILY, seed_task, candidate, removed,
        centers, start, goal, output,
    )


def validation_failure(result: dict[str, Any]) -> str:
    operations = result.get("operations", [])
    if operations:
        last = operations[-1]
        reason = str(last.get("failure_reason", "unknown"))
        frame = last.get("failure_frame")
        return f"{reason}" + (f" at frame {frame}" if frame is not None else "")
    return str(result.get("failure_reason", "validator did not produce an operation result"))


def validate_candidate(
    replay_path: Path, result_path: Path
) -> tuple[bool, dict[str, Any], str]:
    if result_path.is_file():
        result = read_json(result_path)
        return bool(result.get("success")), result, validation_failure(result)
    command = [
        "ros2", "launch", "alfa_robot_moveit_config",
        "v3_dual_arm_5x5_replay_validator.launch.py",
        f"replay_json_path:={replay_path}",
        f"result_json_path:={result_path}",
        "edge_joint_step_deg:=1.0",
        "edge_updown_step_m:=0.01",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=_VALIDATOR_TIMEOUT_S,
        check=False,
    )
    if not result_path.is_file():
        tail = " | ".join(completed.stdout.splitlines()[-3:])
        return False, {}, f"validator_exit_{completed.returncode}: {tail}"
    result = read_json(result_path)
    return bool(result.get("success")), result, validation_failure(result)


def normalize_failure(stage: str, reason: str) -> str:
    if not stage and not reason:
        return "unknown"
    text = f"{stage}: {reason}".strip(": ")
    if "precontact_ik" in text:
        return "precontact_ik"
    if "cartesian_approach" in text:
        return "cartesian_approach"
    if "cartesian_retreat" in text:
        return "cartesian_retreat"
    if "collision:" in text or "edge collision:" in text:
        return "dual_fcl_collision"
    if "bridge" in text:
        return "entry_bridge"
    return text[:240]


def save_progress(path: Path, result: dict[str, Any]) -> None:
    write_json(path, result)


def process_pair(request: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    pair_key = "-".join(f"{int(value):02d}" for value in request["box_ids"])
    pair_dir = _OUTPUT_ROOT / f"pair-{pair_key}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    result_path = pair_dir / "result.json"
    if result_path.is_file():
        existing = read_json(result_path)
        if existing.get("status") in {"success", "exhausted"}:
            existing["resumed"] = True
            return existing

    assert _CONVEYOR is not None
    targets = {int(item["box_id"]): item for item in request["targets"]}
    centers = {
        box_id: tuple(float(value) for value in target["box_pose"]["position_m"])
        for box_id, target in targets.items()
    }
    removed = {int(value) for value in request.get("removed_box_ids", [])}
    start = planner_joints(_CURRENT_STATE)
    seed_task = {"current_state": {"joint_positions": _CURRENT_STATE}}
    attempts: list[dict[str, Any]] = []
    blocked: dict[tuple[Any, ...], str] = {}
    failure_counts: collections.Counter[str] = collections.Counter()
    progress = {
        "schema": SCHEMA,
        "pair": request["box_ids"],
        "request_id": request["request_id"],
        "different_rows": request["different_rows"],
        "status": "in_progress",
        "attempts": attempts,
    }

    for candidate in request["candidates"]:
        rank = int(candidate["rank"])
        candidate_dir = pair_dir / f"candidate-{rank:03d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        left_id = int(candidate["left_box_id"])
        right_id = int(candidate["right_box_id"])
        mode = str(candidate["grasp_mode"])
        updown = float(candidate["common_updown_m"])
        retreat = float(candidate["retreat_distance_m"])
        base_x = float(candidate["base_pose_map"][0])
        base_y = float(candidate["base_pose_map"][1])
        base_yaw = float(candidate["base_pose_map"][2])
        attempt = {
            "candidate_rank": rank,
            "left_box_id": left_id,
            "right_box_id": right_id,
            "grasp_mode": mode,
            "common_updown_m": updown,
            "retreat_distance_m": retreat,
            "base_x_m": base_x,
            "base_y_m": base_y,
            "base_yaw_rad": base_yaw,
        }
        left_prefix = (left_id, "left", mode, updown, base_x, base_y, base_yaw)
        right_prefix = (right_id, "right", mode, updown, base_x, base_y, base_yaw)
        for prefix in (left_prefix, right_prefix):
            cached_global = global_failure(prefix)
            if cached_global:
                blocked[prefix] = cached_global
        if left_prefix in blocked or right_prefix in blocked:
            reason = blocked.get(left_prefix) or blocked.get(right_prefix) or "pruned"
            attempt["outcome"] = "pruned"
            attempt["failure"] = reason
            attempts.append(attempt)
            failure_counts[reason] += 1
            continue

        left_result, left_payload = plan_single_cached(
            seed_task, left_id, "left", mode, updown, base_x,
            removed | {right_id}, centers[left_id], candidate_dir / "left.json", retreat,
            base_y,
            base_yaw,
        )
        right_result, right_payload = plan_single_cached(
            seed_task, right_id, "right", mode, updown, base_x,
            removed | {left_id}, centers[right_id], candidate_dir / "right.json", retreat,
            base_y,
            base_yaw,
        )
        for prefix, side_result in (
            (left_prefix, left_result), (right_prefix, right_result)
        ):
            stage = str(side_result.get("failure_stage", ""))
            if not side_result.get("success") and stage in {
                "precontact_ik", "cartesian_approach"
            }:
                raw_reason = str(side_result.get("failure_reason", ""))
                blocked[prefix] = normalize_failure(
                    stage, raw_reason
                )
                mark_global_failure(prefix, stage, raw_reason)
        if not left_result.get("success") or not right_result.get("success"):
            left_failure = normalize_failure(
                str(left_result.get("failure_stage", "")),
                str(left_result.get("failure_reason", "")),
            ) if not left_result.get("success") else ""
            right_failure = normalize_failure(
                str(right_result.get("failure_stage", "")),
                str(right_result.get("failure_reason", "")),
            ) if not right_result.get("success") else ""
            reason = left_failure or right_failure
            attempt.update({
                "outcome": "single_arm_failure",
                "left_failure": left_failure,
                "right_failure": right_failure,
            })
            attempts.append(attempt)
            failure_counts[reason] += 1
            if len(attempts) % 8 == 0:
                save_progress(result_path, progress)
            continue

        names = [str(value) for value in left_payload["joint_names"]]
        goal = list(start)
        goal[names.index("updown")] = updown
        for index in active_indices(names, "left"):
            goal[index] = float(left_payload["frames"][0]["joints"][index])
        for index in active_indices(names, "right"):
            goal[index] = float(right_payload["frames"][0]["joints"][index])
        bridge_result, bridge = plan_bridge_cached(
            seed_task, candidate, removed, centers, start, goal,
            candidate_dir / "bridge.json",
        )
        if not bridge_result.get("success"):
            reason = normalize_failure(
                str(bridge_result.get("failure_stage", "entry_bridge")),
                str(bridge_result.get("failure_reason", "")),
            )
            attempt.update({"outcome": "bridge_failure", "failure": reason})
            attempts.append(attempt)
            failure_counts[reason] += 1
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
        try:
            operation, terminal = compose_pair_cycle(
                _CONVEYOR, left_task, right_task, list(start),
                set(removed), bridge,
            )
        except ValueError as error:
            reason = f"composition: {error}"
            attempt.update({"outcome": "composition_failure", "failure": reason})
            attempts.append(attempt)
            failure_counts["composition"] += 1
            continue
        if max(abs(a - b) for a, b in zip(terminal, start)) > 1.0e-8:
            attempt.update({
                "outcome": "composition_failure",
                "failure": "cycle does not return to start posture",
            })
            attempts.append(attempt)
            failure_counts["terminal_state"] += 1
            continue
        one_sided = sum(
            bool(frame.get("left_attached")) != bool(frame.get("right_attached"))
            for frame in operation["frames"]
        )
        if one_sided:
            attempt.update({
                "outcome": "composition_failure",
                "failure": f"one-sided attachment frames={one_sided}",
            })
            attempts.append(attempt)
            failure_counts["one_sided_attachment"] += 1
            continue

        operation.update({
            "label": request["request_id"],
            "row": f"{left_target['row_from_top']}+{right_target['row_from_top']}",
            "transition_strategy": "fresh_all_pair_whole_body_bridge",
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
        replay_path = candidate_dir / "replay.json"
        validation_path = candidate_dir / "validation.json"
        write_json(replay_path, replay)
        valid, validation, validation_reason = validate_candidate(
            replay_path, validation_path
        )
        if not valid:
            reason = normalize_failure("dual_validation", validation_reason)
            attempt.update({
                "outcome": "dual_validation_failure",
                "failure": reason,
                "validation_reason": validation_reason,
            })
            attempts.append(attempt)
            failure_counts[reason] += 1
            continue

        attempt.update({
            "outcome": "success",
            "left_planning_ms": float(left_payload.get("total_ms", 0.0)),
            "right_planning_ms": float(right_payload.get("total_ms", 0.0)),
            "bridge_planning_ms": float(bridge.get("total_ms", 0.0)),
            "frame_count": len(operation["frames"]),
            "checked_frames": int(validation.get("checked_frames", 0)),
            "checked_edge_samples": int(validation.get("checked_edge_samples", 0)),
            "one_sided_attachment_frames": one_sided,
        })
        attempts.append(attempt)
        result = {
            **progress,
            "status": "success",
            "success": True,
            "selected": attempt,
            "attempt_count": len(attempts),
            "failure_counts": dict(failure_counts),
            "elapsed_s": time.monotonic() - started,
        }
        save_progress(result_path, result)
        return result

    result = {
        **progress,
        "status": "exhausted",
        "success": False,
        "selected": None,
        "attempt_count": len(attempts),
        "failure_counts": dict(failure_counts),
        "elapsed_s": time.monotonic() - started,
    }
    save_progress(result_path, result)
    return result


def matrix_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    failures: collections.Counter[str] = collections.Counter()
    for result in results:
        if not result.get("success"):
            failures.update(result.get("failure_counts", {}))
    successes = sum(bool(result.get("success")) for result in results)
    return {
        "schema": SCHEMA,
        "model_revision": "robot_v3.2.2-suction",
        "scene_contract": "full 5x5 wall; only the requested pair is absent from obstacles",
        "pair_count": len(results),
        "successes": successes,
        "failures": len(results) - successes,
        "success_rate": successes / len(results) if results else 0.0,
        "total_checked_frames": sum(
            int((result.get("selected") or {}).get("checked_frames", 0))
            for result in results
        ),
        "total_checked_edge_samples": sum(
            int((result.get("selected") or {}).get("checked_edge_samples", 0))
            for result in results
        ),
        "one_sided_attachment_frames": sum(
            int((result.get("selected") or {}).get("one_sided_attachment_frames", 0))
            for result in results
        ),
        "failed_pairs": [
            {
                "pair": result["pair"],
                "attempt_count": result.get("attempt_count", 0),
                "failure_counts": result.get("failure_counts", {}),
            }
            for result in sorted(results, key=lambda value: value["pair"])
            if not result.get("success")
        ],
        "aggregate_failure_reasons": dict(failures.most_common()),
        "results": sorted(results, key=lambda value: value["pair"]),
    }


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "box_a", "box_b", "success", "candidate_rank", "left_box_id",
        "right_box_id", "grasp_mode", "common_updown_m", "retreat_distance_m",
        "base_x_m", "base_y_m", "base_yaw_rad", "attempt_count",
        "checked_frames", "checked_edge_samples",
        "elapsed_s", "failure_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda value: value["pair"]):
            selected = result.get("selected") or {}
            writer.writerow({
                "box_a": result["pair"][0],
                "box_b": result["pair"][1],
                "success": bool(result.get("success")),
                "candidate_rank": selected.get("candidate_rank", ""),
                "left_box_id": selected.get("left_box_id", ""),
                "right_box_id": selected.get("right_box_id", ""),
                "grasp_mode": selected.get("grasp_mode", ""),
                "common_updown_m": selected.get("common_updown_m", ""),
                "retreat_distance_m": selected.get("retreat_distance_m", ""),
                "base_x_m": selected.get("base_x_m", ""),
                "base_y_m": selected.get("base_y_m", ""),
                "base_yaw_rad": selected.get("base_yaw_rad", ""),
                "attempt_count": result.get("attempt_count", 0),
                "checked_frames": selected.get("checked_frames", 0),
                "checked_edge_samples": selected.get("checked_edge_samples", 0),
                "elapsed_s": f"{float(result.get('elapsed_s', 0.0)):.3f}",
                "failure_counts": json.dumps(
                    result.get("failure_counts", {}), ensure_ascii=False,
                    separators=(",", ":"),
                ),
            })


def main() -> int:
    global _FAMILY, _CURRENT_STATE, _OUTPUT_ROOT
    global _BACKEND_SCRIPT, _SCRIPT_DIR, _VALIDATOR_TIMEOUT_S
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--backend-script", type=Path, required=True)
    parser.add_argument("--script-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--validator-timeout", type=float, default=90.0)
    parser.add_argument("--candidate-limit", type=int, default=200)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    template = read_json(args.template)
    _FAMILY = read_json(args.family)
    requests = []
    for left in range(1, 26):
        for right in range(left + 1, 26):
            requests.append({
                "request_id": f"pair-{left:02d}-{right:02d}",
                "box_ids": [left, right],
                "removed_box_ids": [],
                "candidate_limit": args.candidate_limit,
            })
    if args.limit > 0:
        requests = requests[: args.limit]
    contract = {
        "schema": PAIR_SCHEMA,
        "wall": template["wall"],
        "current_state": template["current_state"],
        "requests": requests,
    }
    pair_plan = compile_pair_contract(contract, _FAMILY)
    if pair_plan["compile_failures"]:
        parser.error(f"pair compilation failures: {pair_plan['compile_failures'][:3]}")
    columns = int(pair_plan["wall"]["columns"])
    for request in pair_plan["requests"]:
        active = {int(value) for value in request["box_ids"]}
        removed = {int(value) for value in request["removed_box_ids"]}
        targets = {int(item["box_id"]): item for item in request["targets"]}
        retained = []
        for candidate in request["candidates"]:
            if candidate["grasp_mode"] != "top_suction":
                retained.append(candidate)
                continue
            exposed = True
            for box_id in active:
                target = targets[box_id]
                row = int(target["row_from_top"])
                column = int(target["column_from_left"])
                above = {
                    (above_row - 1) * columns + column
                    for above_row in range(1, row)
                }
                if above - active - removed:
                    exposed = False
                    break
            if exposed:
                retained.append(candidate)
        request["geometric_pruned_candidates"] = (
            len(request["candidates"]) - len(retained)
        )
        request["candidates"] = retained
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.plan, pair_plan)
    _CURRENT_STATE = [
        float(value) for value in pair_plan["current_state"]["joint_positions"]
    ]
    _OUTPUT_ROOT = args.output_root.resolve()
    _BACKEND_SCRIPT = str(args.backend_script.resolve())
    _SCRIPT_DIR = str(args.script_dir.resolve())
    _VALIDATOR_TIMEOUT_S = args.validator_timeout

    completed: list[dict[str, Any]] = []
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=worker_init
    ) as executor:
        futures = {
            executor.submit(process_pair, request): request
            for request in pair_plan["requests"]
        }
        for future in concurrent.futures.as_completed(futures):
            request = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "schema": SCHEMA,
                    "pair": request["box_ids"],
                    "request_id": request["request_id"],
                    "different_rows": request["different_rows"],
                    "status": "exhausted",
                    "success": False,
                    "selected": None,
                    "attempt_count": 0,
                    "failure_counts": {f"worker_exception: {error}": 1},
                    "elapsed_s": 0.0,
                }
            completed.append(result)
            summary = matrix_summary(completed)
            summary["planned_total"] = len(pair_plan["requests"])
            summary["wall_elapsed_s"] = time.monotonic() - started
            write_json(args.summary, summary)
            write_csv(args.csv, completed)
            print(
                f"PROGRESS {len(completed)}/{len(pair_plan['requests'])} "
                f"success={summary['successes']} failures={summary['failures']} "
                f"pair={result['pair'][0]}+{result['pair'][1]} "
                f"status={result['status']} attempts={result.get('attempt_count', 0)}",
                flush=True,
            )

    summary = matrix_summary(completed)
    summary["planned_total"] = len(pair_plan["requests"])
    summary["wall_elapsed_s"] = time.monotonic() - started
    write_json(args.summary, summary)
    write_csv(args.csv, completed)
    print(
        f"RESULT success={summary['successes']}/{summary['pair_count']} "
        f"failures={summary['failures']} summary={args.summary}"
    )
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
