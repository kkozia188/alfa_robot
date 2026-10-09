#!/usr/bin/env python3
"""Plan, compose, validate, and classify one complete X/Y/Yaw case."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from matrix_common import (
    MODEL_REVISION,
    TOOL0_OFFSET_LOCAL_Z_M,
    UPSTREAM_BASE_COMMIT,
    case_slug,
    classify_failure,
    evaluate_quality,
    first_failed_attempt,
    pose_contract,
    read_json,
    write_json,
)


ROOT = Path(__file__).resolve().parent
PICKUP_PLANNER = ROOT / "plan_pose_pickups.py"
CACHE_PLANNER = ROOT / "plan_pose_cache.py"
CONVEYOR_BUILDER = ROOT / "v3_pose_conveyor_shuttle.py"


def run_logged(
    command: list[str], log_path: Path, timeout_s: float
) -> tuple[int, float, bool]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            return_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait()
    return return_code, (time.monotonic() - started) * 1000.0, timed_out


def log_tail(path: Path, lines: int = 12) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def operation_metrics(
    operation: dict[str, Any], joint_names: list[str]
) -> tuple[float, int, float]:
    arm_indices = [
        index for index, name in enumerate(joint_names)
        if name.startswith("left_joint") or name.startswith("right_joint")
    ]
    maximum_step_deg = 0.0
    joint_flips = 0
    base_travel_m = 0.0
    default_pose = [float(value) for value in operation.get("base_pose_map", [0, 0, 0])]
    for previous, current in zip(operation["frames"], operation["frames"][1:]):
        for index in arm_indices:
            step = math.degrees(abs(
                float(current["joints"][index]) - float(previous["joints"][index])
            ))
            maximum_step_deg = max(maximum_step_deg, step)
            joint_flips += step > 180.0 + 1e-6
        before = previous.get("base_pose_map", default_pose)
        after = current.get("base_pose_map", default_pose)
        base_travel_m += math.hypot(
            float(after[0]) - float(before[0]),
            float(after[1]) - float(before[1]),
        )
    return maximum_step_deg, joint_flips, base_travel_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dx-m", type=float, required=True)
    parser.add_argument("--base-dy-m", type=float, required=True)
    parser.add_argument("--yaw-deg", type=float, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--continuation-cache", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--spec", type=Path,
        default=ROOT / "default_matrix.json",
    )
    parser.add_argument("--pickup-attempt-timeout-s", type=float, default=120.0)
    parser.add_argument("--bridge-attempt-timeout-s", type=float, default=360.0)
    parser.add_argument("--stage-wall-timeout-s", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def fail_result(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    stage: str,
    reason: str,
    detail_stage: str = "",
    return_code: int,
    timed_out: bool,
    timings: dict[str, float],
) -> int:
    failed = (
        first_failed_attempt(output_dir)
        if stage in {"pickup_planning", "bridge_planning"} else {}
    )
    failure_stage = str(failed.get("failure_stage", detail_stage or stage))
    failure_reason = str(failed.get("failure_reason", reason))
    failure_class = (
        "case_timeout" if timed_out
        else classify_failure(failure_stage, failure_reason)
    )
    write_json(output_dir / "case-result.json", {
        "schema": "alfa.v322_docking_error_case.v1",
        "case_id": case_slug(args.base_dx_m, args.base_dy_m, args.yaw_deg),
        "model_revision": MODEL_REVISION,
        "tool0_offset_local_z_m": TOOL0_OFFSET_LOCAL_Z_M,
        "upstream_base_commit": UPSTREAM_BASE_COMMIT,
        **pose_contract(args.base_dx_m, args.base_dy_m, args.yaw_deg),
        "status": "failed",
        "failed_stage": stage,
        "failure_class": failure_class,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "failed_attempt": failed,
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_timings_ms": timings,
    })
    print(
        f"CASE failed case={case_slug(args.base_dx_m, args.base_dy_m, args.yaw_deg)} "
        f"stage={stage} class={failure_class}",
        flush=True,
    )
    return 2


def main() -> int:
    args = parse_args()
    spec = read_json(args.spec.resolve())
    thresholds = spec["thresholds"]
    slug = case_slug(args.base_dx_m, args.base_dy_m, args.yaw_deg)
    output_dir = args.output_root.resolve() / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "v322-pose-plan-cache.json"
    bridge_map_path = output_dir / "dual-bridge-map.json"
    replay_path = output_dir / "v322-pose-conveyor-replay.json"
    validation_path = output_dir / "v322-pose-conveyor-validation.json"
    timings: dict[str, float] = {}

    commands: list[tuple[str, list[str]]] = []
    pickup = [
        "/usr/bin/python3", str(PICKUP_PLANNER),
        "--base-dx-m", f"{args.base_dx_m:.6f}",
        "--base-dy-m", f"{args.base_dy_m:.6f}",
        "--yaw-deg", f"{args.yaw_deg:.6f}",
        "--baseline-cache", str(args.baseline_cache.resolve()),
        "--output-dir", str(output_dir),
        "--timeout-s", f"{args.pickup_attempt_timeout_s:.3f}",
    ]
    if args.continuation_cache:
        pickup.extend((
            "--continuation-cache", str(args.continuation_cache.resolve())
        ))
    if args.resume:
        pickup.append("--resume")
    commands.append(("pickup_planning", pickup))

    cache = [
        "/usr/bin/python3", str(CACHE_PLANNER),
        "--base-dx-m", f"{args.base_dx_m:.6f}",
        "--base-dy-m", f"{args.base_dy_m:.6f}",
        "--yaw-deg", f"{args.yaw_deg:.6f}",
        "--pickups-dir", str(output_dir / "pickups"),
        "--baseline-cache", str(args.baseline_cache.resolve()),
        "--output-dir", str(output_dir),
        "--timeout-s", f"{args.bridge_attempt_timeout_s:.3f}",
    ]
    if args.resume:
        cache.append("--resume")
    commands.append(("bridge_planning", cache))

    for stage, command in commands:
        code, elapsed, timed_out = run_logged(
            command, output_dir / f"{stage}.log", args.stage_wall_timeout_s
        )
        timings[f"{stage}_wall_ms"] = elapsed
        if code != 0 or timed_out:
            return fail_result(
                output_dir=output_dir, args=args, stage=stage,
                reason=log_tail(output_dir / f"{stage}.log"),
                return_code=code, timed_out=timed_out, timings=timings,
            )

    bridge_map = read_json(bridge_map_path)
    compose_command = [
        "/usr/bin/python3", str(CONVEYOR_BUILDER),
        "--plan-cache", str(cache_path),
        "--output", str(replay_path),
        "--backoff-m", "2.35",
        "--right-shuttle-m", "1.50",
        "--base-step-m", "0.01",
        "--dual-transition-bridge-map", json.dumps(
            bridge_map, separators=(",", ":")
        ),
    ]
    code, elapsed, timed_out = run_logged(
        compose_command, output_dir / "replay_composition.log", 180.0
    )
    timings["replay_composition_wall_ms"] = elapsed
    if code != 0 or timed_out:
        return fail_result(
            output_dir=output_dir, args=args, stage="replay_composition",
            reason=log_tail(output_dir / "replay_composition.log"),
            return_code=code, timed_out=timed_out, timings=timings,
        )

    validation_command = [
        "ros2", "launch", "alfa_robot_moveit_config",
        "v3_dual_arm_5x5_replay_validator.launch.py",
        f"replay_json_path:={replay_path}",
        f"result_json_path:={validation_path}",
        "edge_joint_step_deg:=1.0",
        "edge_updown_step_m:=0.01",
    ]
    code, elapsed, timed_out = run_logged(
        validation_command, output_dir / "validation.log", 600.0
    )
    timings["validation_wall_ms"] = elapsed
    validation = read_json(validation_path) if validation_path.is_file() else {}
    if code != 0 or timed_out or not validation.get("success"):
        reason = log_tail(output_dir / "validation.log")
        detail_stage = "validation"
        if validation:
            failed_operation = next(
                (item for item in validation.get("operations", []) if not item.get("success")),
                {},
            )
            detail_stage = str(failed_operation.get("failure_stage", detail_stage))
            reason = str(failed_operation.get("failure_reason", reason))
        return fail_result(
            output_dir=output_dir, args=args, stage="validation",
            reason=reason, detail_stage=detail_stage,
            return_code=code, timed_out=timed_out, timings=timings,
        )

    cache_value = read_json(cache_path)
    replay = read_json(replay_path)
    expected = pose_contract(args.base_dx_m, args.base_dy_m, args.yaw_deg)
    observed = {
        tuple(float(value) for value in task["payload"]["base_pose_map"])
        for task in cache_value["tasks"]
    }
    expected_poses = {
        tuple(float(value) for value in expected["upper_base_pose_map"]),
        tuple(float(value) for value in expected["lower_base_pose_map"]),
    }
    pose_contract_ok = (
        len(observed) == len(expected_poses)
        and all(
            any(
                max(abs(left - right) for left, right in zip(actual, target)) <= 1e-6
                for target in expected_poses
            )
            for actual in observed
        )
    )
    if not pose_contract_ok:
        return fail_result(
            output_dir=output_dir, args=args, stage="pose_contract",
            reason=f"expected {sorted(expected_poses)}, observed {sorted(observed)}",
            return_code=1, timed_out=False, timings=timings,
        )
    cycle_operations = [
        operation for operation in replay["operations"]
        if operation.get("kind") in {"dual_conveyor_cycle", "single_conveyor_cycle"}
    ]
    dual_operations = [
        operation for operation in cycle_operations
        if operation.get("kind") == "dual_conveyor_cycle"
    ]
    one_sided = sum(
        bool(frame.get("left_attached")) != bool(frame.get("right_attached"))
        for operation in dual_operations for frame in operation["frames"]
    )
    if len(cycle_operations) != 15 or len(dual_operations) != 10 or one_sided:
        return fail_result(
            output_dir=output_dir, args=args, stage="task_contract",
            reason=(
                f"cycles={len(cycle_operations)}, dual={len(dual_operations)}, "
                f"one_sided={one_sided}"
            ),
            return_code=1, timed_out=False, timings=timings,
        )

    task_core_ms = [
        float(item["selected_task_core_ms"]) for item in replay["box_planning"]
    ]
    bridge_ms = [
        float(task.get("transition_core_ms", 0.0)) for task in cache_value["tasks"]
    ]
    operation_values = [
        operation_metrics(operation, list(replay["joint_names"]))
        for operation in replay["operations"]
    ]
    maximum_joint_step = max(item[0] for item in operation_values)
    joint_flips = sum(item[1] for item in operation_values)
    base_travel_m = sum(item[2] for item in operation_values)
    maximum_tilt = max(
        float(validation.get("maximum_left_tilt_deg", 0.0)),
        float(validation.get("maximum_right_tilt_deg", 0.0)),
    )
    status, findings = evaluate_quality(
        task_core_ms=task_core_ms,
        maximum_bridge_ms=max(bridge_ms, default=0.0),
        maximum_tilt_deg=maximum_tilt,
        maximum_joint_step_deg=maximum_joint_step,
        joint_flip_events=joint_flips,
        thresholds=thresholds,
    )
    result = {
        "schema": "alfa.v322_docking_error_case.v1",
        "case_id": slug,
        "model_revision": MODEL_REVISION,
        "tool0_offset_local_z_m": TOOL0_OFFSET_LOCAL_Z_M,
        "upstream_base_commit": UPSTREAM_BASE_COMMIT,
        **expected,
        "status": status,
        "quality_findings": findings,
        "completed_boxes": 25,
        "cycle_count": len(cycle_operations),
        "dual_cycle_count": len(dual_operations),
        "one_sided_attachment_frames": one_sided,
        "planning": {
            "task_core_total_ms": sum(task_core_ms),
            "task_core_average_ms": sum(task_core_ms) / len(task_core_ms),
            "task_core_max_ms": max(task_core_ms),
            "task_core_below_3s": sum(value < 3000.0 for value in task_core_ms),
            "bridge_core_max_ms": max(bridge_ms, default=0.0),
            "cache_loading_counted": False,
        },
        "trajectory": {
            "maximum_joint_step_deg": maximum_joint_step,
            "joint_flip_events": joint_flips,
            "base_travel_m": base_travel_m,
        },
        "validation": validation,
        "wall_timings_ms": timings,
        "artifacts": {
            "plan_cache": str(cache_path),
            "replay": str(replay_path),
            "validation": str(validation_path),
        },
    }
    write_json(output_dir / "case-result.json", result)
    print(
        f"CASE {status} case={slug} boxes=25/25 "
        f"frames={validation['checked_frames']} findings={findings or ['none']}",
        flush=True,
    )
    return 0 if status in {"passed", "degraded"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
