#!/usr/bin/env python3
"""Run a fast high-risk-group screen for one Y or Yaw docking-error case."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from matrix_common import (
    MODEL_REVISION,
    SENTINEL_GROUP_INDICES,
    TOOL0_OFFSET_LOCAL_Z_M,
    UPSTREAM_BASE_COMMIT,
    case_slug,
    classify_failure,
    first_failed_attempt,
    is_single_axis_error,
    pose_contract,
    read_json,
    write_json,
)


ROOT = Path(__file__).resolve().parent
PLANNER = ROOT / "plan_pose_pickups.py"


def run_logged(command: list[str], log_path: Path, timeout_s: float) -> tuple[int, float, bool]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dx-m", type=float, required=True)
    parser.add_argument("--base-dy-m", type=float, required=True)
    parser.add_argument("--yaw-deg", type=float, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--continuation-cache", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--group-indices",
        default=",".join(map(str, SENTINEL_GROUP_INDICES)),
    )
    parser.add_argument("--planner-timeout-s", type=float, default=45.0)
    parser.add_argument("--case-timeout-s", type=float, default=900.0)
    parser.add_argument("--task-core-target-ms", type=float, default=3000.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if abs(args.base_dx_m) > 1e-12:
        raise SystemExit(
            "X uses run_x_clearance_sweep.py; common base dx is not official"
        )
    if not is_single_axis_error(args.base_dx_m, args.base_dy_m, args.yaw_deg):
        raise SystemExit(
            "single-variable experiment requires at most one nonzero error"
        )
    slug = case_slug(args.base_dx_m, args.base_dy_m, args.yaw_deg)
    output_dir = args.output_root.resolve() / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/python3", str(PLANNER),
        "--base-dx-m", f"{args.base_dx_m:.6f}",
        "--base-dy-m", f"{args.base_dy_m:.6f}",
        "--yaw-deg", f"{args.yaw_deg:.6f}",
        "--baseline-cache", str(args.baseline_cache.resolve()),
        "--output-dir", str(output_dir),
        "--group-indices", args.group_indices,
        "--timeout-s", f"{args.planner_timeout_s:.3f}",
    ]
    if args.continuation_cache:
        command.extend((
            "--continuation-cache", str(args.continuation_cache.resolve())
        ))
    if args.resume:
        command.append("--resume")
    return_code, wall_ms, timed_out = run_logged(
        command, output_dir / "planner.log", args.case_timeout_s
    )
    failed = first_failed_attempt(output_dir)
    task_core_ms: list[float] = []
    pickup_summary_path = output_dir / "pickup-summary.json"
    groups_completed = 0
    if pickup_summary_path.is_file():
        pickup_summary = read_json(pickup_summary_path)
        groups_completed = len(pickup_summary)
        for group in pickup_summary:
            for value in group.get("payloads", []):
                payload = read_json(Path(value))
                task_core_ms.append(float(payload.get("total_ms", 0.0)))
    if timed_out:
        status = "screen_failed"
        failure_class = "case_timeout"
        failure_stage = "case_timeout"
        failure_reason = f"screen exceeded {args.case_timeout_s:.1f} s"
    elif return_code != 0:
        status = "screen_failed"
        failure_stage = str(failed.get("failure_stage", "planner_exit"))
        failure_reason = str(failed.get("failure_reason", f"exit {return_code}"))
        failure_class = classify_failure(failure_stage, failure_reason)
    elif task_core_ms and max(task_core_ms) >= args.task_core_target_ms:
        status = "screen_degraded"
        failure_class = "planning_time_over_target"
        failure_stage = "task_core"
        failure_reason = (
            f"maximum selected task core {max(task_core_ms):.3f} ms is not below "
            f"{args.task_core_target_ms:.3f} ms"
        )
    else:
        status = "screen_passed"
        failure_class = ""
        failure_stage = ""
        failure_reason = ""
    report = {
        "schema": "alfa.v322_docking_error_probe.v1",
        "case_id": slug,
        "model_revision": MODEL_REVISION,
        "tool0_offset_local_z_m": TOOL0_OFFSET_LOCAL_Z_M,
        "upstream_base_commit": UPSTREAM_BASE_COMMIT,
        **pose_contract(args.base_dx_m, args.base_dy_m, args.yaw_deg),
        "scope": "pickup-only sentinel groups; not a complete-task certificate",
        "group_indices": [int(value) for value in args.group_indices.split(",")],
        "status": status,
        "return_code": return_code,
        "timed_out": timed_out,
        "groups_completed": groups_completed,
        "selected_task_count": len(task_core_ms),
        "maximum_task_core_ms": max(task_core_ms, default=0.0),
        "task_cores_below_target": sum(
            value < args.task_core_target_ms for value in task_core_ms
        ),
        "failure_class": failure_class,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "failed_attempt": failed,
        "wall_ms": wall_ms,
        "artifacts": {
            "planner_log": str(output_dir / "planner.log"),
            "pickup_summary": (
                str(pickup_summary_path) if pickup_summary_path.is_file() else ""
            ),
            "pickup_attempts": str(output_dir / "pickup-attempts.json"),
        },
    }
    write_json(output_dir / "probe-result.json", report)
    print(
        f"PROBE {status} case={slug} groups={groups_completed} "
        f"failure={failure_class or 'none'} wall_s={wall_ms / 1000.0:.2f}",
        flush=True,
    )
    return 0 if status != "screen_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
