#!/usr/bin/env python3
"""Plan, compose, and MoveIt/FCL-validate one complete V3 yaw case."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PICKUP_PLANNER = ROOT / "plan_yaw_pickups.py"
CACHE_PLANNER = ROOT / "plan_yaw_cache.py"
CONVEYOR_BUILDER = ROOT / "baseline" / "v3_scoop_5x5_conveyor_shuttle.py"


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


def run(command: list[str]) -> float:
    print("COMMAND " + " ".join(command), flush=True)
    started = time.monotonic()
    subprocess.run(command, check=True)
    return (time.monotonic() - started) * 1000.0


def yaw_slug(yaw_deg: float) -> str:
    rounded = int(round(abs(yaw_deg)))
    sign = "p" if yaw_deg >= 0.0 else "m"
    return f"yaw-{sign}{rounded:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaw-deg", type=float, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--continuation-cache", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.yaw_deg) or not -5.0 <= args.yaw_deg <= 5.0:
        raise SystemExit("--yaw-deg must be finite and in [-5, 5]")
    if abs(args.yaw_deg - round(args.yaw_deg)) > 1e-9:
        raise SystemExit("certification cases require whole-degree yaw values")

    case_dir = (args.output_root.resolve() / yaw_slug(args.yaw_deg))
    case_dir.mkdir(parents=True, exist_ok=True)
    baseline_cache = args.baseline_cache.resolve()
    pickups_dir = case_dir / "pickups"
    cache_path = case_dir / "v322-yaw-plan-cache.json"
    bridge_map_path = case_dir / "dual-bridge-map.json"
    replay_path = case_dir / "v322-yaw-conveyor-replay.json"
    validation_path = case_dir / "v322-yaw-conveyor-validation.json"
    timings: dict[str, float] = {}

    pickup_command = [
        "/usr/bin/python3", str(PICKUP_PLANNER),
        "--yaw-deg", f"{args.yaw_deg:.1f}",
        "--baseline-cache", str(baseline_cache),
        "--output-dir", str(case_dir),
    ]
    if args.continuation_cache:
        pickup_command.extend((
            "--continuation-cache", str(args.continuation_cache.resolve())
        ))
    if args.resume:
        pickup_command.append("--resume")
    timings["pickup_planning_wall_ms"] = run(pickup_command)

    cache_command = [
        "/usr/bin/python3", str(CACHE_PLANNER),
        "--yaw-deg", f"{args.yaw_deg:.1f}",
        "--pickups-dir", str(pickups_dir),
        "--baseline-cache", str(baseline_cache),
        "--output-dir", str(case_dir),
        "--timeout-s", "45.0",
    ]
    if args.resume:
        cache_command.append("--resume")
    timings["bridge_planning_wall_ms"] = run(cache_command)

    timings["replay_composition_wall_ms"] = run([
        "/usr/bin/python3", str(CONVEYOR_BUILDER),
        "--plan-cache", str(cache_path),
        "--output", str(replay_path),
        "--backoff-m", "2.35",
        "--right-shuttle-m", "1.50",
        "--base-step-m", "0.01",
        "--dual-transition-bridge-map", json.dumps(
            read_json(bridge_map_path), separators=(",", ":")
        ),
    ])

    timings["validation_wall_ms"] = run([
        "ros2", "launch", "alfa_robot_moveit_config",
        "v3_dual_arm_5x5_replay_validator.launch.py",
        f"replay_json_path:={replay_path}",
        f"result_json_path:={validation_path}",
        "edge_joint_step_deg:=1.0",
        "edge_updown_step_m:=0.01",
    ])
    validation = read_json(validation_path)
    if not validation.get("success"):
        raise RuntimeError(f"MoveIt/FCL validation failed for yaw {args.yaw_deg:+.0f}")

    cache = read_json(cache_path)
    replay = read_json(replay_path)
    task_yaws = {
        round(math.degrees(float(task["payload"]["base_pose_map"][2])), 4)
        for task in cache["tasks"]
    }
    expected_yaw = round(args.yaw_deg, 4)
    if task_yaws != {expected_yaw}:
        raise RuntimeError(
            f"cache yaw mismatch: expected {expected_yaw}, observed {sorted(task_yaws)}"
        )
    cycle_operations = [
        operation for operation in replay["operations"]
        if operation.get("kind") in ("dual_conveyor_cycle", "single_conveyor_cycle")
    ]
    dual_operations = [
        operation for operation in cycle_operations
        if operation.get("kind") == "dual_conveyor_cycle"
    ]
    one_sided_frames = sum(
        1
        for operation in dual_operations
        for frame in operation["frames"]
        if bool(frame.get("left_attached")) != bool(frame.get("right_attached"))
    )
    if len(cycle_operations) != 15 or len(dual_operations) != 10 or one_sided_frames:
        raise RuntimeError("composed replay violates the 15-cycle/10-dual synchronization contract")
    task_times = [
        float(item["selected_task_core_ms"]) for item in replay["box_planning"]
    ]
    summary = {
        "schema": "alfa.v322_yaw_robustness_case.v1",
        "model_revision": "robot_v3.2.2-suction",
        "tool0_offset_local_z_m": 0.151,
        "upstream_base_commit": "d9c330cef72981390d81ac2b1cd5a6eb9e892195",
        "yaw_error_deg": args.yaw_deg,
        "completed_boxes": 25,
        "cycle_count": len(cycle_operations),
        "dual_cycle_count": len(dual_operations),
        "one_sided_attachment_frames": one_sided_frames,
        "planning": {
            "task_core_total_ms": sum(task_times),
            "task_core_average_ms": sum(task_times) / len(task_times),
            "task_core_max_ms": max(task_times),
            "task_core_below_3s": sum(value < 3000.0 for value in task_times),
            "cache_loading_counted": False,
        },
        "validation": validation,
        "wall_timings": timings,
        "artifacts": {
            "plan_cache": str(cache_path),
            "replay": str(replay_path),
            "validation": str(validation_path),
        },
    }
    write_json(case_dir / "summary.json", summary)
    print(
        f"YAW_CASE PASS yaw={args.yaw_deg:+.0f} boxes=25/25 cycles=15/15 "
        f"frames={validation['checked_frames']} edges={validation['checked_edge_samples']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
