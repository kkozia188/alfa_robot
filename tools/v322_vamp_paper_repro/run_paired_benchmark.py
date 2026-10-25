#!/usr/bin/env python3
"""Run paired OMPL/FCL and native VAMP RRT-Connect trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def parse_range(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", maxsplit=1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--boxes", default="1-25")
    parser.add_argument("--skips", default="0,997,1994,2991,3988")
    parser.add_argument("--domain-id", type=int, default=222)
    parser.add_argument("--trial-timeout", type=float, default=20.0)
    parser.add_argument("--rrt-seconds", type=float, default=2.0)
    parser.add_argument("--vamp-max-iterations", type=int, default=1000000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def box_spec(box_id: int) -> tuple[float, float, str]:
    if not 1 <= box_id <= 25:
        raise ValueError(f"box id outside 1..25: {box_id}")
    row, column = divmod(box_id - 1, 5)
    z = 1.8 - 0.4 * row
    y = 0.8 - 0.4 * column
    side = "left" if column < 2 or box_id == 23 else "right"
    return y, z, side


def endpoint_hash(payload: dict) -> str:
    benchmark = payload["paper_benchmark"]
    encoded = json.dumps(
        {"joint_names": benchmark["joint_names"], "start": benchmark["start"],
         "goal": benchmark["goal"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2.0)


def main() -> int:
    args = parse_args()
    boxes = parse_range(args.boxes)
    skips = parse_range(args.skips)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    log_dir = args.output_dir / "logs"
    raw_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment["ROS_DOMAIN_ID"] = str(args.domain_id)
    environment["ROS_LOCALHOST_ONLY"] = "1"

    trials = []
    for skip in skips:
        for box_id in boxes:
            y, z, side = box_spec(box_id)
            stem = f"box-{box_id:02d}-skip-{skip:06d}"
            output = (raw_dir / f"{stem}.json").resolve()
            log_path = log_dir / f"{stem}.log"
            if output.exists() and args.resume:
                payload = json.loads(output.read_text())
            else:
                if output.exists():
                    raise FileExistsError(f"refusing to overwrite {output}; use --resume")
                command = [
                    "ros2", "launch", "alfa_robot_moveit_config",
                    "v3_chest_carry_single_arm_box_planner.launch.py",
                    "start_rviz:=false", "start_rerun:=false",
                    "start_vehicle_pose_source:=true", "auto_run_once:=true",
                    "task_mode:=paper_rrt_vamp_benchmark", "initial_box_x:=0.9",
                    f"initial_box_y:={y}", f"initial_box_z:={z}", f"side:={side}",
                    "full_box_wall_scene:=true", f"target_box_id:={box_id}",
                    "base_x:=-0.35", "warehouse_enabled:=true",
                    f"rrt_planning_time:={args.rrt_seconds}",
                    "ompl_simplify_solutions:=false",
                    f"vamp_max_iterations:={args.vamp_max_iterations}",
                    f"vamp_max_samples:={args.vamp_max_iterations * 2}",
                    f"paper_halton_skip:={skip}", f"result_json_path:={output}",
                ]
                with log_path.open("w") as log:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        start_new_session=True,
                        text=True,
                    )
                    deadline = time.monotonic() + args.trial_timeout
                    payload = None
                    try:
                        while time.monotonic() < deadline:
                            if output.exists():
                                try:
                                    payload = json.loads(output.read_text())
                                    break
                                except json.JSONDecodeError:
                                    pass
                            if process.poll() is not None:
                                break
                            time.sleep(0.05)
                    finally:
                        stop_process(process)
                if payload is None:
                    raise RuntimeError(f"trial {stem} produced no JSON; see {log_path}")

            benchmark = payload.get("paper_benchmark", {})
            record = {
                "box_id": box_id,
                "halton_skip": skip,
                "side": side,
                "box_center": payload.get("box_center"),
                "endpoint_sha256": endpoint_hash(payload) if benchmark else None,
                "paired_success": bool(payload.get("success")),
                "failure_stage": payload.get("failure_stage", ""),
                "failure_reason": payload.get("failure_reason", ""),
                "execution_order": benchmark.get("execution_order"),
                "ompl_success": benchmark.get("ompl", {}).get("success", False),
                "ompl_ms": benchmark.get("ompl", {}).get("planner_ms"),
                "vamp_success": benchmark.get("vamp", {}).get("success", False),
                "vamp_ms": benchmark.get("vamp", {}).get("planner_ms"),
                "speedup": benchmark.get("speedup_ompl_over_vamp"),
                "result_json": str(output),
            }
            trials.append(record)
            print(
                f"{stem}: OMPL={record['ompl_ms']}ms/{record['ompl_success']} "
                f"VAMP={record['vamp_ms']}ms/{record['vamp_success']}"
            )
            (args.output_dir / "trials.json").write_text(
                json.dumps({"schema": "alfa.v322.vamp_paper_trials.v1", "trials": trials}, indent=2)
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
