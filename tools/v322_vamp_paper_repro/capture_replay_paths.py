#!/usr/bin/env python3
"""Capture one real online RRT-Connect path per box for both backends."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("final_trials_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--domain-id", type=int, default=223)
    parser.add_argument("--trial-timeout", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def box_spec(box_id: int) -> tuple[float, float, str]:
    row, column = divmod(box_id - 1, 5)
    return 0.8 - 0.4 * column, 1.8 - 0.4 * row, (
        "left" if column < 2 or box_id == 23 else "right"
    )


def as_bool(value: str) -> bool:
    return value == "True"


def as_float(value: str) -> float | None:
    return float(value) if value else None


def choose_representatives(path: Path) -> dict[int, dict]:
    grouped = {box_id: [] for box_id in range(1, 26)}
    with path.open() as source:
        for source_row in csv.DictReader(source):
            row = {
                **source_row,
                "box_id": int(source_row["box_id"]),
                "halton_skip": int(source_row["halton_skip"] or 0),
                "endpoint_ready": as_bool(source_row["endpoint_ready"]),
                "ompl_exact": as_bool(source_row["ompl_exact"]),
                "vamp_native_solved": as_bool(source_row["vamp_native_solved"]),
                "vamp_fcl_valid": as_bool(source_row["vamp_fcl_valid"]),
                "paired_certified": as_bool(source_row["paired_certified"]),
                "ompl_ms": as_float(source_row["ompl_ms"]),
                "vamp_ms": as_float(source_row["vamp_ms"]),
            }
            grouped[row["box_id"]].append(row)

    selected = {}
    for box_id, rows in grouped.items():
        ompl_values = [row["ompl_ms"] for row in rows if row["ompl_ms"] is not None]
        vamp_values = [row["vamp_ms"] for row in rows if row["vamp_ms"] is not None]
        ompl_median = statistics.median(ompl_values) if ompl_values else None
        vamp_median = statistics.median(vamp_values) if vamp_values else None

        def distance(row: dict) -> float:
            result = 0.0
            for key, target in (("ompl_ms", ompl_median), ("vamp_ms", vamp_median)):
                value = row[key]
                if value is not None and target is not None and value > 0.0 and target > 0.0:
                    result += abs(math.log(value / target))
            return result

        selected[box_id] = max(
            rows,
            key=lambda row: (
                row["endpoint_ready"],
                row["paired_certified"],
                int(row["ompl_exact"]) + int(row["vamp_native_solved"]),
                row["vamp_fcl_valid"],
                -distance(row),
                -row["halton_skip"],
            ),
        )
    return selected


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
    selected = choose_representatives(args.final_trials_csv.resolve())
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    log_dir = output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["ROS_DOMAIN_ID"] = str(args.domain_id)
    environment["ROS_LOCALHOST_ONLY"] = "1"
    manifest = []

    for box_id in range(1, 26):
        row = selected[box_id]
        skip = row["halton_skip"]
        y, z, side = box_spec(box_id)
        output = raw_dir / f"box-{box_id:02d}-skip-{skip:06d}.json"
        log_path = log_dir / f"box-{box_id:02d}-skip-{skip:06d}.log"
        payload = None
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
                "rrt_planning_time:=2.0", "ompl_simplify_solutions:=false",
                "vamp_max_iterations:=1000000", "vamp_max_samples:=2000000",
                f"paper_halton_skip:={skip}",
                f"result_json_path:={output}",
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
                raise RuntimeError(f"box {box_id} produced no JSON; see {log_path}")

        benchmark = payload.get("paper_benchmark", {})
        ompl = benchmark.get("ompl", {})
        vamp = benchmark.get("vamp", {})
        record = {
            "box_id": box_id,
            "halton_skip": skip,
            "selection_source": row["raw_json"],
            "capture_json": str(output),
            "endpoint_ready": bool(benchmark),
            "ompl_success": bool(ompl.get("success")),
            "ompl_replay_states": int(ompl.get("replay_state_count", 0)),
            "vamp_native_solved": int(vamp.get("audit", {}).get("sparse_states", 0)) >= 2,
            "vamp_fcl_valid": vamp.get("audit", {}).get("fcl_audit_valid") is True,
            "vamp_replay_states": int(vamp.get("replay_state_count", 0)),
            "shortcut_used": bool(benchmark.get("shortcut_used", False)),
            "cartesian_used": bool(benchmark.get("cartesian_used", False)),
            "fallback_used": bool(benchmark.get("fallback_used", False)),
        }
        if record["shortcut_used"] or record["cartesian_used"] or record["fallback_used"]:
            raise RuntimeError(f"box {box_id} violated pure RRT replay contract")
        manifest.append(record)
        print(
            f"box {box_id:02d} skip={skip}: "
            f"OMPL frames={record['ompl_replay_states']} success={record['ompl_success']} | "
            f"VAMP frames={record['vamp_replay_states']} "
            f"native={record['vamp_native_solved']} fcl={record['vamp_fcl_valid']}"
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {"schema": "alfa.v322.vamp_paper_replay_capture.v1", "boxes": manifest},
                indent=2,
            ) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
