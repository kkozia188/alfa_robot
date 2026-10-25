#!/usr/bin/env python3
"""Run paired OMPL/VAMP searches over identical validated full-flow IK endpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any


DEFAULT_SHARED_IK = Path(
    "/home/tim/alfa_robot-v322-pure-rrt-fullflow/results/"
    "v322_pure_rrt_fullflow/final_payloads"
)


def parse_range(value: str) -> list[int]:
    output: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", maxsplit=1))
            output.extend(range(start, end + 1))
        else:
            output.append(int(part))
    return output


def degrees(values: list[float]) -> str:
    return ",".join(f"{math.degrees(value):.12f}" for value in values)


def stop_process(process: subprocess.Popen[str]) -> None:
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--shared-ik-dir", type=Path, default=DEFAULT_SHARED_IK)
    parser.add_argument("--boxes", default="1-25")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rrt-seconds", type=float, default=30.0)
    parser.add_argument("--search-retries", type=int, default=3)
    parser.add_argument("--vamp-workers", type=int, default=1)
    parser.add_argument("--vamp-max-iterations", type=int, default=3000000)
    parser.add_argument("--trial-timeout", type=float, default=900.0)
    parser.add_argument("--domain-id", type=int, default=152)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sphere-world", action="store_true")
    parser.add_argument("--collision-inset", type=float, default=0.002)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    raw_dir, log_dir = output_dir / "raw", output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "trials.json"
    document: dict[str, Any] = {
        "schema": "alfa.v322.shared_ik_paired_trials.v1",
        "contract": {
            "shared_ik_source": str(args.shared_ik_dir.resolve()),
            "source_stage_endpoints_only": True,
            "same_start_goal_per_backend": True,
            "ik_timed": False,
            "historical_path_used": False,
            "shortcut_used": False,
            "cartesian_used": False,
            "fallback_used": False,
            "ompl_collision_backend": "moveit_fcl_mesh",
            "vamp_collision_backend": (
                "cricket_simd_sphere_world" if args.sphere_world else "moveit_fcl_mesh"
            ),
            "vamp_online_fcl_used": not args.sphere_world,
        },
        "trials": [],
    }
    if args.resume and manifest_path.is_file():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = {
        (int(item["round"]), int(item["box_id"]))
        for item in document["trials"] if item.get("success")
    }
    environment = dict(os.environ)
    environment["ROS_DOMAIN_ID"] = str(args.domain_id)
    environment["ROS_LOCALHOST_ONLY"] = "1"
    for round_index in range(args.rounds):
        for box_id in parse_range(args.boxes):
            if (round_index, box_id) in completed:
                continue
            source_path = (args.shared_ik_dir / f"box-{box_id:02d}.json").resolve()
            source = json.loads(source_path.read_text(encoding="utf-8"))
            names = [str(value) for value in source["joint_names"]]
            first = dict(zip(names, map(float, source["frames"][0]["joints"])))
            initial_left = [first[f"left_joint{index}"] for index in range(1, 8)]
            initial_right = [first[f"right_joint{index}"] for index in range(1, 8)]
            x, y, z = map(float, source["box_center"])
            base_x, base_y, base_yaw = map(float, source["base_pose_map"])
            removed_box_ids = ",".join(
                str(value) for value in source.get("removed_box_ids", [])
            )
            stem = f"r{round_index + 1:02d}-box-{box_id:02d}"
            result_path = raw_dir / f"{stem}.json"
            log_path = log_dir / f"{stem}.log"
            skip = round_index * 100003 + box_id * 3989
            command = [
                "ros2", "launch", "alfa_robot_moveit_config",
                "v3_chest_carry_single_arm_box_planner.launch.py",
                "start_rviz:=false", "start_rerun:=false", "start_vehicle_pose_source:=true",
                "auto_run_once:=true",
                "task_mode:=" + (
                    "paper_sphere_world_fullflow_benchmark"
                    if args.sphere_world else "paper_shared_ik_fullflow_benchmark"
                ),
                "end_effector:=scoop", "initial_arm_pose:=zero",
                f"initial_updown:={first['updown']}",
                f"initial_left_arm_joints_deg:={degrees(initial_left)}",
                f"initial_right_arm_joints_deg:={degrees(initial_right)}",
                f"initial_box_x:={x}", f"initial_box_y:={y}", f"initial_box_z:={z}",
                f"side:={source['side']}", f"grasp_mode:={source['grasp_mode']}",
                f"base_x:={base_x}", f"base_y:={base_y}", f"base_yaw:={base_yaw}",
                "full_box_wall_scene:=true", f"target_box_id:={box_id}",
                "warehouse_enabled:=true", "warehouse_opening_x:=-1.18",
                f"collision_inset:={args.collision_inset}",
                f"front_suction_y_offset:={source['front_suction_y_offset']}",
                f"front_suction_z_offset:={source['front_suction_z_offset']}",
                f"top_suction_x_offset:={source['top_suction_x_offset']}",
                f"maximum_carried_box_tilt_deg:={source['maximum_carried_box_tilt_deg']}",
                f"maximum_carried_box_center_z:={source['maximum_carried_box_center_z']}",
                f"rrt_planning_time:={args.rrt_seconds}", "rrt_planning_attempts:=1",
                "ompl_simplify_solutions:=false", f"paper_search_retries:={args.search_retries}",
                f"paper_halton_skip:={skip}", f"vamp_workers:={args.vamp_workers}",
                f"vamp_seed:={20261024 + skip}",
                f"vamp_max_iterations:={args.vamp_max_iterations}",
                f"vamp_max_samples:={args.vamp_max_iterations * 2}", "vamp_range:=0.08",
                f"paper_shared_ik_path:={source_path}", f"result_json_path:={result_path}",
            ]
            if removed_box_ids:
                command.append(f"removed_box_ids:={removed_box_ids}")
            payload: dict[str, Any] | None = None
            if args.resume and result_path.is_file():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                        start_new_session=True, text=True,
                    )
                    deadline = time.monotonic() + args.trial_timeout
                    try:
                        while time.monotonic() < deadline:
                            if result_path.is_file():
                                try:
                                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                                    break
                                except json.JSONDecodeError:
                                    pass
                            if process.poll() is not None:
                                break
                            time.sleep(0.05)
                    finally:
                        stop_process(process)
            benchmark = (payload or {}).get("paper_benchmark", {})
            search = benchmark.get("accumulated_search", {})
            record = {
                "round": round_index,
                "box_id": box_id,
                "side": source["side"], "mode": source["grasp_mode"],
                "base_x": source["base_pose_map"][0],
                "success": bool((payload or {}).get("success")),
                "failure_stage": (payload or {}).get("failure_stage", "timeout"),
                "failure_reason": (payload or {}).get("failure_reason", "no result JSON"),
                "ompl_ms": search.get("ompl_planner_ms"),
                "vamp_ms": search.get("vamp_planner_ms"),
                "speedup": search.get("speedup_ompl_over_vamp"),
                "shared_ik_source": str(source_path),
                "result_json": str(result_path.resolve()),
            }
            document["trials"] = [
                item for item in document["trials"]
                if (int(item["round"]), int(item["box_id"])) != (round_index, box_id)
            ]
            document["trials"].append(record)
            write_json(manifest_path, document)
            print(
                f"round={round_index + 1} box={box_id:02d} success={record['success']} "
                f"OMPL={record['ompl_ms']} VAMP={record['vamp_ms']} speedup={record['speedup']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
