#!/usr/bin/env python3
"""Run matched adaptive manipulation trials for OMPL/FCL and VAMP/SIMD."""

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


LEFT_HOME = "155,-105,20,90,-90,-40,0"
RIGHT_HOME = "-155,105,-20,-90,90,40,0"
STATIONS = (-0.60, -0.35)


def parse_range(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", maxsplit=1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def box_spec(box_id: int) -> tuple[int, int, float, float]:
    row, column = divmod(box_id - 1, 5)
    return row, column, 0.8 - 0.4 * column, 1.8 - 0.4 * row


def candidate_configs(box_id: int) -> list[dict[str, Any]]:
    row, column, y, _ = box_spec(box_id)
    preferred_station = -0.60 if row < 3 else -0.35
    stations = [preferred_station, next(value for value in STATIONS if value != preferred_station)]
    preferred_arm = "left" if y > 1e-9 else "right"
    arms = [preferred_arm, "right" if preferred_arm == "left" else "left"]
    preferred_mode = "top_suction" if row == 4 and 1 <= column <= 3 else "front"
    modes = [preferred_mode, "front" if preferred_mode == "top_suction" else "top_suction"]
    output: list[dict[str, Any]] = []
    for station in stations:
        for mode in modes:
            for arm in arms:
                output.append({"base_x": station, "mode": mode, "side": arm})
    return output


def grasp_offsets(box_id: int, side: str, mode: str) -> tuple[float, float, float]:
    row, _, y, _ = box_spec(box_id)
    if mode == "top_suction":
        return 0.0, 0.0, -0.10
    front_y = (0.08 if side == "left" else -0.08) if abs(y) <= 1e-9 else 0.0
    front_z = 0.12 if row == 4 else (-0.05 if abs(y) <= 1e-9 else 0.0)
    return front_y, front_z, 0.0


def transport_pose(
    box_id: int, base_x: float, side: str, mode: str,
    front_y: float, front_z: float, top_x: float,
) -> list[float]:
    row, _, _, _ = box_spec(box_id)
    sign = 1.0 if side == "left" else -1.0
    if mode == "top_suction":
        center = [base_x + 0.90, sign * 0.40, 0.85]
        return [center[0] + top_x, center[1], center[2] + 0.20, 1.0, 0.0, 0.0, 0.0]
    if row == 4:
        center = [base_x + 1.05, sign * 0.45, 1.10]
    else:
        center = [base_x + 1.00, sign * (0.45 if box_id % 5 == 3 else 0.60), 1.20]
    root = math.sqrt(0.5)
    return [
        center[0] - 0.15,
        center[1] + front_y,
        center[2] + front_z,
        0.0, root, 0.0, root,
    ]


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
    parser.add_argument("--boxes", default="1-25")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--domain-id", type=int, default=232)
    parser.add_argument("--rrt-seconds", type=float, default=3.0)
    parser.add_argument("--search-retries", type=int, default=3)
    parser.add_argument("--vamp-max-iterations", type=int, default=1000000)
    parser.add_argument("--vamp-workers", type=int, default=1)
    parser.add_argument("--candidate-limit", type=int, default=1)
    parser.add_argument("--trial-timeout", type=float, default=180.0)
    parser.add_argument("--max-configs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.rounds < 1 or args.search_retries < 1 or args.max_configs < 1
            or args.vamp_workers < 1 or args.candidate_limit < 1):
        parser.error("rounds, search retries, and max configs must be positive")

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    log_dir = output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "trials.json"
    manifest: dict[str, Any] = {
        "schema": "alfa.v322.vamp_paper_adaptive_trials.v1",
        "contract": {
            "shared_online_ik": True,
            "ik_timed": False,
            "same_start_goal_per_backend": True,
            "same_sampler_per_backend": False,
            "comparison_scope": "integrated_industrial_backends",
            "planners": ["ompl_rrt_connect_fcl", "vamp_rrt_connect_exact_fcl"],
            "stages": ["to_precontact", "precontact_to_contact",
                       "contact_to_retreat_loaded", "retreat_to_transport_loaded",
                       "released_transport_to_home"],
            "base_stations_m": list(STATIONS),
            "shortcut_used": False,
            "cartesian_used": False,
            "historical_path_used": False,
            "fallback_used": False,
            "ik_candidate_limit_per_stage": args.candidate_limit,
        },
        "trials": [],
    }
    if args.resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = {(int(item["round"]), int(item["box_id"])) for item in manifest["trials"]}
    environment = dict(os.environ)
    environment["ROS_DOMAIN_ID"] = str(args.domain_id)
    environment["ROS_LOCALHOST_ONLY"] = "1"

    for round_index in range(args.rounds):
        for box_id in parse_range(args.boxes):
            if (round_index, box_id) in completed:
                continue
            row, column, y, z = box_spec(box_id)
            config_attempts: list[dict[str, Any]] = []
            selected: dict[str, Any] | None = None
            for config_index, config in enumerate(candidate_configs(box_id)[:args.max_configs]):
                front_y, front_z, top_x = grasp_offsets(box_id, config["side"], config["mode"])
                pose = transport_pose(
                    box_id, config["base_x"], config["side"], config["mode"],
                    front_y, front_z, top_x,
                )
                stem = f"r{round_index + 1:02d}-box-{box_id:02d}-config-{config_index:02d}"
                result_path = raw_dir / f"{stem}.json"
                log_path = log_dir / f"{stem}.log"
                skip = round_index * 100003 + box_id * 3989 + config_index * 499
                command = [
                    "ros2", "launch", "alfa_robot_moveit_config",
                    "v3_chest_carry_single_arm_box_planner.launch.py",
                    "start_rviz:=false", "start_rerun:=false",
                    "start_vehicle_pose_source:=true", "auto_run_once:=true",
                    "task_mode:=paper_adaptive_multistage_benchmark",
                    "end_effector:=scoop", "initial_arm_pose:=zero",
                    f"initial_left_arm_joints_deg:={LEFT_HOME}",
                    f"initial_right_arm_joints_deg:={RIGHT_HOME}",
                    "initial_updown:=-0.3", "initial_box_x:=0.9",
                    f"initial_box_y:={y}", f"initial_box_z:={z}",
                    f"side:={config['side']}", f"grasp_mode:={config['mode']}",
                    f"base_x:={config['base_x']}", "base_y:=0.0", "base_yaw:=0.0",
                    "full_box_wall_scene:=true", f"target_box_id:={box_id}",
                    "warehouse_enabled:=true", "warehouse_opening_x:=-1.18",
                    "maximum_carried_box_tilt_deg:=8.0",
                    "maximum_carried_box_center_z:=2.2",
                    f"front_suction_y_offset:={front_y}",
                    f"front_suction_z_offset:={front_z}",
                    f"top_suction_x_offset:={top_x}",
                    f"retreat_distance:={0.20 if config['mode'] == 'top_suction' else (0.30 if row == 4 else 0.35)}",
                    "place_tcp_pose:=" + ",".join(f"{value:.12f}" for value in pose),
                    f"precontact_candidate_limit:={args.candidate_limit}",
                    f"rrt_planning_time:={args.rrt_seconds}",
                    "rrt_planning_attempts:=1", "ompl_simplify_solutions:=false",
                    "vamp_range:=0.08",
                    f"vamp_workers:={args.vamp_workers}",
                    f"vamp_max_iterations:={args.vamp_max_iterations}",
                    f"vamp_max_samples:={args.vamp_max_iterations * 2}",
                    f"paper_halton_skip:={skip}",
                    f"paper_search_retries:={args.search_retries}",
                    f"result_json_path:={result_path}",
                ]
                payload: dict[str, Any] | None = None
                if args.resume and result_path.is_file():
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    with log_path.open("w", encoding="utf-8") as log:
                        process = subprocess.Popen(
                            command, stdout=log, stderr=subprocess.STDOUT,
                            env=environment, start_new_session=True, text=True,
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
                accumulated = benchmark.get("accumulated_candidate_search", {})
                record = {
                    "config_index": config_index,
                    **config,
                    "halton_skip": skip,
                    "success": bool((payload or {}).get("success")),
                    "failure_stage": (payload or {}).get("failure_stage", "timeout"),
                    "failure_reason": (payload or {}).get("failure_reason", "no result JSON"),
                    "ompl_ms": accumulated.get("ompl_planner_ms"),
                    "vamp_ms": accumulated.get("vamp_planner_ms"),
                    "speedup": accumulated.get("speedup_ompl_over_vamp"),
                    "result_json": str(result_path),
                }
                config_attempts.append(record)
                print(
                    f"{stem} station={config['base_x']:+.2f} {config['side']} "
                    f"{config['mode']} success={record['success']} "
                    f"OMPL={record['ompl_ms']} VAMP={record['vamp_ms']}",
                    flush=True,
                )
                if record["success"]:
                    selected = record
                    break
            trial = {
                "round": round_index,
                "box_id": box_id,
                "row": row + 1,
                "column": column + 1,
                "box_center": [0.9, y, z],
                "success": selected is not None,
                "selected": selected,
                "config_attempts": config_attempts,
            }
            manifest["trials"].append(trial)
            write_json(manifest_path, manifest)
    return 0 if all(item["success"] for item in manifest["trials"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
