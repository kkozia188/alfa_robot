#!/usr/bin/env python3
"""Run OMPL and VAMP full flows from the same previously validated IK endpoints."""

from __future__ import annotations

import argparse
import hashlib
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
    result: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", maxsplit=1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def degrees(values: list[float]) -> str:
    return ",".join(f"{math.degrees(value):.12f}" for value in values)


def stage_endpoints(payload: dict[str, Any]) -> dict[str, str]:
    endpoints: dict[str, list[float]] = {}
    for frame in payload.get("frames", []):
        endpoints[str(frame["stage"])] = [float(value) for value in frame["joints"]]
    return {
        stage: hashlib.sha256(
            json.dumps(values, separators=(",", ":")).encode()
        ).hexdigest()
        for stage, values in endpoints.items()
    }


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


def run_one(
    source: dict[str, Any], backend: str, result_path: Path, log_path: Path,
    args: argparse.Namespace, environment: dict[str, str], round_index: int,
) -> dict[str, Any] | None:
    names = [str(value) for value in source["joint_names"]]
    first = source["frames"][0]
    precontact = [frame for frame in source["frames"] if frame["stage"] == "rrt_to_precontact"][-1]
    first_map = dict(zip(names, map(float, first["joints"])))
    pre_map = dict(zip(names, map(float, precontact["joints"])))
    side = str(source["side"])
    initial_left = [first_map[f"left_joint{index}"] for index in range(1, 8)]
    initial_right = [first_map[f"right_joint{index}"] for index in range(1, 8)]
    pre_arm = [pre_map[f"{side}_joint{index}"] for index in range(1, 8)]
    box_id = int(source["target_box_id"])
    x, y, z = map(float, source["box_center"])
    base_x, base_y, base_yaw = map(float, source["base_pose_map"])
    retreat_distance = 0.20 if source["grasp_mode"] == "top_suction" else (0.30 if box_id > 20 else 0.35)
    seed = 20261024 + round_index * 100003 + box_id * 3989
    command = [
        "ros2", "launch", "alfa_robot_moveit_config",
        "v3_chest_carry_single_arm_box_planner.launch.py",
        "start_rviz:=false", "start_rerun:=false", "start_vehicle_pose_source:=true",
        "auto_run_once:=true", "task_mode:=full_extract", "end_effector:=scoop",
        "initial_arm_pose:=zero", f"initial_updown:={pre_map['updown']}",
        f"initial_left_arm_joints_deg:={degrees(initial_left)}",
        f"initial_right_arm_joints_deg:={degrees(initial_right)}",
        f"precontact_arm_joints_deg:={degrees(pre_arm)}",
        f"initial_box_x:={x}", f"initial_box_y:={y}", f"initial_box_z:={z}",
        f"side:={side}", f"grasp_mode:={source['grasp_mode']}",
        f"base_x:={base_x}", f"base_y:={base_y}", f"base_yaw:={base_yaw}",
        "full_box_wall_scene:=true", f"target_box_id:={box_id}",
        "warehouse_enabled:=true", "warehouse_opening_x:=-1.18",
        f"front_suction_y_offset:={source['front_suction_y_offset']}",
        f"front_suction_z_offset:={source['front_suction_z_offset']}",
        f"top_suction_x_offset:={source['top_suction_x_offset']}",
        f"retreat_distance:={retreat_distance}",
        "place_tcp_pose:=" + ",".join(str(value) for value in source["place_tcp_pose"]),
        "continuous_sequence:=false", "continuous_plan_approach:=true",
        "pure_rrt_full_flow:=true", f"pure_rrt_backend:={backend}",
        "ompl_simplify_solutions:=false", f"rrt_planning_time:={args.rrt_seconds}",
        "rrt_planning_attempts:=1", "precontact_candidate_limit:=8",
        f"maximum_carried_box_tilt_deg:={source['maximum_carried_box_tilt_deg']}",
        f"maximum_carried_box_center_z:={source['maximum_carried_box_center_z']}",
        f"vamp_workers:={args.vamp_workers}", f"vamp_seed:={seed}",
        f"vamp_max_iterations:={args.vamp_max_iterations}",
        f"vamp_max_samples:={args.vamp_max_iterations * 2}", "vamp_range:=0.08",
        f"result_json_path:={result_path}",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment,
            start_new_session=True, text=True,
        )
        payload: dict[str, Any] | None = None
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
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--shared-ik-dir", type=Path, default=DEFAULT_SHARED_IK)
    parser.add_argument("--boxes", default="1-25")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rrt-seconds", type=float, default=30.0)
    parser.add_argument("--vamp-workers", type=int, default=1)
    parser.add_argument("--vamp-max-iterations", type=int, default=3000000)
    parser.add_argument("--trial-timeout", type=float, default=900.0)
    parser.add_argument("--domain-id", type=int, default=142)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    raw_dir, log_dir = output_dir / "raw", output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "trials.json"
    document: dict[str, Any] = {
        "schema": "alfa.v322.shared_ik_fullflow_trials.v1",
        "contract": {
            "shared_ik_source": str(args.shared_ik_dir.resolve()),
            "shared_precontact_endpoint_only": True,
            "historical_path_used": False,
            "shortcut_used": False,
            "cartesian_used": False,
            "fallback_used": False,
            "ik_timed": False,
            "base_stations_m": [-0.60, -0.35],
        },
        "trials": [],
    }
    if args.resume and manifest_path.is_file():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = {(int(item["round"]), int(item["box_id"])) for item in document["trials"]}
    environment = dict(os.environ)
    environment["ROS_DOMAIN_ID"] = str(args.domain_id)
    environment["ROS_LOCALHOST_ONLY"] = "1"
    for round_index in range(args.rounds):
        for box_id in parse_range(args.boxes):
            if (round_index, box_id) in completed:
                continue
            source_path = args.shared_ik_dir / f"box-{box_id:02d}.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            backend_records: dict[str, Any] = {}
            payloads: dict[str, dict[str, Any]] = {}
            for backend in ("ompl", "vamp"):
                stem = f"r{round_index + 1:02d}-box-{box_id:02d}-{backend}"
                result_path = raw_dir / f"{stem}.json"
                log_path = log_dir / f"{stem}.log"
                payload = None
                if args.resume and result_path.is_file():
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    payload = run_one(
                        source, backend, result_path, log_path, args, environment, round_index
                    )
                if payload:
                    payloads[backend] = payload
                segments = (payload or {}).get("segment_search", [])
                backend_records[backend] = {
                    "success": bool((payload or {}).get("success")),
                    "failure_stage": (payload or {}).get("failure_stage", "timeout"),
                    "failure_reason": (payload or {}).get("failure_reason", "no result JSON"),
                    "rrt_search_ms": sum(float(item.get("joint_rrt_ms") or 0.0) for item in segments),
                    "ik_ms_excluded": float((payload or {}).get("metrics", {}).get("ik_ms", 0.0)),
                    "result_json": str(result_path),
                    "stage_endpoints": stage_endpoints(payload) if payload else {},
                }
                print(
                    f"round={round_index + 1} box={box_id:02d} backend={backend} "
                    f"success={backend_records[backend]['success']} "
                    f"rrt_ms={backend_records[backend]['rrt_search_ms']:.3f}",
                    flush=True,
                )
            common_stages = sorted(
                set(backend_records["ompl"]["stage_endpoints"])
                & set(backend_records["vamp"]["stage_endpoints"])
            )
            matched = {
                stage: backend_records["ompl"]["stage_endpoints"][stage]
                == backend_records["vamp"]["stage_endpoints"][stage]
                for stage in common_stages
            }
            ompl_ms = backend_records["ompl"]["rrt_search_ms"]
            vamp_ms = backend_records["vamp"]["rrt_search_ms"]
            trial = {
                "round": round_index,
                "box_id": box_id,
                "side": source["side"], "mode": source["grasp_mode"],
                "base_x": source["base_pose_map"][0],
                "shared_ik_source": str(source_path.resolve()),
                "backends": backend_records,
                "both_success": all(item["success"] for item in backend_records.values()),
                "matched_stage_endpoints": matched,
                "all_common_stage_endpoints_match": bool(matched) and all(matched.values()),
                "speedup_ompl_over_vamp": ompl_ms / vamp_ms if vamp_ms > 0.0 else None,
            }
            document["trials"].append(trial)
            write_json(manifest_path, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
