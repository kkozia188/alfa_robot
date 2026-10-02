#!/usr/bin/env python3
"""Evaluate V3 dual-arm stage planning over an x/y/yaw docking-error matrix."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, TransformStamped
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from robot_motion_interfaces.action import ExecuteMotionStage
from robot_motion_interfaces.msg import DualArmPoseTargets
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from docking_error_common import (
    PHASES,
    STAGES,
    build_summary,
    infer_planning_phases,
    matrix_samples,
    read_json,
    selected_cycles,
    validate_config,
    write_json,
    write_summary_bundle,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pose_to_dict(pose: Pose) -> dict[str, list[float]]:
    return {
        "position": [pose.position.x, pose.position.y, pose.position.z],
        "orientation_xyzw": [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
    }


def pose_from_catalog(value: dict[str, Any]) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, value["position"])
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = map(float, value["orientation"])
    return pose


class BenchmarkClient(Node):
    def __init__(self) -> None:
        super().__init__("v3_docking_error_benchmark")
        self._pose = [0.0, 0.0, 0.0]
        self._catalog: dict[str, Any] | None = None
        self._catalog_version = 0
        self._lock = threading.Lock()
        self._tf = TransformBroadcaster(self)
        self.create_timer(0.02, self._publish_tf)
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            String,
            "/v3_box_wall_grasp_demo/wall_target_catalog",
            self._on_catalog,
            qos,
        )
        self.action = ActionClient(self, ExecuteMotionStage, "/motion/execute_stage")

    def set_pose(self, pose: list[float]) -> int:
        if len(pose) != 3 or any(not math.isfinite(value) for value in pose):
            raise ValueError("base pose must contain finite x/y/yaw")
        with self._lock:
            version = self._catalog_version
            self._pose = list(pose)
        self._publish_tf()
        return version

    def current_pose(self) -> list[float]:
        with self._lock:
            return list(self._pose)

    def _publish_tf(self) -> None:
        pose = self.current_pose()
        stamp = self.get_clock().now().to_msg()
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = stamp
        map_to_odom.header.frame_id = "map"
        map_to_odom.child_frame_id = "odom"
        map_to_odom.transform.rotation.w = 1.0
        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = stamp
        odom_to_base.header.frame_id = "odom"
        odom_to_base.child_frame_id = "base_footprint"
        odom_to_base.transform.translation.x = pose[0]
        odom_to_base.transform.translation.y = pose[1]
        half = pose[2] * 0.5
        odom_to_base.transform.rotation.z = math.sin(half)
        odom_to_base.transform.rotation.w = math.cos(half)
        self._tf.sendTransform([map_to_odom, odom_to_base])

    def _on_catalog(self, message: String) -> None:
        try:
            value = json.loads(message.data)
            if value.get("frame_id") != "base_link" or len(value.get("boxes", [])) != 25:
                return
        except (json.JSONDecodeError, TypeError):
            return
        with self._lock:
            self._catalog = value
            self._catalog_version += 1

    def wait_for_catalog_after(self, version: int, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._catalog is not None and self._catalog_version >= version + 2:
                    return copy.deepcopy(self._catalog)
            time.sleep(0.02)
        raise TimeoutError("fresh base_link target catalog was not published")

    @staticmethod
    def _wait_future(future: Any, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if future.done():
                return True
            time.sleep(0.02)
        return future.done()

    def send_stage(self, goal: ExecuteMotionStage.Goal, name: str, timeout: float) -> dict[str, Any]:
        started = time.monotonic()
        future = self.action.send_goal_async(goal)
        if not self._wait_future(future, timeout):
            return self._timeout_result(name, started, "goal acceptance timeout")
        handle = future.result()
        if handle is None or not handle.accepted:
            return {
                "stage": name,
                "ok": False,
                "elapsed_ms": (time.monotonic() - started) * 1000.0,
                "failure_stage": name.lower(),
                "error_message": "goal rejected",
                "error_detail": "",
                "diagnostic": "",
                "timed_out": False,
            }
        result_future = handle.get_result_async()
        if not self._wait_future(result_future, timeout):
            handle.cancel_goal_async()
            return self._timeout_result(name, started, "action result timeout")
        wrapped = result_future.result()
        if wrapped is None:
            return self._timeout_result(name, started, "missing action result")
        result = wrapped.result
        diagnostic = str(result.diagnostic)
        failure_stage = ""
        if not result.ok and result.error.detail:
            failure_stage = str(result.error.detail).split(":", 1)[0]
        return {
            "stage": name,
            "ok": bool(result.ok),
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
            "goal_status": int(wrapped.status),
            "goal_status_succeeded": int(wrapped.status) == GoalStatus.STATUS_SUCCEEDED,
            "error_code": int(result.error.code),
            "error_message": str(result.error.message),
            "error_detail": str(result.error.detail),
            "error_retryable": bool(result.error.retryable),
            "error_source": str(result.error.source),
            "diagnostic": diagnostic,
            "failure_stage": failure_stage,
            "timed_out": False,
        }

    @staticmethod
    def _timeout_result(name: str, started: float, detail: str) -> dict[str, Any]:
        return {
            "stage": name,
            "ok": False,
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
            "failure_stage": name.lower(),
            "error_message": "timeout",
            "error_detail": detail,
            "diagnostic": "",
            "timed_out": True,
        }


def planner_command(config: dict[str, Any], rrd_path: Path | None) -> list[str]:
    args = {
        "x": "0.85",
        "box_id": "20",
        "arm": "auto",
        "auto_run_once": "false",
        "sequence_mode": "false",
        "initial_pose": "home",
        "wall_context": "full",
        "wall_near_x_map": str(float(config["wall_pose_map"]["near_face_x_m"])),
        "wall_center_y": str(float(config["wall_pose_map"].get("center_y_m", 0.0))),
        "wall_bottom_z": str(float(config["wall_pose_map"].get("bottom_z_m", 0.0))),
        "post_extract_policy": "external_handoff",
        "rear_placement_strategy": "named_unloading",
        "height_strategy": "comfort_radius",
        "comfort_ratio_min": "1.10",
        "comfort_ratio_preferred": "1.15",
        "comfort_ratio_max": "1.15",
        "top_shoulder_above_wrist": "0.10",
        "planning_seed": str(int(config.get("planning_seed", 104729))),
        "enable_stage_action": "true",
        "playback_enabled": "false",
        "retain_placed_boxes": "false",
        "execution_backend": "replay",
        "start_rviz": "false",
        "start_rerun": "true" if rrd_path else "false",
        "spawn_viewer": "false",
        "display_rate_hz": str(float(config.get("display_rate_hz", 500.0))),
    }
    if rrd_path:
        args["rerun_recording_path"] = str(rrd_path)
    args.update({str(key): str(value) for key, value in config.get("planner_launch_arguments", {}).items()})
    return [
        "ros2",
        "launch",
        "alfa_robot_moveit_config",
        "v3_box_wall_grasp_demo.launch.py",
        *(f"{key}:={value}" for key, value in args.items()),
    ]


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 12.0), (signal.SIGTERM, 4.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=3.0)


def make_pregrasp_goal(
    cycle: dict[str, Any], catalog: dict[str, Any]
) -> tuple[ExecuteMotionStage.Goal, dict[str, Any]]:
    goal = ExecuteMotionStage.Goal()
    goal.execution_stage = ExecuteMotionStage.Goal.EXECUTION_STAGE_PREGRASP
    mode = (
        DualArmPoseTargets.GRASP_MODE_TOP_SUCTION
        if cycle["grasp_mode"] == "top_suction"
        else DualArmPoseTargets.GRASP_MODE_SIDE_SUCTION
    )
    targets: dict[str, Any] = {}
    for side in ("left", "right"):
        box_id = cycle[f"{side}_catalog_box_id"]
        if box_id is None:
            setattr(goal.targets, f"{side}_grasp_mode", DualArmPoseTargets.GRASP_MODE_NO_MOVE)
            continue
        setattr(goal.targets, f"{side}_grasp_mode", mode)
        key = f"{side}_{'top' if cycle['grasp_mode'] == 'top_suction' else 'side'}"
        pose = pose_from_catalog(catalog["boxes"][box_id][key])
        setattr(goal.targets, f"{side}_pose", pose)
        targets[side] = {
            "catalog_box_id": box_id,
            "motion_box_id": cycle[f"{side}_motion_box_id"],
            "pose_6d_base_link": pose_to_dict(pose),
        }
    return goal, targets


def plain_stage_goal(value: int) -> ExecuteMotionStage.Goal:
    goal = ExecuteMotionStage.Goal()
    goal.execution_stage = value
    return goal


def expected_pose(config: dict[str, Any], cycle: dict[str, Any], error: dict[str, float]) -> list[float]:
    nominal_key = "upper_rows" if cycle["row_from_top"] <= 3 else "lower_rows"
    nominal = config["nominal_base_pose_map"][nominal_key]
    return [
        float(nominal[0]) + error["x_m"],
        float(nominal[1]) + error["y_m"],
        float(nominal[2]) + math.radians(error["yaw_deg"]),
    ]


def root_matches(stage: dict[str, Any], expected: list[float], tolerance: float = 1e-4) -> bool:
    try:
        diagnostic = json.loads(stage["diagnostic"])
        root = diagnostic["planar_root_map"]
        actual = [float(root["x"]), float(root["y"]), float(root["yaw"])]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    yaw_error = math.atan2(math.sin(actual[2] - expected[2]), math.cos(actual[2] - expected[2]))
    return (
        abs(actual[0] - expected[0]) <= tolerance
        and abs(actual[1] - expected[1]) <= tolerance
        and abs(yaw_error) <= tolerance
    )


def run_sample(
    client: BenchmarkClient,
    config: dict[str, Any],
    sample: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    error = {key: float(sample[key]) for key in ("x_m", "y_m", "yaw_deg")}
    record_rrd = sample["sample_id"] in set(config.get("record_rerun_sample_ids", []))
    rrd_path = output_dir / "rerun" / f"{sample['sample_id']}.rrd" if record_rrd else None
    if rrd_path:
        rrd_path.parent.mkdir(parents=True, exist_ok=True)
        rrd_path.unlink(missing_ok=True)
    log_path = output_dir / "logs" / f"{sample['sample_id']}.planner.log"
    ros_log_dir = output_dir / "logs" / "ros" / sample["sample_id"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    initial_cycle = selected_cycles(config)[0]
    client.set_pose(expected_pose(config, initial_cycle, error))
    environment = dict(os.environ)
    environment["ROS_LOG_DIR"] = str(ros_log_dir)
    result: dict[str, Any] = {
        "schema": "alfa.v3_docking_error_sample.v1",
        "sample_id": sample["sample_id"],
        "error": error,
        "started_at": utc_now(),
        "cycles": [],
        "planner_log": str(log_path),
        "rrd": str(rrd_path) if rrd_path else None,
        "cache_loading_counted": False,
    }
    timeout = float(config.get("stage_timeout_s", 180.0))
    startup_timeout = float(config.get("planner_startup_timeout_s", 90.0))
    with log_path.open("w", encoding="utf-8") as planner_log:
        process = subprocess.Popen(
            planner_command(config, rrd_path),
            stdout=planner_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=environment,
        )
        try:
            startup_deadline = time.monotonic() + startup_timeout
            while not client.action.wait_for_server(timeout_sec=0.2):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"planner exited before Action startup with code {process.returncode}"
                    )
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError("/motion/execute_stage did not become available")
            for cycle in selected_cycles(config):
                injected = expected_pose(config, cycle, error)
                version = client.set_pose(injected)
                catalog = client.wait_for_catalog_after(version, startup_timeout)
                goal, target_poses = make_pregrasp_goal(cycle, catalog)
                cycle_result = {
                    **cycle,
                    "nominal_base_pose_map": expected_pose(
                        config, cycle, {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0}
                    ),
                    "injected_base_pose_map": injected,
                    "target_poses": target_poses,
                    "catalog_removed_box_ids": catalog.get("removed_box_ids", []),
                    "stages": [],
                }
                pregrasp = client.send_stage(goal, "PREGRASP", timeout)
                if pregrasp.get("ok") and not root_matches(pregrasp, injected):
                    pregrasp.update(
                        ok=False,
                        failure_stage="tf_contract",
                        error_message="planner used a different planar root",
                        error_detail=f"expected {injected}",
                    )
                cycle_result["stages"].append(pregrasp)
                if pregrasp.get("ok"):
                    goals = (
                        ("APPROACH", ExecuteMotionStage.Goal.EXECUTION_STAGE_APPROACH),
                        ("PLACE", ExecuteMotionStage.Goal.EXECUTION_STAGE_PLACE),
                        ("HOME", ExecuteMotionStage.Goal.EXECUTION_STAGE_HOME),
                    )
                    for name, value in goals:
                        stage = client.send_stage(plain_stage_goal(value), name, timeout)
                        cycle_result["stages"].append(stage)
                        if not stage.get("ok"):
                            break
                cycle_result["ok"] = (
                    len(cycle_result["stages"]) == len(STAGES)
                    and all(stage.get("ok") for stage in cycle_result["stages"])
                )
                cycle_result["planning_phases"] = infer_planning_phases(cycle_result)
                result["cycles"].append(cycle_result)
                write_json(output_dir / "raw" / f"{sample['sample_id']}.json", result)
                if pregrasp.get("ok") and not cycle_result["ok"]:
                    result["stopped_after_stage_failure"] = True
                    break
        except Exception as error_value:  # Preserve partial evidence before propagating.
            result["infrastructure_error"] = f"{type(error_value).__name__}: {error_value}"
        finally:
            stop_process(process)
    result["finished_at"] = utc_now()
    result["planner_exit_code"] = process.returncode
    result["success"] = (
        len(result["cycles"]) == len(selected_cycles(config))
        and all(cycle.get("ok") for cycle in result["cycles"])
        and "infrastructure_error" not in result
    )
    write_json(output_dir / "raw" / f"{sample['sample_id']}.json", result)
    time.sleep(0.8)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    validate_config(config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved-config.json", config)
    samples = matrix_samples(config)
    if args.sample_id:
        requested = set(args.sample_id)
        samples = [sample for sample in samples if sample["sample_id"] in requested]
        missing = requested - {sample["sample_id"] for sample in samples}
        if missing:
            raise SystemExit(f"unknown --sample-id values: {sorted(missing)}")
    if args.limit_samples > 0:
        samples = samples[: args.limit_samples]
    if not samples:
        raise SystemExit("no samples selected")

    rclpy.init(args=[])
    client = BenchmarkClient()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(client)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    records = []
    try:
        for index, sample in enumerate(samples, start=1):
            raw_path = output_dir / "raw" / f"{sample['sample_id']}.json"
            if args.resume and raw_path.is_file():
                record = read_json(raw_path)
                print(f"RESUME {index}/{len(samples)} {sample['sample_id']}", flush=True)
            else:
                print(f"RUN {index}/{len(samples)} {sample['sample_id']}", flush=True)
                record = run_sample(client, config, sample, output_dir)
            records.append(record)
            summary = build_summary(config, records)
            write_summary_bundle(output_dir, summary)
            print(
                f"RESULT sample={sample['sample_id']} success={record.get('success', False)} "
                f"cycles={len(record.get('cycles', []))}/{len(selected_cycles(config))}",
                flush=True,
            )
    finally:
        executor.shutdown(timeout_sec=3.0)
        client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=3.0)
    return 0 if all(record.get("success") for record in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
