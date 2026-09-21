#!/usr/bin/env python3
"""Drive the V3 fixed-wall demo exclusively through /motion/execute_stage."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from robot_motion_interfaces.action import ExecuteMotionStage
from robot_motion_interfaces.msg import DualArmPoseTargets
from std_msgs.msg import String


def wall_rounds() -> list[tuple[int | None, int | None]]:
    rounds: list[tuple[int | None, int | None]] = []
    for row in range(4, -1, -1):
        first = row * 5
        rounds.extend(
            [
                (first + 4, first),
                (first + 3, first + 1),
                ((first + 2, None) if row % 2 == 0 else (None, first + 2)),
            ]
        )
    return rounds


def pose_from_json(value: dict) -> Pose:
    pose = Pose()
    position = value["position"]
    orientation = value["orientation"]
    pose.position.x, pose.position.y, pose.position.z = map(float, position)
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = map(float, orientation)
    return pose


class V3MotionStageWallClient(Node):
    def __init__(self, *, pause_for_vacuum: bool) -> None:
        super().__init__("v3_motion_stage_wall_client")
        self.catalog: dict | None = None
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            String,
            "/v3_box_wall_grasp_demo/wall_target_catalog",
            self._on_catalog,
            qos,
        )
        self.action_client = ActionClient(
            self, ExecuteMotionStage, "/motion/execute_stage"
        )
        self.pause_for_vacuum = pause_for_vacuum

    def _on_catalog(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(f"Invalid wall target catalog: {error}")
            return
        if payload.get("frame_id") != "base_link" or len(payload.get("boxes", [])) != 25:
            self.get_logger().error("Wall target catalog must contain 25 base_link poses")
            return
        self.catalog = payload

    def wait_ready(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and self.catalog is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.catalog is None:
            raise RuntimeError("timed out waiting for wall target catalog")
        if not self.action_client.wait_for_server(timeout_sec=max(0.0, deadline - time.monotonic())):
            raise RuntimeError("/motion/execute_stage is unavailable")

    def target_for(self, box_id: int, side: str, top: bool) -> Pose:
        if self.catalog is None:
            raise RuntimeError("wall target catalog is unavailable")
        box = self.catalog["boxes"][box_id]
        if int(box["box_id"]) != box_id:
            raise RuntimeError(f"catalog index mismatch for box {box_id}")
        return pose_from_json(box[f"{side}_{'top' if top else 'side'}"])

    def make_pregrasp_goal(
        self, left_box: int | None, right_box: int | None, top: bool
    ) -> ExecuteMotionStage.Goal:
        goal = ExecuteMotionStage.Goal()
        goal.execution_stage = ExecuteMotionStage.Goal.EXECUTION_STAGE_PREGRASP
        active_mode = (
            DualArmPoseTargets.GRASP_MODE_TOP_SUCTION
            if top
            else DualArmPoseTargets.GRASP_MODE_SIDE_SUCTION
        )
        goal.targets.left_grasp_mode = (
            active_mode
            if left_box is not None
            else DualArmPoseTargets.GRASP_MODE_NO_MOVE
        )
        goal.targets.right_grasp_mode = (
            active_mode
            if right_box is not None
            else DualArmPoseTargets.GRASP_MODE_NO_MOVE
        )
        if left_box is not None:
            goal.targets.left_pose = self.target_for(left_box, "left", top)
        if right_box is not None:
            goal.targets.right_pose = self.target_for(right_box, "right", top)
        return goal

    def send_stage(self, goal: ExecuteMotionStage.Goal) -> dict:
        started = time.perf_counter()
        goal_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future)
        handle = goal_future.result()
        if handle is None or not handle.accepted:
            return {
                "ok": False,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "diagnostic": "goal rejected",
            }
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            return {
                "ok": False,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "diagnostic": "missing action result",
            }
        result = wrapped.result
        return {
            "ok": bool(result.ok),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "error_code": int(result.error.code),
            "error_message": str(result.error.message),
            "diagnostic": str(result.diagnostic),
        }

    def run_round(
        self,
        index: int,
        pair: tuple[int | None, int | None],
        *,
        grasp_mode: str = "auto",
        top_fallback: bool = False,
    ) -> dict:
        left_box, right_box = pair
        active = [side for side, box in (("left", left_box), ("right", right_box)) if box is not None]
        row = (left_box if left_box is not None else right_box) // 5
        if grasp_mode == "top":
            attempts = [True]
        elif grasp_mode == "side":
            attempts = [False]
        else:
            attempts = [True] if row == 0 else ([False, True] if top_fallback else [False])
        record = {
            "round_index": index,
            "left_box": left_box,
            "right_box": right_box,
            "requested_grasp_mode": grasp_mode,
            "stages": [],
        }
        top = attempts[0]
        pregrasp_ok = False
        for top in attempts:
            result = self.send_stage(self.make_pregrasp_goal(left_box, right_box, top))
            result["stage"] = "PREGRASP"
            result["grasp_mode"] = "top_suction" if top else "side_suction"
            record["stages"].append(result)
            self.get_logger().info(
                f"round={index} boxes={left_box}/{right_box} stage=PREGRASP "
                f"mode={result['grasp_mode']} ok={result['ok']} "
                f"elapsed={result['elapsed_ms']:.1f}ms diagnostic={result.get('diagnostic', '')}"
            )
            if result["ok"]:
                pregrasp_ok = True
                break
        if not pregrasp_ok:
            record["ok"] = False
            return record
        record["selected_grasp_mode"] = "top_suction" if top else "side_suction"
        goals = [
            ExecuteMotionStage.Goal(
                execution_stage=ExecuteMotionStage.Goal.EXECUTION_STAGE_APPROACH
            ),
            ExecuteMotionStage.Goal(
                execution_stage=ExecuteMotionStage.Goal.EXECUTION_STAGE_PLACE
            ),
            ExecuteMotionStage.Goal(
                execution_stage=ExecuteMotionStage.Goal.EXECUTION_STAGE_HOME
            ),
        ]
        names = ["APPROACH", "PLACE", "HOME"]
        for name, goal in zip(names, goals):
            result = self.send_stage(goal)
            result["stage"] = name
            record["stages"].append(result)
            self.get_logger().info(
                f"round={index} boxes={left_box}/{right_box} stage={name} "
                f"ok={result['ok']} elapsed={result['elapsed_ms']:.1f}ms "
                f"diagnostic={result.get('diagnostic', '')}"
            )
            if not result["ok"]:
                record["ok"] = False
                return record
            if self.pause_for_vacuum and name in ("APPROACH", "PLACE"):
                operation = "吸附" if name == "APPROACH" else "释放"
                input(f"由 Autonomy 确认 {','.join(active)} 吸盘已{operation}，按回车继续：")
        record["ok"] = True
        return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--wall", action="store_true", help="run all 15 rounds")
    scope.add_argument(
        "--round-index", type=int, default=0, help="run one round, 0..14"
    )
    parser.add_argument("--pause-for-vacuum", action="store_true")
    parser.add_argument("--grasp-mode", choices=["auto", "side", "top"], default="auto")
    parser.add_argument(
        "--top-fallback",
        action="store_true",
        help="for non-bottom rows, retry PREGRASP with top suction after side failure",
    )
    parser.add_argument("--allow-failure", action="store_true")
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rounds = wall_rounds()
    if not args.wall and not 0 <= args.round_index < len(rounds):
        raise SystemExit("--round-index must be in 0..14")
    rclpy.init(args=[])
    node = V3MotionStageWallClient(pause_for_vacuum=args.pause_for_vacuum)
    summary = {"requested_scope": "wall" if args.wall else "single_round", "rounds": []}
    try:
        node.wait_ready()
        selected = list(enumerate(rounds)) if args.wall else [(args.round_index, rounds[args.round_index])]
        for index, pair in selected:
            record = node.run_round(
                index,
                pair,
                grasp_mode=args.grasp_mode,
                top_fallback=args.top_fallback,
            )
            summary["rounds"].append(record)
            if not record["ok"] and not (args.wall and args.allow_failure):
                break
        summary["completed_rounds"] = sum(1 for item in summary["rounds"] if item["ok"])
        summary["success"] = len(summary["rounds"]) == len(selected) and all(
            item["ok"] for item in summary["rounds"]
        )
        text = json.dumps(summary, indent=2, ensure_ascii=False)
        print(text, flush=True)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(text + "\n")
        if not summary["success"] and not args.allow_failure:
            raise SystemExit(2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
