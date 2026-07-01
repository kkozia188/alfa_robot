#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import rerun as rr

REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER_PATH = REPO_ROOT / "scripts/ik_benchmark/scripts/visualize_rerun.py"


def load_helpers():
    spec = importlib.util.spec_from_file_location("alfa_visualize_rerun_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def yaw_matrix(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    transform = np.eye(4)
    transform[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return transform


def log_axes(path: str, origin: list[float], yaw_rad: float, label: str) -> None:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    x_vec = [0.35 * c, 0.35 * s, 0.0]
    y_vec = [-0.25 * s, 0.25 * c, 0.0]
    rr.log(
        f"{path}/axes",
        rr.Arrows3D(
            origins=[origin, origin],
            vectors=[x_vec, y_vec],
            colors=[[255, 80, 80], [80, 255, 80]],
            labels=[f"{label} +X", f"{label} +Y"],
        ),
    )
    rr.log(f"{path}/label", rr.TextLog(label))


def log_robot(helpers, robot, root: str, joint_map: dict[str, float], base_yaw_rad: float, label: str) -> None:
    rr.log(root, rr.Transform3D(translation=[0.0, 0.0, 0.0], quaternion=helpers.matrix_to_quaternion_xyzw(yaw_matrix(base_yaw_rad)[:3, :3])))
    helpers.log_robot_static_model(robot, root, log_meshes=True)
    transforms = robot.fk(joint_map)
    base_tf = yaw_matrix(base_yaw_rad)
    for link_name, transform in transforms.items():
        helpers.log_transform_matrix(f"{root}/{link_name}", base_tf @ transform)
    log_axes(f"{root}/base_frame_marker", [0.0, 0.0, 0.05], base_yaw_rad, label)


def main() -> int:
    parser = argparse.ArgumentParser(description="预览 ALFA 初始姿态：车体 yaw 与 turn 反向补偿")
    parser.add_argument("--base-yaw-deg", type=float, default=90.0, help="车体相对 world 的 yaw，默认 +90°")
    parser.add_argument("--turn-deg", type=float, default=-90.0, help="turn 关节角，默认 -90°")
    parser.add_argument("--updown", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--save", type=Path, default=REPO_ROOT / "data/ik_benchmark/initial_pose_preview/base_yaw90_turn_minus90.rrd")
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--show-reference", action="store_true", help="同时显示原始 yaw=0/turn=0 姿态作灰色对照")
    args = parser.parse_args()

    helpers = load_helpers()
    robot = helpers.UrdfRobot(helpers.render_current_urdf())

    rr.init("alfa_initial_yaw_turn_preview")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(args.save))
    elif args.connect:
        rr.connect()
    elif args.spawn:
        rr.spawn()

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/info",
        rr.TextLog(
            "初始姿态预览\n"
            f"车体 base yaw = {args.base_yaw_deg:.1f}°\n"
            f"turn = {args.turn_deg:.1f}°\n"
            f"pitch = {args.pitch_deg:.1f}°\n"
            f"updown = {args.updown:.3f} m\n"
            "红箭头为车体 +X，绿箭头为车体 +Y。"
        ),
        static=True,
    )

    joint_map = {
        "pitch": math.radians(args.pitch_deg),
        "turn": math.radians(args.turn_deg),
        "updown": args.updown,
    }
    for side in ("left", "right"):
        for idx in range(1, 7):
            joint_map[f"{side}_v5_joint{idx}"] = 0.0

    helpers.set_sample_time(0)
    log_robot(
        helpers,
        robot,
        "world/robot_base_yaw90_turn_minus90",
        joint_map,
        math.radians(args.base_yaw_deg),
        f"base_yaw={args.base_yaw_deg:.0f} turn={args.turn_deg:.0f}",
    )

    if args.show_reference:
        reference_joint_map = dict(joint_map)
        reference_joint_map["turn"] = 0.0
        log_robot(
            helpers,
            robot,
            "world/reference_yaw0_turn0",
            reference_joint_map,
            0.0,
            "reference yaw=0 turn=0",
        )

    print(f"已生成 Rerun: {args.save}")
    print(f"查看: rerun {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
