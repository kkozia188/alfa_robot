#!/usr/bin/env python3

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose

import sys
sys.path.insert(0, "/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_full_plan")
sys.path.insert(0, "/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_direct_first_pose")
from benchmark_constrained_extract import constraint_metrics  # noqa: E402
from v3_round1_full_plan import result_trajectory, state_rows, validate_payload_path  # noqa: E402
from v3_wall_ik_benchmark import (  # noqa: E402
    ACTIVE_JOINTS, canonical_side_suction_quaternion_wxyz,
    canonical_side_tool_to_box, chassis_front_x, make_scene, wall_center,
)


def goal(poses):
    return GoalToolPose.from_poses({
        frame: Pose(
            position=torch.tensor(data["position"], device="cuda", dtype=torch.float32).reshape(1, 3),
            quaternion=torch.tensor(data["quaternion_wxyz"], device="cuda", dtype=torch.float32).reshape(1, 4),
        )
        for frame, data in poses.items()
    }, ordered_tool_frames=["left_tool0", "right_tool0"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--named-poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    robot = yaml.safe_load(args.robot_config.read_text())
    home = yaml.safe_load(args.named_poses.read_text())["named_poses"]["home"]
    front = chassis_front_x(args.urdf, home)
    scene = make_scene(front, 0.9, {20, 24})
    scene.cuboid = [item for item in scene.cuboid if item.name != "ground"]
    home_state = JointState.from_position(
        torch.tensor([[home[name] for name in ACTIVE_JOINTS]], device="cuda", dtype=torch.float32),
        joint_names=ACTIVE_JOINTS,
    )
    contact_poses = {}
    for side, box_id in (("left", 24), ("right", 20)):
        position = wall_center(front, 0.9, box_id)
        position[0] -= 0.150001
        contact_poses[f"{side}_tool0"] = {
            "position": position.tolist(),
            "quaternion_wxyz": canonical_side_suction_quaternion_wxyz(side).tolist(),
        }

    ik_cfg = InverseKinematicsCfg.create(
        robot=copy.deepcopy(robot), scene_model=scene, collision_cache={"cuboid": 40},
        num_seeds=256, self_collision_check=True, use_cuda_graph=True,
        position_tolerance=0.002, orientation_tolerance=math.radians(1.0),
        override_iters_for_multi_link_ik=500,
        optimizer_collision_activation_distance=0.005,
    )
    ik = InverseKinematics(ik_cfg)
    ik.solve_pose(goal(contact_poses), home_state, return_seeds=8)
    torch.cuda.synchronize()
    started = time.perf_counter()
    ik_result = ik.solve_pose(goal(contact_poses), home_state, return_seeds=8)
    torch.cuda.synchronize()
    ik_wall_ms = (time.perf_counter() - started) * 1000.0
    names = list(ik_result.js_solution.joint_names)
    all_solutions = ik_result.js_solution.position.reshape(-1, len(names))
    success = ik_result.success.reshape(-1).bool()
    successful = torch.nonzero(success, as_tuple=False).reshape(-1)
    report = {
        "description_commit": "6bb184b",
        "tool0_semantic_candidate": {"left_rpy_z_deg": -90, "right_rpy_z_deg": 90},
        "contact_ik": {
            "success": bool(successful.numel()),
            "wall_ms": ik_wall_ms,
            "position_error_mm": float(ik_result.position_error.reshape(-1)[0] * 1000.0),
            "rotation_error_deg": math.degrees(float(ik_result.rotation_error.reshape(-1)[0])),
        },
        "contact_poses": contact_poses,
        "success": False,
    }
    if successful.numel() == 0:
        report["failure_stage"] = "canonical_contact_ik"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return

    active_indices = torch.tensor([names.index(name) for name in ACTIVE_JOINTS], device="cuda")
    active_candidates = all_solutions[successful][:, active_indices]
    current = home_state.position.reshape(-1)
    score = torch.sum((active_candidates[:, 1:] - current[1:]) ** 2, dim=1) + (
        (active_candidates[:, 0] - current[0]) / 0.1
    ) ** 2
    contact_values = active_candidates[int(torch.argmin(score))]
    contact_state = JointState.from_position(
        contact_values.reshape(1, -1), joint_names=ACTIVE_JOINTS
    )
    report["contact_joints"] = contact_values.detach().cpu().tolist()

    planner_cfg = MotionPlannerCfg.create(
        robot=copy.deepcopy(robot), scene_model=scene, collision_cache={"cuboid": 40},
        num_ik_seeds=4, num_trajopt_seeds=2, self_collision_check=True,
        use_cuda_graph=True, position_tolerance=0.002,
        orientation_tolerance=math.radians(1.0),
    )
    planner = MotionPlanner(planner_cfg)
    linear = ToolPoseCriteria.linear_motion(
        axis="x", non_terminal_scale=100.0, project_distance_to_goal=False
    )
    planner.update_tool_pose_criteria({"left_tool0": linear, "right_tool0": linear})
    extract_poses = copy.deepcopy(contact_poses)
    for pose in extract_poses.values():
        pose["position"][0] -= 0.35
    planner.plan_pose(goal(extract_poses), contact_state, max_attempts=3, enable_graph_attempt=1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = planner.plan_pose(
        goal(extract_poses), contact_state, max_attempts=3, enable_graph_attempt=1
    )
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    trajectory = result_trajectory(result)
    frames = [] if trajectory is None else state_rows(trajectory)
    if trajectory is not None and trajectory.joint_names is not None:
        trajectory_names = list(trajectory.joint_names)
        if trajectory_names != ACTIVE_JOINTS:
            indices = [trajectory_names.index(name) for name in ACTIVE_JOINTS]
            frames = [[row[index] for index in indices] for row in frames]
    report["extract"] = {
        "native_success": bool(result.success is not None and result.success.any().item()),
        "wall_ms": wall_ms,
        "solve_ms": float(result.solve_time * 1000.0),
        "frames": len(frames),
    }
    if not frames:
        report["failure_stage"] = "constrained_extract"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return

    metrics = constraint_metrics(planner.kinematics, frames, contact_poses, 0.35)
    tool_to_box = {
        "left": canonical_side_tool_to_box("left"),
        "right": canonical_side_tool_to_box("right"),
    }
    payload_valid, payload_reason = validate_payload_path(
        planner.kinematics, frames, tool_to_box, front, 0.9
    )
    report["extract"]["constraint_metrics"] = metrics
    report["extract"]["exact_payload_valid"] = payload_valid
    report["extract"]["exact_payload_reason"] = payload_reason
    report["frames"] = frames
    report["extract_joints"] = frames[-1]
    report["tool_to_box"] = {side: matrix.tolist() for side, matrix in tool_to_box.items()}
    report["success"] = payload_valid
    if not payload_valid:
        report["failure_stage"] = "extract_exact_payload_validation"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
