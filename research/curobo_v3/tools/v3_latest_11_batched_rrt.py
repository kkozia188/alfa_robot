#!/usr/bin/env python3

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, JointState, Pose

import sys
sys.path.insert(0, "/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_full_plan")
from v3_batched_loaded_search import (  # noqa: E402
    GpuValidity, batched_prm_multi_goal, batched_rrt_connect_multi_goal,
    batched_rrt_multi_goal, densify, loaded_robot,
)
from v3_wall_ik_benchmark import (  # noqa: E402
    ACTIVE_JOINTS, canonical_side_suction_quaternion_wxyz,
    chassis_front_x, make_scene, wall_center,
)


ROUNDS = [
    {"left": 24, "right": 20}, {"left": 23, "right": 21}, {"left": 22},
    {"left": 19, "right": 15}, {"left": 18, "right": 16}, {"right": 17},
    {"left": 14, "right": 10}, {"left": 13, "right": 11}, {"left": 12},
    {"left": 9, "right": 5}, {"left": 8, "right": 6},
]


def goal(poses, ordered_tool_frames=None):
    if ordered_tool_frames is None:
        ordered_tool_frames = ["left_tool0", "right_tool0"]
    return GoalToolPose.from_poses({
        frame: Pose(
            position=torch.tensor(data["position"], device="cuda", dtype=torch.float32).reshape(1, 3),
            quaternion=torch.tensor(data["quaternion_wxyz"], device="cuda", dtype=torch.float32).reshape(1, 4),
        ) for frame, data in poses.items()
    }, ordered_tool_frames=ordered_tool_frames)


def pose_dict(pose):
    return {
        "position": pose.position.reshape(-1, 3)[0].detach().cpu().tolist(),
        "quaternion_wxyz": pose.quaternion.reshape(-1, 4)[0].detach().cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--box-fit", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--named-poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rrt-budget", type=float, default=5.0)
    parser.add_argument(
        "--planner", choices=("rrt", "informed_rrt", "rrt_connect", "prm"),
        default="rrt",
    )
    parser.add_argument("--round", type=int)
    parser.add_argument("--round9-arm", choices=("left", "right"), default="left")
    parser.add_argument("--target-updown", type=float, default=-0.3)
    parser.add_argument("--target-left-deg", nargs=7, type=float,
                        default=[140, -100, -180, 20, -90, -20, 0])
    parser.add_argument("--target-right-deg", nargs=7, type=float,
                        default=[-140, 100, 180, -20, 90, 20, 0])
    args = parser.parse_args()

    robot = yaml.safe_load(args.robot_config.read_text())
    kinematics = robot.get("robot_cfg", robot)["kinematics"]
    joint_names = list(kinematics["cspace"]["joint_names"])
    weights = [
        2.0 if name in {"base_x", "base_y"}
        else 1.0 if name == "base_yaw"
        else 5.0 if name == "updown"
        else 1.0
        for name in joint_names
    ]
    box_fit = json.loads(args.box_fit.read_text())
    home = yaml.safe_load(args.named_poses.read_text())["named_poses"]["home"]
    home_state = JointState.from_position(
        torch.tensor([[home.get(name, 0.0) for name in joint_names]], device="cuda", dtype=torch.float32),
        joint_names=joint_names,
    )
    placement_values = {
        "updown": args.target_updown,
        **{f"left_joint{i + 1}": value for i, value in enumerate(np.deg2rad(
            args.target_left_deg
        ))},
        **{f"right_joint{i + 1}": value for i, value in enumerate(np.deg2rad(
            args.target_right_deg
        ))},
    }
    placement = np.asarray([placement_values.get(name, 0.0) for name in joint_names], dtype=np.float32)
    front = chassis_front_x(args.urdf, home)
    removed = set()
    reports = []
    rounds = copy.deepcopy(ROUNDS)
    if args.round9_arm == "right":
        rounds[8] = {"right": 12}

    for round_index, boxes in enumerate(rounds, start=1):
        if args.round and round_index != args.round:
            removed.update(boxes.values())
            continue
        started_round = time.perf_counter()
        active_sides = tuple(boxes)
        inactive_sides = tuple(side for side in ("left", "right") if side not in active_sides)
        locked_joint_names = [
            f"{side}_joint{joint}" for side in inactive_sides for joint in range(1, 8)
        ]
        locked_indices = [joint_names.index(name) for name in locked_joint_names]
        excluded = removed | set(boxes.values())
        scene = make_scene(front, 0.9, excluded)
        scene.cuboid = [item for item in scene.cuboid if item.name != "ground"]
        row = {"round": round_index, "boxes": boxes, "success": False}
        ik = None
        try:
            ik_robot = copy.deepcopy(robot)
            ik_kinematics = ik_robot.get("robot_cfg", ik_robot)["kinematics"]
            if locked_joint_names:
                ik_kinematics.setdefault("lock_joints", {}).update({
                    name: home[name] for name in locked_joint_names
                })
                ik_kinematics["tool_frames"] = [
                    f"{side}_tool0" for side in active_sides
                ]
            ik_cfg = InverseKinematicsCfg.create(
                robot=ik_robot, scene_model=scene,
                collision_cache={"cuboid": 40},
                num_seeds=1024 if locked_joint_names else 256,
                self_collision_check=True, use_cuda_graph=True,
                position_tolerance=0.002, orientation_tolerance=math.radians(1.0),
                override_iters_for_multi_link_ik=500,
                optimizer_collision_activation_distance=0.005,
            )
            ik = InverseKinematics(ik_cfg)
            ik_joint_names = list(ik.joint_names)
            ik_weights = torch.tensor(
                [weights[joint_names.index(name)] for name in ik_joint_names],
                device="cuda",
            )
            ik_home_state = JointState.from_position(
                torch.tensor(
                    [[home.get(name, 0.0) for name in ik_joint_names]],
                    device="cuda", dtype=torch.float32,
                ),
                joint_names=ik_joint_names,
            )
            target_frames = [f"{side}_tool0" for side in active_sides]
            target_poses = {}
            for side, box_id in boxes.items():
                position = wall_center(front, 0.9, box_id)
                position[0] -= 0.150001
                target_poses[f"{side}_tool0"] = {
                    "position": position.tolist(),
                    "quaternion_wxyz": canonical_side_suction_quaternion_wxyz(side).tolist(),
                }
            ik.solve_pose(goal(target_poses, target_frames), ik_home_state, return_seeds=8)
            torch.cuda.synchronize()
            started = time.perf_counter()
            ik_result = ik.solve_pose(
                goal(target_poses, target_frames), ik_home_state, return_seeds=8
            )
            torch.cuda.synchronize()
            row["contact_ik_ms"] = (time.perf_counter() - started) * 1000.0
            solution_names = list(ik_result.js_solution.joint_names)
            solutions = ik_result.js_solution.position.reshape(-1, len(solution_names))
            success = ik_result.success.reshape(-1).bool()
            successful = torch.nonzero(success, as_tuple=False).reshape(-1)
            if successful.numel() == 0:
                row["failure"] = "contact_ik"
                reports.append(row)
                removed.update(boxes.values())
                continue
            solution_active_indices = torch.tensor(
                [solution_names.index(name) for name in ik_joint_names], device="cuda"
            )
            candidates = solutions[successful][:, solution_active_indices]
            current = ik_home_state.position.reshape(-1)
            score = torch.sum(
                ((candidates - current) * ik_weights) ** 2, dim=1
            )
            contact_solution = candidates[int(torch.argmin(score))]
            contact_values = home_state.position.reshape(-1).clone()
            for solution_index, name in enumerate(ik_joint_names):
                contact_values[joint_names.index(name)] = contact_solution[solution_index]
            contact = JointState.from_position(
                contact_solution.reshape(1, -1), joint_names=ik_joint_names
            )
            contact_actual = ik.compute_kinematics(contact).tool_poses.to_dict()
            contact_poses = {frame: pose_dict(pose) for frame, pose in contact_actual.items()}
            extract_poses = copy.deepcopy(contact_poses)
            for side in active_sides:
                extract_poses[f"{side}_tool0"]["position"][0] -= 0.35

            ik.solve_pose(
                goal(extract_poses, target_frames), contact, return_seeds=8
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            extract_result = ik.solve_pose(
                goal(extract_poses, target_frames), contact, return_seeds=8
            )
            torch.cuda.synchronize()
            row["extract_ik_ms"] = (time.perf_counter() - started) * 1000.0
            extract_names = list(extract_result.js_solution.joint_names)
            extract_solutions = extract_result.js_solution.position.reshape(-1, len(extract_names))
            extract_success = extract_result.success.reshape(-1).bool()
            extract_successful = torch.nonzero(extract_success, as_tuple=False).reshape(-1)
            if extract_successful.numel() == 0:
                row["failure"] = "extract_endpoint_ik"
                reports.append(row)
                removed.update(boxes.values())
                continue
            extract_active_indices = torch.tensor(
                [extract_names.index(name) for name in ik_joint_names], device="cuda"
            )
            extract_candidates = extract_solutions[extract_successful][:, extract_active_indices]
            extract_score = torch.sum(
                ((extract_candidates - contact_solution) * ik_weights) ** 2, dim=1
            )
            extract_solution = extract_candidates[int(torch.argmin(extract_score))]
            extract_values = contact_values.detach().cpu().numpy().astype(np.float32)
            for solution_index, name in enumerate(ik_joint_names):
                extract_values[joint_names.index(name)] = extract_solution[solution_index].item()
            row["extract_ik_candidates"] = int(extract_successful.numel())
            row["locked_joint_names"] = locked_joint_names

            loaded = loaded_robot(robot, box_fit, active_sides=active_sides)
            checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
                robot_config=copy.deepcopy(loaded), scene_model=scene,
                n_cuboids=40, n_meshes=0, collision_activation_distance=0.0,
            ))
            validity = GpuValidity(
                checker, active_sides=active_sides,
                max_box_tilt_deg=89.0, stability_weight=10.0,
                check_ground=True, ground_z=0.0,
                joint_names=joint_names, weights=weights,
            )
            start = torch.tensor(extract_values, device="cuda")
            start_valid = bool(validity.mask(start.reshape(1, -1))[0].item())
            row["start_valid"] = start_valid
            row["ground_collision_checked"] = True
            row["ground_exempt_links"] = ["base_link"]
            if not start_valid:
                row["failure"] = "loaded_start_endpoint"
                reports.append(row)
                removed.update(boxes.values())
                continue

            placement_frames = target_frames
            placement_robot = copy.deepcopy(robot)
            placement_kinematics = placement_robot.get("robot_cfg", placement_robot)["kinematics"]
            if locked_joint_names:
                placement_kinematics.setdefault("lock_joints", {}).update({
                    name: home[name] for name in locked_joint_names
                })
                placement_kinematics["tool_frames"] = placement_frames
            placement_ik = InverseKinematics(InverseKinematicsCfg.create(
                robot=placement_robot, scene_model=scene,
                collision_cache={"cuboid": 40},
                num_seeds=1024 if locked_joint_names else 512,
                self_collision_check=True, use_cuda_graph=False,
                position_tolerance=0.002, orientation_tolerance=math.radians(1.0),
                override_iters_for_multi_link_ik=500,
                optimizer_collision_activation_distance=0.005,
            ))
            placement_full_state = JointState.from_position(
                torch.tensor(placement, device="cuda").reshape(1, -1),
                joint_names=joint_names,
            )
            placement_fk = checker.kinematics.compute_kinematics(
                placement_full_state
            ).tool_poses.to_dict()
            placement_joint_names = list(placement_ik.joint_names)
            start_state = JointState.from_position(
                torch.stack([start[joint_names.index(name)] for name in placement_joint_names])
                .reshape(1, -1),
                joint_names=placement_joint_names,
            )
            placement_poses = {
                frame: pose_dict(placement_fk[frame])
                for frame in placement_frames
            }
            started = time.perf_counter()
            placement_result = placement_ik.solve_pose(
                goal(placement_poses, placement_frames),
                start_state,
                return_seeds=64,
            )
            torch.cuda.synchronize()
            row["placement_ik_ms"] = (time.perf_counter() - started) * 1000.0
            placement_names = list(placement_result.js_solution.joint_names)
            placement_solutions = placement_result.js_solution.position.reshape(
                -1, len(placement_names)
            )
            placement_successful = torch.nonzero(
                placement_result.success.reshape(-1).bool(), as_tuple=False
            ).reshape(-1)
            row["placement_ik_candidates"] = int(placement_successful.numel())
            if placement_successful.numel() == 0:
                row["failure"] = "placement_goal_ik"
                reports.append(row)
                removed.update(boxes.values())
                continue
            placement_active_indices = torch.tensor(
                [placement_names.index(name) for name in placement_joint_names], device="cuda"
            )
            reduced_goals = placement_solutions[placement_successful][:, placement_active_indices]
            goal_candidates = start.reshape(1, -1).repeat(len(reduced_goals), 1)
            for solution_index, name in enumerate(placement_joint_names):
                goal_candidates[:, joint_names.index(name)] = reduced_goals[:, solution_index]
            goal_valid = validity.mask(goal_candidates)
            goal_candidates = goal_candidates[goal_valid]
            row["placement_loaded_valid_candidates"] = int(len(goal_candidates))
            if len(goal_candidates) == 0:
                row["failure"] = "placement_loaded_endpoints"
                reports.append(row)
                removed.update(boxes.values())
                continue
            metric_weights = torch.tensor(weights, device="cuda")
            ranking = torch.argsort(torch.sum(((goal_candidates - start) * metric_weights) ** 2, dim=1))
            retained = []
            for candidate_index in ranking.detach().cpu().tolist():
                candidate = goal_candidates[candidate_index]
                if all(torch.max(torch.abs((candidate - other) * metric_weights)).item() > 0.05
                       for other in retained):
                    retained.append(candidate)
                if len(retained) == 16:
                    break
            goals = torch.stack(retained)
            row["placement_retained_goals"] = len(goals)
            del placement_ik, placement_result
            lower, upper = checker.kinematics.get_joint_limits().position
            search_lower = lower + 1e-5
            search_upper = upper - 1e-5
            if locked_indices:
                search_lower = search_lower.clone()
                search_upper = search_upper.clone()
                search_lower[locked_indices] = start[locked_indices]
                search_upper[locked_indices] = start[locked_indices]
            started = time.perf_counter()
            if args.planner in {"rrt", "informed_rrt"}:
                path, stats = batched_rrt_multi_goal(
                    start, goals, search_lower, search_upper,
                    validity, args.rrt_budget, 20260930 + round_index,
                    apply_shortcut=False,
                    informed_sampling=args.planner == "informed_rrt",
                )
            elif args.planner == "rrt_connect":
                path, stats = batched_rrt_connect_multi_goal(
                    start, goals, search_lower, search_upper,
                    validity, args.rrt_budget, 20260930 + round_index,
                )
            else:
                path, stats = batched_prm_multi_goal(
                    start, goals, search_lower, search_upper,
                    validity, args.rrt_budget, 20260930 + round_index,
                )
            row["planner"] = args.planner
            row["rrt_ms"] = (time.perf_counter() - started) * 1000.0
            row["rrt_stats"] = stats
            row["gpu_states_checked"] = validity.states_checked
            if path is None:
                row["failure"] = f"batched_{args.planner}"
            else:
                dense = densify(path, weights)
                dense_valid, dense_stability = validity.evaluate(
                    torch.tensor(dense, device="cuda", dtype=torch.float32)
                )
                sphere_valid = bool(dense_valid.all().item())
                worst_up_z = 1.0 - float(dense_stability.max().item())
                row.update({
                    "success": sphere_valid,
                    "waypoints": len(path), "frames": dense,
                    "dense_frames": len(dense), "sphere_valid": sphere_valid,
                    "max_box_tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, worst_up_z)))),
                    "box_tilt_limit_deg": 89.0,
                    "box_stability_weight": 10.0,
                })
                if not sphere_valid:
                    row["failure"] = "dense_sphere_validation"
            row["round_ms"] = (time.perf_counter() - started_round) * 1000.0
            reports.append(row)
        except Exception as error:
            row["failure"] = f"{type(error).__name__}: {error}"
            row["round_ms"] = (time.perf_counter() - started_round) * 1000.0
            reports.append(row)
        removed.update(boxes.values())
        if ik is not None:
            del ik
        gc.collect()
        torch.cuda.empty_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"rounds": reports}, indent=2) + "\n")
        print(json.dumps({key: value for key, value in row.items() if key != "frames"}), flush=True)

    summary = {
        "description_commit": "6bb184b",
        "obb_validation": "intentionally_skipped_by_user_request",
        "target_reference": {
            "updown_m": args.target_updown,
            "left_deg": args.target_left_deg,
            "right_deg": args.target_right_deg,
        },
        "rounds": reports,
        "successful_rounds": sum(row["success"] for row in reports),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"successful_rounds": summary["successful_rounds"]}, indent=2))


if __name__ == "__main__":
    main()
