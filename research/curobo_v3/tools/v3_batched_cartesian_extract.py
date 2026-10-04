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

from v3_batched_loaded_search import GpuValidity, densify, loaded_robot
from v3_wall_ik_benchmark import (
    ACTIVE_JOINTS, canonical_side_suction_quaternion_wxyz,
    chassis_front_x, make_scene, wall_center,
)


ROUNDS = [
    {"left": 24, "right": 20}, {"left": 23, "right": 21}, {"left": 22},
    {"left": 19, "right": 15}, {"left": 18, "right": 16}, {"right": 17},
    {"left": 14, "right": 10}, {"left": 13, "right": 11}, {"right": 12},
    {"left": 9, "right": 5}, {"left": 8, "right": 6},
]


def pose_dict(pose):
    return {
        "position": pose.position.reshape(-1, 3)[0].detach().cpu().numpy(),
        "quaternion": pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy(),
    }


def single_goal(poses, frames):
    return GoalToolPose.from_poses({
        frame: Pose(
            position=torch.tensor(poses[frame]["position"], device="cuda", dtype=torch.float32).reshape(1, 3),
            quaternion=torch.tensor(poses[frame]["quaternion"], device="cuda", dtype=torch.float32).reshape(1, 4),
        ) for frame in frames
    }, ordered_tool_frames=frames)


def batch_goal(contact_poses, frames, progress):
    poses = {}
    for frame in frames:
        positions = np.repeat(contact_poses[frame]["position"][None, :], len(progress), axis=0)
        positions[:, 0] -= 0.35 * progress
        quaternions = np.repeat(contact_poses[frame]["quaternion"][None, :], len(progress), axis=0)
        poses[frame] = Pose(
            position=torch.tensor(positions, device="cuda", dtype=torch.float32),
            quaternion=torch.tensor(quaternions, device="cuda", dtype=torch.float32),
        )
    return GoalToolPose.from_poses(poses, ordered_tool_frames=frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--box-fit", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--named-poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", type=int, default=18)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--round", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    robot = yaml.safe_load(args.robot_config.read_text())
    box_fit = json.loads(args.box_fit.read_text())
    home = yaml.safe_load(args.named_poses.read_text())["named_poses"]["home"]
    full_names = list(robot.get("robot_cfg", robot)["kinematics"]["cspace"]["joint_names"])
    full_home = torch.tensor(
        [home.get(name, 0.0) for name in full_names], device="cuda", dtype=torch.float32
    )
    weights = [5.0 if name == "updown" else 1.0 for name in full_names]
    front = chassis_front_x(args.urdf, home)
    removed = set()
    reports = []
    for round_index, boxes in enumerate(ROUNDS, start=1):
        if args.round and round_index != args.round:
            removed.update(boxes.values())
            continue
        active_sides = tuple(boxes)
        inactive_sides = tuple(side for side in ("left", "right") if side not in active_sides)
        locked_names = [
            f"{side}_joint{joint}" for side in inactive_sides for joint in range(1, 8)
        ]
        frames = [f"{side}_tool0" for side in active_sides]
        scene = make_scene(front, 0.9, removed | set(boxes.values()))
        scene.cuboid = [item for item in scene.cuboid if item.name != "ground"]
        row = {"round": round_index, "boxes": boxes, "success": False,
               "locked_joint_names": locked_names}
        started_round = time.perf_counter()
        try:
            ik_robot = copy.deepcopy(robot)
            ik_kinematics = ik_robot.get("robot_cfg", ik_robot)["kinematics"]
            if locked_names:
                ik_kinematics.setdefault("lock_joints", {}).update({
                    name: home[name] for name in locked_names
                })
                ik_kinematics["tool_frames"] = frames
            contact_solver = InverseKinematics(InverseKinematicsCfg.create(
                robot=copy.deepcopy(ik_robot), scene_model=scene,
                collision_cache={"cuboid": 40}, num_seeds=1024 if locked_names else 256,
                self_collision_check=True, use_cuda_graph=True,
                position_tolerance=0.002, orientation_tolerance=math.radians(1.0),
                override_iters_for_multi_link_ik=500,
                optimizer_collision_activation_distance=0.005,
            ))
            active_names = list(contact_solver.joint_names)
            active_home = JointState.from_position(
                torch.stack([full_home[full_names.index(name)] for name in active_names]).reshape(1, -1),
                joint_names=active_names,
            )
            contact_targets = {}
            for side, box_id in boxes.items():
                position = wall_center(front, 0.9, box_id)
                position[0] -= 0.150001
                contact_targets[f"{side}_tool0"] = {
                    "position": position,
                    "quaternion": canonical_side_suction_quaternion_wxyz(side),
                }
            contact_result = contact_solver.solve_pose(
                single_goal(contact_targets, frames), active_home, return_seeds=args.candidates
            )
            successful = torch.nonzero(
                contact_result.success.reshape(-1).bool(), as_tuple=False
            ).reshape(-1)
            if not len(successful):
                row["failure"] = "contact_ik"
                reports.append(row)
                removed.update(boxes.values())
                continue
            result_names = list(contact_result.js_solution.joint_names)
            result_values = contact_result.js_solution.position.reshape(-1, len(result_names))
            active_indices = torch.tensor(
                [result_names.index(name) for name in active_names], device="cuda"
            )
            contact_candidates = result_values[successful][:, active_indices]
            active_weight = torch.tensor(
                [weights[full_names.index(name)] for name in active_names], device="cuda"
            )
            contact_score = torch.sum(
                ((contact_candidates - active_home.position.reshape(-1)) * active_weight) ** 2,
                dim=1,
            )
            contact_active = contact_candidates[int(torch.argmin(contact_score))]
            contact_state = JointState.from_position(
                contact_active.reshape(1, -1), joint_names=active_names
            )
            contact_poses = contact_targets
            progress = np.linspace(1.0 / args.segments, 1.0, args.segments)
            batch_solver = InverseKinematics(InverseKinematicsCfg.create(
                robot=copy.deepcopy(ik_robot), scene_model=scene,
                collision_cache={"cuboid": 40},
                num_seeds=128 if args.segments > 18 else 256,
                self_collision_check=True, use_cuda_graph=False,
                max_batch_size=args.segments,
                position_tolerance=0.002, orientation_tolerance=math.radians(1.0),
                override_iters_for_multi_link_ik=500,
                optimizer_collision_activation_distance=0.005,
            ))
            batch_current = JointState.from_position(
                contact_active.reshape(1, -1).repeat(args.segments, 1),
                joint_names=active_names,
            )
            target_batch = batch_goal(contact_poses, frames, progress)
            batch_solver.solve_pose(target_batch, batch_current, return_seeds=args.candidates)
            torch.cuda.synchronize()
            started_ik = time.perf_counter()
            batch_result = batch_solver.solve_pose(
                target_batch, batch_current, return_seeds=args.candidates
            )
            torch.cuda.synchronize()
            row["batch_ik_ms"] = (time.perf_counter() - started_ik) * 1000.0
            solution_names = list(batch_result.js_solution.joint_names)
            solution_values = batch_result.js_solution.position.reshape(
                args.segments, args.candidates, len(solution_names)
            )
            success_mask = batch_result.success.reshape(args.segments, args.candidates).bool()
            active_result_indices = torch.tensor(
                [solution_names.index(name) for name in active_names], device="cuda"
            )
            solution_values = solution_values[:, :, active_result_indices]
            loaded = copy.deepcopy(robot)
            checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
                robot_config=copy.deepcopy(loaded), scene_model=scene,
                n_cuboids=40, n_meshes=0, collision_activation_distance=0.0,
            ))
            validity = GpuValidity(
                checker, active_sides=active_sides, max_box_tilt_deg=89.0,
                stability_weight=10.0, check_ground=True, ground_z=0.0,
                joint_names=full_names, weights=weights,
            )
            row["payload_collision_deferred_until_extracted"] = True
            contact_layer = []
            for contact_candidate in contact_candidates:
                full_contact = full_home.clone()
                for active_index, name in enumerate(active_names):
                    full_contact[full_names.index(name)] = contact_candidate[active_index]
                contact_layer.append(full_contact)
            contact_tensor = torch.stack(contact_layer)
            contact_valid = validity.mask(contact_tensor)
            contact_layer = [contact_tensor[index] for index in torch.nonzero(
                contact_valid, as_tuple=False
            ).reshape(-1).detach().cpu().tolist()]
            row["contact_candidate_count"] = len(contact_layer)
            if not contact_layer:
                row["failure"] = "contact_collision"
                reports.append(row)
                removed.update(boxes.values())
                continue
            layers = [contact_layer]
            layer_candidate_counts = []
            for layer_index in range(args.segments):
                layer = []
                for candidate_index in torch.nonzero(
                    success_mask[layer_index], as_tuple=False
                ).reshape(-1).detach().cpu().tolist():
                    candidate = full_home.clone()
                    for active_index, name in enumerate(active_names):
                        candidate[full_names.index(name)] = solution_values[
                            layer_index, candidate_index, active_index
                        ]
                    layer.append(candidate)
                if not layer:
                    row["failure"] = f"ik_layer_{layer_index + 1}"
                    break
                candidate_tensor = torch.stack(layer)
                candidate_valid = validity.mask(candidate_tensor)
                layer = [candidate_tensor[index] for index in torch.nonzero(
                    candidate_valid, as_tuple=False
                ).reshape(-1).detach().cpu().tolist()]
                layer_candidate_counts.append(len(layer))
                if not layer:
                    row["failure"] = f"collision_layer_{layer_index + 1}"
                    break
                layers.append(layer)
            row["layer_candidate_counts"] = layer_candidate_counts
            if len(layers) != args.segments + 1:
                reports.append(row)
                removed.update(boxes.values())
                continue
            all_starts, all_goals, edge_meta = [], [], []
            for layer_index in range(len(layers) - 1):
                for source_index, source in enumerate(layers[layer_index]):
                    for target_index, target in enumerate(layers[layer_index + 1]):
                        all_starts.append(source)
                        all_goals.append(target)
                        edge_meta.append((layer_index, source_index, target_index))
            starts = torch.stack(all_starts)
            goals = torch.stack(all_goals)
            torch.cuda.synchronize()
            started_edges = time.perf_counter()
            edge_valid, edge_stability = validity.edges(
                starts, goals, resolution=math.radians(0.5), return_stability=True
            )
            torch.cuda.synchronize()
            row["edge_validation_ms"] = (time.perf_counter() - started_edges) * 1000.0
            metric_weight = torch.tensor(weights, device="cuda")
            edge_length = torch.linalg.vector_norm((goals - starts) * metric_weight, dim=1)
            edge_cost = edge_length * (1.0 + validity.stability_weight * edge_stability)
            costs = [torch.zeros(len(layers[0]), device="cuda")]
            parents = []
            transition_diagnostics = []
            edge_cursor = 0
            for layer_index in range(len(layers) - 1):
                source_count = len(layers[layer_index])
                target_count = len(layers[layer_index + 1])
                matrix_cost = torch.full(
                    (source_count, target_count), torch.inf, device="cuda"
                )
                edge_count = source_count * target_count
                local_valid = edge_valid[edge_cursor:edge_cursor + edge_count].reshape(
                    source_count, target_count
                )
                local_cost = edge_cost[edge_cursor:edge_cursor + edge_count].reshape(
                    source_count, target_count
                )
                matrix_cost[local_valid] = local_cost[local_valid]
                total = costs[-1].reshape(-1, 1) + matrix_cost
                next_cost, next_parent = total.min(dim=0)
                reachable_sources = int(torch.isfinite(costs[-1]).sum().item())
                reachable_targets = int(torch.isfinite(next_cost).sum().item())
                transition_diagnostics.append({
                    "from_layer": layer_index,
                    "to_layer": layer_index + 1,
                    "from_distance_cm": 35.0 * layer_index / args.segments,
                    "to_distance_cm": 35.0 * (layer_index + 1) / args.segments,
                    "source_candidates": source_count,
                    "target_candidates": target_count,
                    "valid_edges": int(local_valid.sum().item()),
                    "reachable_sources": reachable_sources,
                    "reachable_targets": reachable_targets,
                })
                costs.append(next_cost)
                parents.append(next_parent)
                edge_cursor += edge_count
            final_index = int(torch.argmin(costs[-1]).item())
            row["transition_diagnostics"] = transition_diagnostics
            if not torch.isfinite(costs[-1][final_index]):
                row["failure"] = "no_continuous_ik_chain"
                disconnected = next(
                    (item for item in transition_diagnostics
                     if item["reachable_targets"] == 0), None
                )
                row["first_disconnected_transition"] = disconnected
                reports.append(row)
                removed.update(boxes.values())
                continue
            path = [layers[-1][final_index]]
            for layer_index in range(len(parents) - 1, -1, -1):
                final_index = int(parents[layer_index][final_index].item())
                path.append(layers[layer_index][final_index])
            path.reverse()
            path_array = torch.stack(path).detach().cpu().numpy()
            dense = densify(path_array, weights)
            dense_valid, dense_stability = validity.evaluate(torch.tensor(
                dense, device="cuda", dtype=torch.float32
            ))
            row.update({
                "success": bool(dense_valid.all().item()),
                "frames": dense,
                "waypoints": len(path),
                "dense_frames": len(dense),
                "edge_count": len(edge_meta),
                "path_cost": float(costs[-1].min().item()),
                "max_box_tilt_deg": math.degrees(math.acos(max(
                    -1.0, min(1.0, 1.0 - float(dense_stability.max().item()))
                ))),
                "planning_ms": row["batch_ik_ms"] + row["edge_validation_ms"],
            })
            if not row["success"]:
                row["failure"] = "dense_validation"
            row["round_ms"] = (time.perf_counter() - started_round) * 1000.0
            reports.append(row)
        except Exception as error:
            row["failure"] = f"{type(error).__name__}: {error}"
            row["round_ms"] = (time.perf_counter() - started_round) * 1000.0
            reports.append(row)
        removed.update(boxes.values())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"rounds": reports}, indent=2) + "\n")
        print(json.dumps({key: value for key, value in row.items() if key != "frames"}), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    output = {"rounds": reports, "successful_rounds": sum(row["success"] for row in reports)}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"successful_rounds": output["successful_rounds"]}, indent=2))


if __name__ == "__main__":
    main()
