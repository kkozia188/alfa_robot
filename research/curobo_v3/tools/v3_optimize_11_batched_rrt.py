#!/usr/bin/env python3

import argparse
import copy
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState

from v3_batched_loaded_search import GpuValidity, loaded_robot
from v3_wall_ik_benchmark import ACTIVE_JOINTS, chassis_front_x, make_scene


def tensor_state(values):
    return JointState.from_position(
        torch.tensor([values], device="cuda", dtype=torch.float32),
        joint_names=ACTIVE_JOINTS,
    )


def seed_tensor(frames, horizon):
    source = np.asarray(frames, dtype=np.float32)
    source_time = np.linspace(0.0, 1.0, len(source))
    target_time = np.linspace(0.0, 1.0, horizon)
    sampled = np.column_stack([
        np.interp(target_time, source_time, source[:, joint])
        for joint in range(source.shape[1])
    ])
    return torch.tensor(sampled, device="cuda", dtype=torch.float32).view(
        1, 1, horizon, source.shape[1]
    )


def active_frames(trajectory):
    names = list(trajectory.joint_names)
    values = trajectory.position.detach().cpu().numpy().reshape(-1, len(names))
    indices = [names.index(name) for name in ACTIVE_JOINTS]
    return values[:, indices].tolist()


def path_metrics(frames):
    values = np.asarray(frames, dtype=np.float64)
    steps = np.diff(values, axis=0)
    if len(steps) == 0:
        return {"frames": len(values), "joint_l2_length": 0.0, "max_joint_step_deg": 0.0}
    return {
        "frames": len(values),
        "joint_l2_length": float(np.linalg.norm(steps, axis=1).sum()),
        "max_joint_step_deg": float(np.rad2deg(np.abs(steps[:, 1:]).max())),
    }


def tensor_values(value):
    if value is None:
        return None
    return value.detach().cpu().reshape(-1).tolist()


def constraint_summary(metrics):
    if metrics is None or metrics.costs_and_constraints is None:
        return {}
    output = {}
    costs = metrics.costs_and_constraints
    for collection_name in ("constraints", "hybrid_costs_constraints"):
        collection = getattr(costs, collection_name)
        for name, value in zip(collection.names, collection.values):
            output[f"{collection_name}/{name}"] = float(value.detach().max().cpu())
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--box-fit", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--named-poses", type=Path, required=True)
    parser.add_argument("--round", type=int)
    parser.add_argument("--segment-frames", type=int, default=64)
    args = parser.parse_args()

    report = json.loads(args.result.read_text())
    robot = yaml.safe_load(args.robot_config.read_text())
    box_fit = json.loads(args.box_fit.read_text())
    named_poses = yaml.safe_load(args.named_poses.read_text())["named_poses"]
    transition = yaml.safe_load(Path(
        "/mnt/mydisk/ALFA/curobo_v2_ws/src/curobo/curobo/content/configs/task/"
        "trajopt/transition_bspline_trajopt.yml"
    ).read_text())
    transition["transition_model_cfg"]["n_knots"] = 32
    optimizer = yaml.safe_load(Path(
        "/mnt/mydisk/ALFA/curobo_v2_ws/src/curobo/curobo/content/configs/task/"
        "trajopt/lbfgs_bspline_trajopt.yml"
    ).read_text())
    optimizer["rollout"]["constraint_cfg"]["self_collision_cfg"]["weight"] = 1000000.0
    front = chassis_front_x(args.urdf, named_poses["home"])
    removed = set()

    for row in report["rounds"]:
        boxes = row["boxes"]
        if not row.get("frames") or (args.round and row["round"] != args.round):
            removed.update(boxes.values())
            continue

        active_sides = tuple(boxes)
        scene = make_scene(front, 0.9, removed | set(boxes.values()))
        scene.cuboid = [item for item in scene.cuboid if item.name != "ground"]
        configured_robot = loaded_robot(robot, box_fit, active_sides=active_sides)
        planner = MotionPlanner(MotionPlannerCfg.create(
            robot=copy.deepcopy(configured_robot),
            scene_model=scene,
            collision_cache={"cuboid": 40},
            num_ik_seeds=4,
            num_trajopt_seeds=1,
            self_collision_check=True,
            use_cuda_graph=False,
            trajopt_transition_model=copy.deepcopy(transition),
            trajopt_optimizer_configs=[copy.deepcopy(optimizer)],
            optimizer_collision_activation_distance=0.0,
        ))
        checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=copy.deepcopy(configured_robot),
            scene_model=scene,
            n_cuboids=40,
            n_meshes=0,
            collision_activation_distance=0.0,
        ))
        validity = GpuValidity(
            checker, active_sides=active_sides,
            max_box_tilt_deg=89.0, stability_weight=10.0,
            check_ground=True, ground_z=0.0,
        )
        raw_frames = row["frames"]
        optimized_frames = []
        segment_results = []
        success = True
        solve_time_ms = 0.0
        started_all = time.perf_counter()
        segment_start = 0
        result = None
        while segment_start < len(raw_frames) - 1:
            segment_end = min(
                segment_start + args.segment_frames - 1, len(raw_frames) - 1
            )
            segment = raw_frames[segment_start:segment_end + 1]
            seed = seed_tensor(segment, planner.trajopt_solver.action_horizon)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = planner.trajopt_solver.solve_cspace(
                goal_state=tensor_state(segment[-1]),
                current_state=tensor_state(segment[0]),
                seed_traj=seed,
                num_seeds=1,
                return_seeds=1,
                finetune_attempts=2,
            )
            torch.cuda.synchronize()
            segment_wall_ms = (time.perf_counter() - started) * 1000.0
            segment_success = bool(result.success.reshape(-1)[0].item())
            segment_plan = result.get_interpolated_plan()
            segment_frames = [] if segment_plan is None else active_frames(segment_plan)
            segment_results.append({
                "start": segment_start,
                "end": segment_end,
                "success": segment_success,
                "wall_ms": segment_wall_ms,
                "solve_time_ms": float(result.solve_time) * 1000.0,
                "raw_frames": len(segment),
                "optimized_frames": len(segment_frames),
            })
            solve_time_ms += float(result.solve_time) * 1000.0
            if not segment_success or not segment_frames:
                success = False
                optimized_frames = []
                break
            optimized_frames.extend(
                segment_frames if not optimized_frames else segment_frames[1:]
            )
            segment_start = segment_end
        elapsed_ms = (time.perf_counter() - started_all) * 1000.0
        sphere_valid = False
        max_box_tilt_deg = None
        if optimized_frames:
            optimized_valid, optimized_stability = validity.evaluate(torch.tensor(
                optimized_frames, device="cuda", dtype=torch.float32
            ))
            sphere_valid = bool(optimized_valid.all().item())
            worst_up_z = 1.0 - float(optimized_stability.max().item())
            max_box_tilt_deg = float(np.degrees(np.arccos(np.clip(worst_up_z, -1.0, 1.0))))
        row["curobo_optimization"] = {
            "success": success and sphere_valid,
            "planner_success": success,
            "sphere_valid": sphere_valid,
            "ground_collision_checked": True,
            "max_box_tilt_deg": max_box_tilt_deg,
            "box_tilt_limit_deg": 89.0,
            "segment_frames": args.segment_frames,
            "segments": segment_results,
            "wall_ms": elapsed_ms,
            "solver_solve_time_ms": solve_time_ms,
            "feasible": tensor_values(result.feasible) if result is not None else None,
            "cspace_error": tensor_values(result.cspace_error) if result is not None else None,
            "position_error": tensor_values(result.position_error) if result is not None else None,
            "rotation_error": tensor_values(result.rotation_error) if result is not None else None,
            "constraints": constraint_summary(result.metrics) if result is not None else {},
            "interpolated_constraints": constraint_summary(result.interpolated_metrics) if result is not None else {},
            "raw_metrics": path_metrics(raw_frames),
            "optimized_metrics": path_metrics(optimized_frames),
            "frames": optimized_frames if success and sphere_valid else [],
        }
        print(json.dumps({
            "round": row["round"],
            **{key: value for key, value in row["curobo_optimization"].items() if key != "frames"},
        }), flush=True)
        del planner, checker, validity
        gc.collect()
        torch.cuda.empty_cache()
        removed.update(boxes.values())
        args.result.write_text(json.dumps(report, indent=2) + "\n")

    report["optimized_rounds"] = sum(
        item.get("curobo_optimization", {}).get("success", False)
        for item in report["rounds"]
    )
    args.result.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
