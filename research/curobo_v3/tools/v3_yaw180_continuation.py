#!/usr/bin/env python3

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.types import JointState

from v3_batched_loaded_search import GpuValidity, loaded_robot
from v3_wall_ik_benchmark import chassis_front_x, make_scene


ROUNDS = [
    {"left": 24, "right": 20}, {"left": 23, "right": 21}, {"left": 22},
    {"left": 19, "right": 15}, {"left": 18, "right": 16}, {"right": 17},
    {"left": 14, "right": 10}, {"left": 13, "right": 11}, {"left": 12},
    {"left": 9, "right": 5}, {"left": 8, "right": 6},
]


def audit(checker, joint_names, values):
    q = torch.tensor(values, device="cuda", dtype=torch.float32).reshape(1, 1, -1)
    checker.setup_batch_tensors(1, 1)
    state = checker.kinematics.compute_kinematics(
        JointState.from_position(q, joint_names=joint_names)
    )
    spheres = state.robot_spheres.view(1, 1, -1, 4)
    checker.collision_constraint.update_num_spheres(spheres.shape[2], batch_size=1, horizon=1)
    return {
        "self_collision_cost": float(checker.get_self_collision(spheres).sum().item()),
        "scene_collision_cost": float(checker.collision_constraint.forward(state).sum().item()),
        "bound_cost": float(checker.get_bound(q).sum().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--box-fit", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--named-poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    robot = yaml.safe_load(args.robot_config.read_text())
    box_fit = json.loads(args.box_fit.read_text())
    home = yaml.safe_load(args.named_poses.read_text())["named_poses"]["home"]
    joint_names = list(robot.get("robot_cfg", robot)["kinematics"]["cspace"]["joint_names"])
    removed = {box_id for round_boxes in ROUNDS for box_id in round_boxes.values()}
    scene = make_scene(chassis_front_x(args.urdf, home), 0.9, removed)
    scene.cuboid = [item for item in scene.cuboid if item.name != "ground"]
    values = {
        "base_x": 0.0, "base_y": 0.0, "base_yaw": 0.0, "updown": 0.0,
        **{f"left_joint{i + 1}": value for i, value in enumerate(np.deg2rad(
            [130, -105, 10, 90, -90, -40, 0]
        ))},
        **{f"right_joint{i + 1}": value for i, value in enumerate(np.deg2rad(
            [-130, 105, -10, -90, 90, 40, 0]
        ))},
    }
    frames = []
    for yaw in np.linspace(0.0, math.pi, 361):
        values["base_yaw"] = float(yaw)
        frames.append([values.get(name, 0.0) for name in joint_names])
    frame_tensor = torch.tensor(frames, device="cuda", dtype=torch.float32)
    output = {"joint_names": joint_names, "frames": frames, "removed_boxes": sorted(removed)}
    for mode, active_sides in (("released", ()), ("loaded", ("left", "right"))):
        configured = (
            loaded_robot(robot, box_fit, active_sides=active_sides)
            if active_sides else copy.deepcopy(robot)
        )
        checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=configured, scene_model=scene, n_cuboids=40, n_meshes=0,
            collision_activation_distance=0.0,
        ))
        validity = GpuValidity(
            checker, active_sides=active_sides, max_box_tilt_deg=89.0,
            stability_weight=10.0, check_ground=True, ground_z=0.0,
            joint_names=joint_names,
            weights=[2.0, 2.0, 1.0, 5.0] + [1.0] * 14,
        )
        valid, stability = validity.evaluate(frame_tensor)
        invalid = torch.nonzero(~valid, as_tuple=False).reshape(-1)
        first_invalid = None if not len(invalid) else int(invalid[0].item())
        output[mode] = {
            "success": first_invalid is None,
            "checked_frames": len(frames),
            "first_invalid_frame": first_invalid,
            "first_invalid_yaw_deg": None if first_invalid is None else first_invalid * 0.5,
            "max_box_tilt_deg": None if not active_sides else math.degrees(math.acos(
                max(-1.0, min(1.0, 1.0 - float(stability.max().item())))
            )),
            "audit": None if first_invalid is None else audit(
                checker, joint_names, frames[first_invalid]
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
