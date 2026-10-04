#!/usr/bin/env python3

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import yaml

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

from v3_batched_loaded_search import GpuValidity, loaded_robot
from v3_wall_ik_benchmark import chassis_front_x, make_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--box-fit", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--named-poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    robot = yaml.safe_load(args.robot_config.read_text())
    box_fit = json.loads(args.box_fit.read_text())
    home = yaml.safe_load(args.named_poses.read_text())["named_poses"]["home"]
    joint_names = list(robot.get("robot_cfg", robot)["kinematics"]["cspace"]["joint_names"])
    front = chassis_front_x(args.urdf, home)
    removed = set()
    reports = []
    for row in source["rounds"]:
        boxes = row["boxes"]
        excluded = removed | set(boxes.values())
        scene = make_scene(front, 0.9, excluded)
        scene.cuboid = [item for item in scene.cuboid if item.name != "ground"]
        frames = []
        for source_frame in row.get("frames", []):
            values = dict(zip([
                "updown", *[f"left_joint{i}" for i in range(1, 8)],
                *[f"right_joint{i}" for i in range(1, 8)],
            ], source_frame))
            frames.append([values.get(name, 0.0) for name in joint_names])
        transport_frames = len(frames)
        if frames:
            endpoint = frames[-1]
            yaw_index = joint_names.index("base_yaw")
            for step in range(1, 361):
                frame = endpoint.copy()
                frame[yaw_index] = math.pi * step / 360.0
                frames.append(frame)
        configured = loaded_robot(robot, box_fit, active_sides=tuple(boxes))
        checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=copy.deepcopy(configured), scene_model=scene,
            n_cuboids=40, n_meshes=0, collision_activation_distance=0.0,
        ))
        validity = GpuValidity(
            checker, active_sides=tuple(boxes), max_box_tilt_deg=89.0,
            stability_weight=10.0, check_ground=True, ground_z=0.0,
            joint_names=joint_names,
            weights=[2.0, 2.0, 1.0, 5.0] + [1.0] * 14,
        )
        if frames:
            valid, stability = validity.evaluate(torch.tensor(
                frames, device="cuda", dtype=torch.float32
            ))
            invalid = torch.nonzero(~valid, as_tuple=False).reshape(-1)
            first_invalid = None if not len(invalid) else int(invalid[0].item())
            max_tilt = math.degrees(math.acos(max(
                -1.0, min(1.0, 1.0 - float(stability.max().item()))
            )))
        else:
            first_invalid = 0
            max_tilt = None
        success = row.get("success", False) and first_invalid is None
        reports.append({
            **{key: value for key, value in row.items() if key != "frames"},
            "success": success,
            "frames": frames,
            "transport_frames": transport_frames,
            "yaw_frames": max(0, len(frames) - transport_frames),
            "first_invalid_frame": first_invalid,
            "failure_stage": None if success else (
                row.get("failure", "transport")
                if first_invalid is None or first_invalid < transport_frames
                else "yaw180"
            ),
            "first_invalid_yaw_deg": None if first_invalid is None or first_invalid < transport_frames
            else 0.5 * (first_invalid - transport_frames + 1),
            "max_box_tilt_deg": max_tilt,
            "combined_collision_checked": True,
        })
        removed.update(boxes.values())
    output = {
        "joint_names": joint_names,
        "rounds": reports,
        "successful_rounds": sum(row["success"] for row in reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "successful_rounds": output["successful_rounds"],
        "rounds": [{
            "round": row["round"], "success": row["success"],
            "failure_stage": row["failure_stage"],
            "first_invalid_yaw_deg": row["first_invalid_yaw_deg"],
            "frames": len(row["frames"]),
        } for row in reports],
    }, indent=2))


if __name__ == "__main__":
    main()
