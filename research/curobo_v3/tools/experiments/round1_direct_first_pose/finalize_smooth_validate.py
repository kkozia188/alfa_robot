#!/usr/bin/env python3

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.types import JointState

sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_full_plan')
from v3_round1_full_plan import (  # noqa: E402
    ACTIVE_JOINTS,
    BOX_HALF,
    PAIR,
    add_payload_links,
    obb_overlap,
    payload_grid,
    tensor_state,
    update_payloads,
)
from v3_wall_ik_benchmark import make_scene, wall_center  # noqa: E402


def resample_segment(frames, rotary_step=math.radians(0.5), updown_step=0.0025):
    output = [list(frames[0])]
    for target in frames[1:]:
        start = np.asarray(output[-1], dtype=float)
        target = np.asarray(target, dtype=float)
        required = max(
            abs(target[0] - start[0]) / updown_step,
            np.max(np.abs(target[1:] - start[1:])) / rotary_step,
        )
        count = max(1, int(math.ceil(required)))
        for step in range(1, count + 1):
            fraction = step / count
            output.append(((1.0 - fraction) * start + fraction * target).tolist())
    return output


def robot_mask(checker, frames):
    q = torch.tensor(frames, device='cuda', dtype=torch.float32).unsqueeze(0)
    horizon = q.shape[1]
    checker.setup_batch_tensors(1, horizon)
    state = checker.kinematics.compute_kinematics(
        JointState.from_position(q, joint_names=ACTIVE_JOINTS))
    spheres = state.robot_spheres.view(1, horizon, -1, 4)
    checker.collision_constraint.update_num_spheres(
        spheres.shape[2], batch_size=1, horizon=horizon)
    distance = (
        checker.get_self_collision(spheres).reshape(1, horizon, -1).sum(-1)
        + checker.collision_constraint.forward(state).reshape(1, horizon, -1).sum(-1)
        + checker.get_bound(q).reshape(1, horizon, -1).sum(-1)
    )
    return (distance == 0)[0].detach().cpu().numpy(), state


def exact_payload_failures(state, tool_to_box, obstacles):
    tool_poses = state.tool_poses.to_dict()
    box_poses = {}
    for side in ('left', 'right'):
        positions = tool_poses[f'{side}_tool0'].position.reshape(-1, 3).detach().cpu().numpy()
        quaternions = tool_poses[f'{side}_tool0'].quaternion.reshape(-1, 4).detach().cpu().numpy()
        centers = []
        rotations = []
        for position, quaternion in zip(positions, quaternions):
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()
            transform[:3, 3] = position
            box = transform @ tool_to_box[side]
            centers.append(box[:3, 3])
            rotations.append(box[:3, :3])
        box_poses[side] = (centers, rotations)
    failures = {}
    for index in range(len(box_poses['left'][0])):
        for side in ('left', 'right'):
            center = box_poses[side][0][index]
            rotation = box_poses[side][1][index]
            for name, obstacle_center, obstacle_rotation, half in obstacles:
                if obb_overlap(center, rotation, BOX_HALF, obstacle_center, obstacle_rotation, half):
                    failures.setdefault(index, f'payload_{side}_collision:{name}')
                    break
        if obb_overlap(
            box_poses['left'][0][index], box_poses['left'][1][index], BOX_HALF,
            box_poses['right'][0][index], box_poses['right'][1][index], BOX_HALF):
            failures.setdefault(index, 'payload_payload_collision')
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--ik-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.result.read_text())
    summary = json.loads(args.ik_summary.read_text())
    robot = yaml.safe_load(Path(summary['robot_config']).read_text())
    loaded_robot = copy.deepcopy(robot)
    add_payload_links(loaded_robot)
    scene = make_scene(summary['chassis_front_x_m'], summary['wall_distance_m'], set(PAIR))

    smooth = []
    boundaries = {}
    for stage in report['stages']:
        segment = resample_segment(stage['frames'])
        if smooth:
            if max(abs(a - b) for a, b in zip(smooth[-1], segment[0])) > 2e-5:
                raise ValueError(f'boundary mismatch before {stage["name"]}')
            smooth.extend(segment[1:])
        else:
            smooth.extend(segment)
        boundaries[stage['name']] = len(smooth) - 1

    contact_index = boundaries['cartesian_approach_5cm']
    contact = tensor_state(smooth[contact_index])
    loaded_checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=loaded_robot, scene_model=scene, n_cuboids=40, n_meshes=0,
        collision_activation_distance=0.0))
    free_checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=robot, scene_model=scene, n_cuboids=40, n_meshes=0,
        collision_activation_distance=0.0))

    contact_poses = loaded_checker.kinematics.compute_kinematics(contact).tool_poses.to_dict()
    grids = {}
    tool_to_box = {}
    for side, box_id in zip(('left', 'right'), PAIR):
        pose = contact_poses[f'{side}_tool0']
        position = pose.position.reshape(-1, 3)[0].detach().cpu().numpy()
        quaternion = pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy()
        center = wall_center(summary['chassis_front_x_m'], summary['wall_distance_m'], box_id)
        grids[side] = payload_grid(position, quaternion, center)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()
        transform[:3, 3] = position
        box = np.eye(4)
        box[:3, 3] = center
        tool_to_box[side] = np.linalg.inv(transform) @ box
    update_payloads(loaded_checker, grids)

    free_valid, _ = robot_mask(free_checker, smooth[:contact_index + 1])
    loaded_valid, loaded_state = robot_mask(loaded_checker, smooth[contact_index:])
    obstacles = []
    for box_id in range(25):
        if box_id not in PAIR:
            obstacles.append((f'wall_box_{box_id}', wall_center(
                summary['chassis_front_x_m'], summary['wall_distance_m'], box_id),
                np.eye(3), BOX_HALF))
    wall_back = summary['chassis_front_x_m'] + summary['wall_distance_m'] + 0.30 + 1e-6
    obstacles.extend([
        ('ground', np.array([wall_back - 2.0, 0.0, -0.05]), np.eye(3), np.array([2.1, 1.3, 0.05])),
        ('left_wall', np.array([wall_back - 2.0, -1.25, 1.2]), np.eye(3), np.array([2.0, 0.05, 1.2])),
        ('right_wall', np.array([wall_back - 2.0, 1.25, 1.2]), np.eye(3), np.array([2.0, 0.05, 1.2])),
        ('front_wall', np.array([wall_back + 0.05, 0.0, 1.2]), np.eye(3), np.array([0.05, 1.3, 1.2])),
        ('ceiling', np.array([wall_back - 2.0, 0.0, 2.45]), np.eye(3), np.array([2.1, 1.3, 0.05])),
    ])
    payload_failures = exact_payload_failures(loaded_state, tool_to_box, obstacles)
    free_failed = np.flatnonzero(~free_valid).tolist()
    loaded_failed = np.flatnonzero(~loaded_valid).tolist()

    maximum = (0.0, -1, -1)
    for index, (first, second) in enumerate(zip(smooth, smooth[1:])):
        for joint, (a, b) in enumerate(zip(first, second)):
            delta = abs(b - a)
            if delta > maximum[0]:
                maximum = (delta, index, joint)
    delta, frame_index, joint_index = maximum
    validation = {
        'success': not free_failed and not loaded_failed and not payload_failures,
        'source_frame_count': len(report['frames']),
        'frame_count': len(smooth),
        'stage_boundaries': boundaries,
        'free_robot_failed_indices': free_failed,
        'loaded_robot_failed_indices': loaded_failed,
        'payload_failures': {str(key + contact_index): value for key, value in payload_failures.items()},
        'maximum_step_joint': ACTIVE_JOINTS[joint_index],
        'maximum_step_frame': frame_index,
        'maximum_step_raw': delta,
        'maximum_rotary_step_deg': max(
            math.degrees(max(abs(b - a) for a, b in zip(first[1:], second[1:])))
            for first, second in zip(smooth, smooth[1:])),
        'maximum_updown_step_m': max(
            abs(second[0] - first[0]) for first, second in zip(smooth, smooth[1:])),
    }
    report['validated_smooth_path'] = {
        **validation,
        'frames': smooth,
    }
    report['success'] = report['success'] and validation['success']
    if not validation['success']:
        report['failure_stage'] = 'validated_smooth_path'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
