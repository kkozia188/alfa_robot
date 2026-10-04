#!/usr/bin/env python3

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_full_plan')
sys.path.insert(0, str(Path(__file__).parent))
from direct_first_pose_plan import goal  # noqa: E402
from v3_round1_full_plan import (  # noqa: E402
    ACTIVE_JOINTS, PAIR, add_payload_links, audit_state, payload_grid,
    result_trajectory, state_rows, tensor_state, update_payloads,
    validate_payload_path,
)
from v3_wall_ik_benchmark import make_scene, wall_center  # noqa: E402


def planner_cfg(robot, scene, ik_seeds, trajopt_seeds, use_cuda_graph):
    return MotionPlannerCfg.create(
        robot=robot,
        scene_model=scene,
        collision_cache={'cuboid': 40},
        num_ik_seeds=ik_seeds,
        num_trajopt_seeds=trajopt_seeds,
        self_collision_check=True,
        use_cuda_graph=use_cuda_graph,
        position_tolerance=0.002,
        orientation_tolerance=math.radians(1.0),
    )


def pose_dict(poses, frame):
    pose = poses[frame]
    return {
        'position': pose.position.reshape(-1, 3)[0].detach().cpu().tolist(),
        'quaternion_wxyz': pose.quaternion.reshape(-1, 4)[0].detach().cpu().tolist(),
    }


def constraint_metrics(kinematics, frames, contact_poses, distance):
    state = tensor_state(frames[0])
    state.position = torch.tensor(frames, device='cuda', dtype=torch.float32)
    poses = kinematics.compute_kinematics(state).tool_poses.to_dict()
    output = {}
    for frame in ('left_tool0', 'right_tool0'):
        positions = poses[frame].position.reshape(-1, 3).detach().cpu().numpy()
        quaternions = poses[frame].quaternion.reshape(-1, 4).detach().cpu().numpy()
        start = np.asarray(contact_poses[frame]['position'])
        target = start.copy()
        target[0] -= distance
        yz_error = np.linalg.norm(positions[:, 1:3] - start[1:3], axis=1)
        orientation_error = []
        reference = Rotation.from_quat(np.roll(contact_poses[frame]['quaternion_wxyz'], -1))
        for quaternion in quaternions:
            orientation_error.append((reference.inv() * Rotation.from_quat(np.roll(quaternion, -1))).magnitude())
        progress = start[0] - positions[:, 0]
        output[frame] = {
            'maximum_yz_error_mm': float(np.max(yz_error) * 1000.0),
            'maximum_orientation_error_deg': float(np.degrees(np.max(orientation_error))),
            'terminal_position_error_mm': float(np.linalg.norm(positions[-1] - target) * 1000.0),
            'terminal_orientation_error_deg': float(np.degrees(orientation_error[-1])),
            'maximum_backward_progress_mm': float(max(0.0, -np.min(np.diff(progress))) * 1000.0),
            'progress_m': progress.tolist(),
        }
    left_progress = np.asarray(output['left_tool0']['progress_m'])
    right_progress = np.asarray(output['right_tool0']['progress_m'])
    output['maximum_dual_progress_difference_mm'] = float(
        np.max(np.abs(left_progress - right_progress)) * 1000.0)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-result', type=Path, required=True)
    parser.add_argument('--ik-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distance', type=float, default=0.35)
    parser.add_argument('--ik-seeds', type=int, default=16)
    parser.add_argument('--trajopt-seeds', type=int, default=2)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--constraint-scale', type=float, default=1.0)
    args = parser.parse_args()

    source = json.loads(args.source_result.read_text())
    summary = json.loads(args.ik_summary.read_text())
    robot = yaml.safe_load(Path(summary['robot_config']).read_text())
    loaded_robot = copy.deepcopy(robot)
    add_payload_links(loaded_robot)
    scene = make_scene(summary['chassis_front_x_m'], summary['wall_distance_m'], set(PAIR))
    contact = tensor_state(source['stages'][1]['frames'][-1])

    create_started = time.perf_counter()
    planner = MotionPlanner(planner_cfg(
        loaded_robot, scene, args.ik_seeds, args.trajopt_seeds, args.cuda_graph))
    create_ms = (time.perf_counter() - create_started) * 1000.0
    contact_fk = planner.compute_kinematics(contact).tool_poses.to_dict()
    contact_poses = {
        frame: pose_dict(contact_fk, frame)
        for frame in ('left_tool0', 'right_tool0')
    }
    target_poses = copy.deepcopy(contact_poses)
    for frame in target_poses:
        target_poses[frame]['position'][0] -= args.distance

    grids = {}
    tool_to_box = {}
    for side, box_id in zip(('left', 'right'), PAIR):
        pose = contact_fk[f'{side}_tool0']
        position = pose.position.reshape(-1, 3)[0].detach().cpu().numpy()
        quaternion = pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy()
        center = wall_center(summary['chassis_front_x_m'], summary['wall_distance_m'], box_id)
        grids[side] = payload_grid(position, quaternion, center)
        tool_transform = np.eye(4)
        tool_transform[:3, :3] = Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()
        tool_transform[:3, 3] = position
        box_transform = np.eye(4)
        box_transform[:3, 3] = center
        tool_to_box[side] = np.linalg.inv(tool_transform) @ box_transform
    update_payloads(planner, grids)

    linear = ToolPoseCriteria.linear_motion(
        axis='x', non_terminal_scale=args.constraint_scale,
        project_distance_to_goal=False)
    planner.update_tool_pose_criteria({
        'left_tool0': linear,
        'right_tool0': linear,
    })

    counters = {'trajopt_calls': 0, 'graph_calls': 0, 'ik_calls': 0}
    original_trajopt = planner.trajopt_solver.solve_pose
    original_graph = planner.graph_planner.find_path
    original_ik = planner.ik_solver.solve_pose

    def counted_trajopt(*call_args, **call_kwargs):
        counters['trajopt_calls'] += 1
        return original_trajopt(*call_args, **call_kwargs)

    def counted_graph(*call_args, **call_kwargs):
        counters['graph_calls'] += 1
        return original_graph(*call_args, **call_kwargs)

    def counted_ik(*call_args, **call_kwargs):
        counters['ik_calls'] += 1
        return original_ik(*call_args, **call_kwargs)

    planner.trajopt_solver.solve_pose = counted_trajopt
    planner.graph_planner.find_path = counted_graph
    planner.ik_solver.solve_pose = counted_ik

    wall_ms = []
    result = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = planner.plan_pose(
            goal(target_poses), contact,
            max_attempts=args.max_attempts,
            enable_graph_attempt=1,
        )
        wall_ms.append((time.perf_counter() - started) * 1000.0)

    trajectory = result_trajectory(result)
    frames = [] if trajectory is None else state_rows(trajectory)
    robot_failures = []
    payload_success = False
    payload_reason = 'no_trajectory'
    metrics = None
    if frames:
        for index, row in enumerate(frames):
            audit = audit_state(planner.kinematics, tensor_state(row), scene)
            if not audit['valid']:
                robot_failures.append({'frame': index, 'audit': audit})
        payload_success, payload_reason = validate_payload_path(
            planner.kinematics, frames, tool_to_box,
            summary['chassis_front_x_m'], summary['wall_distance_m'])
        metrics = constraint_metrics(planner.kinematics, frames, contact_poses, args.distance)

    standard = ToolPoseCriteria()
    planner.update_tool_pose_criteria({
        'left_tool0': standard,
        'right_tool0': standard,
    })
    output = {
        'native_success': bool(
            result is not None and result.success is not None and result.success.any().item()),
        'validated_success': bool(frames) and not robot_failures and payload_success,
        'create_ms': create_ms,
        'wall_ms_values': wall_ms,
        'wall_ms_min': min(wall_ms),
        'wall_ms_median': statistics.median(wall_ms),
        'distance_m': args.distance,
        'ik_seeds': args.ik_seeds,
        'trajopt_seeds': args.trajopt_seeds,
        'max_attempts': args.max_attempts,
        'repeats': args.repeats,
        'cuda_graph': args.cuda_graph,
        'constraint_scale': args.constraint_scale,
        'points': len(frames),
        'frames': frames,
        'robot_failure_count': len(robot_failures),
        'robot_failures': robot_failures[:20],
        'payload_success': payload_success,
        'payload_reason': payload_reason,
        'constraint_metrics': metrics,
        'calls': counters,
        'solver_total_time': None if result is None else result.total_time,
        'solver_solve_time': None if result is None else result.solve_time,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    printable = copy.deepcopy(output)
    printable.pop('frames', None)
    if printable['constraint_metrics']:
        for frame in ('left_tool0', 'right_tool0'):
            printable['constraint_metrics'][frame].pop('progress_m', None)
    print(json.dumps(printable, indent=2))


if __name__ == '__main__':
    main()
