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

from curobo.motion_planner import MotionPlanner

sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_full_plan')
sys.path.insert(0, str(Path(__file__).parent))
from direct_first_pose_plan import accept_motion_result, planner_cfg  # noqa: E402
from v3_round1_full_plan import (  # noqa: E402
    ACTIVE_JOINTS, PAIR, add_payload_links, guard_limit_roundoff,
    payload_grid, tensor_state, update_payloads,
)
from v3_wall_ik_benchmark import make_scene, wall_center  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-result', type=Path, required=True)
    parser.add_argument('--ik-summary', type=Path, required=True)
    parser.add_argument('--named-poses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--trajopt-seeds', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--trajopt-iters', type=int)
    parser.add_argument('--limit-margin-rad', type=float, default=1e-6)
    args = parser.parse_args()

    source = json.loads(args.source_result.read_text())
    summary = json.loads(args.ik_summary.read_text())
    named = yaml.safe_load(args.named_poses.read_text())['named_poses']
    robot = yaml.safe_load(Path(summary['robot_config']).read_text())
    loaded_robot = copy.deepcopy(robot)
    add_payload_links(loaded_robot)
    scene = make_scene(summary['chassis_front_x_m'], summary['wall_distance_m'], set(PAIR))
    contact = tensor_state(source['stages'][1]['frames'][-1])
    extracted = tensor_state(source['stages'][2]['frames'][-1])
    home = tensor_state([named['home'][name] for name in ACTIVE_JOINTS])

    create_started = time.perf_counter()
    planner = MotionPlanner(planner_cfg(
        loaded_robot, scene, seeds=args.trajopt_seeds,
        use_cuda_graph=args.cuda_graph,
        trajopt_iters=args.trajopt_iters))
    create_ms = (time.perf_counter() - create_started) * 1000.0
    lower, upper = planner.kinematics.get_joint_limits().position
    guarded_home = home.clone()
    guarded_home.position = torch.clamp(
        home.position,
        min=lower + args.limit_margin_rad,
        max=upper - args.limit_margin_rad,
    )
    guard_delta = (guarded_home.position - home.position).reshape(-1).detach().cpu().tolist()
    home = guarded_home

    contact_poses = planner.compute_kinematics(contact).tool_poses.to_dict()
    grids = {}
    tool_to_box = {}
    for side, box_id in zip(('left', 'right'), PAIR):
        pose = contact_poses[f'{side}_tool0']
        position = pose.position.reshape(-1, 3)[0].detach().cpu().numpy()
        quaternion = pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy()
        center = wall_center(summary['chassis_front_x_m'], summary['wall_distance_m'], box_id)
        grids[side] = payload_grid(position, quaternion, center)
        tool = np.eye(4)
        tool[:3, :3] = Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()
        tool[:3, 3] = position
        box = np.eye(4)
        box[:3, 3] = center
        tool_to_box[side] = np.linalg.inv(tool) @ box
    update_payloads(planner, grids)

    counters = {'trajopt_calls': 0, 'graph_calls': 0}
    original_trajopt = planner.trajopt_solver.solve_cspace
    original_graph = planner.graph_planner.find_path

    def counted_trajopt(*call_args, **call_kwargs):
        counters['trajopt_calls'] += 1
        return original_trajopt(*call_args, **call_kwargs)

    def counted_graph(*call_args, **call_kwargs):
        counters['graph_calls'] += 1
        return original_graph(*call_args, **call_kwargs)

    planner.trajopt_solver.solve_cspace = counted_trajopt
    planner.graph_planner.find_path = counted_graph

    plan_ms_values = []
    result = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = planner.plan_cspace(home, extracted, max_attempts=args.max_attempts)
        plan_ms_values.append((time.perf_counter() - started) * 1000.0)
    plan_ms = plan_ms_values[-1]
    native_success = bool(
        result is not None and result.success is not None
        and torch.count_nonzero(result.success).item() > 0)
    frames, method, failures = accept_motion_result(
        planner, result, extracted, home, scene, tool_to_box, summary)
    output = {
        'native_success': native_success,
        'accepted_success': bool(frames),
        'accepted_method': method,
        'create_ms': create_ms,
        'plan_ms': plan_ms,
        'max_attempts': args.max_attempts,
        'trajopt_seeds': args.trajopt_seeds,
        'trajopt_iters': args.trajopt_iters or 100,
        'limit_margin_rad': args.limit_margin_rad,
        'use_cuda_graph': args.cuda_graph,
        'repeats': args.repeats,
        'plan_ms_values': plan_ms_values,
        'plan_ms_min': min(plan_ms_values),
        'plan_ms_median': statistics.median(plan_ms_values),
        'guard_delta': guard_delta,
        'points': len(frames),
        'frames': frames,
        'failures': failures,
        'trajopt_calls': counters['trajopt_calls'],
        'graph_calls': counters['graph_calls'],
        'solver_total_time': None if result is None else result.total_time,
        'solver_solve_time': None if result is None else result.solve_time,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
