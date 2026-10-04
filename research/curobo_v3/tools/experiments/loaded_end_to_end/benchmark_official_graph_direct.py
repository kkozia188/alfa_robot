#!/usr/bin/env python3

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/src/curobo')

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo._src.util.trajectory import TrajInterpolationType

sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_full_plan')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_direct_first_pose')
from v3_interactive_pair_core import (  # noqa: E402
    batch_robot_validation,
    exact_obstacles,
    exact_payload_failures,
)
from v3_round1_full_plan import (  # noqa: E402
    add_payload_links,
    payload_grid,
    result_trajectory,
    state_rows,
    tensor_state,
    update_payloads,
)
from v3_wall_ik_benchmark import make_scene, wall_center  # noqa: E402


def graph_config(args):
    return {
        'graph_planner': {
            'max_nodes': args.max_nodes,
            'steer_buffer_size': args.steer_buffer_size,
            'cspace_similarity_threshold': args.cspace_resolution,
            'sample_rejection_ratio': 10,
            'neighbors_per_node': args.neighbors,
            'feasibility_buffer_size': args.feasibility_buffer_size,
            'new_nodes_per_iteration': args.nodes_per_iteration,
            'max_path_finding_iterations': args.graph_iterations,
            'min_finetune_iterations': 2,
            'use_default_position_heuristic': args.use_default_position_heuristic,
            'exploration_radius': args.exploration_radius,
            'exploration_radius_growth_factor': 1.15,
            'sampler_seed': args.seed,
            'sampler_buffer_size': args.sampler_buffer_size,
            'connect_terminal_nodes_with_nearest': args.connect_terminal_nodes_with_nearest,
            'ellipsoid_projection_method': 'householder',
            'neighbors_per_node_growth_factor': 1.1,
            'new_nodes_per_iteration_growth_factor': 1.05,
        }
    }


def planner_config(robot, scene, args):
    return MotionPlannerCfg.create(
        robot=robot,
        scene_model=scene,
        collision_cache={'cuboid': 40},
        num_ik_seeds=4,
        num_trajopt_seeds=args.num_trajopt_seeds,
        self_collision_check=True,
        use_cuda_graph=True,
        optimizer_collision_activation_distance=args.activation_distance,
        graph_planner_config=graph_config(args),
    )


def update_all_planner_payloads(planner, grids):
    models = [planner.ik_solver.kinematics]
    for rollout in planner.trajopt_solver.core.get_all_rollout_instances():
        models.append(rollout.transition_model.robot_model)
    models.extend((
        planner.graph_planner.feasibility_rollout.transition_model.robot_model,
        planner.graph_planner.auxiliary_rollout.transition_model.robot_model,
    ))
    seen = set()
    for model in models:
        if id(model) in seen:
            continue
        seen.add(id(model))
        params = model.config.kinematics_config
        for side, grid in grids.items():
            params.update_link_spheres(
                f'{side}_payload',
                torch.tensor(grid, device='cuda', dtype=torch.float32),
            )


def payload_grid_from_tool_to_box(tool_to_box, radius):
    rows = []
    for x in (-0.10, 0.0, 0.10):
        for y in (-0.15, -0.05, 0.05, 0.15):
            for z in (-0.15, -0.05, 0.05, 0.15):
                point = tool_to_box @ np.asarray([x, y, z, 1.0])
                rows.append([*point[:3].tolist(), radius])
    return np.asarray(rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--ik-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-nodes', type=int, default=50000)
    parser.add_argument('--steer-buffer-size', type=int, default=20000)
    parser.add_argument('--feasibility-buffer-size', type=int, default=10000)
    parser.add_argument('--nodes-per-iteration', type=int, default=250)
    parser.add_argument('--graph-iterations', type=int, default=40)
    parser.add_argument('--neighbors', type=int, default=20)
    parser.add_argument('--sampler-buffer-size', type=int, default=20000)
    parser.add_argument('--exploration-radius', type=float, default=1.5)
    parser.add_argument('--cspace-resolution', type=float, default=0.02)
    parser.add_argument('--activation-distance', type=float, default=0.02)
    parser.add_argument('--payload-sphere-radius', type=float, default=0.0867)
    parser.add_argument('--num-trajopt-seeds', type=int, default=8)
    parser.add_argument('--direct-trajopt-only', action='store_true')
    parser.add_argument('--skip-direct-trajopt', action='store_true')
    parser.add_argument('--interpolation-steps', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--disable-graph-shortcut', action='store_true')
    parser.add_argument('--connect-terminal-nodes-with-nearest', action='store_true')
    parser.add_argument('--use-default-position-heuristic', action='store_true')
    parser.add_argument('--graph-warmup-iterations', type=int, default=0)
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    summary = json.loads(args.ik_summary.read_text())
    boxes = {side: int(box_id) for side, box_id in source['boxes'].items()}
    removed = set(source.get('removed_before', ())) | set(boxes.values())
    robot = yaml.safe_load(Path(summary['robot_config']).read_text())
    loaded_robot = copy.deepcopy(robot)
    add_payload_links(loaded_robot)
    kinematics_config = loaded_robot.get('robot_cfg', loaded_robot)['kinematics']
    disabled_grid = np.zeros((48, 4), dtype=np.float32)
    disabled_grid[:, 3] = -100.0
    initial_grids = {}
    source_tool_to_box = {
        side: np.asarray(transform, dtype=float)
        for side, transform in source['tool_to_box'].items()
    }
    for side in ('left', 'right'):
        if side in source_tool_to_box:
            initial_grids[side] = payload_grid_from_tool_to_box(
                source_tool_to_box[side], args.payload_sphere_radius)
        else:
            initial_grids[side] = disabled_grid.copy()
        kinematics_config['collision_spheres'][f'{side}_payload'] = [
            {'center': row[:3].tolist(), 'radius': float(row[3])}
            for row in initial_grids[side]
        ]
    scene = make_scene(summary['chassis_front_x_m'], summary['wall_distance_m'], removed)

    create_started = time.perf_counter()
    planner = MotionPlanner(planner_config(loaded_robot, scene, args))
    checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=loaded_robot,
        scene_model=scene,
        n_cuboids=40,
        n_meshes=0,
        collision_activation_distance=0.0,
    ))
    create_ms = (time.perf_counter() - create_started) * 1000.0

    contact = tensor_state(source['frames'][source['attach_index']])
    extracted = tensor_state(source['frames'][source['boundaries']['cartesian_extract_35cm']])
    placement = tensor_state(source['frames'][source['release_index']])
    poses = planner.compute_kinematics(contact).tool_poses.to_dict()
    grids = {}
    tool_to_box = {}
    for side in ('left', 'right'):
        if side not in boxes:
            grids[side] = disabled_grid.copy()
            continue
        box_id = boxes[side]
        pose = poses[f'{side}_tool0']
        position = pose.position.reshape(-1, 3)[0].detach().cpu().numpy()
        quaternion = pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy()
        center = wall_center(summary['chassis_front_x_m'], summary['wall_distance_m'], box_id)
        grids[side] = payload_grid(position, quaternion, center)
        grids[side][:, 3] = args.payload_sphere_radius
        tool = np.eye(4)
        tool[:3, :3] = Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()
        tool[:3, 3] = position
        box = np.eye(4)
        box[:3, 3] = center
        tool_to_box[side] = np.linalg.inv(tool) @ box
    update_all_planner_payloads(planner, grids)
    update_payloads(checker, grids)
    planner.graph_planner.reset_cuda_graph()
    planner.trajopt_solver.reset_cuda_graph()
    if args.disable_graph_shortcut:
        def no_prune(paths, start_idx, goal_idx):
            return planner.graph_planner._find_path_for_index_pairs(
                start_idx, goal_idx, return_length=True)
        planner.graph_planner.path_pruner.prune_path_with_shortcuts = no_prune

    lower, upper = planner.kinematics.get_joint_limits().position
    extracted.position = torch.clamp(extracted.position, lower + 1e-5, upper - 1e-5)
    placement.position = torch.clamp(placement.position, lower + 1e-5, upper - 1e-5)

    warmup_started = time.perf_counter()
    if args.graph_warmup_iterations > 0:
        planner.graph_planner.warmup(
            num_warmup_iterations=args.graph_warmup_iterations)
        planner.graph_planner.reset_buffer()
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

    direct = None
    direct_wall_ms = None
    if not args.skip_direct_trajopt:
        direct_started = time.perf_counter()
        direct = planner.plan_cspace(
            placement,
            extracted,
            max_attempts=1,
            enable_graph_attempt=2,
        )
        torch.cuda.synchronize()
        direct_wall_ms = (time.perf_counter() - direct_started) * 1000.0
    direct_frames = []
    direct_success = direct is not None and bool(direct.success.any().item())
    direct_solve_ms = None if direct is None else direct.solve_time * 1000.0
    direct_robot_failures = []
    direct_payload_failures = {}
    direct_trajectory = None if direct is None else result_trajectory(direct)
    if direct_trajectory is not None:
        direct_frames = state_rows(direct_trajectory)
        direct_robot_failures, direct_state = batch_robot_validation(checker, direct_frames)
        direct_payload_failures = exact_payload_failures(
            direct_state, tool_to_box, exact_obstacles(summary, removed))

    graph = None
    graph_wall_ms = None
    graph_frames = []
    raw_waypoints = []
    raw_waypoint_feasible = []
    raw_edge_diagnostics = []
    raw_path_indices = []
    if not args.direct_trajopt_only:
        graph_started = time.perf_counter()
        graph = planner.graph_planner.find_path(
            extracted.position,
            placement.position,
            interpolate_waypoints=True,
            interpolation_steps=planner.trajopt_solver.action_horizon,
            validate_interpolated_trajectory=False,
        )
        torch.cuda.synchronize()
        graph_wall_ms = (time.perf_counter() - graph_started) * 1000.0
    if graph is not None and graph.success.any():
        dense_graph = planner.graph_planner.get_interpolated_trajectory(
            graph.plan_waypoints,
            graph.success,
            args.interpolation_steps,
            TrajInterpolationType.LINEAR,
        )
        graph_frames = dense_graph[0].detach().cpu().tolist()
        raw_waypoints = graph.plan_waypoints[0].detach().cpu().tolist()
        raw = graph.plan_waypoints[0]
        raw_path_indices = planner.graph_planner._find_path_for_index_pairs(
            [0], [1])[0]
        raw_waypoint_feasible = planner.graph_planner.check_samples_feasibility(
            raw).detach().cpu().tolist()
        for edge_index in range(raw.shape[0] - 1):
            edge = torch.lerp(
                raw[edge_index].unsqueeze(0),
                raw[edge_index + 1].unsqueeze(0),
                torch.linspace(0.0, 1.0, 256, device=raw.device).unsqueeze(1),
            )
            edge_mask = planner.graph_planner.check_samples_feasibility(edge)
            failed = torch.where(~edge_mask)[0].detach().cpu().tolist()
            indexed_start = torch.cat((
                raw[edge_index],
                torch.tensor([float(raw_path_indices[edge_index])], device=raw.device),
            )).unsqueeze(0)
            indexed_goal = torch.cat((
                raw[edge_index + 1],
                torch.tensor([float(raw_path_indices[edge_index + 1])], device=raw.device),
            )).unsqueeze(0)
            connector_end = planner.graph_planner.linear_connector.steer_until_infeasible(
                indexed_start, indexed_goal)[0, :-1]
            connector_line = planner.graph_planner.linear_connector._compute_steering_line_points(
                indexed_start, indexed_goal)[0]
            connector_mask = planner.graph_planner.check_samples_feasibility(connector_line)
            connector_failed = torch.where(~connector_mask)[0].detach().cpu().tolist()
            raw_edge_diagnostics.append({
                'edge': edge_index,
                'failure_count': len(failed),
                'first_failure': None if not failed else failed[0],
                'connector_distance_to_start': float(torch.max(torch.abs(
                    connector_end - raw[edge_index])).item()),
                'connector_distance_to_goal': float(torch.max(torch.abs(
                    connector_end - raw[edge_index + 1])).item()),
                'connector_steps': int(connector_line.shape[0]),
                'connector_failure_count': len(connector_failed),
                'connector_first_failure': (
                    None if not connector_failed else connector_failed[0]),
                'connector_last_failure': (
                    None if not connector_failed else connector_failed[-1]),
            })

    graph_robot_failures = []
    graph_payload_failures = {}
    if graph_frames:
        graph_robot_failures, graph_state = batch_robot_validation(checker, graph_frames)
        graph_payload_failures = exact_payload_failures(
            graph_state, tool_to_box, exact_obstacles(summary, removed))

    trajopt_frames = []
    trajopt_success = False
    trajopt_wall_ms = None
    trajopt_solve_ms = None
    trajopt_robot_failures = []
    trajopt_payload_failures = {}
    if graph is not None and graph.success.any():
        seed_traj = graph.interpolated_waypoints[0:1].unsqueeze(0)
        trajopt_started = time.perf_counter()
        trajopt = planner.trajopt_solver.solve_cspace(
            placement, extracted, seed_traj=seed_traj)
        torch.cuda.synchronize()
        trajopt_wall_ms = (time.perf_counter() - trajopt_started) * 1000.0
        trajopt_solve_ms = trajopt.solve_time * 1000.0
        trajopt_success = bool(trajopt.success.any().item())
        trajectory = result_trajectory(trajopt)
        if trajectory is not None:
            trajopt_frames = state_rows(trajectory)
            trajopt_robot_failures, trajopt_state = batch_robot_validation(
                checker, trajopt_frames)
            trajopt_payload_failures = exact_payload_failures(
                trajopt_state, tool_to_box, exact_obstacles(summary, removed))

    output = {
        'kind': 'v3_official_graph_trajopt_direct_loaded_placement',
        'round': source['round'],
        'boxes': boxes,
        'source': str(args.source),
        'joint_names': planner.joint_names,
        'parameters': vars(args) | {'source': str(args.source), 'ik_summary': str(args.ik_summary), 'output': str(args.output)},
        'create_ms': create_ms,
        'warmup_ms': warmup_ms,
        'direct_trajopt': {
            'success': direct_success,
            'solve_ms': direct_solve_ms,
            'wall_ms': direct_wall_ms,
            'frames': direct_frames,
            'robot_failure_count': len(direct_robot_failures),
            'payload_failure_count': len(direct_payload_failures),
            'first_payload_failure': (
                None if not direct_payload_failures else {
                    'frame': min(direct_payload_failures),
                    'reason': direct_payload_failures[min(direct_payload_failures)],
                }),
        },
        'graph': {
            'success': False if graph is None else bool(graph.success.any().item()),
            'valid_query': None if graph is None else graph.valid_query,
            'solve_ms': None if graph is None else graph.solve_time * 1000.0,
            'wall_ms': graph_wall_ms,
            'path_length': (
                [] if graph is None else graph.path_length.detach().cpu().tolist()),
            'roadmap_nodes': planner.graph_planner.node_manager.n_nodes,
            'raw_waypoints': raw_waypoints,
            'raw_path_indices': raw_path_indices,
            'raw_waypoint_feasible': raw_waypoint_feasible,
            'raw_edge_diagnostics': raw_edge_diagnostics,
            'frames': graph_frames,
            'robot_failure_count': len(graph_robot_failures),
            'payload_failure_count': len(graph_payload_failures),
            'first_payload_failure': (
                None if not graph_payload_failures else {
                    'frame': min(graph_payload_failures),
                    'reason': graph_payload_failures[min(graph_payload_failures)],
                }),
        },
        'trajopt': {
            'success': trajopt_success,
            'solve_ms': trajopt_solve_ms,
            'wall_ms': trajopt_wall_ms,
            'frames': trajopt_frames,
            'robot_failure_count': len(trajopt_robot_failures),
            'payload_failure_count': len(trajopt_payload_failures),
            'first_payload_failure': (
                None if not trajopt_payload_failures else {
                    'frame': min(trajopt_payload_failures),
                    'reason': trajopt_payload_failures[min(trajopt_payload_failures)],
                }),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    printable = copy.deepcopy(output)
    printable['direct_trajopt'].pop('frames')
    printable['graph'].pop('raw_waypoints')
    printable['graph'].pop('frames')
    printable['trajopt'].pop('frames')
    print(json.dumps(printable, indent=2, default=str))


if __name__ == '__main__':
    main()
