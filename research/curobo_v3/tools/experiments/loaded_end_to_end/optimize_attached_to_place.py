#!/usr/bin/env python3

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from curobo.motion_planner import MotionPlanner
from curobo._src.state.state_joint import JointState

sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_full_plan')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_direct_first_pose')
from direct_first_pose_plan import planner_cfg  # noqa: E402
from v3_interactive_pair_core import (  # noqa: E402
    exact_obstacles,
    exact_payload_failures,
)
from v3_round1_full_plan import (  # noqa: E402
    add_payload_links,
    audit_state,
    payload_grid,
    tensor_state,
    update_payloads,
)
from v3_wall_ik_benchmark import make_scene, wall_center  # noqa: E402


TOOL_FRAMES = ('left_tool0', 'right_tool0')


def append_segment(output, segment):
    if output:
        output.extend(segment[1:])
    else:
        output.extend(segment)


def loaded_reference(source):
    if source.get('boundaries') and source.get('attach_index') is not None:
        attach_index = int(source['attach_index'])
        release_index = int(source['release_index'])
        extract_end = int(source['boundaries']['cartesian_extract_35cm'])
        frames = np.asarray(
            source['frames'][attach_index:release_index + 1], dtype=np.float32)
        return frames, extract_end - attach_index + 1
    stages = {stage['name']: stage for stage in source['stages']}
    frames = []
    for name in (
        'cartesian_extract_35cm',
        'loaded_extract_to_first_home',
        'loaded_first_home_to_first_unloading',
    ):
        append_segment(frames, stages[name]['frames'])
    return np.asarray(frames, dtype=np.float32), len(stages['cartesian_extract_35cm']['frames'])


def resample_rows(rows, count):
    source_t = np.linspace(0.0, 1.0, len(rows))
    target_t = np.linspace(0.0, 1.0, count)
    return np.stack([
        np.interp(target_t, source_t, rows[:, joint])
        for joint in range(rows.shape[1])
    ], axis=1).astype(np.float32)


def quat_conjugate(quaternion):
    result = quaternion.clone()
    result[..., 1:] *= -1.0
    return result


def quat_multiply(left, right):
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), dim=-1)


def quat_rotate_inverse(quaternion, vector):
    zeros = torch.zeros_like(vector[..., :1])
    pure = torch.cat((zeros, vector), dim=-1)
    return quat_multiply(quat_multiply(quat_conjugate(quaternion), pure), quaternion)[..., 1:]


def quaternion_distance_squared(left, right):
    dot = torch.sum(left * right, dim=-1).abs().clamp(max=1.0)
    return 1.0 - dot.square()


def tool_tensors(kinematics, q):
    state = JointState(position=q.unsqueeze(0), joint_names=kinematics.joint_names)
    kinematics_state = kinematics.compute_kinematics(state)
    poses = kinematics_state.tool_poses.to_dict()
    return {
        frame: (
            poses[frame].position.reshape(-1, 3),
            poses[frame].quaternion.reshape(-1, 4),
        )
        for frame in TOOL_FRAMES
    }, kinematics_state


def trajectory_metrics(
        q, reference, tool_poses, reference_tool_poses, kinematics_state,
        scene_collision_cost, self_collision_cost, extract_nodes,
        trust_weight, smooth_weight, collision_weight):
    velocity = q[1:] - q[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    jerk = acceleration[1:] - acceleration[:-1]

    left_pos, left_quat = tool_poses['left_tool0']
    right_pos, right_quat = tool_poses['right_tool0']
    ref_left_pos, ref_left_quat = reference_tool_poses['left_tool0']
    ref_right_pos, ref_right_quat = reference_tool_poses['right_tool0']

    extraction_position = (
        torch.mean((left_pos[:extract_nodes] - ref_left_pos[:extract_nodes]).square())
        + torch.mean((right_pos[:extract_nodes] - ref_right_pos[:extract_nodes]).square())
    )
    extraction_orientation = (
        quaternion_distance_squared(
            left_quat[:extract_nodes], ref_left_quat[:extract_nodes]).mean()
        + quaternion_distance_squared(
            right_quat[:extract_nodes], ref_right_quat[:extract_nodes]).mean()
    )
    final_pose = (
        torch.mean((left_pos[-1] - ref_left_pos[-1]).square())
        + torch.mean((right_pos[-1] - ref_right_pos[-1]).square())
        + quaternion_distance_squared(left_quat[-1], ref_left_quat[-1])
        + quaternion_distance_squared(right_quat[-1], ref_right_quat[-1])
    )
    trust = torch.mean((q - reference).square())
    smooth = (
        2.0 * torch.mean(velocity.square())
        + 8.0 * torch.mean(acceleration.square())
        + 2.0 * torch.mean(jerk.square())
    )
    scene_collision = scene_collision_cost.forward(kinematics_state)
    scene_collision = scene_collision[:, extract_nodes:].mean()
    self_collision = self_collision_cost.forward(kinematics_state.robot_spheres).mean()
    losses = {
        'extraction_position': extraction_position,
        'extraction_orientation': extraction_orientation,
        'final_pose': final_pose,
        'trust': trust,
        'smooth': smooth,
        'scene_collision': scene_collision,
        'self_collision': self_collision,
    }
    total = (
        30000.0 * extraction_position
        + 1000.0 * extraction_orientation
        + 5000.0 * final_pose
        + trust_weight * trust
        + smooth_weight * smooth
        + collision_weight * scene_collision
        + collision_weight * self_collision
    )
    return total, losses


def densify(rows, split_index=None, rotational_step=math.radians(0.5), linear_step=0.0025):
    output = [rows[0].tolist()]
    dense_split_index = None
    for segment_index, (start, end) in enumerate(zip(rows[:-1], rows[1:]), start=1):
        delta = np.abs(end - start)
        count = max(
            1,
            int(math.ceil(delta[0] / linear_step)),
            int(math.ceil(np.max(delta[1:]) / rotational_step)),
        )
        for index in range(1, count + 1):
            output.append((start + (end - start) * (index / count)).tolist())
        if split_index is not None and segment_index == split_index:
            dense_split_index = len(output) - 1
    return output, dense_split_index


def pose_errors(optimized, reference):
    result = {}
    for frame in TOOL_FRAMES:
        opt_pos, opt_quat = optimized[frame]
        ref_pos, ref_quat = reference[frame]
        result[frame] = {
            'terminal_position_mm': float(
                torch.linalg.vector_norm(opt_pos[-1] - ref_pos[-1]).detach().cpu() * 1000.0),
            'terminal_orientation_deg': float(torch.rad2deg(
                2.0 * torch.acos(
                    torch.sum(opt_quat[-1] * ref_quat[-1]).abs().clamp(max=1.0)
                )).detach().cpu()),
        }
    return result


def extraction_errors(optimized, reference, extract_nodes):
    result = {}
    for frame in TOOL_FRAMES:
        opt_pos, opt_quat = optimized[frame]
        ref_pos, ref_quat = reference[frame]
        position_error = torch.linalg.vector_norm(
            opt_pos[:extract_nodes] - ref_pos[:extract_nodes], dim=-1)
        orientation_error = 2.0 * torch.acos(
            torch.sum(
                opt_quat[:extract_nodes] * ref_quat[:extract_nodes], dim=-1
            ).abs().clamp(max=1.0)
        )
        result[frame] = {
            'maximum_position_mm': float(position_error.max().detach().cpu() * 1000.0),
            'rms_position_mm': float(
                torch.sqrt(torch.mean(position_error.square())).detach().cpu() * 1000.0),
            'maximum_orientation_deg': float(
                torch.rad2deg(orientation_error.max()).detach().cpu()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--ik-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nodes', type=int, default=192)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--trust-radius-rad', type=float, default=0.02)
    parser.add_argument('--direct-after-extraction', action='store_true')
    parser.add_argument('--extract-node-fraction', type=float, default=0.2)
    parser.add_argument('--trust-weight', type=float, default=10.0)
    parser.add_argument('--smooth-weight', type=float, default=50.0)
    parser.add_argument('--collision-weight', type=float, default=100.0)
    parser.add_argument('--payload-sphere-radius', type=float, default=0.05)
    parser.add_argument('--skip-validation', action='store_true')
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    summary = json.loads(args.ik_summary.read_text())
    if source.get('boxes'):
        boxes = {side: int(box_id) for side, box_id in source['boxes'].items()}
    else:
        source_pair = [
            int(box_id) for box_id in source.get('pair', (24, 20))
            if box_id is not None]
        boxes = dict(zip(('left', 'right'), source_pair))
    pair = tuple(boxes.values())
    removed_before = set(source.get('removed_before', ()))
    removed_boxes = removed_before | set(pair)
    robot = yaml.safe_load(Path(summary['robot_config']).read_text())
    loaded_robot = copy.deepcopy(robot)
    add_payload_links(loaded_robot)
    scene = make_scene(
        summary['chassis_front_x_m'], summary['wall_distance_m'], removed_boxes)
    planner = MotionPlanner(planner_cfg(loaded_robot, scene, seeds=2))

    full_reference, extract_frame_count = loaded_reference(source)
    if args.direct_after_extraction:
        extract_nodes = max(3, int(round(args.extract_node_fraction * args.nodes)))
        extract_reference = resample_rows(full_reference[:extract_frame_count], extract_nodes)
        transport_nodes = args.nodes - extract_nodes + 1
        alpha = np.linspace(0.0, 1.0, transport_nodes, dtype=np.float32)[:, None]
        transport_reference = (
            extract_reference[-1][None, :] * (1.0 - alpha)
            + full_reference[-1][None, :] * alpha)
        reference_np = np.concatenate(
            (extract_reference, transport_reference[1:]), axis=0).astype(np.float32)
    else:
        reference_np = resample_rows(full_reference, args.nodes)
        extract_nodes = max(3, int(round(
            (extract_frame_count - 1) / (len(full_reference) - 1) * (args.nodes - 1))) + 1)
    reference = torch.tensor(reference_np, device='cuda', dtype=torch.float32)
    reference_tool_poses, _ = tool_tensors(planner.kinematics, reference)

    contact_fk = planner.compute_kinematics(tensor_state(reference_np[0])).tool_poses.to_dict()
    payload_grids = {}
    tool_to_box = {}
    disabled_grid = np.zeros((48, 4), dtype=np.float32)
    disabled_grid[:, 3] = -100.0
    for side in ('left', 'right'):
        if side not in boxes:
            payload_grids[side] = disabled_grid.copy()
            continue
        box_id = boxes[side]
        pose = contact_fk[f'{side}_tool0']
        position = pose.position.reshape(-1, 3)[0].detach().cpu().numpy()
        quaternion = pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy()
        center = wall_center(summary['chassis_front_x_m'], summary['wall_distance_m'], box_id)
        payload_grids[side] = payload_grid(position, quaternion, center)
        payload_grids[side][:, 3] = args.payload_sphere_radius
        tool = np.eye(4)
        tool[:3, :3] = Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()
        tool[:3, 3] = position
        box = np.eye(4)
        box[:3, 3] = center
        tool_to_box[side] = np.linalg.inv(tool) @ box
    update_payloads(planner, payload_grids)

    constraint_manager = planner.trajopt_solver.core.auxiliary_rollout.constraint_manager
    scene_collision_cost = constraint_manager.get_cost('scene_collision')
    self_collision_cost = constraint_manager.get_cost('self_collision')
    scene_collision_cost.setup_batch_tensors(1, args.nodes)
    self_collision_cost.setup_batch_tensors(1, args.nodes)
    scene_collision_cost.config.activation_distance[:] = 0.02

    limits = planner.kinematics.get_joint_limits()
    lower = limits.position_lower_limits.to(device='cuda', dtype=torch.float32) + 1e-5
    upper = limits.position_upper_limits.to(device='cuda', dtype=torch.float32) - 1e-5
    delta_raw = torch.nn.Parameter(torch.zeros_like(reference[1:-1]))
    optimization_mask = torch.ones_like(delta_raw)
    optimization_mask[:extract_nodes - 1] = 0.0
    optimizer = torch.optim.Adam([delta_raw], lr=args.learning_rate)
    history = []
    best_total = float('inf')
    best_delta_raw = delta_raw.detach().clone()
    torch.cuda.synchronize()
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    cuda_start.record()
    started = time.perf_counter()
    for iteration in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        interior = reference[1:-1] + (
            args.trust_radius_rad * torch.tanh(delta_raw) * optimization_mask)
        q = torch.cat((reference[:1], interior, reference[-1:]), dim=0)
        tool_poses, kinematics_state = tool_tensors(planner.kinematics, q)
        total, losses = trajectory_metrics(
            q, reference, tool_poses, reference_tool_poses, kinematics_state,
            scene_collision_cost, self_collision_cost, extract_nodes,
            trust_weight=args.trust_weight,
            smooth_weight=args.smooth_weight,
            collision_weight=args.collision_weight)
        should_sample = iteration % 25 == 0 or iteration == args.iterations - 1
        if should_sample:
            total_value = float(total.detach().cpu())
            if total_value < best_total:
                best_total = total_value
                best_delta_raw = delta_raw.detach().clone()
        total.backward()
        torch.nn.utils.clip_grad_norm_([delta_raw], max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            candidate = reference[1:-1] + args.trust_radius_rad * torch.tanh(delta_raw)
            projected = candidate.clamp(lower, upper)
            ratio = ((projected - reference[1:-1]) / args.trust_radius_rad).clamp(-0.999, 0.999)
            delta_raw.copy_(torch.atanh(ratio))
            delta_raw.mul_(optimization_mask)
        if should_sample:
            history.append({
                'iteration': iteration,
                'total': total_value,
                **{name: float(value.detach().cpu()) for name, value in losses.items()},
            })
    cuda_end.record()
    torch.cuda.synchronize()
    optimize_ms = (time.perf_counter() - started) * 1000.0
    optimize_cuda_ms = cuda_start.elapsed_time(cuda_end)

    optimized_interior = reference[1:-1] + (
        args.trust_radius_rad * torch.tanh(best_delta_raw) * optimization_mask)
    optimized = torch.cat((reference[:1], optimized_interior, reference[-1:]), dim=0)
    optimized_tool_poses, _ = tool_tensors(planner.kinematics, optimized)
    optimized_np = optimized.detach().cpu().numpy()
    densify_started = time.perf_counter()
    dense_frames, dense_extract_end = densify(optimized_np, split_index=extract_nodes - 1)
    densify_ms = (time.perf_counter() - densify_started) * 1000.0

    robot_failures = []
    robot_validation_ms = 0.0
    payload_validation_ms = 0.0
    payload_failures = {}
    if args.skip_validation:
        payload_success = True
        payload_reason = 'skipped'
    else:
        robot_validation_started = time.perf_counter()
        for index, row in enumerate(dense_frames):
            audit = audit_state(planner.kinematics, tensor_state(row), scene)
            if dense_extract_end is not None and index <= dense_extract_end:
                audit['world_collisions'] = [
                    collision for collision in audit['world_collisions']
                    if collision['link'] not in ('left_payload', 'right_payload')
                ]
                audit['valid'] = not (
                    audit['self_collisions'] or audit['world_collisions']
                    or audit['joint_limit_violations'])
            if not audit['valid']:
                robot_failures.append({'frame': index, 'audit': audit})
        robot_validation_ms = (time.perf_counter() - robot_validation_started) * 1000.0
        payload_validation_started = time.perf_counter()
        dense_q = torch.tensor(
            dense_frames, device='cuda', dtype=torch.float32).unsqueeze(0)
        dense_state = planner.kinematics.compute_kinematics(JointState(
            position=dense_q, joint_names=planner.kinematics.joint_names))
        payload_failures = exact_payload_failures(
            dense_state, tool_to_box, exact_obstacles(summary, removed_boxes))
        payload_validation_ms = (time.perf_counter() - payload_validation_started) * 1000.0
        payload_success = not payload_failures
        if payload_failures:
            first_payload_failure = min(payload_failures)
            payload_reason = f'{payload_failures[first_payload_failure]}@{first_payload_failure}'
        else:
            payload_reason = ''
    reference_velocity = np.diff(reference_np, axis=0)
    optimized_velocity = np.diff(optimized_np, axis=0)
    reference_acceleration = np.diff(reference_velocity, axis=0)
    optimized_acceleration = np.diff(optimized_velocity, axis=0)
    home_avoidance = None
    if source.get('boundaries') and 'loaded_return_home' in source['boundaries']:
        home = np.asarray(source['frames'][source['boundaries']['loaded_return_home']])
        reference_home_distance = np.linalg.norm(reference_np[extract_nodes:] - home, axis=1)
        optimized_home_distance = np.linalg.norm(optimized_np[extract_nodes:] - home, axis=1)
        home_avoidance = {
            'reference_minimum_l2_rad': float(reference_home_distance.min()),
            'optimized_minimum_l2_rad': float(optimized_home_distance.min()),
            'reference_path_l2': float(
                np.linalg.norm(np.diff(reference_np, axis=0), axis=1).sum()),
            'optimized_path_l2': float(
                np.linalg.norm(np.diff(optimized_np, axis=0), axis=1).sum()),
        }
    output = {
        'kind': 'v3_loaded_attached_to_place_joint_optimization',
        'source': str(args.source),
        'round': source.get('round', 1),
        'boxes': boxes,
        'pair': list(pair),
        'removed_before': sorted(removed_before),
        'joint_names': planner.kinematics.joint_names,
        'nodes': args.nodes,
        'extract_nodes': extract_nodes,
        'iterations': args.iterations,
        'learning_rate': args.learning_rate,
        'trust_radius_rad': args.trust_radius_rad,
        'direct_after_extraction': args.direct_after_extraction,
        'trust_weight': args.trust_weight,
        'smooth_weight': args.smooth_weight,
        'collision_weight': args.collision_weight,
        'payload_sphere_radius': args.payload_sphere_radius,
        'optimization_ms': optimize_ms,
        'optimization_cuda_ms': optimize_cuda_ms,
        'densify_ms': densify_ms,
        'robot_validation_ms': robot_validation_ms,
        'payload_validation_ms': payload_validation_ms,
        'validation_skipped': args.skip_validation,
        'best_total': best_total,
        'extraction_nodes_frozen': True,
        'history': history,
        'reference_frames': reference_np.tolist(),
        'optimized_frames': optimized_np.tolist(),
        'dense_frames': dense_frames,
        'dense_frame_count': len(dense_frames),
        'dense_extract_end': dense_extract_end,
        'robot_collision_failure_count': len(robot_failures),
        'robot_collision_failures': robot_failures[:20],
        'payload_success': payload_success,
        'payload_reason': payload_reason,
        'payload_failure_count': len(payload_failures),
        'validated_success': (
            None if args.skip_validation else not robot_failures and payload_success),
        'pose_errors': pose_errors(optimized_tool_poses, reference_tool_poses),
        'extraction_errors': extraction_errors(
            optimized_tool_poses, reference_tool_poses, extract_nodes),
        'smoothness': {
            'reference_velocity_rms': float(np.sqrt(np.mean(reference_velocity ** 2))),
            'optimized_velocity_rms': float(np.sqrt(np.mean(optimized_velocity ** 2))),
            'reference_acceleration_rms': float(np.sqrt(np.mean(reference_acceleration ** 2))),
            'optimized_acceleration_rms': float(np.sqrt(np.mean(optimized_acceleration ** 2))),
            'maximum_reference_deviation_rad': float(np.max(np.abs(optimized_np - reference_np))),
        },
        'home_avoidance': home_avoidance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    printable = copy.deepcopy(output)
    for key in ('reference_frames', 'optimized_frames', 'dense_frames'):
        printable.pop(key)
    print(json.dumps(printable, indent=2))


if __name__ == '__main__':
    main()
