#!/usr/bin/env python3

import argparse
import copy
import gc
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

sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_full_plan')
sys.path.insert(0, str(Path(__file__).parent))
from direct_first_pose_plan import accept_motion_result, planner_cfg  # noqa: E402
from v3_round1_full_plan import (  # noqa: E402
    ACTIVE_JOINTS,
    PAIR,
    add_payload_links,
    payload_grid,
    tensor_state,
    update_payloads,
    validate_payload_path,
)
from v3_wall_ik_benchmark import make_scene, wall_center  # noqa: E402


def append_segment(output, segment):
    if not segment:
        return
    if output:
        delta = max(abs(a - b) for a, b in zip(output[-1], segment[0]))
        if delta > 2e-5:
            raise ValueError(f'trajectory boundary mismatch: {delta}')
        output.extend(segment[1:])
    else:
        output.extend(segment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-result', type=Path, required=True)
    parser.add_argument('--placement-source', type=Path, required=True)
    parser.add_argument('--ik-summary', type=Path, required=True)
    parser.add_argument('--named-poses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_result.read_text())
    placement_source = json.loads(args.placement_source.read_text())
    summary = json.loads(args.ik_summary.read_text())
    named = yaml.safe_load(args.named_poses.read_text())['named_poses']
    robot = yaml.safe_load(Path(summary['robot_config']).read_text())
    scene = make_scene(summary['chassis_front_x_m'], summary['wall_distance_m'], set(PAIR))
    loaded_robot = copy.deepcopy(robot)
    add_payload_links(loaded_robot)

    contact = tensor_state(source['stages'][1]['frames'][-1])
    extracted = tensor_state(source['stages'][2]['frames'][-1])
    home = tensor_state([named['home'][name] for name in ACTIVE_JOINTS])
    placement = placement_source['first_unloading_fast_probe']['placement_reference']['frames']

    planner = MotionPlanner(planner_cfg(loaded_robot, scene, seeds=16))
    contact_poses = planner.compute_kinematics(contact).tool_poses.to_dict()
    payload_grids = {}
    tool_to_box = {}
    for side, box_id in zip(('left', 'right'), PAIR):
        pose = contact_poses[f'{side}_tool0']
        position = pose.position.reshape(-1, 3)[0].detach().cpu().numpy()
        quaternion = pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy()
        center = wall_center(summary['chassis_front_x_m'], summary['wall_distance_m'], box_id)
        payload_grids[side] = payload_grid(position, quaternion, center)
        tool = np.eye(4)
        tool[:3, :3] = Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()
        tool[:3, 3] = position
        box = np.eye(4)
        box[:3, 3] = center
        tool_to_box[side] = np.linalg.inv(tool) @ box
    update_payloads(planner, payload_grids)

    started = time.perf_counter()
    result = planner.plan_cspace(home, extracted, max_attempts=12)
    return_frames, method, failures = accept_motion_result(
        planner, result, extracted, home, scene, tool_to_box, summary)
    return_ms = (time.perf_counter() - started) * 1000.0
    placement_ok, placement_reason = validate_payload_path(
        planner.kinematics,
        placement,
        tool_to_box,
        summary['chassis_front_x_m'],
        summary['wall_distance_m'],
    )

    output = copy.deepcopy(source)
    output['kind'] = 'v3_round1_direct_first_home_then_first_unloading'
    output['source_result'] = str(args.source_result)
    output['stages'] = output['stages'][:3]
    output['stages'].append({
        'name': 'loaded_extract_to_first_home',
        'success': bool(return_frames),
        'method': method,
        'wall_ms': return_ms,
        'failures': failures,
        'frames': return_frames,
    })
    output['stages'].append({
        'name': 'loaded_first_home_to_first_unloading',
        'success': placement_ok,
        'method': 'validated_four_progress_reference',
        'reason': placement_reason,
        'frames': placement,
    })
    output['success'] = bool(return_frames) and placement_ok
    output['failure_stage'] = None if output['success'] else (
        'loaded_extract_to_first_home' if not return_frames
        else 'loaded_first_home_to_first_unloading')
    output['loaded_return_attempt'] = {
        'wall_ms': return_ms,
        'method': method,
        'failures': failures,
    }
    if output['success']:
        frames = []
        boundaries = {}
        for stage in output['stages']:
            append_segment(frames, stage['frames'])
            boundaries[stage['name']] = len(frames) - 1
        output['frames'] = frames
        output['stage_boundaries'] = boundaries
    else:
        output.pop('frames', None)
        output.pop('stage_boundaries', None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({
        'success': output['success'],
        'failure_stage': output['failure_stage'],
        'return_ms': return_ms,
        'return_method': method,
        'return_points': len(return_frames),
        'placement_valid': placement_ok,
        'placement_points': len(placement),
        'total_points': len(output.get('frames', [])),
        'first_failures': failures[:3],
    }, indent=2))
    planner.destroy()
    del planner
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
