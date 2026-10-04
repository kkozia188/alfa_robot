#!/usr/bin/env python3

import argparse
import copy
import json
from pathlib import Path


def append_segment(output, segment):
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
    parser.add_argument('--optimized-return', type=Path, required=True)
    parser.add_argument('--placement-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_result.read_text())
    optimized = json.loads(args.optimized_return.read_text())
    placement_source = json.loads(args.placement_source.read_text())
    if not optimized['native_success'] or not optimized['accepted_success']:
        raise ValueError('optimized loaded return is not valid')
    placement = placement_source['first_unloading_fast_probe']['placement_reference']
    if not placement['success']:
        raise ValueError('placement reference is not valid')

    output = copy.deepcopy(source)
    output['kind'] = 'v3_round1_optimized_first_home_then_first_unloading'
    output['source_result'] = str(args.source_result)
    output['optimized_loaded_return_source'] = str(args.optimized_return)
    output['stages'] = output['stages'][:3]
    output['stages'].append({
        'name': 'loaded_extract_to_first_home',
        'success': True,
        'method': 'curobo_motion_planner_2seed_cuda_graph_steady',
        'wall_ms': optimized['plan_ms_median'],
        'cold_capture_ms': optimized['plan_ms_values'][0],
        'steady_plan_ms_values': optimized['plan_ms_values'][1:],
        'limit_margin_rad': optimized['limit_margin_rad'],
        'trajopt_seeds': optimized['trajopt_seeds'],
        'trajopt_iters': optimized['trajopt_iters'],
        'trajopt_calls': optimized['trajopt_calls'],
        'graph_calls': optimized['graph_calls'],
        'frames': optimized['frames'],
    })
    output['stages'].append({
        'name': 'loaded_first_home_to_first_unloading',
        'success': True,
        'method': 'validated_four_progress_reference',
        'reason': '',
        'frames': placement['frames'],
    })
    output['success'] = True
    output['failure_stage'] = None
    frames = []
    boundaries = {}
    for stage in output['stages']:
        append_segment(frames, stage['frames'])
        boundaries[stage['name']] = len(frames) - 1
    output['frames'] = frames
    output['stage_boundaries'] = boundaries
    timed_stages = [stage for stage in output['stages'] if stage.get('wall_ms') is not None]
    output['planning_timing'] = {
        'steady_total_ms': sum(stage['wall_ms'] for stage in timed_stages),
        'cold_total_ms': sum(
            stage.get('cold_capture_ms', stage['wall_ms']) for stage in timed_stages),
        'stages': {
            stage['name']: {
                'steady_ms': stage['wall_ms'],
                'cold_ms': stage.get('cold_capture_ms', stage['wall_ms']),
            }
            for stage in timed_stages
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({
        'success': True,
        'frames': len(frames),
        'stage_boundaries': boundaries,
        'planning_timing': output['planning_timing'],
    }, indent=2))


if __name__ == '__main__':
    main()
