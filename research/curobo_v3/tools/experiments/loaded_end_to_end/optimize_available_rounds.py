#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--ik-summary', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--nodes', type=int, default=192)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--trust-radius-rad', type=float, default=0.01)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--trust-weight', type=float, default=10.0)
    parser.add_argument('--smooth-weight', type=float, default=50.0)
    parser.add_argument('--collision-weight', type=float, default=100.0)
    parser.add_argument('--payload-sphere-radius', type=float, default=0.05)
    args = parser.parse_args()

    script = Path(__file__).with_name('optimize_attached_to_place.py')
    args.output_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for round_number in range(1, 12):
        source_path = args.source_root / f'round_{round_number:02d}.json'
        source = json.loads(source_path.read_text())
        if not source.get('frames') or not source.get('boundaries'):
            entries.append({
                'round': round_number,
                'status': 'missing_complete_reference',
                'source_success': bool(source.get('success')),
                'source_failure_stage': source.get('failure_stage'),
                'source': str(source_path),
            })
            continue
        output_path = args.output_root / f'round_{round_number:02d}.json'
        command = [
            sys.executable, str(script),
            '--source', str(source_path),
            '--ik-summary', str(args.ik_summary),
            '--output', str(output_path),
            '--nodes', str(args.nodes),
            '--iterations', str(args.iterations),
            '--trust-radius-rad', str(args.trust_radius_rad),
            '--learning-rate', str(args.learning_rate),
            '--trust-weight', str(args.trust_weight),
            '--smooth-weight', str(args.smooth_weight),
            '--collision-weight', str(args.collision_weight),
            '--payload-sphere-radius', str(args.payload_sphere_radius),
        ]
        print(f'round {round_number}: optimizing', flush=True)
        completed = subprocess.run(
            command, cwd=Path.cwd(), env=os.environ.copy(),
            text=True, capture_output=True)
        log_path = args.output_root / f'round_{round_number:02d}.log'
        log_path.write_text(completed.stdout + completed.stderr)
        if completed.returncode != 0:
            entries.append({
                'round': round_number,
                'status': 'optimizer_error',
                'source': str(source_path),
                'log': str(log_path),
                'returncode': completed.returncode,
            })
            continue
        result = json.loads(output_path.read_text())
        first_collision = result['history'][0]['scene_collision']
        last_collision = result['history'][-1]['scene_collision']
        entries.append({
            'round': round_number,
            'status': 'optimized',
            'source_success': bool(source.get('success')),
            'source_failure_stage': source.get('failure_stage'),
            'boxes': result['boxes'],
            'validated_success': result['validated_success'],
            'optimization_ms': result['optimization_ms'],
            'collision_cost_before': first_collision,
            'collision_cost_after': last_collision,
            'collision_cost_reduction_percent': (
                0.0 if first_collision == 0.0 else
                100.0 * (first_collision - last_collision) / first_collision),
            'reference_velocity_rms': result['smoothness']['reference_velocity_rms'],
            'optimized_velocity_rms': result['smoothness']['optimized_velocity_rms'],
            'reference_acceleration_rms': result['smoothness']['reference_acceleration_rms'],
            'optimized_acceleration_rms': result['smoothness']['optimized_acceleration_rms'],
            'output': str(output_path),
        })
        print(
            f'round {round_number}: validated={result["validated_success"]} '
            f'time={result["optimization_ms"]:.1f}ms', flush=True)
    summary = {
        'kind': 'v3_loaded_end_to_end_available_rounds',
        'nodes': args.nodes,
        'iterations': args.iterations,
        'trust_radius_rad': args.trust_radius_rad,
        'learning_rate': args.learning_rate,
        'trust_weight': args.trust_weight,
        'smooth_weight': args.smooth_weight,
        'collision_weight': args.collision_weight,
        'payload_sphere_radius': args.payload_sphere_radius,
        'entries': entries,
        'optimized_count': sum(item['status'] == 'optimized' for item in entries),
        'validated_count': sum(item.get('validated_success', False) for item in entries),
        'missing_reference_count': sum(
            item['status'] == 'missing_complete_reference' for item in entries),
    }
    summary_path = args.output_root / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
