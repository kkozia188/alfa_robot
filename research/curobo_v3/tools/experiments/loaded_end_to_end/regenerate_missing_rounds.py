#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_full_plan')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_direct_first_pose')
from v3_interactive_pair_core import InteractivePairPlanner  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--named-poses', type=Path, required=True)
    parser.add_argument('--placement-reference', type=Path, required=True)
    parser.add_argument('--payload-fit', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--rounds', type=int, nargs='+', required=True)
    parser.add_argument('--direct-loaded-to-placement', action='store_true')
    args = parser.parse_args()

    planner = InteractivePairPlanner(
        args.summary,
        args.named_poses,
        args.placement_reference,
        payload_fit_path=args.payload_fit,
        defer_payload_collision_until_extracted=True,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for round_number in args.rounds:
        print(f'round {round_number}: replanning', flush=True)
        result = planner.plan_round(
            round_number,
            progress=lambda stage, message: print(
                f'round {round_number} [{stage}] {message}', flush=True),
            sequence_prefix=True,
            direct_loaded_to_placement=args.direct_loaded_to_placement,
        )
        output = args.output_root / f'round_{round_number:02d}.json'
        output.write_text(json.dumps(result, indent=2) + '\n')
        entries.append({
            'round': round_number,
            'success': result['success'],
            'failure_stage': result.get('failure_stage'),
            'output': str(output),
            'timings': result.get('timings', {}),
        })
        print(
            f'round {round_number}: success={result["success"]} '
            f'failure={result.get("failure_stage")}', flush=True)
    summary = {'entries': entries}
    (args.output_root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
