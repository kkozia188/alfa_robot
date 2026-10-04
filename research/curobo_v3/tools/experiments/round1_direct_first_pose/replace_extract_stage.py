#!/usr/bin/env python3

import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-result', type=Path, required=True)
    parser.add_argument('--constrained-extract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_result.read_text())
    constrained = json.loads(args.constrained_extract.read_text())
    if not constrained['native_success'] or not constrained['validated_success']:
        raise ValueError('constrained extract did not pass validation')
    output = copy.deepcopy(source)
    output['kind'] = 'v3_round1_official_constrained_extract'
    output['constrained_extract_source'] = str(args.constrained_extract)
    output['stages'] = output['stages'][:3]
    output['stages'][2] = {
        'name': 'cartesian_extract_35cm',
        'success': True,
        'method': 'curobo_official_constrained_trajopt',
        'wall_ms': constrained['wall_ms_median'],
        'cold_capture_ms': constrained['wall_ms_values'][0],
        'steady_plan_ms_values': constrained['wall_ms_values'][1:],
        'constraint_scale': constrained['constraint_scale'],
        'ik_seeds': constrained['ik_seeds'],
        'trajopt_seeds': constrained['trajopt_seeds'],
        'constraint_metrics': constrained['constraint_metrics'],
        'frames': constrained['frames'],
    }
    output['success'] = False
    output['failure_stage'] = 'loaded_return_not_replanned'
    output.pop('frames', None)
    output.pop('stage_boundaries', None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({
        'success': True,
        'extract_points': len(constrained['frames']),
        'extract_ms': constrained['wall_ms_median'],
        'new_extract_endpoint': constrained['frames'][-1],
    }, indent=2))


if __name__ == '__main__':
    main()
