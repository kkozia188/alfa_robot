"""Native collision/negative regression using a current diagnostic state, NOT replanning.

The supplied result must be from this model. Saved joints are regression inputs only;
this tool never passes them to a route search or counts them as a full task success.
"""
import argparse
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from curobo_core.contracts import PlanRequest, PlannerAssets
from curobo_core.planner import FullCyclePlanner, CycleBlocked
from curobo_core.scene import Pose, SceneSnapshot, SceneStore
from curobo_core.sequential import _SequentialTask
from v3_plan_cycle import gpu_processes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with open('/tmp/sevenova-curobo-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        other = gpu_processes()
        if other:
            raise SystemExit('GPU occupied: '+str(other))
        document = json.loads(args.result.read_text())
        request = PlanRequest.from_dict(document['request'])
        result = document['result'] if 'result' in document else document['results'][0]
        snapshot = SceneSnapshot.from_dict(result['final_snapshot'])
        if len(snapshot.attachments) != 2:
            raise ValueError('regression fixture requires both attached boxes')
        planner = FullCyclePlanner(PlannerAssets())
        planner.set_snapshot(request.snapshot)
        task = _SequentialTask(planner, request, lambda text: None)
        task.q = np.asarray(result['frames'][-1])
        task.store = SceneStore(snapshot)
        records = []
        started = time.perf_counter()
        original_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            for tf32 in (False, True):
                torch.backends.cuda.matmul.allow_tf32 = tf32
                for count in (1, 2, 16, 64):
                    passed = bool(task.validity().mask(task.tensor([task.q]*count)).all().item())
                    records.append({'case': 'batch_parity', 'tf32': tf32, 'points': count, 'passed': passed})
                    assert passed, records[-1]
            lower = next(item for item in snapshot.attachments if item.parent_link == 'left_tool0')
            box = task.box_pose(task.q, lower)
            for object_id in ('warehouse_top_door_leaf', 'wall_box_16'):
                obstacle = snapshot.object(object_id)
                bad = replace(obstacle, pose=Pose(tuple(box[:3, 3])))
                task.store = SceneStore(replace(snapshot, objects=tuple(
                    bad if item.object_id == object_id else item for item in snapshot.objects)))
                task.enter('left_extract')
                try:
                    task.audit([task.q], list(range(1, 8)), task.q)
                except CycleBlocked as error:
                    records.append({'case': object_id, 'passed': error.stage == 'left_extract', 'error': str(error)})
                else:
                    raise AssertionError('obstacle intersection was accepted')
            task.store = SceneStore(snapshot)
            task.enter('left_extract')
            drifted = task.q.copy()
            drifted[8] += .0001
            try:
                task.audit([drifted], list(range(1, 8)), task.q)
            except CycleBlocked as error:
                records.append({'case': 'support_drift', 'passed': error.stage == 'left_extract', 'error': str(error)})
            else:
                raise AssertionError('support drift was accepted')
            task.request = replace(request, targets=(('left_tool0', Pose((100., 0., 0.))), request.targets[1]))
            try:
                task.place('left', list(range(1, 8)))
            except CycleBlocked as error:
                records.append({'case': 'unreachable_unloading', 'passed': error.stage == 'left_place', 'error': str(error)})
            else:
                raise AssertionError('unreachable target was accepted')
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original_tf32
            torch.cuda.synchronize()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'kind': 'native_guard_regression_not_task_acceptance',
                'fixture': str(args.result), 'records': records, 'wall_ms': (time.perf_counter()-started)*1000,
                'passed': len(records) == 12 and all(record['passed'] for record in records)}, indent=2)+'\n')
        print(args.output.read_text())


if __name__ == '__main__':
    main()
