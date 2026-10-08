"""Read-only, CPU independent-FK verification of a completed sequential result.

This replays all joint rows through the URDF; it does not trust reported errors,
reuse a planning cache, or claim continuous/dynamic/hardware certification.
Native collision certificates remain in the original run's stage audits.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yourdfpy

from curobo_core.adapter import pose_matrix, matrix_pose
from curobo_core.contracts import ACTIVE_JOINTS, PlanRequest, PlannerAssets
from curobo_core.scene import Pose


def pose_error(a, b):
    return float(np.linalg.norm(a[:3, 3]-b[:3, 3])), float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())


def verify(request, result, robot):
    assert result['success'], 'planning did not complete'
    assert result['request_id'] == request.identity, 'request hash differs'
    assert result['scene_id'] == request.snapshot.identity, 'scene differs'
    assert tuple(result['joint_names']) == ACTIVE_JOINTS, 'not the fixed 15-variable model'
    assert request.snapshot.state.base_pose == Pose(), 'mobile base request'
    assert result['time_parameterized'] is False
    q = np.array(result['frames'])
    assert q.ndim == 2 and q.shape[1] == 15 and np.isfinite(q).all()
    n = len(q)
    phases, payloads = result['phases'], result['attachments_by_frame']
    assert len(phases) == len(payloads) == len(result['payload']) == n
    sequence = [phase for i, phase in enumerate(phases) if i == 0 or phase != phases[i-1]]
    expected = ['initialize', 'right_precontact', 'right_contact', 'right_attach',
                'left_precontact', 'left_contact', 'left_attach', 'left_extract', 'left_place',
                'left_release', 'left_retreat', 'left_park', 'right_lower', 'right_extract',
                'right_place', 'right_release']
    assert sequence == expected, sequence
    assert np.max(np.abs(q[0]-request.snapshot.state.ordered(ACTIVE_JOINTS))) < 1e-9
    names = {joint.name: joint for joint in robot.robot.joints}
    for i, name in enumerate(ACTIVE_JOINTS):
        limit = names[name].limit
        assert q[:, i].min() >= limit.lower-1e-6 and q[:, i].max() <= limit.upper+1e-6, name
    tools = {side: np.empty((n, 4, 4)) for side in ('left', 'right')}
    for i, row in enumerate(q):
        values = dict(zip(ACTIVE_JOINTS, row))
        robot.update_cfg({name: values.get(name, 0.) for name in robot.actuated_joint_names})
        for side in tools:
            tools[side][i] = robot.get_transform(side+'_tool0', robot.base_link)
    events = result['predicted_scene_events']
    assert [e['phase'] for e in events] == ['right_attach', 'left_attach', 'left_release', 'right_release']
    upper, lower = f'wall_box_{request.upper_box:02d}', f'wall_box_{request.lower_box:02d}'
    assert [e['object_id'] for e in events] == [upper, lower, lower, upper]
    assert len({e['frame_index'] for e in events}) == 4
    by_frame = {e['frame_index']: e for e in events}
    attached, released, attachment_defs = {}, [], {}
    box_reference = {}
    max_tilt, min_bottom = 0., math.inf
    max_support_drift = max_rigid_error = 0.
    release_checks = []
    corner_signs = np.array([(x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    begin = phases.index('right_attach')
    park_end = len(phases)-1-list(reversed(phases)).index('left_park')
    support_frozen = [0]+list(range(8, 15))
    max_frozen = float(np.abs(q[begin:park_end+1, support_frozen]-q[begin, support_frozen]).max())
    if result.get("support_joint_freeze_required", True):
        assert max_frozen <= 1e-6
    # New requests constrain the held box pose, not its elbow configuration.
    # The following per-frame box FK checks remain mandatory.
    assert np.abs(q[park_end:, 1:8]-q[park_end, 1:8]).max() <= 1e-6
    for i in range(n):
        if i in by_frame:
            event = by_frame[i]
            assert phases[i] == event['phase']
            object_id = event['object_id']
            if event['phase'].endswith('_attach'):
                item = next(a for a in payloads[i] if a['object_id'] == object_id)
                attached[object_id] = item
                attachment_defs[object_id] = item
                side = item['parent_link'].removesuffix('_tool0')
                box = tools[side][i] @ pose_matrix(Pose(**item['tool_to_object']))
                source = request.snapshot.object(object_id)
                assert np.allclose(item['dimensions_m'], source.dimensions_m, atol=0., rtol=0.)
                assert pose_error(box, pose_matrix(source.pose))[0] <= 1e-6
                assert pose_error(box, pose_matrix(source.pose))[1] <= 1e-6
                box_reference[object_id] = box.copy()
            else:
                item = attached.pop(object_id)
                side = item['parent_link'].removesuffix('_tool0')
                relative = pose_matrix(Pose(**item['tool_to_object']))
                box = tools[side][i] @ relative
                recorded = pose_matrix(Pose(**event['world_pose']))
                translation, angle = pose_error(box, recorded)
                assert translation <= 1e-6 and angle <= 1e-6, 'release not actual FK'
                target = pose_matrix(dict(request.targets)[side+'_tool0']) @ relative
                translation, angle = pose_error(box, target)
                assert translation <= .001 and angle <= math.radians(.5), 'release target error'
                release_checks.append({'side': side, 'world_pose': asdict(matrix_pose(box)),
                                       'translation_error_m': translation, 'rotation_error_deg': math.degrees(angle)})
                released.append(object_id)
        assert {a['object_id'] for a in payloads[i]} == set(attached), f'attachment lifecycle at {i}'
        assert bool(result['payload'][i]) == bool(attached)
        for a in payloads[i]:
            object_id = a['object_id']
            assert a == attachment_defs[object_id], 'attachment changed within rigid segment'
            side = a['parent_link'].removesuffix('_tool0')
            relative = pose_matrix(Pose(**a['tool_to_object']))
            box = tools[side][i] @ relative
            rigid_error = np.linalg.norm(np.linalg.inv(tools[side][i]) @ box-relative)
            max_rigid_error = max(max_rigid_error, float(rigid_error))
            corners = (corner_signs*np.array(a['dimensions_m'])/2) @ box[:3, :3].T + box[:3, 3]
            bottom = float(corners[:, 2].min())
            tilt = math.degrees(math.acos(float(np.clip(box[2, 2], -1., 1.))))
            min_bottom, max_tilt = min(min_bottom, bottom), max(max_tilt, tilt)
            assert bottom >= request.snapshot.policy.ground_z_m-1e-6
            assert tilt <= request.snapshot.policy.max_box_tilt_deg+1e-6
            if object_id == upper and i <= park_end:
                translation, angle = pose_error(box, box_reference[upper])
                max_support_drift = max(max_support_drift, translation)
                assert translation <= .001 and angle <= math.radians(.5)
    assert released == [lower, upper] and not attached
    assert not result['final_snapshot']['attachments']
    assert not {upper, lower}.intersection(a['object_id'] for a in result['final_snapshot']['objects'])
    line_checks = []
    for phase, side, direction, distance in (
        ('left_extract', 'left', np.array([-1., 0., 0.]), .36),
        ('right_extract', 'right', np.array([-1., 0., 0.]), .36),
        ('right_lower', 'right', np.array([0., 0., -1.]),
         request.snapshot.object(upper).pose.position[2]-request.snapshot.object(lower).pose.position[2])):
        indices = [i for i, p in enumerate(phases) if p == phase]
        transforms = tools[side][indices]
        origin = tools[side][indices[0]-1]
        delta = transforms[:, :3, 3]-origin[:3, 3]
        along = delta @ direction
        line_error = float(np.linalg.norm(delta-along[:, None]*direction, axis=1).max())
        angle = float(Rotation.from_matrix(np.transpose(transforms[:, :3, :3], (0, 2, 1)) @ origin[:3, :3]).magnitude().max())
        assert line_error <= .001 and angle <= math.radians(.5)
        assert along.min() >= -1e-5 and np.diff(along).min() >= -1e-5 and abs(along[-1]-distance) <= .001
        check = {'phase': phase, 'points': len(indices), 'line_error_m': line_error,
                 'rotation_error_deg': math.degrees(angle), 'distance_m': float(along[-1])}
        if phase.endswith('extract'):
            object_id = lower if side == 'left' else upper
            a = attachment_defs[object_id]
            box = transforms[-1] @ pose_matrix(Pose(**a['tool_to_object']))
            cs = (corner_signs*np.array(a['dimensions_m'])/2) @ box[:3, :3].T+box[:3, 3]
            source = request.snapshot.object(object_id)
            clearance = source.pose.position[0]-source.dimensions_m[0]/2-float(cs[:, 0].max())
            assert clearance > 0.
            check['wall_clearance_m'] = clearance
        line_checks.append(check)
    return {'passed': True, 'seed': result['seed'], 'repeat': result['repeat'], 'frames_checked': n,
            'max_frozen_joint_error': max_frozen, 'support_drift_m': max_support_drift,
            'max_rigid_transform_error': max_rigid_error, 'max_box_tilt_deg': max_tilt,
            'minimum_box_bottom_m': min_bottom, 'release_checks': release_checks, 'line_checks': line_checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, default=PlannerAssets().urdf)
    parser.add_argument('--require-nine', action='store_true')
    args = parser.parse_args()
    document = json.loads(args.input.read_text())
    results = document['results'] if 'results' in document else [document['result']]
    request = PlanRequest.from_dict(document['request'])
    robot = yourdfpy.URDF.load(args.urdf, load_meshes=False, build_scene_graph=True)
    reports = []
    from dataclasses import replace
    for result in results:
        try:
            reports.append(verify(replace(request, seed=result['seed']), result, robot))
        except (AssertionError, ValueError, KeyError, StopIteration) as error:
            reports.append({'passed': False, 'seed': result.get('seed'), 'repeat': result.get('repeat'),
                            'error': str(error) or type(error).__name__})
    protocol = [(r.get('seed'), r.get('repeat')) for r in results] == [(s, n) for s in (11, 29, 41) for n in (1, 2, 3)]
    passed = all(r['passed'] for r in reports) and (protocol or not args.require_nine)
    output = {'kind': 'independent_cpu_urdf_fk_audit', 'passed': passed, 'nine_run_protocol': protocol,
              'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'urdf_sha256': hashlib.sha256(args.urdf.read_bytes()).hexdigest(), 'results': reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'passed': passed, 'runs_checked': len(reports), 'failures': [r for r in reports if not r['passed']]}))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
