import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ipc_left_prefix'))
from model import JOINT_NAMES,MOTOR_SIGNS,export_prefix,motor_positions,read_library,select_path,time_parameterize
from server import State


class PrefixControlTest(unittest.TestCase):
    def record(self):
        return dict(model_positions=[[0.]*7,[.01,.02,.03,0.,0.,.04,0.],[.02,.03,.04,0.,0.,.05,0.]],
                    phases=['home','approach','contact_approach'])

    def test_export_stops_before_attachment_and_forbids_following_segments(self):
        names=['base_x','base_y','base_yaw','updown']+JOINT_NAMES+[f'right_joint{i}' for i in range(1,8)]
        phases=['home','approach','contact_approach','attach','extract','transport','release','return_home']
        frames=[[0.]*18 for _ in phases]
        for index in [1,2,3]:frames[index][4]=.01*min(index,2)
        for index in range(4,8):frames[index][4]=99.
        source=dict(success=True,contact_roll_policy='corrected',joint_names=names,phases=phases,frames=frames,task=dict(left=18,right=22))
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'source.json';path.write_text(json.dumps(source));record=export_prefix(path)
        self.assertEqual(len(record['model_positions']),3)
        self.assertEqual(record['model_positions'][-1][0],.02)
        self.assertEqual(record['phases'][-1],'contact_approach')

    def test_named_mapping_and_return_reverse_only_recorded_prefix(self):
        rows=self.record()['model_positions']
        self.assertEqual(motor_positions([1,2,3,4,5,6,7]),[1,2,3,4,5,6,7])
        backward=select_path(self.record(),motor_positions(rows[-1]),'home')
        self.assertEqual(backward[-1],motor_positions(rows[0]))
        self.assertEqual(backward[1:],list(reversed([motor_positions(row) for row in rows])))
        with self.assertRaises(ValueError):select_path(self.record(),[1.]*7,'forward')
        with self.assertRaises(ValueError):select_path(self.record(),[1.]*7,'home')
        with self.assertRaises(ValueError):select_path(self.record(),[0.]*7,'extract')

    def test_timing_respects_limits_and_rest_at_vertices(self):
        points=time_parameterize(self.record()['model_positions'],speed_deg_s=5.)
        self.assertTrue(all(len(point['positions'])==7 for point in points))
        self.assertTrue(all(end['time_s']>start['time_s'] for start,end in zip(points,points[1:])))
        self.assertLessEqual(max(abs(value) for point in points for value in point['velocities']),math.radians(5.)+1e-8)
        self.assertLessEqual(max(abs(value) for point in points for value in point['accelerations']),math.radians(10.)+1e-8)
        self.assertEqual(points[-1]['velocities'],[0.]*7)
        for start,end in zip(points,points[1:]):
            jerk=max(abs(b-a)/(end['time_s']-start['time_s']) for a,b in zip(start['accelerations'],end['accelerations']))
            self.assertLessEqual(jerk,math.radians(50.)+1e-6)

    def test_invalid_or_post_contact_library_is_rejected(self):
        data=dict(joint_names=JOINT_NAMES,motor_signs=MOTOR_SIGNS,tasks=dict(sample=self.record()))
        data['tasks']['sample']['phases'][1]='extract'
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'library.json';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):read_library(path)

    def test_scope_and_feedback_gate_block_actual_publish(self):
        state=State(dict(tasks=dict(sample=self.record())),'pending')
        sent=[];state.publisher=lambda points:sent.append(points)
        with self.assertRaises(ValueError):state.send('sample','forward',5.)
        state.scope='left7_joint_replay'
        with self.assertRaises(ValueError):state.send('sample','forward',5.)
        self.assertEqual(sent,[])

    def test_ready_sends_only_seven_axis_prefix_and_locks_until_arrival(self):
        state=State(dict(tasks=dict(sample=self.record())),'left7_joint_replay')
        now=time.monotonic();state.positions=[0.]*7;state.received=now;state.stamp_age=0.
        state.samples.extend([(now-.25,[0.]*7),(now,[0.]*7)])
        state.controller=True;state.controller_received=now;state.subscribers=1
        state.enable_state='ENABLED';state.enable_received=now
        state.lease_client=object();state.console_status={'lease':'you'};state.console_received=now
        state.motor_bounds=[(-math.pi,math.pi)]*7
        sent=[];state.publisher=lambda points:sent.append(points)
        result=state.send('sample','forward',5.)
        self.assertEqual(result['joint_names'],JOINT_NAMES)
        self.assertEqual(sent[0][-1]['positions'],motor_positions(self.record()['model_positions'][-1]))
        with self.assertRaises(ValueError):state.send('sample','home',5.)
        self.assertEqual(len(sent),1)


if __name__=='__main__':
    unittest.main()
