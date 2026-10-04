import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from v3_round1_full_plan import (
    ACTIVE_JOINTS, PAIR, add_payload_links, audit_state, linear_joint_path,
    payload_grid, tensor_state, update_payloads, validate_payload_path,
)
from v3_wall_ik_benchmark import make_scene, wall_center

parser=argparse.ArgumentParser()
parser.add_argument('--result',type=Path,required=True)
parser.add_argument('--ik-summary',type=Path,required=True)
parser.add_argument('--named-poses',type=Path,required=True)
args=parser.parse_args()
report=json.loads(args.result.read_text()); summary=json.loads(args.ik_summary.read_text())
named=yaml.safe_load(args.named_poses.read_text())['named_poses']
robot=yaml.safe_load(Path(summary['robot_config']).read_text()); loaded=copy.deepcopy(robot); add_payload_links(loaded)
scene=make_scene(summary['chassis_front_x_m'],summary['wall_distance_m'],set(PAIR))
config=InverseKinematicsCfg.create(robot=loaded,scene_model=scene,collision_cache={'cuboid':40},num_seeds=4,self_collision_check=True,use_cuda_graph=False)
ik=InverseKinematics(config)
contact_state=tensor_state(summary['rounds'][0]['collision_free']['joints'])
poses=ik.compute_kinematics(contact_state).tool_poses.to_dict(); grids={}; tool_to_box={}
for side,box_id in zip(('left','right'),PAIR):
 pose=poses[f'{side}_tool0']; position=pose.position.reshape(-1,3)[0].detach().cpu().numpy(); quaternion=pose.quaternion.reshape(-1,4)[0].detach().cpu().numpy()
 center=wall_center(summary['chassis_front_x_m'],summary['wall_distance_m'],box_id); grids[side]=payload_grid(position,quaternion,center)
 rotation=Rotation.from_quat(np.roll(quaternion,-1)).as_matrix(); tool=np.eye(4);tool[:3,:3]=rotation;tool[:3,3]=position;box=np.eye(4);box[:3,3]=center;tool_to_box[side]=np.linalg.inv(tool)@box
update_payloads(ik,grids)
extracted=tensor_state(report['stages'][1]['frames'][-1]); home=tensor_state([named['home'][n] for n in ACTIVE_JOINTS]); unloading=tensor_state([named['unloading'][n] for n in ACTIVE_JOINTS])
output={}
for name,start,goal in (('extracted_to_home',extracted,home),('home_to_unloading',home,unloading)):
 frames=linear_joint_path(start,goal,maximum_step=np.deg2rad(0.5)); audits=[audit_state(ik.kinematics,tensor_state(row),scene) for row in frames]
 exact=[validate_payload_path(ik.kinematics,[row],tool_to_box,summary['chassis_front_x_m'],summary['wall_distance_m']) for row in frames]
 failed=[i for i,(audit,payload) in enumerate(zip(audits,exact)) if not audit['valid'] or not payload[0]]
 output[name]={'success':not failed,'frame_count':len(frames),'failed_frame_indices':failed,'frames':frames,
  'first_failure':None if not failed else {'frame':failed[0],'audit':audits[failed[0]],'payload_reason':exact[failed[0]][1]}}
report['staged_shortcut_probe']=output; args.result.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:{x:v[x] for x in ('success','frame_count','failed_frame_indices','first_failure')} for k,v in output.items()},indent=2))
