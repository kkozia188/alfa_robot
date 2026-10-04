import argparse
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from v3_round1_full_plan import (
    ACTIVE_JOINTS, PAIR, add_payload_links, audit_state, payload_grid,
    tensor_state, update_payloads, validate_payload_path,
)
from v3_wall_ik_benchmark import make_scene, wall_center


parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--result',type=Path,required=True)
parser.add_argument('--ik-summary',type=Path,required=True)
args=parser.parse_args()
cache=json.loads(subprocess.check_output([
    'git','-C',str(args.repo),'show',
    'd949b69:ros2_ws/src/alfa_robot_moveit_config/config/v3_stage_wall_cache.json']))
entry=cache['entries'][0]
summary=json.loads(args.ik_summary.read_text())
scene=make_scene(summary['chassis_front_x_m'],summary['wall_distance_m'],set(PAIR))
robot=yaml.safe_load(Path(summary['robot_config']).read_text())
unloaded_cfg=InverseKinematicsCfg.create(robot=robot,scene_model=scene,collision_cache={'cuboid':40},num_seeds=4,self_collision_check=True,use_cuda_graph=False)
unloaded=InverseKinematics(unloaded_cfg)
loaded_robot=copy.deepcopy(robot);add_payload_links(loaded_robot)
loaded_cfg=InverseKinematicsCfg.create(robot=loaded_robot,scene_model=scene,collision_cache={'cuboid':40},num_seeds=4,self_collision_check=True,use_cuda_graph=False)
loaded=InverseKinematics(loaded_cfg)
cache_names=cache['joint_names']
def convert(frame):
 values=dict(zip(cache_names,frame['joints'])); return [values[name] for name in ACTIVE_JOINTS]
pregrasp=[convert(x) for x in entry['pregrasp']]
approach=[convert(x) for x in entry['approach']]
place=[convert(x) for x in entry['place']]
contact=tensor_state(place[0]); poses=loaded.compute_kinematics(contact).tool_poses.to_dict();grids={};tool_to_box={}
for side,box_id in zip(('left','right'),PAIR):
 pose=poses[f'{side}_tool0'];position=pose.position.reshape(-1,3)[0].detach().cpu().numpy();quaternion=pose.quaternion.reshape(-1,4)[0].detach().cpu().numpy();center=wall_center(summary['chassis_front_x_m'],summary['wall_distance_m'],box_id)
 grids[side]=payload_grid(position,quaternion,center);rotation=Rotation.from_quat(np.roll(quaternion,-1)).as_matrix();tool=np.eye(4);tool[:3,:3]=rotation;tool[:3,3]=position;box=np.eye(4);box[:3,3]=center;tool_to_box[side]=np.linalg.inv(tool)@box
update_payloads(loaded,grids)
output={'source':'d949b69:v3_stage_wall_cache.json entry 1','cache_entry':{k:entry[k] for k in ('round','left','right','initial_pose','unloading_pose','selected_candidate')},'segments':{}}
for name,solver,frames,payload in (
 ('pregrasp',unloaded,pregrasp,False),('approach',unloaded,approach,False),('place',loaded,place,True)):
 failed=[];first=None
 for index,row in enumerate(frames):
  audit=audit_state(solver.kinematics,tensor_state(row),scene);exact=(True,'') if not payload else validate_payload_path(solver.kinematics,[row],tool_to_box,summary['chassis_front_x_m'],summary['wall_distance_m'])
  if not audit['valid'] or not exact[0]:
   failed.append(index)
   if first is None:first={'frame':index,'audit':audit,'payload_reason':exact[1]}
 output['segments'][name]={'success':not failed,'frame_count':len(frames),'failed_frame_indices':failed,'first_failure':first,'frames':frames}
 print(name,'success',not failed,'frames',len(frames),'failed',len(failed),flush=True)
result=json.loads(args.result.read_text());result['historical_cache_seed_audit']=output;args.result.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:{x:v[x] for x in ('success','frame_count','first_failure')} for k,v in output['segments'].items()},indent=2))
