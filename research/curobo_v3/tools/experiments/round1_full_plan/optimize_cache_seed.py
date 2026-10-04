import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from v3_round1_full_plan import (
    ACTIVE_JOINTS, PAIR, add_payload_links, audit_state, payload_grid,
    result_trajectory, state_rows, tensor_state, update_payloads, validate_payload_path,
)
from v3_wall_ik_benchmark import make_scene, wall_center


def seed_tensor(frames, horizon):
    source=np.asarray(frames,dtype=np.float32)
    source_t=np.linspace(0.0,1.0,len(source));target_t=np.linspace(0.0,1.0,horizon)
    sampled=np.column_stack([np.interp(target_t,source_t,source[:,joint]) for joint in range(source.shape[1])])
    return torch.tensor(sampled.astype(np.float32),device='cuda').view(1,1,horizon,source.shape[1])


def planner_config(robot,scene):
    return MotionPlannerCfg.create(robot=robot,scene_model=scene,collision_cache={'cuboid':40},
        num_ik_seeds=16,num_trajopt_seeds=1,self_collision_check=True,use_cuda_graph=False,
        position_tolerance=0.002,orientation_tolerance=np.deg2rad(1.0))


parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,required=True);parser.add_argument('--ik-summary',type=Path,required=True);args=parser.parse_args()
report=json.loads(args.result.read_text());summary=json.loads(args.ik_summary.read_text());seed=report['historical_cache_seed_audit']['segments']
scene=make_scene(summary['chassis_front_x_m'],summary['wall_distance_m'],set(PAIR));robot=yaml.safe_load(Path(summary['robot_config']).read_text())
free_frames=seed['pregrasp']['frames']+seed['approach']['frames'][1:];loaded_frames=seed['place']['frames']
free=MotionPlanner(planner_config(robot,scene));free_horizon=free.trajopt_solver.action_horizon
started=time.perf_counter();free_result=free.trajopt_solver.solve_cspace(tensor_state(free_frames[-1]),tensor_state(free_frames[0]),seed_traj=seed_tensor(free_frames,free_horizon),num_seeds=1,return_seeds=1,finetune_attempts=3);free_wall=(time.perf_counter()-started)*1000
free_plan=result_trajectory(free_result);free_output=[] if free_plan is None else state_rows(free_plan);free_bad=[] if free_plan is None else [i for i,row in enumerate(free_output) if not audit_state(free.kinematics,tensor_state(row),scene)['valid']]
loaded_robot=copy.deepcopy(robot);add_payload_links(loaded_robot);loaded=MotionPlanner(planner_config(loaded_robot,scene))
contact=tensor_state(loaded_frames[0]);poses=loaded.compute_kinematics(contact).tool_poses.to_dict();grids={};tool_to_box={}
for side,box_id in zip(('left','right'),PAIR):
 pose=poses[f'{side}_tool0'];position=pose.position.reshape(-1,3)[0].detach().cpu().numpy();quaternion=pose.quaternion.reshape(-1,4)[0].detach().cpu().numpy();center=wall_center(summary['chassis_front_x_m'],summary['wall_distance_m'],box_id);grids[side]=payload_grid(position,quaternion,center);rotation=Rotation.from_quat(np.roll(quaternion,-1)).as_matrix();tool=np.eye(4);tool[:3,:3]=rotation;tool[:3,3]=position;box=np.eye(4);box[:3,3]=center;tool_to_box[side]=np.linalg.inv(tool)@box
update_payloads(loaded,grids);loaded_horizon=loaded.trajopt_solver.action_horizon
started=time.perf_counter();loaded_result=loaded.trajopt_solver.solve_cspace(tensor_state(loaded_frames[-1]),tensor_state(loaded_frames[0]),seed_traj=seed_tensor(loaded_frames,loaded_horizon),num_seeds=1,return_seeds=1,finetune_attempts=3);loaded_wall=(time.perf_counter()-started)*1000
loaded_plan=result_trajectory(loaded_result);loaded_output=[] if loaded_plan is None else state_rows(loaded_plan);loaded_bad=[]
if loaded_plan is not None:
 for index,row in enumerate(loaded_output):
  audit=audit_state(loaded.kinematics,tensor_state(row),scene);exact=validate_payload_path(loaded.kinematics,[row],tool_to_box,summary['chassis_front_x_m'],summary['wall_distance_m'])
  if not audit['valid'] or not exact[0]: loaded_bad.append({'frame':index,'audit':audit,'payload_reason':exact[1]})
output={
 'free':{'success':free_plan is not None and not free_bad,'planner_success':free_plan is not None,'wall_ms':free_wall,'points':len(free_output),'failed_frames':free_bad,'frames':free_output},
 'loaded':{'success':loaded_plan is not None and not loaded_bad,'planner_success':loaded_plan is not None,'wall_ms':loaded_wall,'points':len(loaded_output),'failed_frames':loaded_bad[:20],'frames':loaded_output},
 'seed_horizons':{'free':free_horizon,'loaded':loaded_horizon},
}
report['curobo_seed_optimization']=output;args.result.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:{x:v[x] for x in ('success','planner_success','wall_ms','points','failed_frames')} for k,v in output.items() if isinstance(v,dict) and 'success' in v},indent=2))
