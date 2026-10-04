import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from v3_round1_full_plan import (
    PAIR, add_payload_links, audit_state, payload_grid, result_trajectory,
    state_rows, tensor_state, update_payloads, validate_payload_path,
)
from v3_wall_ik_benchmark import make_scene, wall_center


def seed_tensor(frames,horizon):
    source=np.asarray(frames,dtype=np.float32);source_t=np.linspace(0,1,len(source));target_t=np.linspace(0,1,horizon)
    sampled=np.column_stack([np.interp(target_t,source_t,source[:,joint]) for joint in range(source.shape[1])]).astype(np.float32)
    return torch.tensor(sampled,device='cuda').view(1,1,horizon,source.shape[1])


def cfg(robot,scene):
    return MotionPlannerCfg.create(robot=robot,scene_model=scene,collision_cache={'cuboid':40},num_ik_seeds=16,num_trajopt_seeds=1,self_collision_check=True,use_cuda_graph=False,position_tolerance=.002,orientation_tolerance=math.radians(1))


def split_frames(frames,max_source_points):
    boundaries=list(range(0,len(frames)-1,max_source_points-1))
    if boundaries[-1] != len(frames)-1: boundaries.append(len(frames)-1)
    return [frames[boundaries[i]:boundaries[i+1]+1] for i in range(len(boundaries)-1)]


parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,required=True);parser.add_argument('--ik-summary',type=Path,required=True);parser.add_argument('--chunk-points',type=int,default=90);args=parser.parse_args()
report=json.loads(args.result.read_text());summary=json.loads(args.ik_summary.read_text());safe=report['historical_cache_seed_audit']['segments'];scene=make_scene(summary['chassis_front_x_m'],summary['wall_distance_m'],set(PAIR));robot=yaml.safe_load(Path(summary['robot_config']).read_text())
free_frames=safe['pregrasp']['frames']+safe['approach']['frames'][1:];loaded_frames=safe['place']['frames']
free=MotionPlanner(cfg(robot,scene));loaded_robot=copy.deepcopy(robot);add_payload_links(loaded_robot);loaded=MotionPlanner(cfg(loaded_robot,scene))
contact=tensor_state(loaded_frames[0]);poses=loaded.compute_kinematics(contact).tool_poses.to_dict();grids={};tool_to_box={}
for side,box_id in zip(('left','right'),PAIR):
 pose=poses[f'{side}_tool0'];position=pose.position.reshape(-1,3)[0].detach().cpu().numpy();quaternion=pose.quaternion.reshape(-1,4)[0].detach().cpu().numpy();center=wall_center(summary['chassis_front_x_m'],summary['wall_distance_m'],box_id);grids[side]=payload_grid(position,quaternion,center);rotation=Rotation.from_quat(np.roll(quaternion,-1)).as_matrix();tool=np.eye(4);tool[:3,:3]=rotation;tool[:3,3]=position;box=np.eye(4);box[:3,3]=center;tool_to_box[side]=np.linalg.inv(tool)@box
update_payloads(loaded,grids)

def validate(mode,planner,frames):
    failures=[]
    for index,row in enumerate(frames):
        audit=audit_state(planner.kinematics,tensor_state(row),scene)
        payload=(True,'') if mode=='free' else validate_payload_path(planner.kinematics,[row],tool_to_box,summary['chassis_front_x_m'],summary['wall_distance_m'])
        if not audit['valid'] or not payload[0]:failures.append({'frame':index,'audit':audit,'payload_reason':payload[1]})
    return failures

output={'segments':{},'success':True};combined=[]
for mode,planner,source in (('free',free,free_frames),('loaded',loaded,loaded_frames)):
    accepted=[];records=[]
    for index,chunk in enumerate(split_frames(source,args.chunk_points)):
        started=time.perf_counter();result=planner.trajopt_solver.solve_cspace(tensor_state(chunk[-1]),tensor_state(chunk[0]),seed_traj=seed_tensor(chunk,planner.trajopt_solver.action_horizon),num_seeds=1,return_seeds=1,finetune_attempts=2);elapsed=(time.perf_counter()-started)*1000
        trajectory=result_trajectory(result);optimized=[] if trajectory is None else state_rows(trajectory);failures=[] if trajectory is None else validate(mode,planner,optimized)
        use_optimized=trajectory is not None and not failures
        chosen=optimized if use_optimized else chunk
        fallback_failures=[] if use_optimized else validate(mode,planner,chosen)
        if fallback_failures: output['success']=False
        if accepted: accepted.extend(chosen[1:])
        else: accepted.extend(chosen)
        records.append({'chunk':index+1,'source_points':len(chunk),'planner_success':trajectory is not None,'optimized_points':len(optimized),'accepted_method':'curobo_trajopt' if use_optimized else 'validated_seed_fallback','optimization_wall_ms':elapsed,'optimized_failures':failures[:5],'fallback_failures':fallback_failures[:5]})
        print(mode,index+1,records[-1]['accepted_method'],'source',len(chunk),'optimized',len(optimized),flush=True)
    output['segments'][mode]={'success':not any(x['fallback_failures'] for x in records),'records':records,'frames':accepted,'source_points':len(source),'accepted_points':len(accepted),'optimized_chunks':sum(x['accepted_method']=='curobo_trajopt' for x in records)}
    if combined: combined.extend(accepted[1:])
    else: combined.extend(accepted)
output['frames']=combined;output['total_frames']=len(combined);output['stage_boundaries']={'free_end':len(output['segments']['free']['frames'])-1,'loaded_begin':len(output['segments']['free']['frames'])-1,'loaded_end':len(combined)-1}
report['chunked_curobo_optimization']=output;args.result.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:{x:v[x] for x in ('success','source_points','accepted_points','optimized_chunks')} for k,v in output['segments'].items()}|{'full_success':output['success'],'total_frames':output['total_frames']},indent=2))
