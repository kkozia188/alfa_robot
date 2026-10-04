#!/usr/bin/env python3

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose

EXPERIMENT_ROOT = Path('/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_full_plan')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools')
sys.path.insert(0, str(EXPERIMENT_ROOT))
from v3_round1_full_plan import (  # noqa: E402
    ACTIVE_JOINTS, PAIR, add_payload_links, audit_state, guard_limit_roundoff,
    linear_joint_path, payload_grid, result_trajectory, state_rows, tensor_state,
    update_payloads, validate_payload_path,
)
from v3_wall_ik_benchmark import make_scene, wall_center  # noqa: E402


def pose_json(pose) -> dict:
    return {
        'position': pose.position.reshape(-1, 3)[0].detach().cpu().tolist(),
        'quaternion_wxyz': pose.quaternion.reshape(-1, 4)[0].detach().cpu().tolist(),
    }


def goal(poses: dict[str, dict]) -> GoalToolPose:
    return GoalToolPose.from_poses({
        frame: Pose(
            position=torch.tensor(data['position'], device='cuda').reshape(1, 3),
            quaternion=torch.tensor(data['quaternion_wxyz'], device='cuda').reshape(1, 4),
        ) for frame, data in poses.items()
    }, ordered_tool_frames=['left_tool0', 'right_tool0'])


def closest_successful_solution(result, current: JointState, maximum_rotary_step=None):
    if result is None or result.js_solution is None or result.success is None:
        return None, {'reason': 'ik_returned_no_result'}
    solutions = result.js_solution.position.reshape(-1, len(ACTIVE_JOINTS))
    success = result.success.reshape(-1).bool()
    if success.numel() != solutions.shape[0]:
        return None, {
            'reason': 'ik_result_shape_mismatch',
            'success_shape': list(result.success.shape),
            'solution_shape': list(result.js_solution.position.shape),
        }
    successful_indices = torch.nonzero(success, as_tuple=False).reshape(-1)
    if successful_indices.numel() == 0:
        return None, {'reason': 'ik_no_successful_candidate'}
    candidates = solutions[successful_indices]
    current_values = current.position.reshape(-1)
    delta = torch.abs(candidates - current_values)
    rotary_max = torch.max(delta[:, 1:], dim=1).values
    normalized_square = torch.sum(delta[:, 1:] ** 2, dim=1) + (delta[:, 0] / 0.10) ** 2
    score = rotary_max * 1000.0 + normalized_square
    selected_local = int(torch.argmin(score).item())
    selected = candidates[selected_local]
    selected_index = int(successful_indices[selected_local].item())
    selected_rotary_max = float(rotary_max[selected_local].item())
    diagnostics = {
        'successful_candidates': int(successful_indices.numel()),
        'selected_candidate': selected_index,
        'maximum_rotary_step_deg': math.degrees(selected_rotary_max),
        'updown_step_m': float(delta[selected_local, 0].item()),
    }
    if maximum_rotary_step is not None and selected_rotary_max > maximum_rotary_step:
        diagnostics['reason'] = 'closest_ik_branch_exceeds_continuity_limit'
        return None, diagnostics
    return selected, diagnostics


def planner_cfg(robot, scene, seeds=4, use_cuda_graph=False, trajopt_iters=None):
    trajopt_configs = ['trajopt/lbfgs_bspline_trajopt.yml']
    if trajopt_iters is not None:
        config_path = Path(
            '/mnt/mydisk/ALFA/curobo_v2_ws/src/curobo/curobo/content/configs/task/'
            'trajopt/lbfgs_bspline_trajopt.yml')
        config = yaml.safe_load(config_path.read_text())
        config['optimizer']['num_iters'] = trajopt_iters
        trajopt_configs = [config]
    return MotionPlannerCfg.create(
        robot=robot, scene_model=scene, collision_cache={'cuboid': 40},
        trajopt_optimizer_configs=trajopt_configs,
        num_ik_seeds=64, num_trajopt_seeds=seeds, self_collision_check=True,
        use_cuda_graph=use_cuda_graph, position_tolerance=.002,
        orientation_tolerance=math.radians(1.0),
    )


def audit_frames(kinematics, frames, scene, tool_to_box=None, summary=None):
    failures=[]
    for index, row in enumerate(frames):
        audit=audit_state(kinematics, tensor_state(row), scene)
        payload=(True, '') if tool_to_box is None else validate_payload_path(
            kinematics, [row], tool_to_box,
            summary['chassis_front_x_m'], summary['wall_distance_m'])
        if not audit['valid'] or not payload[0]:
            failures.append({'frame': index, 'audit': audit, 'payload_reason': payload[1]})
    return failures


def accept_motion_result(planner, result, start, goal_state, scene, tool_to_box=None, summary=None):
    trajectory=result_trajectory(result)
    if trajectory is not None:
        frames=state_rows(trajectory);failures=audit_frames(planner.kinematics, frames, scene, tool_to_box, summary)
        if not failures:return frames, 'curobo_motion_planner', []
    if result is not None and result.interpolated_trajectory is not None:
        rejected=result.get_interpolated_plan().position.reshape(-1, len(ACTIVE_JOINTS))
        lower,upper=planner.kinematics.get_joint_limits().position
        projected=torch.clamp(rejected,min=lower+1e-6,max=upper-1e-6)
        frames=projected.detach().cpu().tolist();failures=audit_frames(planner.kinematics, frames, scene, tool_to_box, summary)
        if not failures:return frames, 'curobo_bounded_projection_repair', []
    frames=linear_joint_path(start, goal_state, maximum_step=math.radians(.5));failures=audit_frames(planner.kinematics, frames, scene, tool_to_box, summary)
    if not failures:return frames, 'validated_joint_shortcut', []
    return [], '', failures[:20]


def solve_pose_line(ik, start_state, start_poses, end_poses, steps):
    frames=[start_state.position.reshape(-1).detach().cpu().tolist()];current=start_state;selection=[]
    for index in range(1, steps+1):
        fraction=index/steps;poses={}
        for frame in ('left_tool0','right_tool0'):
            begin=np.asarray(start_poses[frame]['position']);end=np.asarray(end_poses[frame]['position'])
            poses[frame]={'position':((1-fraction)*begin+fraction*end).tolist(),
                          'quaternion_wxyz':end_poses[frame]['quaternion_wxyz']}
        result=ik.solve_pose(goal(poses), current_state=current, return_seeds=64)
        solution,diagnostics=closest_successful_solution(
            result,current,maximum_rotary_step=math.radians(15.0))
        diagnostics['step']=index;selection.append(diagnostics)
        if solution is None:return frames, f'ik_failed@{index}/{steps}:{diagnostics["reason"]}',selection
        current=JointState.from_position(solution.reshape(1,-1),joint_names=ACTIVE_JOINTS)
        frames.append(solution.detach().cpu().tolist())
    return frames, '',selection


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ik-summary',type=Path,required=True);parser.add_argument('--named-poses',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    summary=json.loads(args.ik_summary.read_text());robot=yaml.safe_load(Path(summary['robot_config']).read_text());named=yaml.safe_load(args.named_poses.read_text())['named_poses'];scene=make_scene(summary['chassis_front_x_m'],summary['wall_distance_m'],set(PAIR))
    report={'success':False,'kind':'v3_round1_direct_first_pose_plan','joint_names':ACTIVE_JOINTS,'pair':list(PAIR),'wall_distance_m':summary['wall_distance_m'],'chassis_front_x_m':summary['chassis_front_x_m'],'robot_config':summary['robot_config'],'stages':[]}
    started_total=time.perf_counter();home=tensor_state([named['home'][name] for name in ACTIVE_JOINTS]);unloading=tensor_state([named['unloading'][name] for name in ACTIVE_JOINTS]);contact=tensor_state(summary['rounds'][0]['collision_free']['joints'])
    planner=MotionPlanner(planner_cfg(robot,scene));planner.ik_solver.config.use_lm_seed=False;home,_=guard_limit_roundoff(home,planner.kinematics);unloading,_=guard_limit_roundoff(unloading,planner.kinematics);contact,_=guard_limit_roundoff(contact,planner.kinematics)
    contact_fk=planner.compute_kinematics(contact).tool_poses.to_dict();contact_poses={frame:pose_json(contact_fk[frame]) for frame in ('left_tool0','right_tool0')};pregrasp_poses=copy.deepcopy(contact_poses)
    for frame in pregrasp_poses:
        rotation=Rotation.from_quat(np.roll(pregrasp_poses[frame]['quaternion_wxyz'],-1)).as_matrix();pregrasp_poses[frame]['position']=(np.asarray(pregrasp_poses[frame]['position'])-rotation[:,2]*.05).tolist()
    started=time.perf_counter();retreat_frames,reason,pregrasp_selection=solve_pose_line(
        planner.ik_solver,contact,contact_poses,pregrasp_poses,5)
    if reason:
        report['failure_stage']='pregrasp_reverse_ik';report['pregrasp_ik']={'selection':pregrasp_selection,'reason':reason};report['total_ms']=(time.perf_counter()-started_total)*1000;args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return
    pregrasp=tensor_state(retreat_frames[-1]);pregrasp,_=guard_limit_roundoff(pregrasp,planner.kinematics)
    approach_frames=list(reversed(retreat_frames))
    approach_failures=audit_frames(planner.kinematics,approach_frames,scene)
    report['pregrasp_ik']={'wall_ms':(time.perf_counter()-started)*1000,'joints':pregrasp.position.reshape(-1).detach().cpu().tolist(),'pose_target':pregrasp_poses,'selection':pregrasp_selection,'construction':'reverse_from_verified_contact'}
    started=time.perf_counter();motion=planner.plan_cspace(pregrasp,home,max_attempts=5);frames,method,failures=accept_motion_result(planner,motion,home,pregrasp,scene)
    report['stages'].append({'name':'home_to_pregrasp','success':bool(frames),'method':method,'wall_ms':(time.perf_counter()-started)*1000,'failures':failures,'frames':frames})
    if not frames:report['failure_stage']='home_to_pregrasp';report['total_ms']=(time.perf_counter()-started_total)*1000;args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'failure_stage':report['failure_stage'],'failures':failures},indent=2));return
    report['stages'].append({'name':'cartesian_approach_5cm','success':not approach_failures,'method':'reverse_curobo_collision_ik_1cm_nearest_branch','wall_ms':report['pregrasp_ik']['wall_ms'],'reason':'','selection':list(reversed(pregrasp_selection)),'failures':approach_failures[:20],'frames':approach_frames})
    if approach_failures:report['failure_stage']='cartesian_approach_5cm';report['total_ms']=(time.perf_counter()-started_total)*1000;args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'failure_stage':report['failure_stage'],'failures':approach_failures[:3]},indent=2));return
    contact=tensor_state(approach_frames[-1]);contact_fk=planner.compute_kinematics(contact).tool_poses.to_dict();payload_grids={};tool_to_box={}
    for side,box_id in zip(('left','right'),PAIR):
        pose=contact_fk[f'{side}_tool0'];position=pose.position.reshape(-1,3)[0].detach().cpu().numpy();quaternion=pose.quaternion.reshape(-1,4)[0].detach().cpu().numpy();center=wall_center(summary['chassis_front_x_m'],summary['wall_distance_m'],box_id);payload_grids[side]=payload_grid(position,quaternion,center);rotation=Rotation.from_quat(np.roll(quaternion,-1)).as_matrix();tool=np.eye(4);tool[:3,:3]=rotation;tool[:3,3]=position;box=np.eye(4);box[:3,3]=center;tool_to_box[side]=np.linalg.inv(tool)@box
    planner.destroy();del planner;gc.collect();torch.cuda.empty_cache();loaded_robot=copy.deepcopy(robot);add_payload_links(loaded_robot);loaded_ik=InverseKinematics(InverseKinematicsCfg.create(robot=loaded_robot,scene_model=scene,collision_cache={'cuboid':40},num_seeds=64,self_collision_check=True,use_cuda_graph=False,position_tolerance=.002,orientation_tolerance=math.radians(1),override_iters_for_multi_link_ik=300));loaded_ik.config.use_lm_seed=False;update_payloads(loaded_ik,payload_grids)
    contact_actual={frame:pose_json(contact_fk[frame]) for frame in ('left_tool0','right_tool0')};extract_targets=copy.deepcopy(contact_actual)
    for frame in extract_targets:extract_targets[frame]['position'][0]-=.35
    started=time.perf_counter();extract_frames,reason,extract_selection=solve_pose_line(loaded_ik,contact,contact_actual,extract_targets,35);extract_audit=audit_frames(loaded_ik.kinematics,extract_frames,scene,tool_to_box,summary)
    report['stages'].append({'name':'cartesian_extract_35cm','success':not reason and not extract_audit,'method':'curobo_loaded_collision_ik_1cm_nearest_branch','wall_ms':(time.perf_counter()-started)*1000,'reason':reason,'selection':extract_selection,'failures':extract_audit[:20],'frames':extract_frames})
    if reason or extract_audit:report['failure_stage']='cartesian_extract_35cm';report['total_ms']=(time.perf_counter()-started_total)*1000;args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'failure_stage':report['failure_stage'],'reason':reason,'failures':extract_audit[:3]},indent=2));return
    extracted=tensor_state(extract_frames[-1]);loaded_ik.destroy();del loaded_ik;gc.collect();torch.cuda.empty_cache();loaded=MotionPlanner(planner_cfg(loaded_robot,scene,seeds=8));update_payloads(loaded,payload_grids);started=time.perf_counter();placement=loaded.plan_cspace(unloading,extracted,max_attempts=8);place_frames,method,failures=accept_motion_result(loaded,placement,extracted,unloading,scene,tool_to_box,summary)
    report['stages'].append({'name':'loaded_to_first_unloading','success':bool(place_frames),'method':method,'wall_ms':(time.perf_counter()-started)*1000,'failures':failures,'frames':place_frames});report['success']=bool(place_frames);report['failure_stage']=None if place_frames else 'loaded_to_first_unloading';report['total_ms']=(time.perf_counter()-started_total)*1000
    if report['success']:
        combined=[]
        for stage in report['stages']:
            combined.extend(stage['frames'][1:] if combined else stage['frames'])
        report['frames']=combined;report['stage_boundaries']={};offset=0
        for stage in report['stages']:offset+=len(stage['frames'])-(1 if offset else 0);report['stage_boundaries'][stage['name']]=offset-1
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'success':report['success'],'failure_stage':report['failure_stage'],'total_ms':report['total_ms'],'stages':[{k:stage.get(k) for k in ('name','success','method','wall_ms','reason')}|{'points':len(stage['frames'])} for stage in report['stages']]},indent=2))


if __name__=='__main__':main()
