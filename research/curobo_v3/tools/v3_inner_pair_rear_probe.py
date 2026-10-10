#!/usr/bin/env python3
"""Full dual-arm inner-column pair placed at the vehicle rear before release."""
import argparse, fcntl, json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch, yaml
from curobo_core.contracts import ACTIVE_JOINTS, PlannerAssets
from curobo_core.planner import FullCyclePlanner
from curobo_core.scene import Pose, SceneStore
from curobo_core.sequential import _SequentialTask, completion, sequential_request
from v3_batched_loaded_search import densify
from v3_plan_cycle import gpu_processes

def local_snapshot(planner,column):
 base=sequential_request(planner);base_y=(column-2)*.4;cleared=({17,22} if column!=2 else set());objects=[]
 for item in base.snapshot.objects:
  if item.object_id in ('left_wall','right_wall') or item.object_id in {f'wall_box_{i:02d}' for i in cleared}:continue
  if item.object_id=='ground':objects.append(item);continue
  pos=list(item.pose.position);pos[1]-=base_y;objects.append(replace(item,pose=Pose(tuple(pos),item.pose.quaternion_wxyz)))
 return replace(base.snapshot,objects=tuple(objects))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--column',type=int,choices=(1,2,3),required=True);p.add_argument('--seed',type=int,default=11);p.add_argument('--start-pose',choices=('home','second_home','support_home'),default='home');p.add_argument('--output',type=Path,required=True);a=p.parse_args();upper=20+a.column;lower=15+a.column
 with open('/tmp/sevenova-curobo-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);occupied=gpu_processes();
  if occupied:raise RuntimeError('GPU occupied: '+'; '.join(occupied))
  planner=FullCyclePlanner(PlannerAssets());snapshot=local_snapshot(planner,a.column)
  if a.start_pose!='home':
   poses=yaml.safe_load(planner.args.named_poses.read_text())['named_poses']
   pose=dict(poses['second_home'])
   if a.start_pose=='support_home':
    pose['updown']=planner.home['updown']
    pose.update({f'right_joint{i}':planner.home[f'right_joint{i}'] for i in range(1,8)})
   snapshot=replace(snapshot,state=snapshot.state.with_positions(pose))
  planner.set_snapshot(snapshot);request=SimpleNamespace(mode='sequential_unload',tasks=(('left',lower),('right',upper)),targets=sequential_request(planner).targets,upper_box=upper,lower_box=lower,extraction_m=.36,seed=a.seed,support_elbow_rise_m=.18,cartesian_geometry_tolerance_m=.0001,snapshot=snapshot);task=_SequentialTask(planner,request,print);left=list(range(1,8));right=[0]+list(range(8,15));upper_id=f'wall_box_{upper:02d}';lower_id=f'wall_box_{lower:02d}'
  try:
   task.enter('initialize');task.audit([task.q],[],task.q);task.append([task.q]);task.contact('right',upper_id,right);held=next(x for x in task.store.snapshot().attachments if x.object_id==upper_id);task.support_pose=task.box_pose(task.q,held);task.contact('left',lower_id,left);task.extract('left',left);task.place('left',left);task.enter('left_retreat');direction=-(task.tool(task.q,'left')[:3,:3]@task.released_offsets['left'][:3,3]);direction/=np.linalg.norm(direction);rows=task.cartesian('left',tuple(direction),.05,left)
   if rows is None:task.fail('left empty retreat failed')
   task.append(rows)
   poses=yaml.safe_load(planner.args.named_poses.read_text())['named_poses'];park=task.q.copy();park[left]=[poses['second_home'][ACTIVE_JOINTS[i]] for i in left];distance=.4;lowered=park.copy();lowered[0]-=distance;support=task.support_pose;task.support_pose=None;future=task.validity();task.support_pose=support;sweep=np.asarray(densify(np.array([park,lowered])))
   if not bool(future.mask(task.tensor(sweep)).all()):task.fail('second_home future lowering sweep failed')
   task.enter('left_park');path=task.search([park],left)
   if path is None:task.fail('left park search failed')
   task.audit(path,left,task.q);task.append(path);task.support_pose=None;task.enter('right_lower');rows=task.cartesian('right',(0.,0.,-1.),distance,right,lift_end=float(task.q[0]-distance))
   if rows is None:task.fail('right lower failed')
   task.append(rows);task.extract('right',right);task.place('right',right)
   if not completion(task.report['predicted_scene_events'],task.store.snapshot(),upper_id,lower_id):task.fail('incomplete lifecycle')
   task.report['success']=True
  except Exception as error:
   task.report.setdefault('error',{'code':'PROBE_FAILED','stage':task.phase,'message':str(error)});task.report['success']=False
  finally:
   task.close_stage_timing();task.report.update(seed=a.seed,column=a.column,start_pose=a.start_pose,final_snapshot=task.store.snapshot().to_dict());a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'result':task.report},indent=2)+'\n');print(json.dumps({'column':a.column,'success':task.report['success'],'stage':task.phase,'error':task.report.get('error'),'events':[(x['phase'],x['object_id']) for x in task.report['predicted_scene_events']]},indent=2));raise SystemExit(0 if task.report['success'] else 1)
if __name__=='__main__':main()
