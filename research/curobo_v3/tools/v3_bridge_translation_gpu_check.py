#!/usr/bin/env python3
"""Validate yaw-zero base translations with both arms parked and middle cells empty."""
import argparse, fcntl, json, math
from dataclasses import replace
from pathlib import Path
import numpy as np, torch, yaml
from curobo_core.contracts import ACTIVE_JOINTS, PlannerAssets
from curobo_core.planner import FullCyclePlanner
from curobo_core.sequential import sequential_request
from v3_batched_loaded_search import GpuValidity
from v3_plan_cycle import gpu_processes

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with open('/tmp/sevenova-curobo-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);occupied=gpu_processes();
  if occupied:raise RuntimeError('GPU occupied: '+'; '.join(occupied))
  planner=FullCyclePlanner(PlannerAssets());req=sequential_request(planner);cleared={16,17,18,21,22,23};snapshot=replace(req.snapshot,objects=tuple(x for x in req.snapshot.objects if x.object_id not in ('left_wall','right_wall') and x.object_id not in {f'wall_box_{i:02d}' for i in cleared}));planner.set_snapshot(snapshot);names=list(planner.mobile_robot.get('robot_cfg',planner.mobile_robot)['kinematics']['cspace']['joint_names']);checker=planner.collision_checker(dict(req.tasks),payload=False,mobile=True);validity=GpuValidity(checker,check_ground=True,ground_z=snapshot.policy.ground_z_m,joint_names=names,weights=[2.,2.,1.,5.]+[1.]*14);park=yaml.safe_load(planner.args.named_poses.read_text())['named_poses']['second_home'];base=np.zeros(len(names));mapping={**park,'base_x':0.,'base_y':0.,'base_yaw':0.};base[:]=[mapping.get(n,0.) for n in names];reports=[]
  for target in (-.5,.5):
   steps=math.ceil(abs(target)/.01);rows=np.repeat(base[None],steps+1,axis=0);rows[:,names.index('base_y')]=np.linspace(0.,target,steps+1);mask=validity.mask(torch.tensor(rows,device='cuda',dtype=torch.float32)).cpu().numpy();reports.append({'target_base_y_m':target,'frames':len(rows),'valid':bool(mask.all()),'first_invalid':None if mask.all() else int(np.flatnonzero(~mask)[0])})
  output={'kind':'bridge_y_translation_gpu_check','yaw_deg':0.,'cleared_boxes':sorted(cleared),'reports':reports};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2));raise SystemExit(0 if all(x['valid'] for x in reports) else 1)
if __name__=='__main__':main()
