#!/usr/bin/env python3
"""Independent structural/bridge audit of a composed five-column result."""
import argparse, json, math
from pathlib import Path
import numpy as np, yaml, yourdfpy
from scipy.spatial.transform import Rotation
from curobo_core.adapter import pose_matrix
from curobo_core.scene import Pose
from curobo_core.contracts import ACTIVE_JOINTS, PlannerAssets
from v3_five_column_translation_probe import chassis_hull

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.input.read_text());errors=[];rows=np.asarray(d['frames'],dtype=float);names=d['joint_names'];idx={n:i for i,n in enumerate(names)}
 if not d.get('success'):errors.append('result not successful')
 if rows.ndim!=2 or rows.shape[1]!=18 or not np.isfinite(rows).all():errors.append('invalid frame matrix')
 for name in ('base_x','base_y','base_yaw',*ACTIVE_JOINTS):
  if name not in idx:errors.append('missing '+name)
 if np.max(np.abs(rows[:,idx['base_x']]))>1e-9:errors.append('base_x moved')
 if np.max(np.abs(rows[:,idx['base_yaw']]))>1e-9:errors.append('base_yaw moved')
 if rows[:,idx['base_y']].min() < -.500001 or rows[:,idx['base_y']].max() > .500001:errors.append('base_y bound')
 assets=PlannerAssets();home=yaml.safe_load(assets.named_poses.read_text())['named_poses']['home'];hull=chassis_hull(assets,home);minimum=rows[:,idx['base_y']].min()+hull[:,1].min();maximum=rows[:,idx['base_y']].max()+hull[:,1].max()
 if minimum < -1.050001 or maximum > 1.050001:errors.append('chassis projection left bridge')
 robot=yourdfpy.URDF.load(assets.urdf,load_meshes=False,build_scene_graph=True);joints={j.name:j for j in robot.robot.joints}
 for name in ACTIVE_JOINTS:
  values=rows[:,idx[name]];limit=joints[name].limit
  if values.min()<limit.lower-1e-6 or values.max()>limit.upper+1e-6:errors.append('joint bound '+name)
 expected_order=[2,1,3,4,0]
 if d.get('order')!=expected_order or [x['column'] for x in d.get('segments',[])]!=expected_order:errors.append('column order')
 if d.get('remaining_wall_boxes')!=[f'wall_box_{i:02d}' for i in range(15)]:errors.append('remaining boxes')
 lifecycle=d.get('lifecycle',[])
 if len(lifecycle)!=20:errors.append(f'lifecycle count {len(lifecycle)}')
 for box in range(15,25):
  events=[x for x in lifecycle if x['object_id']==f'wall_box_{box:02d}']
  if len(events)!=2:errors.append(f'box {box} lifecycle count')
  if events and not (0<=events[0]['global_frame_index']<=events[-1]['global_frame_index']<len(rows)):errors.append(f'box {box} lifecycle order')
  if len(events)==2:
   release=events[-1]
   if not release['phase'].endswith('_release'):errors.append(f'box {box} not released at rear')
   if 'world_pose' not in release or 'target_world_pose' not in release:errors.append(f'box {box} missing rear pose')
   else:
    actual=pose_matrix(Pose(**release['world_pose']));target=pose_matrix(Pose(**release['target_world_pose']))
    position=float(np.linalg.norm(actual[:3,3]-target[:3,3]));angle=float(Rotation.from_matrix(actual[:3,:3].T@target[:3,:3]).magnitude())
    if position>.001 or angle>math.radians(.5):errors.append(f'box {box} rear pose error')
 phases=d['phases'];base_steps=np.abs(np.diff(rows[:,idx['base_y']]))
 if len(base_steps) and base_steps.max()>.010001:errors.append('base interpolation step')
 joint_rows=rows[:,[idx[n] for n in ACTIVE_JOINTS]];max_joint_step=float(np.abs(np.diff(joint_rows,axis=0)).max()) if len(rows)>1 else 0.
 if max_joint_step>math.radians(.51):errors.append(f'joint step {max_joint_step}')
 report={'passed':not errors,'errors':errors,'frames_checked':len(rows),'lifecycle_events':len(lifecycle),'bridge_projection_y_m':[float(minimum),float(maximum)],'base_y_range_m':[float(rows[:,idx['base_y']].min()),float(rows[:,idx['base_y']].max())],'max_base_step_m':float(base_steps.max()) if len(base_steps) else 0.,'max_joint_step_deg':math.degrees(max_joint_step),'order':d.get('order'),'remaining_wall_boxes':d.get('remaining_wall_boxes')}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
