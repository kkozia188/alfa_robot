"""One bounded diagnostic: same baseline solver optimization, top-32 vs all 512 outputs."""
import fcntl,json,math
from pathlib import Path
import numpy as np,torch
from curobo.types import JointState
from curobo_core.contracts import PlanRequest,PlannerAssets,ACTIVE_JOINTS
from curobo_core.planner import FullCyclePlanner
from curobo_core.sequential import _SequentialTask,pose_error
from curobo_core.scene import SceneSnapshot,SceneStore,Pose,AttachedObject
from curobo_core.adapter import pose_matrix,matrix_pose
from curobo_core.backend import goal
from v3_plan_cycle import gpu_processes
root=Path('artifacts/sequential-unload/baseline-restore-20261007')
with open('/tmp/sevenova-curobo-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not gpu_processes(),gpu_processes()
 d=json.loads((root/'endpoint-guided-fixed-swivel-seed11-20261008.json').read_text());r=d['results'][0];req=PlanRequest.from_dict(d['request'])
 p=FullCyclePlanner(PlannerAssets());p.set_snapshot(req.snapshot);t=_SequentialTask(p,req,lambda _:None)
 t.q=np.asarray(r['frames'][-1]);t.store=SceneStore(SceneSnapshot.from_dict(r['final_snapshot']))
 upper=next(a for a in t.store.snapshot().attachments if a.parent_link=='right_tool0');t.support_pose=t.box_pose(t.q,upper)
 g=next(g for g in r['contact_geometry'] if g['side']=='left' and abs(g['roll_rad']-math.pi/6)<1e-9)
 target=pose_matrix(Pose(**g['target']));item=t.store.snapshot().object('wall_box_17')
 t.store.attach(AttachedObject(item.object_id,item.dimensions_m,'left_tool0',matrix_pose(np.linalg.inv(target)@pose_matrix(item.pose)),('left_link7',)),t.store.snapshot().revision)
 endpoint=target.copy();endpoint[0,3]-=req.extraction_m;t.enter('left_extract_target');t.ik('left',endpoint,list(range(1,8)))
 solver=p._cached_solver;poses={'left_tool0':{'position':endpoint[:3,3],'quaternion':matrix_pose(endpoint).quaternion_wxyz}}
 state=JointState.from_position(t.tensor(t.q[1:8])[None],joint_names=list(ACTIVE_JOINTS[1:8]));v=t.validity();reports={}
 for returned in (32,512):
  solver.reset_seed();result=solver.solve_pose(goal(poses,frames=['left_tool0']),state,return_seeds=returned)
  names=list(result.js_solution.joint_names);values=result.js_solution.position.reshape(-1,len(names));rows=np.repeat(t.q[None],len(values),axis=0);rows[:,1:8]=values[:,[names.index(n) for n in ACTIVE_JOINTS[1:8]]].cpu().numpy()
  finite=np.isfinite(rows).all(1);valid=np.zeros(len(rows),bool);valid[finite]=v.mask(t.tensor(rows[finite])).cpu().numpy();errors=[pose_error(t.tool(q,'left'),endpoint) if ok else (float('inf'),float('inf')) for q,ok in zip(rows,finite)]
  pose_ok=np.array([x<=.002 and a<=math.radians(1) for x,a in errors]);good=finite&valid&pose_ok
  reports[str(returned)]={'native_successes':int(result.success.sum()),'outputs':len(rows),'finite':int(finite.sum()),'pose_ok_2mm_1deg':int(pose_ok.sum()),'full_valid':int(valid.sum()),'certified':int(good.sum()),'minimum_position_error_m':float(min(x for x,a in errors))}
 out={'scope':'same +30deg endpoint, same baseline 512-seed/500-iteration solver; return count only changes exposed optimized outputs','reports':reports}
 (root/'endpoint-return32-vs-512.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
