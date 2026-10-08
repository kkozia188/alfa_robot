"""Compare the known successful support/endpoint with the restored baseline IK."""
import fcntl,json,math
from dataclasses import asdict
from pathlib import Path
import numpy as np
from curobo_core.contracts import PlanRequest,PlannerAssets
from curobo_core.planner import FullCyclePlanner
from curobo_core.sequential import _SequentialTask,pose_error
from curobo_core.scene import SceneStore,AttachedObject,Pose
from curobo_core.adapter import pose_matrix,matrix_pose
from v3_plan_cycle import gpu_processes
root=Path('artifacts/sequential-unload')
with open('/tmp/sevenova-curobo-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not gpu_processes(),gpu_processes()
 d=json.loads((root/'elbow-profile-20261007/optimized-nine-runs-s11-r1.json').read_text());r=d['result'];req=PlanRequest.from_dict(d['request']);p=FullCyclePlanner(PlannerAssets());p.set_snapshot(req.snapshot);t=_SequentialTask(p,req,lambda _:None)
 support_i=max(i for i,x in enumerate(r['phases']) if x=='right_attach');t.q=np.asarray(r['frames'][support_i]);t.store=SceneStore(req.snapshot)
 upper=AttachedObject(**{**r['attachments_by_frame'][support_i][0],'tool_to_object':Pose(**r['attachments_by_frame'][support_i][0]['tool_to_object'])});t.store.attach(upper,t.store.snapshot().revision);t.support_pose=t.box_pose(t.q,upper)
 g=next(x for x in r['contact_geometry'] if x['side']=='left' and x['selected']);target=pose_matrix(Pose(**g['target']));item=t.store.snapshot().object('wall_box_17');lower=AttachedObject(item.object_id,item.dimensions_m,'left_tool0',matrix_pose(np.linalg.inv(target)@pose_matrix(item.pose)),('left_link7',));t.store.attach(lower,t.store.snapshot().revision)
 endpoint=target.copy();endpoint[0,3]-=req.extraction_m;t.enter('left_extract_target');found=t.ik('left',endpoint,list(range(1,8)))
 old_endpoint=np.asarray(r['frames'][max(i for i,x in enumerate(r['phases']) if x=='left_extract')]);v=t.validity();valid=bool(v.mask(t.tensor(old_endpoint[None])).item());error=pose_error(t.tool(old_endpoint,'left'),endpoint)
 out={'old_support_right_joints':t.q[8:].tolist(),'old_endpoint_is_valid_now':valid,'old_endpoint_pose_error_m_rad':error,'restored_baseline_ik_candidates':len(found),'attempt':r if False else t.report['attempts'][-1]}
 (root/'baseline-restore-20261007/old-support-baseline-ik.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
