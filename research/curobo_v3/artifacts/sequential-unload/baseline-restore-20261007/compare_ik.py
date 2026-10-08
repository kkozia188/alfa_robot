"""Same inputs for unmodified f044 factory and the extended cached entry."""
import fcntl,gc,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
import torch
from curobo.types import JointState
from curobo_core.backend import CuroboBackend,goal
from curobo_core.contracts import PlannerAssets,ACTIVE_JOINTS
from curobo_core.scene import Pose
from v3_wall_ik_benchmark import canonical_side_suction_quaternion_wxyz
from v3_plan_cycle import gpu_processes
root=Path('artifacts/sequential-unload/baseline-restore-20261007')
with open('/tmp/sevenova-curobo-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not gpu_processes(),gpu_processes()
 spec=importlib.util.spec_from_file_location('curobo_core.f044_reference_backend',root/'f044-backend.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 assets=PlannerAssets();tasks={'left':24,'right':20};reports=[];outputs={};snapshot=None
 for label,cls in [('f044_unmodified',module.CuroboBackend),('restored_cached_entry',CuroboBackend)]:
  planner=cls(assets)
  if snapshot is None:snapshot=planner.snapshot
  else:planner.set_snapshot(snapshot)
  poses={}
  for side,box_id in tasks.items():
   box=snapshot.object(f'wall_box_{box_id:02d}');pos=np.array(box.pose.position);pos[0]-=box.dimensions_m[0]/2
   poses[side+'_tool0']={'position':pos,'quaternion':canonical_side_suction_quaternion_wxyz(side)}
  state=JointState.from_position(torch.tensor(planner.home_values[None],device='cuda'),joint_names=list(ACTIVE_JOINTS));last=None
  for repeat in range(2):
   torch.cuda.synchronize();started=time.perf_counter();solver=planner.ik_solver(tasks);torch.cuda.synchronize();init_ms=(time.perf_counter()-started)*1000
   same=solver is last;solver.reset_seed();torch.cuda.synchronize();started=time.perf_counter();result=solver.solve_pose(goal(poses),state,return_seeds=32);torch.cuda.synchronize();solve_ms=(time.perf_counter()-started)*1000
   started=time.perf_counter();names=list(result.js_solution.joint_names);values=result.js_solution.position.reshape(-1,len(names));successful=result.success.reshape(-1).bool();candidates=values[successful][:,[names.index(name) for name in ACTIVE_JOINTS]].cpu().numpy();weights=np.array([5.]+[1.]*14);candidates=candidates[np.argsort(np.sum(((candidates-planner.home_values)*weights)**2,axis=1))];filter_ms=(time.perf_counter()-started)*1000
   reports.append({'implementation':label,'repeat':repeat+1,'same_cached_object':same,'initialization_or_cache_lookup_ms':init_ms,'solve_ms':solve_ms,'filter_ms':filter_ms,'returned':int(result.success.numel()),'native_successes':len(candidates)})
   outputs[label+str(repeat)]=candidates.tolist();last=solver
  del last,solver,planner,result;gc.collect();torch.cuda.empty_cache()
 a=np.array(outputs['f044_unmodified0']);b=np.array(outputs['restored_cached_entry0']);equal=a.shape==b.shape and np.allclose(a,b,atol=1e-6,rtol=0.)
 document={'scope':'identical default model, scene, joint state, targets, native seed reset, budgets and return/filter contract',
  'position_tolerance_m':.002,'orientation_tolerance_deg':1.,'num_seeds':512,'multi_link_iterations':500,'return_seeds':32,
  'input_scene':snapshot.to_dict(),'goal_poses':{n:{'position':np.asarray(v['position']).tolist(),'quaternion':np.asarray(v['quaternion']).tolist()} for n,v in poses.items()},
  'reports':reports,'candidates_match_1e_6':bool(equal),'max_candidate_difference':float(np.abs(a-b).max()) if a.shape==b.shape and a.size else None,
  'first_initialization_includes_first_use_cuda_costs':True,'outputs':outputs}
 (root/'same-input-ik-comparison.json').write_text(json.dumps(document,indent=2));print(json.dumps({k:v for k,v in document.items() if k not in ('input_scene','goal_poses','outputs')},indent=2))
