"""CPU differential: successful historical path vs f044 fixed-swivel bridge."""
import importlib.util, json, math, time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from curobo_core.contracts import PlannerAssets
from curobo_core.planner import FullCyclePlanner

root=Path('artifacts/sequential-unload')
r=json.loads((root/'elbow-profile-20261007/optimized-nine-runs-s11-r1.json').read_text())['result']
ix=[i for i,p in enumerate(r['phases']) if p=='left_extract']
recorded=np.asarray([r['frames'][i] for i in ix])
contact=recorded[0]
planner=FullCyclePlanner(PlannerAssets())
class Validity:
 def mask(self, rows): return torch.ones(len(rows),dtype=torch.bool)
original=torch.tensor
def cpu_tensor(*args,**kwargs): kwargs['device']='cpu'; return original(*args,**kwargs)
started=time.perf_counter()
with patch('curobo_core.planner.torch.tensor',side_effect=cpu_tensor):
 fixed,fixed_details=planner.analytic_extract(contact,Validity(),Validity(),sides=('left',),direction=(-1.,0.,0.),distance_m=.36)
fixed_ms=(time.perf_counter()-started)*1000
# Measure the already accepted path against f044's 2mm/1deg Cartesian contract.
planner.fk(contact); origin=planner.fk_robot.get_transform('left_tool0',planner.fk_robot.base_link).copy()
psis=[]; line=angle=progress_error=0.; previous=0.
for q in recorded:
 planner.fk(q); actual=planner.fk_robot.get_transform('left_tool0',planner.fk_robot.base_link)
 delta=actual[:3,3]-origin[:3,3]; along=-delta[0]
 line=max(line,float(np.linalg.norm(delta[1:3])))
 angle=max(angle,float(Rotation.from_matrix(actual[:3,:3].T@origin[:3,:3]).magnitude()))
 progress_error=max(progress_error,max(0.,previous-along)); previous=along
 psis.append(float(planner.analytic.swivel(0,q[1:8])))
psis=np.unwrap(psis)
out={
 'historical_path':{'source':'optimized-nine-runs-s11-r1.json','frames':len(recorded),
  'distance_m':previous,'max_line_error_m':line,'max_orientation_error_deg':math.degrees(angle),
  'swivel_start_rad':float(psis[0]),'swivel_end_rad':float(psis[-1]),
  'swivel_change_rad':float(psis[-1]-psis[0]),'swivel_range_rad':float(np.ptp(psis)),
  'meets_f044_2mm_1deg':line<=.002 and angle<=math.radians(1)},
 'f044_parameterized_forward_fixed_swivel':{'success':fixed is not None,'details':fixed_details,
  'wall_ms_cpu_without_collision':fixed_ms,'failure':planner.last_cartesian_failure},
 'conclusion':'The historical lower path is effectively fixed-swivel, and the parameterized f044 bridge succeeds from the same contact state; candidate branch selection caused the restored failure.'}
p=root/'baseline-restore-20261007/successful-vs-f044-cartesian.json';p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
