#!/usr/bin/env python3
import csv, hashlib, json, math, statistics
from pathlib import Path
BASE=Path(__import__('os').environ.get('OUTPUT_DIR', '/home/astesia/v3-scoop-seed-comparison/fixed-v3'))
SEEDS=[104729,104743,104759,104761,104773,104779,104789,104801]

def load(v,seed):
 d=BASE/v/f'seed-{seed}'; stem=d/f'v3-scoop-{v}-seed-{seed}'
 return json.loads((Path(str(stem)+'-summary.json')).read_text()), json.loads((Path(str(stem)+'-plan-cache.json')).read_text()), list(csv.DictReader(open(str(stem)+'-metrics.csv')))

def norm_task(t):
 return {
  'box_id':t['box_id'],'side':t['side'],'updown':t['updown'],
  'frames':t['payload']['frames'],'transition_frames':t['transition_frames'],
  'place_arm_joints_deg':t.get('place_arm_joints_deg'),
 }
report={'schema':'alfa.v3_scoop_fixed_seed_c_d_comparison.v1','date':'2026-09-25','seeds':SEEDS,'acceptance':{'all_seeds_must_pass':True,'completed_boxes':25,'validated_transitions':25,'maximum_selected_core_planning_s':5.0},'runs':[]}
all_pass=True
for seed in SEEDS:
 cs,cp,cm=load('C',seed); ds,dp,dm=load('D',seed)
 ct=[float(r['selected_planner_s']) for r in cm]; dt=[float(r['selected_planner_s']) for r in dm]
 c_hash=[hashlib.sha256(json.dumps(norm_task(t),sort_keys=True,separators=(',',':')).encode()).hexdigest() for t in cp['tasks']]
 d_hash=[hashlib.sha256(json.dumps(norm_task(t),sort_keys=True,separators=(',',':')).encode()).hexdigest() for t in dp['tasks']]
 paths_equal=c_hash==d_hash
 cst=[b['planning_search']['selected_task'] for b in cs['boxes']]; dst=[b['planning_search']['selected_task'] for b in ds['boxes']]
 planner_delta=[abs(float(a['planner_wall_ms'])-float(b['planner_wall_ms'])) for a,b in zip(cst,dst)]
 process_delta=[abs(float(a['process_wall_ms'])-float(b['process_wall_ms'])) for a,b in zip(cst,dst)]
 row={
  'seed':seed,
  'C':{'completed_boxes':cs['completed_boxes'],'validated_transitions':cs['validated_transitions'],'average_s':statistics.fmean(ct),'median_s':statistics.median(ct),'p95_s':sorted(ct)[math.ceil(.95*25)-1],'maximum_s':max(ct),'at_or_below_5s':sum(x<=5 for x in ct),'joint_flip_events':sum(int(r['joint_flip_events']) for r in cm),'inverted_frames':sum(int(b['carried_box_orientation']['inverted_frames']) for b in cs['boxes'])},
  'D':{'completed_boxes':ds['completed_boxes'],'validated_transitions':ds['validated_transitions'],'average_s':statistics.fmean(dt),'median_s':statistics.median(dt),'p95_s':sorted(dt)[math.ceil(.95*25)-1],'maximum_s':max(dt),'at_or_below_5s':sum(x<=5 for x in dt),'joint_flip_events':sum(int(r['joint_flip_events']) for r in dm),'inverted_frames':sum(int(b['carried_box_orientation']['inverted_frames']) for b in ds['boxes'])},
  'selected_trajectories_equal':paths_equal,
  'different_box_ids':[cp['tasks'][i]['box_id'] for i,(a,b) in enumerate(zip(c_hash,d_hash)) if a!=b],
  'stage_timing_delta_ms':{'planner_max':max(planner_delta),'planner_median':statistics.median(planner_delta),'process_max':max(process_delta),'process_median':statistics.median(process_delta)},
 }
 passed=all(row[v]['completed_boxes']==25 and row[v]['validated_transitions']==25 and row[v]['at_or_below_5s']==25 and row[v]['joint_flip_events']==0 and row[v]['inverted_frames']==0 for v in ('C','D')) and paths_equal
 row['passed']=passed; all_pass &= passed; report['runs'].append(row)
report['passed']=all_pass
(BASE/'comparison-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
raise SystemExit(0 if all_pass else 1)
