import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from v3_round1_full_plan import (
    PAIR, add_payload_links, audit_state, payload_grid, tensor_state,
    update_payloads, validate_payload_path,
)
from v3_wall_ik_benchmark import make_scene, wall_center


parser = argparse.ArgumentParser()
parser.add_argument('--result', type=Path, required=True)
parser.add_argument('--ik-summary', type=Path, required=True)
args = parser.parse_args()
report = json.loads(args.result.read_text())
summary = json.loads(args.ik_summary.read_text())
robot_data = yaml.safe_load(Path(summary['robot_config']).read_text())
loaded_robot = copy.deepcopy(robot_data)
add_payload_links(loaded_robot)
scene = make_scene(summary['chassis_front_x_m'], summary['wall_distance_m'], set(PAIR))
config = InverseKinematicsCfg.create(
    robot=loaded_robot, scene_model=scene, collision_cache={'cuboid': 40},
    num_seeds=4, self_collision_check=True, use_cuda_graph=False,
)
ik = InverseKinematics(config)
contact = summary['rounds'][0]['collision_free']
contact_state = tensor_state(contact['joints'])
poses = ik.compute_kinematics(contact_state).tool_poses.to_dict()
grids = {}
tool_to_box = {}
for side, box_id in zip(('left','right'), PAIR):
    pose = poses[f'{side}_tool0']
    position = pose.position.reshape(-1,3)[0].detach().cpu().numpy()
    quaternion = pose.quaternion.reshape(-1,4)[0].detach().cpu().numpy()
    center = wall_center(summary['chassis_front_x_m'], summary['wall_distance_m'], box_id)
    grids[side] = payload_grid(position, quaternion, center)
    rotation = Rotation.from_quat(np.roll(quaternion,-1)).as_matrix()
    tool = np.eye(4); tool[:3,:3]=rotation; tool[:3,3]=position
    box = np.eye(4); box[:3,3]=center
    tool_to_box[side]=np.linalg.inv(tool)@box
update_payloads(ik, grids)
frames = report['stages'][-1]['rejected_frames']
first_failure = None
failure_counts = {}
sampled = {}
for index,row in enumerate(frames):
    audit = audit_state(ik.kinematics, tensor_state(row), scene)
    exact_ok, exact_reason = validate_payload_path(
        ik.kinematics, [row], tool_to_box,
        summary['chassis_front_x_m'], summary['wall_distance_m'])
    reasons=[]
    reasons += [f"self:{x['links'][0]}<->{x['links'][1]}" for x in audit['self_collisions']]
    reasons += [f"world:{x['link']}<->{x['obstacle']}" for x in audit['world_collisions']]
    reasons += [f"limit:{x}" for x in audit['joint_limit_violations']]
    if not exact_ok: reasons.append(exact_reason)
    for reason in reasons: failure_counts[reason]=failure_counts.get(reason,0)+1
    if reasons and first_failure is None:
        first_failure={'frame':index,'reasons':reasons,'audit':audit,'exact_payload_reason':exact_reason}
    if reasons and len(sampled)<8:
        sampled[index]={'reasons':reasons,'audit':audit,'exact_payload_reason':exact_reason}
result={'checked_frames':len(frames),'first_failure':first_failure,
        'failure_counts':dict(sorted(failure_counts.items(),key=lambda x:-x[1])),
        'sampled_failures':sampled}
report['loaded_rejected_path_audit']=result
args.result.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(result,indent=2))
