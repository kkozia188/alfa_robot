#!/usr/bin/env python3
"""CPU-only direct-y station and analytic-IK probe for all five wall columns."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from curobo_core.adapter import pose_matrix
from curobo_core.contracts import ACTIVE_JOINTS, PlannerAssets
from curobo_core.planner import FullCyclePlanner
from curobo_core.scene import Pose
from curobo_core.sequential import sequential_request
from v3_wall_ik_benchmark import canonical_side_suction_quaternion_wxyz, transform, urdf_link_transforms


def chassis_hull(assets, home):
    root, transforms = urdf_link_transforms(assets.urdf, home)
    points = []
    for link in root.findall("link"):
        if link.attrib["name"] != "base_link":
            continue
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            origin = collision.find("origin")
            local = transform(origin.get("xyz") if origin is not None else None,
                              origin.get("rpy") if origin is not None else None)
            geometry = trimesh.load(mesh.get("filename"), force="mesh", process=False)
            vertices = np.c_[np.asarray(geometry.vertices), np.ones(len(geometry.vertices))]
            points.append((transforms["base_link"] @ local @ vertices.T).T[:, :2])
    points = np.concatenate(points)
    return points[ConvexHull(points).vertices]


def target(side, xyz, roll=0.):
    matrix = pose_matrix(Pose(tuple(xyz), tuple(canonical_side_suction_quaternion_wxyz(side))))
    matrix[:3, :3] = Rotation.from_rotvec((roll, 0., 0.)).as_matrix() @ matrix[:3, :3]
    return matrix


def analytic(planner, world_to_carriage, side, matrix, base_y, seed):
    local = matrix.copy(); local[1, 3] -= base_y
    index = ("left", "right").index(side)
    solutions = []
    for psi in np.linspace(-math.pi, math.pi, 64, endpoint=False):
        solutions.extend(planner.analytic.solve(index, world_to_carriage @ local, psi, seed))
    if not solutions:
        return {"solutions": 0, "minimum_joint_distance": None}
    rows = np.asarray(solutions)
    distance = np.sum((rows-seed)**2, axis=1)
    return {"solutions": len(rows), "minimum_joint_distance": float(distance.min()),
            "best": rows[int(np.argmin(distance))].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warehouse-width-m", type=float, default=2.4)
    parser.add_argument("--bridge-side-inset-m", type=float, default=.15)
    args = parser.parse_args()
    assets = PlannerAssets(); planner = FullCyclePlanner(assets); request = sequential_request(planner)
    home_named = yaml.safe_load(assets.named_poses.read_text())["named_poses"]["home"]
    home = np.asarray([home_named[name] for name in ACTIVE_JOINTS])
    planner.fk(home)
    world_to_carriage = np.linalg.inv(planner.fk_robot.get_transform("arm_carriage", planner.fk_robot.base_link))
    hull = chassis_hull(assets, home_named)
    bridge_half = (args.warehouse_width_m-2*args.bridge_side_inset_m)/2
    minimum_base_y = -bridge_half-float(hull[:, 1].min())
    maximum_base_y = bridge_half-float(hull[:, 1].max())
    right_seed, left_seed = home[8:15], home[1:8]
    wall_face = request.snapshot.object("wall_box_22").pose.position[0]-.15
    result = {"kind":"five_column_direct_y_cpu_probe","yaw_deg":0.,"base_x_m":0.,
              "warehouse_width_m":args.warehouse_width_m,"bridge_width_m":2*bridge_half,
              "bridge_side_inset_m":args.bridge_side_inset_m,
              "base_y_limits_m":[minimum_base_y,maximum_base_y],
              "scope":"bridge footprint and 64-angle analytic precontact IK only; no collision/full-cycle claim",
              "roles":{"support":"right","transport":"left"},"columns":[]}
    for column in range(5):
        world_y=(column-2)*.4
        base_y=float(np.clip(world_y,minimum_base_y,maximum_base_y))
        footprint_y=hull[:,1]+base_y
        support_target=target("right",(wall_face-.05,world_y,1.7673550912141804))
        support=analytic(planner,world_to_carriage,"right",support_target,base_y,right_seed)
        transport_rows=[]
        for roll in (0.,math.pi/6,-math.pi/6,math.pi/4,-math.pi/4):
            matrix=target("left",(wall_face-.05,world_y,1.4289049959004567),roll)
            answer=analytic(planner,world_to_carriage,"left",matrix,base_y,left_seed)
            transport_rows.append({"roll_rad":roll,**answer})
        transport=max(transport_rows,key=lambda row:(row["solutions"]>0,
                      -math.inf if row["minimum_joint_distance"] is None else -row["minimum_joint_distance"]))
        result["columns"].append({"column":column,"upper_box":20+column,"lower_box":15+column,
            "world_y_m":world_y,"base_y_m":base_y,"target_local_y_m":world_y-base_y,
            "bridge_left_clearance_m":float(footprint_y.min()+bridge_half),
            "bridge_right_clearance_m":float(bridge_half-footprint_y.max()),
            "support_precontact":support,"transport_precontact":transport,
            "both_analytic_reachable":support["solutions"]>0 and transport["solutions"]>0})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"base_y_limits_m":result["base_y_limits_m"],"columns":[{
        key:row[key] for key in ("column","world_y_m","base_y_m","target_local_y_m","bridge_left_clearance_m","bridge_right_clearance_m","both_analytic_reachable")}
        for row in result["columns"]]},indent=2))

if __name__=="__main__":main()
