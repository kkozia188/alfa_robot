import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yourdfpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    robot = yourdfpy.URDF.load(args.urdf, load_meshes=False, build_scene_graph=True)
    joints = {joint.attrib["name"]: joint for joint in ET.parse(args.urdf).getroot().findall("joint")}
    random = np.random.default_rng(20261003)
    cases = {}
    inputs = []
    for side_id, side in enumerate(("left", "right")):
        names = [f"{side}_joint{index}" for index in range(1, 8)]
        limits = [joints[name].find("limit").attrib for name in names]
        lower = np.array([float(limit["lower"]) for limit in limits])
        upper = np.array([float(limit["upper"]) for limit in limits])
        samples = random.uniform(lower + 0.001, upper - 0.001, (1000, 7))
        samples = np.vstack((np.zeros(7), samples))
        for sample_id, values in enumerate(samples):
            state = dict(zip(names, values))
            state["updown"] = float(random.uniform(-1.0, 0.0))
            robot.update_cfg({name: state.get(name, 0.0) for name in robot.actuated_joint_names})
            target = robot.get_transform(f"{side}_tool0", "arm_carriage").copy()
            cases[(sample_id, side_id)] = (target, state)
            inputs.append(" ".join(map(str, [sample_id, side_id, *values, *target.reshape(-1)])))
    output = subprocess.run([str(args.binary.resolve())], input="\n".join(inputs), text=True,
                            capture_output=True, check=True).stdout
    reports = {side: {"fk_errors_m": [], "fk_errors_rad": [], "ik_errors_m": [],
                     "ik_errors_rad": [], "solve_us": [], "no_solution": [], "solutions": 0}
               for side in ("left", "right")}
    for line in output.splitlines():
        fields = line.split()
        sample_id, side_id = map(int, fields[1:3])
        side = ("left", "right")[side_id]
        report = reports[side]
        if fields[0] == "sample":
            report["fk_errors_m"].append(float(fields[3]))
            report["fk_errors_rad"].append(float(fields[4]))
            report["solve_us"].append(float(fields[6]))
            if int(fields[5]) == 0:
                report["no_solution"].append(sample_id)
        else:
            target, source_state = cases[(sample_id, side_id)]
            state = source_state.copy()
            state.update({f"{side}_joint{index}": float(value)
                          for index, value in enumerate(fields[3:], 1)})
            robot.update_cfg({name: state.get(name, 0.0) for name in robot.actuated_joint_names})
            actual = robot.get_transform(f"{side}_tool0", "arm_carriage")
            report["ik_errors_m"].append(float(np.linalg.norm(actual[:3, 3] - target[:3, 3])))
            report["ik_errors_rad"].append(float(Rotation.from_matrix(
                actual[:3, :3].T @ target[:3, :3]).magnitude()))
            report["solutions"] += 1
    summary = {"urdf": str(args.urdf.resolve()), "seed": 20261003, "arms": {}}
    for side, report in reports.items():
        summary["arms"][side] = {
            "samples": len(report["fk_errors_m"]), "ik_no_solution": report["no_solution"],
            "solutions_verified": report["solutions"],
            "max_analytic_vs_urdf_fk_mm": max(report["fk_errors_m"]) * 1000,
            "max_analytic_vs_urdf_fk_deg": np.degrees(max(report["fk_errors_rad"])),
            "max_ik_urdf_roundtrip_mm": max(report["ik_errors_m"], default=float("inf")) * 1000,
            "max_ik_urdf_roundtrip_deg": np.degrees(max(report["ik_errors_rad"], default=float("inf"))),
            "mean_analytic_ik_us": float(np.mean(report["solve_us"])),
        }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
