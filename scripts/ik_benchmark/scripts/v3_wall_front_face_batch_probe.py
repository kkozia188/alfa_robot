#!/usr/bin/env python3
"""Probe all five wall rows using front-face poses, without passing box IDs to IK."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from alfa_robot_rerun.visualize_rerun import UrdfRobot


def front_face(box, world_to_base):
    face = np.eye(4)
    face[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    face[:3, 3] = np.asarray(box["center"]) - np.array([box["size"][0] / 2, 0, 0])
    local = world_to_base @ face
    roll, pitch, yaw = Rotation.from_matrix(local[:3, :3]).as_euler("xyz")
    return dict(zip(("x", "y", "z", "roll", "pitch", "yaw"),
                    [*local[:3, 3], roll, pitch, yaw]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text())
    names = scene["joint_names"]
    robot = UrdfRobot((args.scene.parent / "robot.urdf").read_text())
    initial = dict(zip(names, scene["frames"][0]["joints"]))
    world_to_base = np.linalg.inv(robot.fk(initial)["base_link"])
    boxes = {item["box_id"]: item for item in scene["wall_boxes"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in range(4, -1, -1):
        start = row * 5
        for left_id, right_id in ((start + 4, start), (start + 3, start + 1),
                                  (start + 2, None)):
            number = len(records) + 1
            request = {"left": front_face(boxes[left_id], world_to_base),
                       "right": front_face(boxes[right_id], world_to_base)
                       if right_id is not None else None}
            request_file = args.output_dir / f"round_{number:02}_front_faces.json"
            result_file = args.output_dir / f"round_{number:02}_side_ik.json"
            request_file.write_text(json.dumps(request, indent=2) + "\n")
            command = ["python3", str(args.runner), "--scene", str(args.scene),
                       "--srdf", str(args.srdf), "--backend", str(args.backend),
                       "--dual-face-ik-request", str(request_file),
                       "--output", str(result_file)]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            if completed.returncode:
                raise RuntimeError(f"round {number}: {completed.stdout}\n{completed.stderr}")
            result = json.loads(result_file.read_text())
            pair_count = sum(len(item["shared_heights"]) for item in result["initials"])
            record = {"round": number, "left": left_id, "right": right_id,
                      "side_ik_shared_heights": pair_count,
                      "requires_top_ik_fallback": pair_count == 0,
                      "request": str(request_file.resolve()),
                      "result": str(result_file)}
            records.append(record)
            print(json.dumps(record), flush=True)
    summary = {"kind": "v3_front_face_wall_side_ik_only", "rounds": records,
               "side_ik_success": sum(item["side_ik_shared_heights"] > 0 for item in records),
               "scope": "IK only; no path, dynamic wall, top fallback or placement validation"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
