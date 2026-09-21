#!/usr/bin/env python3
"""Record the complete verified eleven-round V3 side-suction sequence."""

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from alfa_robot_rerun.visualize_rerun import (
    UrdfRobot,
    log_robot_state,
    log_robot_static_model,
    matrix_to_quaternion_xyzw,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text())
    sequence = json.loads(args.sequence.read_text())
    if not sequence.get("success") or len(sequence.get("boundaries", [])) != 11:
        raise ValueError("requires the verified eleven-round sequence")
    names = sequence["joint_names"]
    boxes = {box["box_id"]: box for box in scene["wall_boxes"]}
    robot = UrdfRobot((args.scene.parent / "robot.urdf").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("v3_complete_11_round_side_suction", spawn=False)
    rr.save(str(args.output))
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", contents=["/world/**"],
                              name="11-round complete task",
                              eye_controls=rrb.EyeControls3D(
                                  position=[-2.8, 3.5, 2.7], look_target=[0.4, 0, 1.05],
                                  eye_up=[0, 0, 1])),
            rrb.TextDocumentView(origin="/status", name="Task stage"),
            column_shares=[.80, .20]),
        rrb.TimePanel(timeline="full_task", expanded=True, fps=60, play_state="Playing"),
        collapse_panels=True))
    log_robot_static_model(robot, "world/robot")
    for obstacle in scene["environment"]["boxes"]:
        rr.log("world/environment/" + obstacle["id"], rr.Boxes3D(
            centers=[obstacle["center"]], half_sizes=[np.asarray(obstacle["size"]) / 2],
            colors=[[110, 135, 165, 35]]), static=True)
    removed = set()
    attachment_offsets = {}
    prior_stage = None
    for index, frame in enumerate(sequence["frames"]):
        rr.set_time("full_task", sequence=index)
        joints = dict(zip(names, frame["joints"]))
        log_robot_state(robot, joints, "world/robot")
        transforms = robot.fk(joints)
        if frame["stage"] == "ATTACH":
            attachment_offsets = {}
            active_sides = ["left"] + (["right"] if len(frame["box_ids"]) == 2 else [])
            for side, box_id in zip(active_sides, frame["box_ids"]):
                box_pose = np.eye(4)
                box_pose[:3, 3] = boxes[box_id]["center"]
                attachment_offsets[side] = (box_id,
                    np.linalg.inv(transforms[f"{side}_tool0"]) @ box_pose)
        if frame["stage"] == "RELEASE":
            removed.update(frame["box_ids"])
            attachment_offsets = {}
        active_ids = set(frame["box_ids"]) if frame["attached"] else set()
        remaining = [box for box_id, box in boxes.items()
                     if box_id not in removed and box_id not in active_ids]
        rr.log("world/wall", rr.Boxes3D(
            centers=[box["center"] for box in remaining],
            half_sizes=[np.asarray(box["size"]) / 2 for box in remaining],
            colors=[[45, 170, 190, 32] for _ in remaining],
            labels=[str(box["box_id"]) for box in remaining]))
        centers, sizes, rotations, labels = [], [], [], []
        if frame["attached"]:
            for side, (box_id, offset) in attachment_offsets.items():
                pose = transforms[f"{side}_tool0"] @ offset
                centers.append(pose[:3, 3])
                sizes.append(np.asarray(boxes[box_id]["size"]) / 2)
                rotations.append(matrix_to_quaternion_xyzw(pose[:3, :3]))
                labels.append(f"{side} box {box_id}")
        rr.log("world/carried_boxes", rr.Boxes3D(
            centers=centers, half_sizes=sizes, quaternions=rotations,
            colors=[[65, 210, 110, 200] for _ in centers], labels=labels))
        if prior_stage != frame["stage"] or index % 30 == 0:
            rr.log("status", rr.TextDocument(
                f"## Round {frame['round']}/11 · {frame['stage']}\n\n"
                f"Targets: `{frame['box_ids']}`\n\n"
                f"Attached: `{frame['attached']}` · Visible: `{frame['visible']}`\n\n"
                f"Removed boxes: `{sorted(removed)}`\n\n"
                f"Frame: {index + 1}/{len(sequence['frames'])}",
                media_type=rr.MediaType.MARKDOWN))
        prior_stage = frame["stage"]
    rr.get_global_data_recording().flush()
    print(json.dumps({"rrd": str(args.output), "frames": len(sequence["frames"]),
                      "rounds": len(sequence["boundaries"]),
                      "removed_boxes": sorted(removed),
                      "maximum_rotary_step_deg": sequence["maximum_rotary_step_deg"],
                      "maximum_updown_step_m": sequence["maximum_updown_step_m"]}, indent=2))


if __name__ == "__main__":
    main()
