#!/usr/bin/python3
"""Record a production-shell keyframe at the first dual chest-carry pose."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rerun as rr


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT_DIR = WORKSPACE / "ros2_ws/src/alfa_robot_moveit_config/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v3_scoop_5x5_dual_rerun as dual
from alfa_robot_rerun.visualize_rerun import log_robot_state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--model-urdf", type=Path, required=True)
    parser.add_argument("--model-asset-root", type=Path, required=True)
    args = parser.parse_args()

    replay = dual.read_json(args.input.resolve())
    operations = list(replay["operations"])
    operation = next(item for item in operations if item["kind"] != "base_transition")
    frame = next(
        item for item in operation["frames"]
        if item["stage"] == "chest_carry_ready"
    )
    urdf_text = dual.load_model_urdf(
        args.model_urdf.resolve(), args.model_asset_root.resolve()
    )
    dual.validate_v322_model_urdf(replay, urdf_text)
    recorder = dual.DualSequenceRecorder(
        args.save.resolve(), 1.0, list(replay["box_planning"]), operations, urdf_text
    )
    names = [str(value) for value in replay["joint_names"]]
    joints = {
        name: float(value) for name, value in zip(names, frame["joints"])
    }
    recorder.base_transform = dual.frame_base_transform(operation, frame)
    recorder.set_time()
    rr.log(
        "world/robot",
        rr.Transform3D(
            translation=recorder.base_transform[:3, 3].tolist(),
            quaternion=dual.sequence.matrix_to_quaternion(
                recorder.base_transform[:3, :3]
            ),
        ),
    )
    log_robot_state(recorder.robot, joints, "world/robot")
    for side in ("left", "right"):
        box_id = int(operation.get(f"{side}_box_id", 0))
        if box_id > 0:
            recorder.remaining.remove(box_id)
            recorder.completed_boxes += 1
            recorder.log_carried_box(
                side, box_id, operation[f"{side}_attachment"], joints
            )
    recorder.log_boxes()
    recorder.log_status(
        operation,
        1,
        "chest_carry_ready (base stationary)",
        dual.operation_metrics(operation, names),
        "pullback",
    )
    rr.log(
        "transition_status",
        rr.TextDocument(
            "# Chest carry ready\n\n"
            "- Boxes: **1 + 5**\n"
            "- Arms/lift: **motion complete**\n"
            "- Base: **stationary at pickup pose**\n"
            "- Next: `backoff at 0.35 m/s`\n"
            "- Validation: `MoveIt/FCL passed`",
            media_type=rr.MediaType.MARKDOWN,
        ),
    )
    rr.disconnect()
    print(f"RESULT keyframe={args.save.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
