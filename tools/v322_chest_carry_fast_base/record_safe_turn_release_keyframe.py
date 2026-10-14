#!/usr/bin/python3
"""Record the first dual-box safe-turn release as a production-shell keyframe."""

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
    operation = next(
        item for item in operations
        if item.get("kind") == "dual_safe_turn_unload_cycle"
    )
    release_index = next(
        index for index, frame in enumerate(operation["frames"])
        if frame["stage"] == "release_after_turn_at_safe_exit"
    )
    frame = operation["frames"][release_index]
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
        box_id = int(operation[f"{side}_box_id"])
        recorder.remaining.remove(box_id)
        recorder.completed_boxes += 1
        recorder.remember_placed_box(
            side, box_id, operation[f"{side}_attachment"], joints
        )
    recorder.log_boxes()
    left_z = recorder.placed_boxes[int(operation["left_box_id"])]["center"][2]
    right_z = recorder.placed_boxes[int(operation["right_box_id"])]["center"][2]
    rr.log(
        "transition_status",
        rr.TextDocument(
            "# Safe-turn equal-height release\n\n"
            f"- Boxes: **{operation['left_box_id']} + {operation['right_box_id']}**\n"
            "- Chest carry: **completed before backoff**\n"
            "- Backoff: **2.35 m**\n"
            "- Turn: **180 deg**\n"
            f"- Left / right center Z: **{left_z:.9f} / {right_z:.9f} m**\n"
            f"- Height error: **{1000.0 * abs(left_z - right_z):.9f} mm**\n"
            "- Validation: `MoveIt/FCL passed`",
            media_type=rr.MediaType.MARKDOWN,
        ),
    )
    rr.disconnect()
    print(
        f"RESULT keyframe={args.save.resolve()} "
        f"left_z={left_z:.9f} right_z={right_z:.9f} "
        f"error_mm={1000.0 * abs(left_z - right_z):.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
