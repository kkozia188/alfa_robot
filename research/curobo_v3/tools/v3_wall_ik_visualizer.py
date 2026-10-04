#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from curobo.scene import Cuboid, Scene
from curobo.types import ContentPath, JointState, Pose
from curobo.viewer import ViserVisualizer


LOCKED_JOINTS = {
    "head_joint": 0.0,
    "head_pitch_joint": 0.0,
    "active_suspension_joint": 0.0,
    "caster01_joint": 0.0,
    "wheel01_joint": 0.0,
    "caster02_joint": 0.0,
    "wheel02_joint": 0.0,
    "caster03_joint": 0.0,
    "wheel03_joint": 0.0,
    "caster04_joint": 0.0,
    "wheel04_joint": 0.0,
}


def wall_center(front_x: float, wall_distance: float, box_id: int) -> np.ndarray:
    return np.array([
        front_x + wall_distance + 0.15,
        (box_id % 5 - 2) * 0.41,
        0.20 + (box_id // 5) * 0.41,
    ])


def wall_scene(front_x: float, wall_distance: float, targets: set[int]) -> Scene:
    cuboids = []
    for box_id in range(25):
        center = wall_center(front_x, wall_distance, box_id)
        color = [0.95, 0.35, 0.12, 0.8] if box_id in targets else [0.55, 0.60, 0.68, 0.22]
        cuboids.append(Cuboid(
            name=f"box_{box_id:02d}",
            pose=[*center.tolist(), 1, 0, 0, 0],
            dims=[0.30, 0.40, 0.40],
            color=color,
        ))
    return Scene(cuboid=cuboids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--visual-config", type=Path, required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    successful_records = [
        item for item in summary["rounds"] if item["collision_free"]["success"]
    ]
    if not successful_records:
        raise RuntimeError("summary has no collision-free IK result")
    records = {item["round"]: item for item in successful_records}
    if args.round not in records:
        args.round = successful_records[0]["round"]
    visualizer = ViserVisualizer(
        content_path=ContentPath(robot_config_file=str(args.visual_config)),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_control_frames=False,
        visualize_robot_spheres=True,
    )
    labels = {
        item["round"]: (
            f"Round {item['round']:02d} | "
            f"L{item['left_box'] if item['left_box'] is not None else '-'} "
            f"R{item['right_box'] if item['right_box'] is not None else '-'} | "
            f"{item['suction_mode']}"
        )
        for item in successful_records
    }
    label_to_round = {label: round_number for round_number, label in labels.items()}
    with visualizer._server.gui.add_folder("IK Results"):
        round_selector = visualizer._server.gui.add_dropdown(
            "Round",
            options=[labels[item["round"]] for item in successful_records],
            initial_value=labels[args.round],
        )
        status = visualizer._server.gui.add_markdown("")

    def show_round(round_number: int) -> None:
        record = records[round_number]
        result = record["collision_free"]
        full_names = list(result["joint_names"]) + list(LOCKED_JOINTS)
        full_values = list(result["joints"]) + list(LOCKED_JOINTS.values())
        visualizer.add_scene(wall_scene(
            summary["chassis_front_x_m"],
            summary["wall_distance_m"],
            {
                box_id
                for box_id in (record["left_box"], record["right_box"])
                if box_id is not None
            },
        ))
        visualizer.set_joint_state(JointState.from_position(
            torch.tensor(full_values, device="cuda", dtype=torch.float32),
            joint_names=full_names,
        ))
        for frame, pose_data in result["fk"].items():
            visualizer.add_frame(
                f"/ik_tcp/{frame}",
                Pose(
                    position=torch.tensor(pose_data["position"], device="cuda").reshape(1, 3),
                    quaternion=torch.tensor(
                        pose_data["quaternion_wxyz"], device="cuda"
                    ).reshape(1, 4),
                ),
                scale=0.12,
            )
        status.content = (
            f"**Round {round_number:02d}**  "
            f"Left: `{record['left_box']}`  Right: `{record['right_box']}`  "
            f"Mode: `{record['suction_mode']}`  "
            f"Initial: `{record.get('initial_pose', 'home')}`  "
            f"IK: `{result['wall_ms']:.2f} ms`"
        )
        print(
            f"show round={round_number} left={record['left_box']} "
            f"right={record['right_box']}"
        )

    @round_selector.on_update
    def _on_round_change(_) -> None:
        show_round(label_to_round[round_selector.value])

    show_round(args.round)
    print(f"V3 cuRobo IK selector ready: http://localhost:{args.port}")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
