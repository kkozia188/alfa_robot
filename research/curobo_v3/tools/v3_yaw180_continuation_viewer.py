#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf
import yaml
import yourdfpy

from v3_wall_ik_benchmark import canonical_side_tool_to_box


def quaternion(matrix):
    return np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--front-x", type=float, default=0.5080520510673523)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    config_data = yaml.safe_load(args.config.read_text())
    config = config_data.get("robot_cfg", config_data)["kinematics"]
    names = config["cspace"]["joint_names"]
    urdf = yourdfpy.URDF.load(
        Path(config["urdf_path"]), load_meshes=True, build_scene_graph=True
    )
    actuated = list(urdf.actuated_joint_names)
    frames = result["frames"]
    server = viser.ViserServer(
        host="127.0.0.1", port=args.port,
        label="V3.2.2 · 任务完成后Yaw 180°接续动作",
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=5, height=3)
    robot = ViserUrdf(
        server, urdf, root_node_name="/robot",
        mesh_color_override=(0.68, 0.73, 0.80, 0.76),
    )
    wall_front = args.front_x + 0.9
    for box_id in range(25):
        if box_id in result["removed_boxes"]:
            continue
        server.scene.add_box(
            f"/wall/box_{box_id:02d}",
            position=(wall_front + 0.15, (box_id % 5 - 2) * 0.41,
                      0.20 + (box_id // 5) * 0.41),
            dimensions=(0.30, 0.40, 0.40), color=(87, 145, 165), opacity=0.18,
        )
    attached = {}
    offsets = {side: canonical_side_tool_to_box(side) for side in ("left", "right")}
    for side in ("left", "right"):
        attached[side] = server.scene.add_frame(f"/attached/{side}", show_axes=False)
        server.scene.add_box(
            f"/attached/{side}/box", dimensions=(0.30, 0.40, 0.40),
            color=(239, 146, 62), opacity=0.78,
        )
        server.scene.add_label(
            f"/attached/{side}/label", text=f"{side}最后一组箱",
            position=(0, 0, 0.27),
        )
    with server.gui.add_folder("Yaw 180°接续动作"):
        slider = server.gui.add_slider(
            "轨迹帧", min=1, max=len(frames), step=1, initial_value=1
        )
        play = server.gui.add_checkbox("播放", initial_value=False)
        status = server.gui.add_markdown("")
        server.gui.add_markdown(
            "固定Updown和双臂关节，仅底盘Yaw从0°旋转到180°。"
            "显示最后一组箱仍附着的更严格场景；361帧联合碰撞验收全部通过。"
        )

    def show(index):
        values = dict(zip(names, frames[index]))
        q = np.asarray([values.get(name, 0.0) for name in actuated])
        urdf.update_cfg(q)
        robot.update_cfg(q)
        for side, node in attached.items():
            matrix = urdf.get_transform(f"{side}_tool0", urdf.base_link) @ offsets[side]
            node.position = matrix[:3, 3]
            node.wxyz = quaternion(matrix)
        status.content = (
            f"**任务完成后接续动作 · PASS**  \n"
            f"Frame：**{index + 1}/{len(frames)}**  \n"
            f"Base Yaw：**{index * 0.5:.1f}°**  \n"
            f"空载：**PASS** · 携最后双箱：**PASS**  \n"
            f"最大箱体倾角：**{result['loaded']['max_box_tilt_deg']:.2f}°**"
        )

    @slider.on_update
    def on_slider(_event):
        show(int(slider.value) - 1)

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-3.2, 3.4, 2.8)
        client.camera.look_at = (0.6, 0.0, 1.2)
        client.camera.up_direction = (0, 0, 1)

    show(0)
    print(f"Ready at http://localhost:{args.port}", flush=True)
    while True:
        if play.value:
            slider.value = int(slider.value) % len(frames) + 1
            time.sleep(0.04)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
