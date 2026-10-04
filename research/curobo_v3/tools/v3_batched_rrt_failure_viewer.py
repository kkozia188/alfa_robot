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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--front-x", type=float, default=0.5080520510673523)
    parser.add_argument("--port", type=int, default=8085)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    config_data = yaml.safe_load(args.config.read_text())
    config = config_data.get("robot_cfg", config_data)["kinematics"]
    frames = np.asarray(result["frames"])
    collision_index = int(result["exact_payload_reason"].rsplit("@", 1)[1])
    urdf = yourdfpy.URDF.load(
        Path(config["urdf_path"]), load_meshes=True, build_scene_graph=True
    )
    actuated = list(urdf.actuated_joint_names)
    names = config["cspace"]["joint_names"]
    server = viser.ViserServer(host="127.0.0.1", port=args.port,
                               label="V3.2.2 · Batched RRT精确碰撞否决")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=5, height=3)
    server.scene.add_frame("/robot", show_axes=False)
    robot = ViserUrdf(server, urdf, root_node_name="/robot",
                      mesh_color_override=(0.68, 0.73, 0.80, 0.75))
    wall_front = args.front_x + 0.9
    for box_id in range(25):
        if box_id in (20, 24):
            continue
        server.scene.add_box(
            f"/wall/box_{box_id:02d}",
            position=(wall_front + 0.15, (box_id % 5 - 2) * 0.41,
                      0.20 + (box_id // 5) * 0.41),
            dimensions=(0.30, 0.40, 0.40),
            color=(220, 48, 52) if box_id == 19 else (87, 145, 165),
            opacity=0.92 if box_id == 19 else 0.16,
        )
        if box_id == 19:
            server.scene.add_label("/wall/box_19/label", text="首次碰撞：19号箱",
                                   position=(wall_front + 0.15, 0.82, 1.92))
    attached = {}
    offsets = {side: canonical_side_tool_to_box(side) for side in ("left", "right")}
    for side in ("left", "right"):
        attached[side] = server.scene.add_frame(f"/attached/{side}", show_axes=True,
                                                axes_length=0.09, axes_radius=0.003)
        server.scene.add_box(f"/attached/{side}/box", dimensions=(0.30, 0.40, 0.40),
                             color=(239, 146, 62), opacity=0.76)
        server.scene.add_label(f"/attached/{side}/label", text=f"{side}附着箱",
                               position=(0, 0, 0.27))
    with server.gui.add_folder("Batched RRT结果"):
        slider = server.gui.add_slider("轨迹帧", min=1, max=len(frames), step=1,
                                       initial_value=collision_index + 1)
        start_button = server.gui.add_button("抽离起点")
        collision_button = server.gui.add_button("首次真实碰撞")
        end_button = server.gui.add_button("规划终点")
        status = server.gui.add_markdown("")
        server.gui.add_markdown(
            "球模型全程判定无碰撞；真实箱体OBB在红色19号箱处否决。"
        )

    def show(index):
        values = dict(zip(names, frames[index]))
        q = np.asarray([values.get(name, 0.0) for name in actuated])
        urdf.update_cfg(q)
        robot.update_cfg(q)
        centers = {}
        for side, node in attached.items():
            matrix = urdf.get_transform(f"{side}_tool0", urdf.base_link) @ offsets[side]
            node.position = matrix[:3, 3]
            node.wxyz = quaternion(matrix)
            centers[side] = matrix[:3, 3]
        status.content = (
            f"第 **{index + 1}/{len(frames)}** 帧  \n"
            f"{'**真实OBB首次碰撞**' if index == collision_index else '球模型认为合法'}  \n"
            f"左箱中心：`{centers['left'].round(3).tolist()}`"
        )

    @slider.on_update
    def on_slider(_event):
        show(int(slider.value) - 1)

    @start_button.on_click
    def on_start(_event):
        slider.value = 1

    @collision_button.on_click
    def on_collision(_event):
        slider.value = collision_index + 1

    @end_button.on_click
    def on_end(_event):
        slider.value = len(frames)

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-3.0, 3.4, 2.8)
        client.camera.look_at = (0.9, 0.0, 1.4)
        client.camera.up_direction = (0, 0, 1)

    show(collision_index)
    print(f"Ready at http://localhost:{args.port}; collision_frame={collision_index + 1}", flush=True)
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
