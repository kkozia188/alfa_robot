#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser
from viser.extras import ViserUrdf
import yaml
import yourdfpy


def quaternion(matrix):
    return np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--front-x", type=float, default=0.5080520510673523)
    parser.add_argument("--wall-distance", type=float, default=0.9)
    parser.add_argument("--port", type=int, default=8084)
    args = parser.parse_args()

    report = json.loads(args.result.read_text())
    config_data = yaml.safe_load(args.config.read_text())
    config = config_data.get("robot_cfg", config_data)["kinematics"]
    frames = np.asarray(report["frames"], dtype=float)
    names = config["cspace"]["joint_names"]
    urdf = yourdfpy.URDF.load(
        Path(config["urdf_path"]), load_meshes=True, build_scene_graph=True
    )
    actuated = list(urdf.actuated_joint_names)

    server = viser.ViserServer(
        host="127.0.0.1", port=args.port,
        label="V3.2.2 · 规范侧吸35cm抽离",
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground_grid", width=5.0, height=3.0)
    server.scene.add_frame("/robot", show_axes=False)
    robot = ViserUrdf(
        server, urdf, root_node_name="/robot",
        mesh_color_override=(0.68, 0.73, 0.80, 0.72),
    )

    sphere_root = server.scene.add_frame(
        "/robot_collision_spheres", show_axes=False, visible=False
    )
    sphere_frames = {}
    sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    for link_name, spheres in config["collision_spheres"].items():
        sphere_frames[link_name] = server.scene.add_frame(
            f"/robot_collision_spheres/{link_name}", show_axes=False
        )
        centers = np.asarray([sphere["center"] for sphere in spheres])
        radii = np.asarray([sphere["radius"] for sphere in spheres])
        rotations = np.zeros((len(spheres), 4))
        rotations[:, 0] = 1.0
        server.scene.add_batched_meshes_simple(
            f"/robot_collision_spheres/{link_name}/spheres",
            vertices=sphere_mesh.vertices, faces=sphere_mesh.faces,
            batched_positions=centers, batched_wxyzs=rotations,
            batched_scales=radii, batched_colors=(73, 169, 207), opacity=0.52,
        )

    wall_root = server.scene.add_frame("/wall", show_axes=False)
    wall_front = args.front_x + args.wall_distance
    for box_id in range(25):
        if box_id in (20, 24):
            continue
        server.scene.add_box(
            f"/wall/box_{box_id:02d}",
            position=(wall_front + 0.15, (box_id % 5 - 2) * 0.41,
                      0.20 + (box_id // 5) * 0.41),
            dimensions=(0.30, 0.40, 0.40), color=(87, 145, 165), opacity=0.18,
        )
    for side, box_id in (("left", 24), ("right", 20)):
        server.scene.add_box(
            f"/wall/original_{side}",
            position=(wall_front + 0.15, (box_id % 5 - 2) * 0.41,
                      0.20 + (box_id // 5) * 0.41),
            dimensions=(0.30, 0.40, 0.40), color=(70, 190, 204), opacity=0.10,
        )

    attached_frames = {}
    tool_to_box = {side: np.asarray(matrix) for side, matrix in report["tool_to_box"].items()}
    for side in ("left", "right"):
        attached_frames[side] = server.scene.add_frame(
            f"/attached/{side}", show_axes=True, axes_length=0.09, axes_radius=0.003
        )
        server.scene.add_box(
            f"/attached/{side}/box", dimensions=(0.30, 0.40, 0.40),
            color=(239, 146, 62), opacity=0.75,
        )
        server.scene.add_label(
            f"/attached/{side}/label", text=f"{side}附着箱",
            position=(0.0, 0.0, 0.27),
        )

    with server.gui.add_folder("规范抽离"):
        frame_slider = server.gui.add_slider(
            "轨迹帧", min=1, max=len(frames), step=1, initial_value=len(frames)
        )
        contact = server.gui.add_button("接触起点")
        extracted = server.gui.add_button("35cm抽离终点")
        play = server.gui.add_checkbox("播放", initial_value=False)
        show_spheres = server.gui.add_checkbox("显示机器人碰撞球", initial_value=False)
        status = server.gui.add_markdown("")
        server.gui.add_markdown(
            "青色透明箱为原箱位，橙色为随Tool0刚性附着的箱体。"
        )

    lock = threading.Lock()

    def show_frame(index):
        with lock, server.atomic():
            values = dict(zip(names, frames[index]))
            urdf.update_cfg(np.asarray([values.get(name, 0.0) for name in actuated]))
            robot.update_cfg(np.asarray([values.get(name, 0.0) for name in actuated]))
            for link_name, node in sphere_frames.items():
                matrix = urdf.get_transform(link_name, urdf.base_link)
                node.position = matrix[:3, 3]
                node.wxyz = quaternion(matrix)
            centers = {}
            for side, node in attached_frames.items():
                matrix = urdf.get_transform(f"{side}_tool0", urdf.base_link) @ tool_to_box[side]
                node.position = matrix[:3, 3]
                node.wxyz = quaternion(matrix)
                centers[side] = matrix[:3, 3]
            progress = report["extract"]["constraint_metrics"]["left_tool0"]["progress_m"][index]
            status.content = (
                f"第 **{index + 1}/{len(frames)}** 帧  \n"
                f"抽离进度：**{progress * 100:.2f}cm**  \n"
                f"左箱中心：`{centers['left'].round(3).tolist()}`  \n"
                f"右箱中心：`{centers['right'].round(3).tolist()}`"
            )

    @frame_slider.on_update
    def on_frame(_event):
        show_frame(int(frame_slider.value) - 1)

    @contact.on_click
    def on_contact(_event):
        play.value = False
        frame_slider.value = 1

    @extracted.on_click
    def on_extracted(_event):
        play.value = False
        frame_slider.value = len(frames)

    @show_spheres.on_update
    def on_spheres(_event):
        sphere_root.visible = show_spheres.value

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-3.2, 3.4, 2.8)
        client.camera.look_at = (1.0, 0.0, 1.45)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    show_frame(len(frames) - 1)
    print(f"Ready at http://localhost:{args.port}; frames={len(frames)}", flush=True)
    while True:
        if play.value:
            frame_slider.value = int(frame_slider.value) % len(frames) + 1
            time.sleep(0.04)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
