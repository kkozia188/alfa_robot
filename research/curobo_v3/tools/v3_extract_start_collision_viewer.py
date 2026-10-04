#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser
from viser.extras import ViserUrdf
import yaml
import yourdfpy


JOINT_NAMES = [
    "updown",
    *[f"left_joint{index}" for index in range(1, 8)],
    *[f"right_joint{index}" for index in range(1, 8)],
]


def quaternion(matrix):
    return np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--front-x", type=float, default=0.5080520510673523)
    parser.add_argument("--wall-distance", type=float, default=0.9)
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()

    config_data = yaml.safe_load(args.config.read_text())
    config = config_data.get("robot_cfg", config_data)["kinematics"]
    source = json.loads(args.source.read_text())
    audit = json.loads(args.audit.read_text())
    values = source["frames"][source["boundaries"]["cartesian_extract_35cm"]]
    state = dict(zip(source["joint_names"], values))
    urdf = yourdfpy.URDF.load(
        Path(config["urdf_path"]), load_meshes=True, build_scene_graph=True
    )
    actuated = list(urdf.actuated_joint_names)
    joint_values = np.asarray([state.get(name, 0.0) for name in actuated])
    urdf.update_cfg(joint_values)

    server = viser.ViserServer(
        host="127.0.0.1", port=args.port,
        label="V3.2.2 · 历史抽离起点碰撞诊断",
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground_grid", width=5.0, height=3.0)
    server.scene.add_frame("/robot", show_axes=False)
    robot = ViserUrdf(
        server, urdf, root_node_name="/robot",
        mesh_color_override=(0.68, 0.73, 0.80, 0.72),
    )
    robot.update_cfg(joint_values)

    sphere_root = server.scene.add_frame(
        "/robot_collision_spheres", show_axes=False, visible=False
    )
    sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    for link_name, spheres in config["collision_spheres"].items():
        matrix = urdf.get_transform(link_name, urdf.base_link)
        server.scene.add_frame(
            f"/robot_collision_spheres/{link_name}",
            position=matrix[:3, 3], wxyz=quaternion(matrix), show_axes=False,
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
    removed = {20, 24}
    start_audit = audit.get("states", {}).get("start", audit.get("start", {}))
    hit_by_side = start_audit["obb_hits"] if "obb_hits" in start_audit else start_audit["obb_wall_hits"]
    hit_ids = set(hit_by_side["left"]) | set(hit_by_side["right"])
    wall_front = args.front_x + args.wall_distance
    wall_centers = {}
    for box_id in range(25):
        if box_id in removed:
            continue
        center = np.array([
            wall_front + 0.15,
            (box_id % 5 - 2) * 0.41,
            0.20 + (box_id // 5) * 0.41,
        ])
        wall_centers[box_id] = center
        hit = box_id in hit_ids
        server.scene.add_box(
            f"/wall/box_{box_id:02d}", position=center,
            dimensions=(0.30, 0.40, 0.40),
            color=(220, 48, 52) if hit else (86, 145, 165),
            opacity=0.90 if hit else 0.18,
        )
        if hit:
            server.scene.add_label(
                f"/wall/box_{box_id:02d}/label", text=f"碰撞箱 {box_id}",
                position=center + np.array([0.0, 0.0, 0.28]),
            )

    attached_root = server.scene.add_frame("/attached", show_axes=False)
    attached_centers = {}
    for side, color in (("left", (239, 146, 62)), ("right", (239, 146, 62))):
        tool = urdf.get_transform(f"{side}_tool0", urdf.base_link)
        old_tool_to_box = np.asarray(source["tool_to_box"][side], dtype=float)
        tool_offset_correction = np.eye(4)
        tool_offset_correction[2, 3] = -(0.151 - 0.13585)
        box_transform = tool @ tool_offset_correction @ old_tool_to_box
        attached_centers[side] = box_transform[:3, 3]
        server.scene.add_frame(
            f"/attached/{side}", position=box_transform[:3, 3],
            wxyz=quaternion(box_transform), show_axes=True,
            axes_length=0.10, axes_radius=0.003,
        )
        server.scene.add_box(
            f"/attached/{side}/box", dimensions=(0.30, 0.40, 0.40),
            color=color, opacity=0.72,
        )
        server.scene.add_label(
            f"/attached/{side}/label", text=f"{side}附着箱",
            position=(0.0, 0.0, 0.28),
        )

    container_root = server.scene.add_frame("/container", show_axes=False)
    wall_back = wall_front + 0.30 + 1e-6
    for name, center, size in (
        ("left_wall", [wall_back - 2.0, -1.25, 1.2], [4.0, 0.10, 2.4]),
        ("right_wall", [wall_back - 2.0, 1.25, 1.2], [4.0, 0.10, 2.4]),
        ("front_wall", [wall_back + 0.05, 0.0, 1.2], [0.10, 2.6, 2.4]),
        ("ceiling", [wall_back - 2.0, 0.0, 2.45], [4.2, 2.6, 0.10]),
    ):
        server.scene.add_box(
            f"/container/{name}", position=center, dimensions=size,
            color=(149, 160, 174), opacity=0.06,
        )

    with server.gui.add_folder("碰撞诊断"):
        show_spheres = server.gui.add_checkbox("显示机器人碰撞球", initial_value=False)
        show_wall = server.gui.add_checkbox("显示其他箱墙", initial_value=True)
        show_container = server.gui.add_checkbox("显示集装箱边界", initial_value=True)
        perspective = server.gui.add_button("斜视角")
        top = server.gui.add_button("俯视碰撞间隙")
        server.gui.add_markdown(
            f"**历史V3.1.1 cuRobo数值IK/约束TrajOpt抽离状态，不可执行**  \n"
            f"左附着箱中心：`{attached_centers['left'].round(3).tolist()}`  \n"
            f"右附着箱中心：`{attached_centers['right'].round(3).tolist()}`  \n"
            f"左箱碰撞：**{hit_by_side['left']}**；右箱碰撞：**{hit_by_side['right']}**  \n"
            "同一旧关节状态在V3.2.2下产生约90°末端滚转差。  \n"
            "红色为被撞箱体，橙色为附着箱。"
        )

    @show_spheres.on_update
    def on_spheres(_event):
        sphere_root.visible = show_spheres.value

    @show_wall.on_update
    def on_wall(_event):
        wall_root.visible = show_wall.value

    @show_container.on_update
    def on_container(_event):
        container_root.visible = show_container.value

    def set_perspective(client):
        client.camera.position = (-3.2, 3.4, 2.8)
        client.camera.look_at = (1.15, 0.0, 1.55)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    def set_top(client):
        client.camera.position = (1.1, 0.0, 5.2)
        client.camera.look_at = (1.25, 0.0, 1.5)
        client.camera.up_direction = (1.0, 0.0, 0.0)

    @perspective.on_click
    def on_perspective(_event):
        for client in server.get_clients().values():
            set_perspective(client)

    @top.on_click
    def on_top(_event):
        for client in server.get_clients().values():
            set_top(client)

    @server.on_client_connect
    def on_connect(client):
        set_perspective(client)

    for client in server.get_clients().values():
        set_perspective(client)
    print(f"Ready at http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
