#!/usr/bin/env python3

import argparse
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser
import yaml
import yourdfpy


def explosion_offset(link_name: str) -> np.ndarray:
    if link_name == "arm_carriage":
        return np.array([0.0, 0.0, 0.55])
    if link_name == "head_yaw":
        return np.array([0.0, 0.0, 1.0])
    if link_name == "head":
        return np.array([0.0, 0.0, 1.35])
    for side, direction in (("left", 1.0), ("right", -1.0)):
        if link_name.startswith(f"{side}_link"):
            index = int(link_name.removeprefix(f"{side}_link"))
            return np.array([0.0, direction * (0.35 + 0.32 * index), 0.48 + 0.09 * index])
        if link_name == f"{side}_suction":
            return np.array([0.0, direction * 2.95, 1.2])
        if link_name == f"{side}_tool0":
            return np.array([0.0, direction * 3.2, 1.28])
    return np.zeros(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--named-poses", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    data = yaml.safe_load(args.config.read_text())
    config = data.get("robot_cfg", data)["kinematics"]
    named_poses = yaml.safe_load(args.named_poses.read_text())["named_poses"]
    home = named_poses["home"]
    urdf_path = Path(config["urdf_path"])
    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=True, build_scene_graph=True)
    locked_joints = config.get("lock_joints") or {}
    joint_values = {
        name: float(home.get(name, locked_joints.get(name, 0.0)))
        for name in urdf.actuated_joint_names
    }
    urdf.update_cfg(np.array([joint_values[name] for name in urdf.actuated_joint_names]))

    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="ALFA V3.2.2 · cuRobo碰撞爆炸图")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=9.0, height=7.0)
    link_root = server.scene.add_frame("/exploded", show_axes=False)
    connector_root = server.scene.add_frame("/connectors", show_axes=False)
    sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

    link_frames = {}
    mesh_nodes = []
    sphere_nodes = {}
    label_nodes = {}
    link_names = list(urdf.link_map)
    for link_name in link_names:
        link_frames[link_name] = server.scene.add_frame(
            f"/exploded/{link_name}", show_axes=False
        )

    geometry_index = 0
    for geometry_node in urdf.scene.graph.nodes_geometry:
        link_name = urdf.scene.graph.transforms.parents[geometry_node]
        if link_name not in link_frames:
            continue
        local_transform, geometry_name = urdf.scene.graph.get(
            frame_to=geometry_node, frame_from=link_name
        )
        mesh = urdf.scene.geometry[geometry_name].copy()
        mesh.apply_transform(local_transform)
        mesh_nodes.append(server.scene.add_mesh_simple(
            f"/exploded/{link_name}/visual_{geometry_index:03d}",
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=(185, 194, 204),
            opacity=0.72,
        ))
        geometry_index += 1

    for link_name, spheres in config["collision_spheres"].items():
        if link_name not in link_frames or not spheres:
            continue
        positions = np.asarray([sphere["center"] for sphere in spheres], dtype=float)
        radii = np.asarray([sphere["radius"] for sphere in spheres], dtype=float)
        quaternions = np.zeros((len(spheres), 4), dtype=float)
        quaternions[:, 0] = 1.0
        sphere_nodes[link_name] = server.scene.add_batched_meshes_simple(
            f"/exploded/{link_name}/collision_spheres",
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            batched_positions=positions,
            batched_wxyzs=quaternions,
            batched_scales=radii,
            batched_colors=(70, 178, 215),
            opacity=0.65,
        )
        label_nodes[link_name] = server.scene.add_label(
            f"/exploded/{link_name}/label",
            text=f"{link_name} · {len(spheres)}球",
            position=(0.0, 0.0, 0.12),
            visible=False,
        )

    connector_handles = {}
    for joint_name, joint in urdf.joint_map.items():
        connector_handles[joint_name] = server.scene.add_line_segments(
            f"/connectors/{joint_name}",
            points=np.zeros((1, 2, 3)),
            colors=(94, 107, 122),
            line_width=1.5,
        )

    with server.gui.add_folder("爆炸图"):
        amount = server.gui.add_slider("爆炸程度", min=0.0, max=1.5, step=0.05, initial_value=1.0)
        show_mesh = server.gui.add_checkbox("显示半透明实体", initial_value=True)
        show_spheres = server.gui.add_checkbox("显示碰撞球", initial_value=True)
        show_connectors = server.gui.add_checkbox("显示关节连线", initial_value=True)
        show_labels = server.gui.add_checkbox("显示链接名称", initial_value=False)
        fit_view = server.gui.add_button("最佳视角")
        reset = server.gui.add_button("恢复默认爆炸图")
        server.gui.add_markdown(
            "**V3.2.2 · Home姿态**  \n"
            "保持289球拟合不变，仅改变显示位置。左臂向+Y、右臂向-Y，升降与头部向+Z展开。"
        )

    update_lock = threading.Lock()

    def link_transform(link_name: str, scale: float) -> tuple[np.ndarray, np.ndarray]:
        matrix = urdf.get_transform(link_name, urdf.base_link)
        position = matrix[:3, 3] + explosion_offset(link_name) * scale
        quaternion = np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)
        return position, quaternion

    def update_scene() -> None:
        with update_lock, server.atomic():
            scale = float(amount.value)
            positions = {}
            for link_name, frame in link_frames.items():
                position, quaternion = link_transform(link_name, scale)
                positions[link_name] = position
                frame.position = position
                frame.wxyz = quaternion
            for joint_name, joint in urdf.joint_map.items():
                if joint.parent in positions and joint.child in positions:
                    connector_handles[joint_name].points = np.asarray(
                        [[positions[joint.parent], positions[joint.child]]]
                    )

    def set_camera(client) -> None:
        client.camera.position = (-7.8, 0.0, 3.8)
        client.camera.look_at = (0.2, 0.0, 1.55)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    @amount.on_update
    def on_amount(_event):
        update_scene()

    @show_mesh.on_update
    def on_mesh(_event):
        for node in mesh_nodes:
            node.visible = show_mesh.value

    @show_spheres.on_update
    def on_spheres(_event):
        for node in sphere_nodes.values():
            node.visible = show_spheres.value

    @show_connectors.on_update
    def on_connectors(_event):
        connector_root.visible = show_connectors.value

    @show_labels.on_update
    def on_labels(_event):
        for node in label_nodes.values():
            node.visible = show_labels.value

    @reset.on_click
    def on_reset(_event):
        amount.value = 1.0

    @fit_view.on_click
    def on_fit_view(_event):
        for client in server.get_clients().values():
            set_camera(client)

    @server.on_client_connect
    def on_connect(client):
        set_camera(client)

    update_scene()
    for client in server.get_clients().values():
        set_camera(client)
    print(f"Ready at http://localhost:{args.port}; links={len(link_frames)} spheres={sum(len(v) for v in config['collision_spheres'].values())}", flush=True)
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
