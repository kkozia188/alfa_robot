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


def matrix_quaternion(matrix):
    return np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--box-fit", type=Path, required=True)
    parser.add_argument("--named-poses", type=Path, required=True)
    parser.add_argument("--attachment-source", type=Path)
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    config_data = yaml.safe_load(args.config.read_text())
    config = config_data.get("robot_cfg", config_data)["kinematics"]
    box_fit = json.loads(args.box_fit.read_text())
    named = yaml.safe_load(args.named_poses.read_text())["named_poses"]["home"]
    if args.attachment_source:
        attachment = json.loads(args.attachment_source.read_text())["tool_to_box"]
    else:
        attachment = {
            "left": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.15], [0, 0, 0, 1]],
            "right": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.15], [0, 0, 0, 1]],
        }
    urdf_path = Path(config["urdf_path"])
    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=True, build_scene_graph=True)
    joint_values = np.array([named.get(name, 0.0) for name in urdf.actuated_joint_names])
    urdf.update_cfg(joint_values)

    server = viser.ViserServer(host="127.0.0.1", port=args.port,
                               label="V3.2.2 · 最终整机与双附着箱碰撞模型")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=5.0, height=4.0)
    server.scene.add_frame("/robot", show_axes=False)
    robot = ViserUrdf(server, urdf, root_node_name="/robot",
                      mesh_color_override=(0.72, 0.76, 0.82, 0.72))
    robot.update_cfg(joint_values)

    sphere_root = server.scene.add_frame("/robot_collision_spheres", show_axes=False)
    link_frames = {}
    sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    for link_name, spheres in config["collision_spheres"].items():
        matrix = urdf.get_transform(link_name, urdf.base_link)
        frame = server.scene.add_frame(
            f"/robot_collision_spheres/{link_name}",
            position=matrix[:3, 3], wxyz=matrix_quaternion(matrix), show_axes=False,
        )
        link_frames[link_name] = frame
        centers = np.asarray([sphere["center"] for sphere in spheres])
        radii = np.asarray([sphere["radius"] for sphere in spheres])
        quaternions = np.zeros((len(spheres), 4))
        quaternions[:, 0] = 1.0
        server.scene.add_batched_meshes_simple(
            f"/robot_collision_spheres/{link_name}/spheres",
            vertices=sphere_mesh.vertices, faces=sphere_mesh.faces,
            batched_positions=centers, batched_wxyzs=quaternions,
            batched_scales=radii, batched_colors=(71, 171, 210), opacity=0.57,
        )

    payload_root = server.scene.add_frame("/attached_boxes", show_axes=False)
    box_centers = np.asarray(box_fit["centers"])
    box_radii = np.asarray(box_fit["radii"])
    box_quaternions = np.zeros((len(box_centers), 4))
    box_quaternions[:, 0] = 1.0
    for side, color in (("left", (225, 135, 70)), ("right", (214, 107, 72))):
        tool = urdf.get_transform(f"{side}_tool0", urdf.base_link)
        box_transform = tool @ np.asarray(attachment[side])
        server.scene.add_frame(
            f"/attached_boxes/{side}", position=box_transform[:3, 3],
            wxyz=matrix_quaternion(box_transform), show_axes=True,
            axes_length=0.09, axes_radius=0.003,
        )
        server.scene.add_box(
            f"/attached_boxes/{side}/mesh", dimensions=(0.30, 0.40, 0.40),
            color=color, opacity=0.48,
        )
        server.scene.add_batched_meshes_simple(
            f"/attached_boxes/{side}/collision_spheres",
            vertices=sphere_mesh.vertices, faces=sphere_mesh.faces,
            batched_positions=box_centers, batched_wxyzs=box_quaternions,
            batched_scales=box_radii, batched_colors=(226, 76, 113), opacity=0.62,
        )

    robot_sphere_count = sum(len(rows) for rows in config["collision_spheres"].values())
    with server.gui.add_folder("最终碰撞模型"):
        show_mesh = server.gui.add_checkbox("显示机器人实体", initial_value=True)
        show_robot_spheres = server.gui.add_checkbox("显示机器人碰撞球", initial_value=True)
        show_boxes = server.gui.add_checkbox("显示附着箱实体与碰撞球", initial_value=True)
        fit_view = server.gui.add_button("正面最佳视角")
        server.gui.add_markdown(
            f"机器人：**{robot_sphere_count}球**  \n"
            f"BaseLink：**{len(config['collision_spheres']['base_link'])}球**  \n"
            f"每个箱体：**{len(box_centers)}球**  \n"
            f"箱体覆盖：**{box_fit['metrics']['coverage'] * 100:.2f}%**  \n"
            f"箱体外凸P95：**{box_fit['metrics']['protrusion_p95_mm']:.2f}mm**"
        )

    @show_mesh.on_update
    def on_mesh(_event):
        robot.show_visual = show_mesh.value

    @show_robot_spheres.on_update
    def on_robot_spheres(_event):
        sphere_root.visible = show_robot_spheres.value

    @show_boxes.on_update
    def on_boxes(_event):
        payload_root.visible = show_boxes.value

    def set_camera(client):
        client.camera.position = (-4.2, 0.0, 2.4)
        client.camera.look_at = (0.35, 0.0, 1.05)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    @fit_view.on_click
    def on_fit_view(_event):
        for client in server.get_clients().values():
            set_camera(client)

    @server.on_client_connect
    def on_connect(client):
        set_camera(client)

    for client in server.get_clients().values():
        set_camera(client)

    print(f"Ready at http://localhost:{args.port}; robot={robot_sphere_count} box={len(box_centers)}x2", flush=True)
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
