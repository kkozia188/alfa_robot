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

from v3_batched_loaded_search import loaded_robot
from v3_wall_ik_benchmark import ACTIVE_JOINTS, canonical_side_tool_to_box


def quaternion(matrix: np.ndarray) -> np.ndarray:
    return np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--box-fit", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()

    robot_config = yaml.safe_load(args.config.read_text())
    box_fit = json.loads(args.box_fit.read_text())
    audit = json.loads(args.audit.read_text())
    loaded_config = loaded_robot(robot_config, box_fit, active_sides=("left", "right"))
    config = loaded_config.get("robot_cfg", loaded_config)["kinematics"]
    values = audit["best"]["joints"]
    state = dict(zip(ACTIVE_JOINTS, values))

    urdf = yourdfpy.URDF.load(
        Path(config["urdf_path"]), load_meshes=True, build_scene_graph=True
    )
    actuated = list(urdf.actuated_joint_names)
    joint_values = np.asarray([state.get(name, 0.0) for name in actuated])
    urdf.update_cfg(joint_values)

    server = viser.ViserServer(
        host="127.0.0.1",
        port=args.port,
        label="抽离终点 · 箱体与40球对齐检查",
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground_grid", width=5.0, height=3.0)
    robot = ViserUrdf(
        server,
        urdf,
        root_node_name="/robot",
        mesh_color_override=(0.68, 0.73, 0.80, 0.76),
    )
    robot.update_cfg(joint_values)

    sphere_root = server.scene.add_frame(
        "/collision_spheres", show_axes=False, visible=True
    )
    sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    for link_name, spheres in config["collision_spheres"].items():
        if not spheres:
            continue
        matrix = urdf.get_transform(link_name, urdf.base_link)
        server.scene.add_frame(
            f"/collision_spheres/{link_name}",
            position=matrix[:3, 3],
            wxyz=quaternion(matrix),
            show_axes=False,
        )
        centers = np.asarray([sphere["center"] for sphere in spheres])
        radii = np.asarray([sphere["radius"] for sphere in spheres])
        rotations = np.zeros((len(spheres), 4))
        rotations[:, 0] = 1.0
        server.scene.add_batched_meshes_simple(
            f"/collision_spheres/{link_name}/spheres",
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            batched_positions=centers,
            batched_wxyzs=rotations,
            batched_scales=radii,
            batched_colors=(69, 169, 207),
            opacity=0.48,
        )

    front_x = float(audit["front_x"])
    wall_distance = float(audit["wall_distance"])
    wall_front = front_x + wall_distance
    removed = set(audit["task"].values())
    wall_root = server.scene.add_frame("/wall", show_axes=False)
    for box_id in range(25):
        if box_id in removed:
            continue
        server.scene.add_box(
            f"/wall/box_{box_id:02d}",
            position=(
                wall_front + 0.15,
                (box_id % 5 - 2) * 0.41,
                0.20 + (box_id // 5) * 0.41,
            ),
            dimensions=(0.30, 0.40, 0.40),
            color=(86, 145, 165),
            opacity=0.20,
        )

    attached_root = server.scene.add_frame("/attached", show_axes=False)
    actual_up = {}
    for side in ("left", "right"):
        tool = urdf.get_transform(f"{side}_tool0", urdf.base_link)
        box_transform = tool @ canonical_side_tool_to_box(side)
        server.scene.add_frame(
            f"/attached/{side}",
            position=box_transform[:3, 3],
            wxyz=quaternion(box_transform),
            show_axes=True,
            axes_length=0.12,
            axes_radius=0.004,
        )
        server.scene.add_box(
            f"/attached/{side}/box",
            dimensions=(0.30, 0.40, 0.40),
            color=(239, 146, 62),
            opacity=0.25,
        )

        tool_rotation = tool[:3, :3]
        local_up = canonical_side_tool_to_box(side)[:3, 2]
        world_up = tool_rotation @ local_up
        actual_up[side] = world_up
        origin = box_transform[:3, 3]
        server.scene.add_line_segments(
            f"/stability/{side}_computed_up",
            points=np.asarray([[origin, origin + world_up * 0.38]]),
            colors=(245, 166, 35),
            thickness=0.018,
        )
        server.scene.add_line_segments(
            f"/stability/{side}_required_up",
            points=np.asarray([[origin, origin + np.array([0.0, 0.0, 0.38])]]),
            colors=(48, 190, 96),
            thickness=0.012,
        )
        server.scene.add_label(
            f"/stability/{side}_computed_label",
            text=f"{side}: 程序计算上方向",
            position=origin + world_up * 0.43,
        )

    with server.gui.add_folder("拒绝原因"):
        show_spheres = server.gui.add_checkbox("显示40球附着模型与机器人球", initial_value=True)
        show_wall = server.gui.add_checkbox("显示箱墙", initial_value=True)
        server.gui.add_markdown(
            "## 🟡 该候选没有碰撞\n"
            "当前展示：修正变换后重新审计的单个候选  \n"
            "最优候选自碰撞：**0** · 箱墙碰撞：**0** · 关节越限：**0**  \n"
            f"左侧计算上方向 Z：**{actual_up['left'][2]:.8f}**  \n"
            f"右侧计算上方向 Z：**{actual_up['right'][2]:.8f}**  \n"
            "门限要求：**Z ≥ 0.01745（倾斜不超过89°）**  \n"
            "黄色线是程序计算的箱体上方向，绿色线是世界 +Z。"
        )

    @show_spheres.on_update
    def on_spheres(_event):
        sphere_root.visible = show_spheres.value

    @show_wall.on_update
    def on_wall(_event):
        wall_root.visible = show_wall.value

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-2.8, 3.2, 2.8)
        client.camera.look_at = (0.9, 0.0, 1.45)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    print(f"Ready at http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
