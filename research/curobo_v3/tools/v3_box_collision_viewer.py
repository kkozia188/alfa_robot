#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import trimesh
import viser

from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh


def build_feature_spheres(
    dimensions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    half_extents = dimensions / 2.0
    np.random.seed(20261003)
    torch.manual_seed(20261003)
    voxel_result = fit_spheres_to_mesh(
        trimesh.creation.box(extents=dimensions),
        num_spheres=20,
        fit_type=SphereFitType.VOXEL,
        compute_metrics=True,
    )
    centers = voxel_result.centers.detach().cpu().numpy().tolist()
    radii = voxel_result.radii.detach().cpu().numpy().reshape(-1).tolist()
    groups = ["voxel"] * len(centers)

    shortest_half_extent = float(np.min(dimensions) / 2.0)
    edge_radius = shortest_half_extent * 0.14
    for free_axis in range(3):
        constrained_axes = [axis for axis in range(3) if axis != free_axis]
        for first_sign in (-1.0, 1.0):
            for second_sign in (-1.0, 1.0):
                center = np.zeros(3)
                center[constrained_axes[0]] = first_sign * (
                    half_extents[constrained_axes[0]] - edge_radius
                )
                center[constrained_axes[1]] = second_sign * (
                    half_extents[constrained_axes[1]] - edge_radius
                )
                centers.append(center)
                radii.append(edge_radius)
                groups.append("edge")

    corner_radius = shortest_half_extent * 0.09
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            for z_sign in (-1.0, 1.0):
                centers.append(
                    np.array(
                        [
                            x_sign * (half_extents[0] - corner_radius),
                            y_sign * (half_extents[1] - corner_radius),
                            z_sign * (half_extents[2] - corner_radius),
                        ]
                    )
                )
                radii.append(corner_radius)
                groups.append("corner")

    metrics = {
        "coverage": float(voxel_result.metrics.coverage),
        "surface_gap_p95_mm": float(voxel_result.metrics.surface_gap_p95 * 1000.0),
        "max_gap_mm": float(voxel_result.metrics.max_uncovered_gap * 1000.0),
    }
    return np.asarray(centers), np.asarray(radii), groups, metrics


def quaternion_matrix(wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(wxyz, dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def obb_overlap(
    position_a: np.ndarray,
    rotation_a: np.ndarray,
    position_b: np.ndarray,
    rotation_b: np.ndarray,
    half_extents: np.ndarray,
) -> tuple[bool, float, str]:
    relative_rotation = rotation_a.T @ rotation_b
    absolute_rotation = np.abs(relative_rotation) + 1e-9
    translation = rotation_a.T @ (position_b - position_a)
    minimum_overlap = np.inf
    minimum_axis = ""

    for axis in range(3):
        radius_a = half_extents[axis]
        radius_b = half_extents @ absolute_rotation[axis, :]
        overlap = radius_a + radius_b - abs(translation[axis])
        if overlap < 0.0:
            return False, -overlap, f"A{axis}"
        if overlap < minimum_overlap:
            minimum_overlap, minimum_axis = overlap, f"A{axis}"

    for axis in range(3):
        radius_a = half_extents @ absolute_rotation[:, axis]
        radius_b = half_extents[axis]
        overlap = radius_a + radius_b - abs(translation @ relative_rotation[:, axis])
        if overlap < 0.0:
            return False, -overlap, f"B{axis}"
        if overlap < minimum_overlap:
            minimum_overlap, minimum_axis = overlap, f"B{axis}"

    for axis_a in range(3):
        for axis_b in range(3):
            if abs(relative_rotation[axis_a, axis_b]) > 1.0 - 1e-8:
                continue
            radius_a = (
                half_extents[(axis_a + 1) % 3] * absolute_rotation[(axis_a + 2) % 3, axis_b]
                + half_extents[(axis_a + 2) % 3]
                * absolute_rotation[(axis_a + 1) % 3, axis_b]
            )
            radius_b = (
                half_extents[(axis_b + 1) % 3]
                * absolute_rotation[axis_a, (axis_b + 2) % 3]
                + half_extents[(axis_b + 2) % 3]
                * absolute_rotation[axis_a, (axis_b + 1) % 3]
            )
            projected = abs(
                translation[(axis_a + 2) % 3] * relative_rotation[(axis_a + 1) % 3, axis_b]
                - translation[(axis_a + 1) % 3] * relative_rotation[(axis_a + 2) % 3, axis_b]
            )
            overlap = radius_a + radius_b - projected
            if overlap < 0.0:
                return False, -overlap, f"A{axis_a}xB{axis_b}"
            if overlap < minimum_overlap:
                minimum_overlap, minimum_axis = overlap, f"A{axis_a}xB{axis_b}"

    return True, minimum_overlap, minimum_axis


def sphere_collision(
    position_a: np.ndarray,
    rotation_a: np.ndarray,
    position_b: np.ndarray,
    rotation_b: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> tuple[bool, float, tuple[int, int]]:
    world_a = centers @ rotation_a.T + position_a
    world_b = centers @ rotation_b.T + position_b
    deltas = world_a[:, None, :] - world_b[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    clearances = distances - radii[:, None] - radii[None, :]
    closest_flat = int(np.argmin(clearances))
    closest_pair = np.unravel_index(closest_flat, clearances.shape)
    minimum_clearance = float(clearances[closest_pair])
    return minimum_clearance <= 0.0, minimum_clearance, closest_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--box-fit", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    fit = json.loads(args.box_fit.read_text())
    dimensions = np.asarray(fit["dimensions_m"], dtype=float)
    centers, radii, sphere_groups, voxel_metrics = build_feature_spheres(dimensions)
    half_extents = dimensions / 2.0
    overflow_axis = np.maximum(np.abs(centers) + radii[:, None] - half_extents, 0.0)
    overflow = overflow_axis.max(axis=1)
    protruding = overflow > 1e-6
    sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

    server = viser.ViserServer(
        host="127.0.0.1",
        port=args.port,
        label="双箱体碰撞检查 · OBB 对比人工特征球",
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=1.6, height=1.6, cell_size=0.05)

    box_handles = []
    handle_offset = np.array([0.0, -0.34, 0.34])
    group_colors = {
        "voxel": (64, 170, 215),
        "edge": (245, 166, 35),
        "corner": (228, 63, 80),
    }
    for name, position, color in (
        ("box_a", (-0.24, 0.0, 0.24), (235, 145, 66)),
        ("box_b", (0.24, 0.0, 0.24), (58, 166, 120)),
    ):
        control_position = np.asarray(position) + handle_offset
        box_local_position = -handle_offset
        control = server.scene.add_transform_controls(
            f"/{name}",
            scale=0.18,
            position=control_position,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            translation_limits=((-0.8, 0.8), (-0.8, 0.8), (-0.2, 1.0)),
            rotation_limits=((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
        )
        real_geometry = server.scene.add_box(
            f"/{name}/real_geometry",
            dimensions=dimensions,
            color=color,
            opacity=0.28,
            position=box_local_position,
        )
        exact_collision_overlay = server.scene.add_box(
            f"/{name}/exact_collision",
            dimensions=dimensions * 1.005,
            color=(235, 45, 58),
            opacity=0.34,
            position=box_local_position,
            visible=False,
        )
        sphere_only_overlay = server.scene.add_box(
            f"/{name}/sphere_only_collision",
            dimensions=dimensions * 1.01,
            color=(245, 190, 48),
            opacity=0.26,
            position=box_local_position,
            visible=False,
        )
        sphere_handles = []
        for index, (center, radius, group) in enumerate(zip(centers, radii, sphere_groups)):
            sphere_handles.append(
                server.scene.add_mesh_simple(
                    f"/{name}/collision/sphere_{index:02d}",
                    vertices=sphere_mesh.vertices * radius + center + box_local_position,
                    faces=sphere_mesh.faces,
                    color=group_colors[group],
                    opacity=0.52,
                )
            )
        box_handles.append(
            {
                "control": control,
                "real": real_geometry,
                "exact_overlay": exact_collision_overlay,
                "sphere_overlay": sphere_only_overlay,
                "spheres": sphere_handles,
            }
        )

    with server.gui.add_folder("碰撞状态"):
        status = server.gui.add_markdown("正在计算……")
        server.gui.add_markdown(
            "**红色箱体**：真实 OBB 已碰撞  \n"
            "**黄色箱体**：只有球模型判为碰撞（误报）  \n"
            "箱体外侧上方的坐标轴是移动旋转手柄。"
        )
    with server.gui.add_folder("显示设置"):
        show_real = server.gui.add_checkbox("真实箱体", initial_value=True)
        show_spheres = server.gui.add_checkbox("VOXEL内部球 + 边角球", initial_value=True)
        show_controls = server.gui.add_checkbox("移动旋转手柄", initial_value=True)
        server.gui.add_markdown(
            f"尺寸：**{dimensions[0]:.2f} × {dimensions[1]:.2f} × {dimensions[2]:.2f}m**  \n"
            f"总球数：**{len(centers)}**（VOXEL内部20 / 边中点12 / 角点8）  \n"
            f"VOXEL球直径：**{2*radii[0]*1000:.1f}mm**，内部覆盖：**{voxel_metrics['coverage']*100:.2f}%**  \n"
            f"边/角球直径：**{2*radii[20]*1000:.1f} / {2*radii[32]*1000:.1f}mm**  \n"
            f"轴向外凸球：**{int(protruding.sum())}/{len(centers)}**"
        )

    @show_real.on_update
    def on_show_real(_event):
        for handles in box_handles:
            handles["real"].visible = show_real.value

    @show_spheres.on_update
    def on_show_spheres(_event):
        for handles in box_handles:
            for sphere in handles["spheres"]:
                sphere.visible = show_spheres.value

    @show_controls.on_update
    def on_show_controls(_event):
        for handles in box_handles:
            handles["control"].visible = show_controls.value

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (1.0, 0.9, 0.8)
        client.camera.look_at = (0.0, 0.0, 0.2)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    previous_poses = None
    print(f"Ready at http://localhost:{args.port}; drag both box gizmos", flush=True)
    while True:
        rotations = [
            quaternion_matrix(np.asarray(handles["control"].wxyz, dtype=float))
            for handles in box_handles
        ]
        positions = [
            np.asarray(handles["control"].position, dtype=float)
            + rotations[index] @ box_local_position
            for index, handles in enumerate(box_handles)
        ]
        poses = tuple(np.concatenate((positions[index], rotations[index].reshape(-1))) for index in range(2))
        if previous_poses is not None and all(
            np.allclose(current, previous, atol=1e-8)
            for current, previous in zip(poses, previous_poses)
        ):
            time.sleep(0.02)
            continue
        previous_poses = tuple(pose.copy() for pose in poses)

        exact_hit, exact_value, exact_axis = obb_overlap(
            positions[0], rotations[0], positions[1], rotations[1], half_extents
        )
        sphere_hit, sphere_clearance, sphere_pair = sphere_collision(
            positions[0], rotations[0], positions[1], rotations[1], centers, radii
        )

        if exact_hit and sphere_hit:
            verdict = "🔴 **真实碰撞，球模型也检测到**"
        elif exact_hit:
            verdict = "🟣 **球模型漏检：真实箱体已经碰撞**"
        elif sphere_hit:
            verdict = "🟡 **球模型误报：真实箱体仍有间隙**"
        else:
            verdict = "🟢 **完全分离**"

        for handles in box_handles:
            handles["exact_overlay"].visible = exact_hit
            handles["sphere_overlay"].visible = sphere_hit and not exact_hit

        exact_measure = (
            f"穿入近似下界 **{exact_value * 1000:.2f} mm**"
            if exact_hit
            else f"SAT 分离间隔 **{exact_value * 1000:.2f} mm**"
        )
        sphere_measure = (
            f"球体重叠 **{-sphere_clearance * 1000:.2f} mm**"
            if sphere_hit
            else f"最近球间隙 **{sphere_clearance * 1000:.2f} mm**"
        )
        status.content = (
            f"## {verdict}\n"
            f"真实 OBB：{exact_measure}，关键轴 `{exact_axis}`  \n"
            f"人工特征球：{sphere_measure}，最近球 `A{sphere_pair[0]:02d} ↔ B{sphere_pair[1]:02d}`"
        )
        time.sleep(0.02)


if __name__ == "__main__":
    main()
