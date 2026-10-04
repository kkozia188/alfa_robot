#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np
import torch
import trimesh
import viser
import yaml
import yourdfpy

from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo._src.geom.sphere_fit.metrics import compute_sphere_fit_metrics


METHODS = (
    ("morphit", SphereFitType.MORPHIT, (226, 84, 145)),
    ("voxel", SphereFitType.VOXEL, (59, 177, 109)),
    ("surface", SphereFitType.SURFACE, (62, 137, 210)),
)


def link_mesh(urdf: yourdfpy.URDF, link_name: str) -> trimesh.Trimesh:
    meshes = []
    for geometry_node in urdf.scene.graph.nodes_geometry:
        if urdf.scene.graph.transforms.parents[geometry_node] != link_name:
            continue
        transform, geometry_name = urdf.scene.graph.get(
            frame_to=geometry_node, frame_from=link_name
        )
        mesh = urdf.scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"no visual mesh for {link_name}")
    return trimesh.util.concatenate(meshes)


def add_mesh(server, name, mesh, offset, color, opacity):
    return server.scene.add_mesh_simple(
        name,
        vertices=mesh.vertices,
        faces=mesh.faces,
        color=color,
        opacity=opacity,
        position=offset,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    config_data = yaml.safe_load(args.config.read_text())
    config = config_data.get("robot_cfg", config_data)["kinematics"]
    urdf = yourdfpy.URDF.load(
        Path(config["urdf_path"]), load_meshes=True, build_scene_graph=True
    )
    meshes = {
        "base_link": link_mesh(urdf, "base_link"),
        "box": trimesh.creation.box(extents=(0.30, 0.40, 0.40)),
    }
    offsets = {
        "base_link": np.array([0.0, -1.0, -meshes["base_link"].bounds[0, 2]]),
        "box": np.array([0.0, 1.15, 0.20]),
    }
    surface_radius = {"base_link": 0.020, "box": 0.015}
    saved = {}
    if args.output.exists():
        try:
            saved = json.loads(args.output.read_text())
        except (json.JSONDecodeError, OSError):
            saved = {}

    def saved_count(target, method, default):
        return int(saved.get(target, {}).get("methods", {}).get(method, {}).get("requested", default))

    server = viser.ViserServer(
        host="127.0.0.1", port=args.port, label="V3.2.2 · BaseLink与箱体组合球拟合"
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=5.0, height=4.0)
    add_mesh(server, "/base_link/mesh", meshes["base_link"], offsets["base_link"],
             (172, 182, 194), 0.70)
    add_mesh(server, "/box/mesh", meshes["box"], offsets["box"],
             (198, 153, 82), 0.62)
    server.scene.add_label(
        "/base_link/title", text="BaseLink", position=offsets["base_link"] + np.array([0, 0, 1.0])
    )
    server.scene.add_label(
        "/box/title", text="标准箱体 0.30×0.40×0.40m",
        position=offsets["box"] + np.array([0, 0, 0.30])
    )

    sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    fit_handles = {target: {} for target in meshes}
    fit_cache = {}
    fit_results = {target: {} for target in meshes}
    fit_lock = threading.Lock()

    with server.gui.add_folder("BaseLink组合"):
        base_counts = {
            "morphit": server.gui.add_number("MorphIt球数", min=0, max=256, step=1,
                                                initial_value=saved_count("base_link", "morphit", 20)),
            "voxel": server.gui.add_number("VOXEL球数", min=0, max=256, step=1,
                                              initial_value=saved_count("base_link", "voxel", 20)),
            "surface": server.gui.add_number("SURFACE球数", min=0, max=256, step=1,
                                                initial_value=saved_count("base_link", "surface", 0)),
        }
        base_iterations = server.gui.add_number("MorphIt迭代次数", min=1, max=5000, step=10,
                                                  initial_value=200)
        base_coverage = server.gui.add_number("MorphIt覆盖权重", min=0.0, max=100000.0,
                                                step=10.0, initial_value=1000.0)
        base_protrusion = server.gui.add_number("MorphIt外凸权重", min=0.0, max=100000.0,
                                                  step=1.0, initial_value=10.0)
        base_status = server.gui.add_markdown("等待拟合")
    with server.gui.add_folder("箱体组合"):
        box_counts = {
            "morphit": server.gui.add_number("MorphIt球数", min=0, max=256, step=1,
                                                initial_value=saved_count("box", "morphit", 24)),
            "voxel": server.gui.add_number("VOXEL球数", min=0, max=256, step=1,
                                              initial_value=saved_count("box", "voxel", 24)),
            "surface": server.gui.add_number("SURFACE球数", min=0, max=256, step=1,
                                                initial_value=saved_count("box", "surface", 0)),
        }
        box_iterations = server.gui.add_number("MorphIt迭代次数", min=1, max=5000, step=10,
                                                 initial_value=200)
        box_coverage = server.gui.add_number("MorphIt覆盖权重", min=0.0, max=100000.0,
                                               step=10.0, initial_value=1000.0)
        box_protrusion = server.gui.add_number("MorphIt外凸权重", min=0.0, max=100000.0,
                                                 step=1.0, initial_value=10.0)
        box_status = server.gui.add_markdown("等待拟合")
    with server.gui.add_folder("显示与执行"):
        show_morphit = server.gui.add_checkbox("显示MorphIt（粉）", initial_value=True)
        show_voxel = server.gui.add_checkbox("显示VOXEL（绿）", initial_value=True)
        show_surface = server.gui.add_checkbox("显示SURFACE（蓝）", initial_value=True)
        refit = server.gui.add_button("按当前球数重新组合拟合")
        fit_state = server.gui.add_markdown("准备中")
        server.gui.add_markdown(
            "球数设为 **0** 即停用该方法。MorphIt与VOXEL拟合实体内部，"
            "SURFACE使用固定小球覆盖表面；组合指标对所有已启用球统一计算。"
        )

    def render_method(target, method, centers, radii, color):
        old = fit_handles[target].get(method)
        if old is not None:
            old.remove()
        if len(centers) == 0:
            fit_handles[target][method] = None
            return
        quaternions = np.zeros((len(centers), 4))
        quaternions[:, 0] = 1.0
        fit_handles[target][method] = server.scene.add_batched_meshes_simple(
            f"/{target}/fits/{method}",
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            batched_positions=centers + offsets[target],
            batched_wxyzs=quaternions,
            batched_scales=radii,
            batched_colors=color,
            opacity=0.60,
        )

    def fit_one(target, method_name, fit_type, count, morphit_params):
        key = (target, method_name, count, *morphit_params)
        if key not in fit_cache:
            torch.manual_seed(20260929 + count)
            np.random.seed(20260929 + count)
            fit_cache[key] = fit_spheres_to_mesh(
                meshes[target],
                num_spheres=count,
                surface_radius=surface_radius[target],
                fit_type=fit_type,
                iterations=int(morphit_params[0]),
                coverage_weight=float(morphit_params[1]),
                protrusion_weight=float(morphit_params[2]),
                compute_metrics=False,
            )
        result = fit_cache[key]
        return (
            result.centers.detach().cpu().numpy(),
            result.radii.detach().cpu().numpy().reshape(-1),
            float(result.fit_time_s or 0.0),
            result.debug_info,
        )

    def target_report(target):
        centers = [value["centers"] for value in fit_results[target].values() if len(value["centers"])]
        radii = [value["radii"] for value in fit_results[target].values() if len(value["radii"])]
        if not centers:
            return {"num_spheres": 0}
        all_centers = np.concatenate(centers, axis=0)
        all_radii = np.concatenate(radii, axis=0)
        metrics = compute_sphere_fit_metrics(
            meshes[target], all_centers, all_radii,
            n_interior=5000, n_surface=4000, n_sphere_surface=120,
        )
        return {
            "num_spheres": int(len(all_centers)),
            "coverage": metrics.coverage,
            "protrusion": metrics.protrusion,
            "protrusion_p95_mm": metrics.protrusion_dist_p95 * 1000.0,
            "surface_gap_p95_mm": metrics.surface_gap_p95 * 1000.0,
            "max_gap_mm": metrics.max_uncovered_gap * 1000.0,
            "methods": {
                name: {
                    "requested": value["requested"],
                    "actual": int(len(value["centers"])),
                    "fit_time_s": value["fit_time_s"],
                    "iterations": value["iterations"],
                    "coverage_weight": value["coverage_weight"],
                    "protrusion_weight": value["protrusion_weight"],
                    "debug_info": value["debug_info"],
                }
                for name, value in fit_results[target].items()
            },
        }

    def status_text(report):
        if report.get("num_spheres", 0) == 0:
            return "未启用拟合方法"
        return (
            f"**组合 {report['num_spheres']} 球**  \n"
            f"体积采样覆盖：**{report['coverage'] * 100:.2f}%**  \n"
            f"球面外凸占比：**{report['protrusion'] * 100:.2f}%**  \n"
            f"外凸P95：**{report['protrusion_p95_mm']:.2f}mm**  \n"
            f"表面缺口P95：**{report['surface_gap_p95_mm']:.2f}mm**  \n"
            f"最大缺口：**{report['max_gap_mm']:.2f}mm**"
        )

    def run_fit():
        if not fit_lock.acquire(blocking=False):
            return
        try:
            fit_state.content = "正在拟合并计算组合指标..."
            controls = {"base_link": base_counts, "box": box_counts}
            morphit_controls = {
                "base_link": (base_iterations, base_coverage, base_protrusion),
                "box": (box_iterations, box_coverage, box_protrusion),
            }
            for target in ("base_link", "box"):
                fit_results[target] = {}
                morphit_params = tuple(control.value for control in morphit_controls[target])
                for method_name, fit_type, color in METHODS:
                    count = int(controls[target][method_name].value)
                    if count <= 0:
                        render_method(target, method_name, np.empty((0, 3)), np.empty(0), color)
                        continue
                    centers, radii, fit_time_s, debug_info = fit_one(
                        target, method_name, fit_type, count, morphit_params
                    )
                    fit_results[target][method_name] = {
                        "requested": count,
                        "centers": centers,
                        "radii": radii,
                        "fit_time_s": fit_time_s,
                        "iterations": int(morphit_params[0]),
                        "coverage_weight": float(morphit_params[1]),
                        "protrusion_weight": float(morphit_params[2]),
                        "debug_info": debug_info,
                    }
                    render_method(target, method_name, centers, radii, color)
            reports = {target: target_report(target) for target in meshes}
            base_status.content = status_text(reports["base_link"])
            box_status.content = status_text(reports["box"])
            fit_state.content = "组合拟合完成"
            args.output.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                "description_commit": "bd1e455",
                "base_link": reports["base_link"],
                "box": reports["box"],
            }
            args.output.write_text(json.dumps(serializable, indent=2) + "\n")
        except Exception as error:
            fit_state.content = f"拟合失败：`{type(error).__name__}: {error}`"
        finally:
            fit_lock.release()

    @refit.on_click
    def on_refit(_event):
        threading.Thread(target=run_fit, daemon=True).start()

    def update_visibility(method, visible):
        for target in fit_handles:
            node = fit_handles[target].get(method)
            if node is not None:
                node.visible = visible

    @show_morphit.on_update
    def on_morphit(_event):
        update_visibility("morphit", show_morphit.value)

    @show_voxel.on_update
    def on_voxel(_event):
        update_visibility("voxel", show_voxel.value)

    @show_surface.on_update
    def on_surface(_event):
        update_visibility("surface", show_surface.value)

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-3.8, 3.8, 2.8)
        client.camera.look_at = (0.0, 0.15, 0.45)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    threading.Thread(target=run_fit, daemon=True).start()
    print(f"Ready at http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
