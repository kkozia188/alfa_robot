#!/usr/bin/env python3

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh
import viser
from viser.extras import ViserUrdf
import yaml
import yourdfpy

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, JointState, Pose

from v3_batched_loaded_search import (
    GpuValidity, batched_rrt_multi_goal, densify, loaded_robot,
    batched_rrt_connect_multi_goal, batched_prm_multi_goal,
)
from v3_search_helpers import audit_state
from v3_wall_ik_benchmark import (
    ACTIVE_JOINTS, canonical_side_suction_quaternion_wxyz,
    canonical_side_tool_to_box, chassis_front_x, make_scene, wall_center,
)


DEFAULT_BOX_FIT = (
    Path(__file__).resolve().parents[1]
    / "generated/v3_suction_v322_6bb184b/frozen_collision_model/box_voxel_edge_corner_final.json"
)
DEFAULT_MOBILE_ROBOT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "generated/v3_analytic_071cb95/mobile18/alfa_v322_suction_mobile18.yml"
)
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_CONFIG = WORKSPACE_ROOT / "generated/v3_analytic_071cb95/alfa_v322_suction_final.yml"
DEFAULT_URDF = WORKSPACE_ROOT / "generated/v3_analytic_071cb95/model/robot_meter_meshes.urdf"
DEFAULT_NAMED_POSES = WORKSPACE_ROOT / "models/robot_description_analytic_071cb95/config/named_poses_analytic_proxy.yaml"
DEFAULT_TARGET_POSES = WORKSPACE_ROOT / "generated/current_carry_target_6d.json"


def tasks():
    output = []
    task_rows = [4, 3, 2]
    left_columns = [4, 3, 2]
    right_columns = [0, 1, 2]
    for left_row in task_rows:
        for right_row in task_rows:
            if abs(left_row - right_row) > 1:
                continue
            for left_column in left_columns:
                for right_column in right_columns:
                    left_box = left_row * 5 + left_column
                    right_box = right_row * 5 + right_column
                    if left_box == right_box:
                        continue
                    label = (
                        f"L{left_box:02d}(排{5-left_row}/列{5-left_column}) · "
                        f"R{right_box:02d}(排{5-right_row}/列{5-right_column})"
                    )
                    output.append((label, {"left": left_box, "right": right_box}))
    return output


def goal(poses):
    frames = ["left_tool0", "right_tool0"]
    return GoalToolPose.from_poses({
        frame: Pose(
            position=torch.tensor(poses[frame]["position"], device="cuda", dtype=torch.float32).reshape(1, 3),
            quaternion=torch.tensor(poses[frame]["quaternion"], device="cuda", dtype=torch.float32).reshape(1, 4),
        ) for frame in frames
    }, ordered_tool_frames=frames)


def pose_dict(pose):
    return {
        "position": pose.position.reshape(-1, 3)[0].detach().cpu().numpy(),
        "quaternion": pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy(),
    }


def quaternion(matrix):
    return np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)


from curobo_core.backend import CuroboBackend


class InteractivePlanner(CuroboBackend):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--mobile-robot-config", type=Path, default=DEFAULT_MOBILE_ROBOT_CONFIG)
    parser.add_argument("--box-fit", type=Path, default=DEFAULT_BOX_FIT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--named-poses", type=Path, default=DEFAULT_NAMED_POSES)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    planner = InteractivePlanner(args)
    mobile_config_data = yaml.safe_load(args.mobile_robot_config.read_text())
    mobile_config = mobile_config_data.get("robot_cfg", mobile_config_data)["kinematics"]
    mobile_names = mobile_config["cspace"]["joint_names"]
    urdf = yourdfpy.URDF.load(
        Path(mobile_config["urdf_path"]), load_meshes=True, build_scene_graph=True
    )
    actuated = list(urdf.actuated_joint_names)
    server = viser.ViserServer(
        host="127.0.0.1", port=args.port,
        label="V3.2.2 · 60组合实时规划",
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=5, height=3)
    robot = ViserUrdf(
        server, urdf, root_node_name="/robot",
        mesh_color_override=(0.68, 0.73, 0.80, 0.76),
    )
    wall_root = server.scene.add_frame("/wall", show_axes=False)
    attached = {}
    offsets = {side: canonical_side_tool_to_box(side) for side in ("left", "right")}
    for side in ("left", "right"):
        attached[side] = server.scene.add_frame(f"/attached/{side}", show_axes=False)
        server.scene.add_box(
            f"/attached/{side}/box", dimensions=(0.30, 0.40, 0.40),
            color=(239, 146, 62), opacity=0.78,
        )
    labels = [item[0] for item in planner.task_options]
    with server.gui.add_folder("60组合实时规划"):
        task_selector = server.gui.add_dropdown("组合任务", options=labels, initial_value=labels[0])
        random_button = server.gui.add_button("1. 随机抽离终点")
        plan_button = server.gui.add_button("2. 实时计算并播放", disabled=True)
        slider = server.gui.add_slider("轨迹帧", min=1, max=1, step=1, initial_value=1)
        play = server.gui.add_checkbox("播放", initial_value=False)
        inspect_failure = server.gui.add_button("定位首个失败帧", disabled=True)
        status = server.gui.add_markdown(
            "请选择任务并生成随机抽离终点。  \n"
            f"附着箱碰撞：**{planner.box_fit.get('name', args.box_fit.stem)} · "
            f"{planner.box_fit['actual_spheres']}球**"
        )
    current = {"frames": [], "transport_frames": 0, "wall_root": wall_root}
    collision_markers = server.scene.add_frame("/collision_markers", show_axes=False)
    sphere_mesh = trimesh.creation.icosphere(subdivisions=2)

    def show_failure_markers(index):
        nonlocal collision_markers
        collision_markers.remove()
        collision_markers = server.scene.add_frame("/collision_markers", show_axes=False)
        failure = planner.result.get("failure") if planner.result else None
        if failure is None or index != failure["frame_index"]:
            return
        audit = failure["audit"]
        hit_indices = {entry["sphere_index"] for entry in audit["world_collisions"]}
        for entry in audit["self_collisions"]:
            hit_indices.update(entry["sphere_indices"])
        for sphere_index in hit_indices:
            sphere = failure["spheres"][sphere_index]
            server.scene.add_mesh_simple(
                f"/collision_markers/sphere_{sphere_index}",
                vertices=sphere_mesh.vertices * sphere[3], faces=sphere_mesh.faces,
                position=sphere[:3], color=(245, 35, 45), opacity=0.9,
            )
        obstacles = {entry["obstacle"] for entry in audit["world_collisions"]}
        for obstacle in planner.scene(planner.selected_task).cuboid:
            if obstacle.name in obstacles:
                server.scene.add_box(
                    f"/collision_markers/{obstacle.name}", dimensions=obstacle.dims,
                    position=obstacle.pose[:3], wxyz=obstacle.pose[3:7],
                    color=(245, 35, 45), opacity=0.35,
                )

    def render_wall(boxes):
        current["wall_root"].remove()
        current["wall_root"] = server.scene.add_frame("/wall", show_axes=False)
        for obstacle in planner.scene(boxes).cuboid:
            if not obstacle.name.startswith("wall_box_"):
                server.scene.add_box(
                    f"/wall/{obstacle.name}", dimensions=obstacle.dims,
                    position=obstacle.pose[:3], wxyz=obstacle.pose[3:7],
                    color=(149, 160, 174), opacity=0.06,
                )
        wall_front = planner.front + 0.9
        for box_id in range(25):
            if box_id in boxes.values():
                continue
            server.scene.add_box(
                f"/wall/box_{box_id:02d}",
                position=(wall_front + 0.15, (box_id % 5 - 2) * 0.41,
                          0.20 + (box_id // 5) * 0.41),
                dimensions=(0.30, 0.40, 0.40), color=(87, 145, 165), opacity=0.18,
            )

    def show(values):
        mapping = dict(zip(mobile_names, values))
        q = np.asarray([mapping.get(name, 0.0) for name in actuated])
        urdf.update_cfg(q)
        robot.update_cfg(q)
        for side, node in attached.items():
            matrix = urdf.get_transform(f"{side}_tool0", urdf.base_link) @ offsets[side]
            node.position = matrix[:3, 3]
            node.wxyz = quaternion(matrix)

    def show_frame(index):
        if not current["frames"]:
            return
        show(current["frames"][index])
        show_failure_markers(index)
        if planner.result:
            phase = "搬运到新目标" if index < current["transport_frames"] else "Yaw 180°"
            status.content = (
                f"**{'轨迹有效' if planner.result['success'] else '诊断回放：存在失败帧'} · {phase}**  \n"
                f"Frame：**{index + 1}/{len(current['frames'])}**  \n"
                f"随机起点生成：**{planner.random_start['generation_ms']:.1f}ms**  \n"
                f"目标IK：**{planner.result['target_ik_ms']:.1f}ms** · "
                f"搜索：**{planner.result['search_ms']:.1f}ms**  \n"
                f"本次总计算：**{planner.result['total_ms']:.1f}ms** · "
                f"最大箱体倾角：**{planner.result['max_box_tilt_deg']:.1f}°**"
            )
            failure = planner.result.get("failure")
            if failure:
                audit = failure["audit"]
                pairs = [
                    f"{entry['link']} ↔ {entry['obstacle']}：穿入{entry['penetration_mm']:.2f}mm"
                    for entry in audit["world_collisions"]
                ] + [
                    f"{' ↔ '.join(entry['links'])}：穿入{entry['penetration_mm']:.2f}mm"
                    for entry in audit["self_collisions"]
                ]
                status.content += (
                    f"  \n首个失败：**{failure['stage']} · frame {failure['frame_index']+1} · "
                    f"Yaw {failure['yaw_deg']:.1f}°**  \n"
                    + "  \n".join(pairs)
                )

    def start_job(action):
        if planner.busy:
            return
        planner.busy = True
        random_button.disabled = True
        plan_button.disabled = True
        play.value = False

        def worker():
            try:
                boxes = planner.selected_task
                if action == "random":
                    status.content = "**正在实时生成随机抽离终点...**"
                    planner.random_start = planner.random_extract_start(boxes)
                    planner.result = None
                    inspect_failure.disabled = True
                    values15 = planner.random_start["values"]
                    mapping = dict(zip(ACTIVE_JOINTS, values15))
                    values18 = [mapping.get(name, 0.0) for name in mobile_names]
                    current["frames"] = [values18]
                    current["transport_frames"] = 1
                    slider.max = 1
                    slider.value = 1
                    show(values18)
                    status.content = (
                        f"**随机抽离终点已生成**  \n"
                        f"耗时：**{planner.random_start['generation_ms']:.1f}ms** · "
                        f"合法候选：**{planner.random_start['candidate_count']}**  \n"
                        "点击“实时计算并播放”开始本次规划。"
                    )
                    plan_button.disabled = False
                else:
                    status.content = "**正在实时规划搬运与Yaw 180°...**"
                    planner.result = planner.plan(boxes, planner.random_start["values"])
                    current["frames"] = planner.result["frames"]
                    current["transport_frames"] = planner.result["transport_frames"]
                    slider.max = len(current["frames"])
                    failure = planner.result.get("failure")
                    inspect_failure.disabled = failure is None
                    slider.value = failure["frame_index"] + 1 if failure else 1
                    show_frame(int(slider.value) - 1)
                    play.value = failure is None
            except Exception as error:
                status.content = f"**计算失败**  \n`{type(error).__name__}: {error}`"
            finally:
                planner.busy = False
                random_button.disabled = False
                if planner.random_start is not None:
                    plan_button.disabled = False

        threading.Thread(target=worker, daemon=True).start()

    @task_selector.on_update
    def on_task(_event):
        planner.selected_task = planner.task_map[task_selector.value]
        planner.random_start = None
        planner.result = None
        inspect_failure.disabled = True
        current["frames"] = []
        plan_button.disabled = True
        render_wall(planner.selected_task)
        status.content = "任务已切换，请生成新的随机抽离终点。"

    @random_button.on_click
    def on_random(_event):
        start_job("random")

    @plan_button.on_click
    def on_plan(_event):
        start_job("plan")

    @inspect_failure.on_click
    def on_inspect_failure(_event):
        if planner.result and planner.result.get("failure"):
            play.value = False
            slider.value = planner.result["failure"]["frame_index"] + 1
            show_frame(int(slider.value) - 1)

    @slider.on_update
    def on_slider(_event):
        show_frame(min(int(slider.value) - 1, len(current["frames"]) - 1))

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-3.2, 3.4, 2.8)
        client.camera.look_at = (0.6, 0.0, 1.2)
        client.camera.up_direction = (0, 0, 1)

    render_wall(planner.selected_task)
    home_map = {**{name: 0.0 for name in mobile_names}, **planner.home}
    show([home_map.get(name, 0.0) for name in mobile_names])
    print(f"Ready at http://localhost:{args.port}; tasks={len(planner.task_options)}", flush=True)
    while True:
        if play.value and current["frames"]:
            failure = planner.result.get("failure") if planner.result else None
            if failure and int(slider.value) >= failure["frame_index"] + 1:
                play.value = False
            else:
                slider.value = int(slider.value) % len(current["frames"]) + 1
            time.sleep(0.04)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
