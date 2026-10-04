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


ROUNDS = [
    {"left": 24, "right": 20}, {"left": 23, "right": 21}, {"left": 22},
    {"left": 19, "right": 15}, {"left": 18, "right": 16}, {"right": 17},
    {"left": 14, "right": 10}, {"left": 13, "right": 11}, {"left": 12},
    {"left": 9, "right": 5}, {"left": 8, "right": 6},
]


def quaternion(matrix):
    return np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)


def label(index, row):
    boxes = "/".join(f"{side[0].upper()}{box}" for side, box in row["boxes"].items())
    success = row.get("curobo_optimization", {}).get("success", row["success"])
    return f"{index:02d} {boxes} {'SUCCESS' if success else 'FAILED'}"


def display_frames(row):
    optimization = row.get("curobo_optimization", {})
    return optimization.get("frames") or row.get("frames") or []


def display_success(row):
    return row.get("curobo_optimization", {}).get("success", row["success"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--prm-result", type=Path)
    parser.add_argument("--rrt-connect-result", type=Path)
    parser.add_argument("--informed-rrt-result", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--named-poses", type=Path, required=True)
    parser.add_argument("--front-x", type=float, default=0.5080520510673523)
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--title", default="RRT / PRM 对比")
    args = parser.parse_args()
    report = json.loads(args.result.read_text())
    reports = {"RRT": report}
    if args.prm_result:
        reports["PRM"] = json.loads(args.prm_result.read_text())
    if args.rrt_connect_result:
        reports["RRTConnect"] = json.loads(args.rrt_connect_result.read_text())
    if args.informed_rrt_result:
        reports["InformedRRT"] = json.loads(args.informed_rrt_result.read_text())
    rounds = report["rounds"]
    config_data = yaml.safe_load(args.config.read_text())
    config = config_data.get("robot_cfg", config_data)["kinematics"]
    names = config["cspace"]["joint_names"]
    home = yaml.safe_load(args.named_poses.read_text())["named_poses"]["home"]
    urdf = yourdfpy.URDF.load(
        Path(config["urdf_path"]), load_meshes=True, build_scene_graph=True
    )
    actuated = list(urdf.actuated_joint_names)
    home_row = [home.get(name, 0.0) for name in names]
    server = viser.ViserServer(host="127.0.0.1", port=args.port,
                               label=f"V3.2.2 · {args.title}")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=5, height=3)
    wall_back = args.front_x + 0.9 + 0.30 + 1e-6
    container_root = server.scene.add_frame("/container_collision", show_axes=False)
    container_geometry = [
        ("left_wall", (wall_back - 2.0, -1.25, 1.2), (4.0, 0.1, 2.4)),
        ("right_wall", (wall_back - 2.0, 1.25, 1.2), (4.0, 0.1, 2.4)),
        ("front_wall", (wall_back + 0.05, 0.0, 1.2), (0.1, 2.6, 2.4)),
        ("ceiling", (wall_back - 2.0, 0.0, 2.45), (4.2, 2.6, 0.1)),
    ]
    for name, position, dimensions in container_geometry:
        server.scene.add_box(
            f"/container_collision/{name}", position=position,
            dimensions=dimensions, color=(45, 112, 170), opacity=0.10,
        )
        server.scene.add_label(
            f"/container_collision/{name}/label", text=name,
            position=(position[0], position[1], position[2]),
        )
    server.scene.add_frame("/robot", show_axes=False)
    robot = ViserUrdf(server, urdf, root_node_name="/robot",
                      mesh_color_override=(0.68, 0.73, 0.80, 0.74))
    wall_root = server.scene.add_frame("/wall", show_axes=False)
    attached_root = server.scene.add_frame("/attached", show_axes=False)
    attached = {}
    offsets = {side: canonical_side_tool_to_box(side) for side in ("left", "right")}
    for side in ("left", "right"):
        attached[side] = server.scene.add_frame(f"/attached/{side}", show_axes=False)
        server.scene.add_box(f"/attached/{side}/box", dimensions=(0.30, 0.40, 0.40),
                             color=(239, 146, 62), opacity=0.76)
        server.scene.add_label(f"/attached/{side}/label", text=f"{side}附着箱",
                               position=(0, 0, 0.27))
    options = [label(index, row) for index, row in enumerate(rounds, start=1)]
    with server.gui.add_folder(f"11轮{args.title}"):
        planner_selector = server.gui.add_dropdown(
            "规划器", options=list(reports), initial_value="RRT"
        )
        selector = server.gui.add_dropdown("任务", options=options, initial_value=options[0])
        slider = server.gui.add_slider("轨迹帧", min=1, max=max(
            len(display_frames(row)) for planner_report in reports.values()
            for row in planner_report["rounds"] if display_frames(row)
        ), step=1, initial_value=1)
        play = server.gui.add_checkbox("播放", initial_value=False)
        show_container = server.gui.add_checkbox("集装箱碰撞体", initial_value=True)
        status = server.gui.add_markdown("")
        server.gui.add_markdown(
            "本视图从35cm抽离后的数值IK构型直接开始搜索；展示的是原始Batched RRT路径，"
            "每轮使用同一起点和最多16个携箱合法终点；RRT/InformedRRT显示单树路径，"
            "RRTConnect显示起点树与目标森林连接路径，PRM显示道路图最短路；"
            "所有对比均不使用Shortcut或TrajOpt。"
            "左右侧吸滚转与附着变换互为镜像，末端横向抓取且箱体保持朝上；"
            "地面仅豁免base_link正常支撑接触；按要求不执行真实箱体OBB验收。"
        )
    current = {"round": 0, "frames": [], "wall_root": wall_root, "planner": "RRT"}

    def active_rounds():
        return reports[current["planner"]]["rounds"]

    def remaining_boxes(round_index):
        removed = set()
        for prior in ROUNDS[:round_index]:
            removed.update(prior.values())
        removed.update(ROUNDS[round_index].values())
        return [box for box in range(25) if box not in removed]

    def render_wall(round_index):
        current["wall_root"].remove()
        current["wall_root"] = server.scene.add_frame("/wall", show_axes=False)
        wall_front = args.front_x + 0.9
        for box in remaining_boxes(round_index):
            server.scene.add_box(
                f"/wall/box_{box:02d}",
                position=(wall_front + 0.15, (box % 5 - 2) * 0.41,
                          0.20 + (box // 5) * 0.41),
                dimensions=(0.30, 0.40, 0.40), color=(87, 145, 165), opacity=0.18,
            )

    def show_frame(index):
        row = active_rounds()[current["round"]]
        frames = current["frames"]
        values = dict(zip(names, frames[index]))
        q = np.asarray([values.get(name, 0.0) for name in actuated])
        urdf.update_cfg(q)
        robot.update_cfg(q)
        for side, node in attached.items():
            node.visible = side in row["boxes"] and display_success(row)
            if node.visible:
                matrix = urdf.get_transform(f"{side}_tool0", urdf.base_link) @ offsets[side]
                node.position = matrix[:3, 3]
                node.wxyz = quaternion(matrix)
        if display_success(row):
            optimization = row.get("curobo_optimization", {})
            planner_name = row.get("planner", current["planner"].lower()).upper()
            planner_stats = row.get("rrt_stats", {})
            if "batch_ik_ms" in row:
                status.content = (
                    f"**批量IK分层图 · Round {row['round']}/11 · SUCCESS**  \n"
                    f"Boxes：`{row['boxes']}`  \n"
                    f"Frame：**{index + 1}/{len(frames)}**  \n"
                    f"批量联合IK：**{row['batch_ik_ms']:.1f}ms** · "
                    f"GPU验边：**{row['edge_validation_ms']:.1f}ms**  \n"
                    f"核心规划：**{row['planning_ms']:.1f}ms** · "
                    f"层数：**{row.get('waypoints', 0)}** · 候选边：**{row.get('edge_count', 0):,}**  \n"
                    f"最大箱体倾角：**{row.get('max_box_tilt_deg', 0.0):.2f}°** · "
                    f"附着箱碰撞：**抽离后启用**"
                )
            else:
                status.content = (
                f"**{planner_name} · Round {row['round']}/11 · SUCCESS**  \n"
                f"Boxes：`{row['boxes']}`  \n"
                f"Frame：**{index + 1}/{len(frames)}**  \n"
                f"抽离后IK：**{row.get('extract_ik_ms', 0.0):.1f}ms** · "
                f"放置目标IK：**{row.get('placement_ik_ms', 0.0):.1f}ms**  \n"
                f"目标数：**{row.get('placement_retained_goals', 1)}** · "
                f"命中：**{planner_stats.get('selected_goal', 0)}** · "
                f"搜索：**{planner_stats.get('search_ms', row['rrt_ms']):.1f}ms**  \n"
                f"节点：**{planner_stats.get('roadmap_nodes', planner_stats.get('tree_nodes', 0)):,}** · "
                f"有效边/连接：**{planner_stats.get('valid_edges', planner_stats.get('solutions_found', 0)):,}** · "
                f"路点：**{planner_stats.get('raw_waypoints', row.get('waypoints', 0))}**  \n"
                f"帧数：**{len(frames)}** · "
                f"最大箱体倾角：**{optimization.get('max_box_tilt_deg', row.get('max_box_tilt_deg', 0.0)):.1f}°** · "
                f"硬门限：**{row.get('box_tilt_limit_deg', 0.0):.1f}°**  \n"
                f"GPU checks：**{row['gpu_states_checked']:,}**"
                )
            if row.get("transport_frames") is not None:
                transport_frames = row["transport_frames"]
                if index < transport_frames:
                    phase = "抽离终点 → 新携箱目标"
                    yaw = 0.0
                else:
                    phase = "底盘Yaw 180°接续"
                    yaw = 0.5 * (index - transport_frames + 1)
                status.content += (
                    f"  \n阶段：**{phase}** · Base Yaw：**{yaw:.1f}°**"
                )
        else:
            status.content = (
                f"**Round {row['round']}/11 · FAILED**  \n"
                f"Boxes：`{row['boxes']}`  \n"
                f"Reason：`{row.get('failure', 'unknown')}`  \n"
                "本轮没有RRT路径。"
            )

    def select_round(index):
        current["round"] = index
        row = active_rounds()[index]
        current["frames"] = display_frames(row) or [home_row]
        slider.max = len(current["frames"])
        slider.value = 1
        render_wall(index)
        show_frame(0)

    @selector.on_update
    def on_select(_event):
        play.value = False
        select_round(options.index(selector.value))

    @planner_selector.on_update
    def on_planner(_event):
        play.value = False
        current["planner"] = planner_selector.value
        select_round(current["round"])

    @slider.on_update
    def on_slider(_event):
        show_frame(min(int(slider.value) - 1, len(current["frames"]) - 1))

    @show_container.on_update
    def on_container(_event):
        container_root.visible = show_container.value

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-3.2, 3.4, 2.8)
        client.camera.look_at = (0.8, 0.0, 1.35)
        client.camera.up_direction = (0, 0, 1)

    select_round(0)
    summary = {name: value.get("optimized_rounds", value["successful_rounds"])
               for name, value in reports.items()}
    print(f"Ready at http://localhost:{args.port}; success={summary}", flush=True)
    while True:
        if play.value and display_success(active_rounds()[current["round"]]):
            slider.value = int(slider.value) % len(current["frames"]) + 1
            time.sleep(0.04)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
