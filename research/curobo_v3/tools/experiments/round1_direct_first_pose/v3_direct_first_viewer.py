#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf
import yaml
import yourdfpy


LOCKED = {
    'head_joint': 0.0,
    'head_pitch_joint': 0.0,
    'active_suspension_joint': 0.0,
    'caster01_joint': 0.0,
    'wheel01_joint': 0.0,
    'caster02_joint': 0.0,
    'wheel02_joint': 0.0,
    'caster03_joint': 0.0,
    'wheel03_joint': 0.0,
    'caster04_joint': 0.0,
    'wheel04_joint': 0.0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    report = json.loads(args.result.read_text())
    validated = report['validated_smooth_path']
    if not report['success'] or not validated['success']:
        raise RuntimeError('viewer only accepts a fully validated successful result')
    frames = np.asarray(validated['frames'], dtype=float)
    boundaries = validated['stage_boundaries']
    names = report['joint_names']
    config = yaml.safe_load(Path(report['robot_config']).read_text())['kinematics']
    urdf_path = Path(config['urdf_path'])

    server = viser.ViserServer(
        host='127.0.0.1', port=args.port,
        label='ALFA V3 · cuRobo优化版完整规划')
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/floor_grid', width=6.0, height=5.0)
    server.scene.add_frame('/robot', show_axes=False)
    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=True, build_scene_graph=True)
    robot = ViserUrdf(
        server, urdf, root_node_name='/robot',
        mesh_color_override=(0.64, 0.70, 0.77, 0.78))
    actuated = list(robot.get_actuated_joint_names())

    def values_at(index):
        values = dict(zip(names, frames[index].tolist()))
        return np.asarray([values.get(name, LOCKED.get(name, 0.0)) for name in actuated])

    def set_robot(index):
        robot.update_cfg(values_at(index))

    front = report['chassis_front_x_m'] + report['wall_distance_m']
    target_ids = set(report['pair'])
    static_targets = {}
    carried = {}
    for box_id in range(25):
        center = (front + 0.15, (box_id % 5 - 2) * 0.41, 0.20 + (box_id // 5) * 0.41)
        target = box_id in target_ids
        handle = server.scene.add_box(
            f'/boxes/box_{box_id:02}', position=center, dimensions=(0.30, 0.40, 0.40),
            color=(232, 126, 46) if target else (122, 141, 168),
            opacity=0.74 if target else 0.18)
        if target:
            static_targets[box_id] = handle
            server.scene.add_label(
                f'/boxes/box_{box_id:02}/label', text=f'目标 {box_id}',
                position=(-0.16, 0.0, 0.24))

    wall_back = front + 0.30 + 1e-6
    container = [
        ('ground', (wall_back - 2.0, 0.0, -0.05), (4.2, 2.6, 0.1)),
        ('left_wall', (wall_back - 2.0, -1.25, 1.2), (4.0, 0.1, 2.4)),
        ('right_wall', (wall_back - 2.0, 1.25, 1.2), (4.0, 0.1, 2.4)),
        ('front_wall', (wall_back + 0.05, 0.0, 1.2), (0.1, 2.6, 2.4)),
        ('ceiling', (wall_back - 2.0, 0.0, 2.45), (4.2, 2.6, 0.1)),
    ]
    for name, center, dimensions in container:
        server.scene.add_box(
            f'/container/{name}', position=center, dimensions=dimensions,
            color=(180, 150, 102), opacity=0.06 if name != 'ground' else 0.12)

    attach_index = boundaries['cartesian_approach_5cm']
    set_robot(attach_index)
    tool_to_box = {}
    for side, box_id in zip(('left', 'right'), report['pair']):
        tool = urdf.get_transform(f'{side}_tool0', urdf.base_link)
        box = np.eye(4)
        box[:3, 3] = np.asarray([
            front + 0.15,
            (box_id % 5 - 2) * 0.41,
            0.20 + (box_id // 5) * 0.41,
        ])
        tool_to_box[side] = np.linalg.inv(tool) @ box
        carried[side] = server.scene.add_box(
            f'/carried/{side}', dimensions=(0.30, 0.40, 0.40),
            color=(236, 80, 45) if side == 'left' else (36, 126, 211),
            opacity=0.82, visible=False)

    trace_indices = np.unique(np.linspace(0, len(frames) - 1, min(360, len(frames))).astype(int))
    traces = {'left': [], 'right': []}
    for index in trace_indices:
        set_robot(int(index))
        for side in traces:
            traces[side].append(urdf.get_transform(f'{side}_tool0', urdf.base_link)[:3, 3])
    for side, points in traces.items():
        server.scene.add_spline_catmull_rom(
            f'/tcp_trace/{side}', points=np.asarray(points),
            color=(226, 73, 57) if side == 'left' else (40, 124, 202),
            thickness=0.008, segments=max(100, len(points)))

    labels = [
        ('初始', 0),
        ('预抓取', boundaries['home_to_pregrasp']),
        ('吸附', boundaries['cartesian_approach_5cm']),
        ('抽离完成', boundaries['cartesian_extract_35cm']),
        ('携箱回第一初始位', boundaries['loaded_extract_to_first_home']),
        ('第一放置位', boundaries['loaded_first_home_to_first_unloading']),
    ]
    with server.gui.add_folder('L24 / R20 完整任务'):
        server.gui.add_markdown(
            '**优化版验证通过** · 0.9 m 箱墙 · 双臂侧吸 · 集装箱碰撞开启')
        frame_slider = server.gui.add_slider(
            '帧', min=1, max=len(frames), step=1, initial_value=1)
        play = server.gui.add_checkbox('播放', initial_value=True)
        speed = server.gui.add_slider('速度', min=0.1, max=3.0, step=0.1, initial_value=1.0)
        with server.gui.add_folder('阶段定位'):
            buttons = [server.gui.add_button(label) for label, _ in labels]
        show_trace = server.gui.add_checkbox('显示末端轨迹', initial_value=True)
        status = server.gui.add_markdown('')
        server.gui.add_markdown(
            f'总帧数 `{len(frames)}` · 最大旋转步长 '
            f'`{validated["maximum_rotary_step_deg"]:.3f}°` · '
            f'最大 Updown 步长 `{validated["maximum_updown_step_m"] * 1000.0:.3f} mm`')
        timing = report.get('planning_timing')
        if timing:
            stage_timing = timing['stages']
            total = timing['steady_total_ms']
            extraction = stage_timing['cartesian_extract_35cm']['steady_ms']
            timing_labels = {
                'home_to_pregrasp': '初始位 → 预抓取',
                'cartesian_approach_5cm': '5 cm 接近',
                'cartesian_extract_35cm': '35 cm 约束抽离',
                'loaded_extract_to_first_home': '携箱回初始位',
            }
            bottleneck_name, bottleneck_data = max(
                stage_timing.items(), key=lambda item: item[1]['steady_ms'])
            constrained_stage = next(
                stage for stage in report['stages']
                if stage['name'] == 'cartesian_extract_35cm')
            constrained_metrics = constrained_stage.get('constraint_metrics')
            constraint_text = ''
            if constrained_metrics:
                maximum_line_error = max(
                    constrained_metrics['left_tool0']['maximum_yz_error_mm'],
                    constrained_metrics['right_tool0']['maximum_yz_error_mm'])
                maximum_orientation_error = max(
                    constrained_metrics['left_tool0']['maximum_orientation_error_deg'],
                    constrained_metrics['right_tool0']['maximum_orientation_error_deg'])
                constraint_text = (
                    f'\n\n约束抽离最大横向偏差 `{maximum_line_error:.3f} mm`，'
                    f'最大姿态偏差 `{maximum_orientation_error:.3f}°`。')
            server.gui.add_markdown(
                '**规划耗时（常驻稳态）**\n\n'
                f'- 初始位 → 预抓取：`{stage_timing["home_to_pregrasp"]["steady_ms"] / 1000.0:.3f} s`\n'
                f'- 5 cm 接近：`{stage_timing["cartesian_approach_5cm"]["steady_ms"] / 1000.0:.3f} s`\n'
                f'- 35 cm 抽离：`{extraction / 1000.0:.3f} s`\n'
                f'- 携箱回初始位：`{stage_timing["loaded_extract_to_first_home"]["steady_ms"] / 1000.0:.3f} s`\n'
                f'- 合计：`{total / 1000.0:.3f} s`\n\n'
                f'当前瓶颈：{timing_labels[bottleneck_name]}，占 '
                f'`{100.0 * bottleneck_data["steady_ms"] / total:.1f}%`。'
                f'{constraint_text}')

    lock = threading.Lock()
    phases = [
        (boundaries['home_to_pregrasp'], '第一初始位 → 预抓取', '#2676bd'),
        (boundaries['cartesian_approach_5cm'], '笛卡尔接近 5 cm', '#6d55a3'),
        (boundaries['cartesian_extract_35cm'], '携箱水平抽离 35 cm', '#d97706'),
        (boundaries['loaded_extract_to_first_home'], '携箱回第一初始位', '#21815d'),
        (boundaries['loaded_first_home_to_first_unloading'], '第一初始位 → 第一放置位', '#146b72'),
    ]

    def show(index):
        with lock, server.atomic():
            set_robot(index)
            attached = index >= attach_index
            for handle in static_targets.values():
                handle.visible = not attached
            for side, handle in carried.items():
                handle.visible = attached
                if attached:
                    transform = urdf.get_transform(f'{side}_tool0', urdf.base_link) @ tool_to_box[side]
                    handle.position = transform[:3, 3]
                    handle.wxyz = np.roll(Rotation.from_matrix(transform[:3, :3]).as_quat(), 1)
            phase, color = phases[-1][1], phases[-1][2]
            for end, label, phase_color in phases:
                if index <= end:
                    phase, color = label, phase_color
                    break
            status.content = (
                f'第 **{index + 1}/{len(frames)}** 帧\n\n'
                f'<span style="color:{color}">**{phase}**</span>')

    @frame_slider.on_update
    def on_frame(event):
        show(int(frame_slider.value) - 1)

    for button, (_, target) in zip(buttons, labels):
        @button.on_click
        def on_click(event, target=target):
            play.value = False
            frame_slider.value = target + 1

    @show_trace.on_update
    def on_trace(event):
        server.scene.set_global_visibility('/tcp_trace', show_trace.value)

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-3.4, 3.6, 2.65)
        client.camera.look_at = (0.55, 0.0, 1.25)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    show(0)
    print(json.dumps({
        'url': f'http://localhost:{args.port}',
        'frames': len(frames),
        'stage_boundaries': boundaries,
    }, ensure_ascii=False), flush=True)
    while True:
        if play.value:
            frame_slider.value = int(frame_slider.value) % len(frames) + 1
            time.sleep(0.02 / speed.value)
        else:
            time.sleep(0.05)


if __name__ == '__main__':
    main()
