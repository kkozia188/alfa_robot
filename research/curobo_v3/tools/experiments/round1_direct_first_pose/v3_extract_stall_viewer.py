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

from v3_interactive_pair_core import ACTIVE_JOINTS
from v3_wall_ik_benchmark import wall_center


LOCKED = {
    'head_joint': 0.0, 'head_pitch_joint': 0.0,
    'active_suspension_joint': 0.0,
    'caster01_joint': 0.0, 'wheel01_joint': 0.0,
    'caster02_joint': 0.0, 'wheel02_joint': 0.0,
    'caster03_joint': 0.0, 'wheel03_joint': 0.0,
    'caster04_joint': 0.0, 'wheel04_joint': 0.0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8082)
    args = parser.parse_args()
    reports = {item['round']: item for item in json.loads(args.diagnostics.read_text())['reports']}
    summary = json.loads(args.summary.read_text())
    robot_config = yaml.safe_load(Path(summary['robot_config']).read_text())['kinematics']
    urdf = yourdfpy.URDF.load(
        Path(robot_config['urdf_path']), load_meshes=True, build_scene_graph=True)
    server = viser.ViserServer(
        host='127.0.0.1', port=args.port,
        label='ALFA V3 · 抽离TrajOpt失败诊断')
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/floor_grid', width=6.0, height=5.0)
    server.scene.add_frame('/robot', show_axes=False)
    robot = ViserUrdf(
        server, urdf, root_node_name='/robot',
        mesh_color_override=(0.63, 0.69, 0.76, 0.80))
    actuated = list(robot.get_actuated_joint_names())
    static_boxes = {}
    labels = {}
    for box_id in range(25):
        center = wall_center(summary['chassis_front_x_m'], summary['wall_distance_m'], box_id)
        static_boxes[box_id] = server.scene.add_box(
            f'/boxes/box_{box_id:02}', position=center,
            dimensions=(0.30, 0.40, 0.40), color=(122, 141, 168), opacity=0.22)
        labels[box_id] = server.scene.add_label(
            f'/boxes/box_{box_id:02}/label', text=str(box_id),
            position=(-0.16, 0.0, 0.24))
    front = summary['chassis_front_x_m'] + summary['wall_distance_m']
    wall_back = front + 0.30 + 1e-6
    for name, center, dimensions in [
        ('ground', (wall_back - 2.0, 0.0, -0.05), (4.2, 2.6, 0.1)),
        ('left_wall', (wall_back - 2.0, -1.25, 1.2), (4.0, 0.1, 2.4)),
        ('right_wall', (wall_back - 2.0, 1.25, 1.2), (4.0, 0.1, 2.4)),
        ('front_wall', (wall_back + 0.05, 0.0, 1.2), (0.1, 2.6, 2.4)),
        ('ceiling', (wall_back - 2.0, 0.0, 2.45), (4.2, 2.6, 0.1)),
    ]:
        server.scene.add_box(
            f'/container/{name}', position=center, dimensions=dimensions,
            color=(180, 150, 102), opacity=0.06 if name != 'ground' else 0.12)
    carried = {
        'left': server.scene.add_box(
            '/carried/left', dimensions=(0.30, 0.40, 0.40),
            color=(236, 80, 45), opacity=0.82, visible=False),
        'right': server.scene.add_box(
            '/carried/right', dimensions=(0.30, 0.40, 0.40),
            color=(36, 126, 211), opacity=0.82, visible=False),
    }
    state = {
        'report': None, 'frames': None, 'tool_to_box': {}, 'lines': [],
        'maximum_jump_deg': 0.0, 'maximum_jump_step': 0,
        'maximum_jump_joint': '',
    }
    lock = threading.Lock()

    options = {'第8组 · L13/R11': 8, '第9组 · L12': 9}
    with server.gui.add_folder('约束TrajOpt失败复现'):
        selector = server.gui.add_dropdown('任务', list(options), initial_value=next(iter(options)))
        frame_slider = server.gui.add_slider('抽离距离 / cm', min=0, max=35, step=1, initial_value=0)
        play = server.gui.add_checkbox('播放连续IK可行轨迹', initial_value=True)
        speed = server.gui.add_slider('播放速度', min=0.1, max=2.0, step=0.1, initial_value=0.6)
        status = server.gui.add_markdown('')

    def update_robot(values):
        mapping = dict(zip(ACTIVE_JOINTS, values))
        robot.update_cfg(np.asarray([
            mapping.get(name, LOCKED.get(name, 0.0)) for name in actuated]))

    def removed_before(round_number):
        removed = set()
        for item in summary['rounds']:
            if item['round'] >= round_number:
                break
            if item['left_box'] is not None: removed.add(item['left_box'])
            if item['right_box'] is not None: removed.add(item['right_box'])
        return removed

    def show(index):
        report = state['report']
        if report is None:
            return
        index = max(0, min(index, len(state['frames']) - 1))
        with lock, server.atomic():
            update_robot(state['frames'][index])
            for side, handle in carried.items():
                handle.visible = side in state['tool_to_box']
                if handle.visible:
                    transform = (
                        urdf.get_transform(f'{side}_tool0', urdf.base_link)
                        @ state['tool_to_box'][side])
                    handle.position = transform[:3, 3]
                    handle.wxyz = np.roll(
                        Rotation.from_matrix(transform[:3, :3]).as_quat(), 1)
            continuity = (
                '连续性通过' if state['maximum_jump_deg'] <= 15.0
                else '存在严重换枝，不可直接执行')
            status.content = (
                '**单次约束TrajOpt：失败**\n\n'
                f'**逐点碰撞IK：已验证 `{index}/35 cm`**\n\n'
                f'最大单步变化 `{state["maximum_jump_deg"]:.2f}°`，'
                f'发生在 `{state["maximum_jump_step"]} cm` 的 '
                f'`{state["maximum_jump_joint"]}`；**{continuity}**。\n\n'
                '所有离散姿态均无机器人球或真实箱体OBB碰撞；'
                '但只有连续性通过时才能作为可执行回退轨迹。')

    def load(round_number):
        report = reports[round_number]
        state['report'] = report
        state['frames'] = report['frames']
        maximum = (0.0, 0, 0)
        for step, (first, second) in enumerate(zip(report['frames'], report['frames'][1:]), 1):
            delta = np.abs(np.asarray(second) - np.asarray(first))
            joint = int(np.argmax(delta[1:])) + 1
            value = float(np.degrees(delta[joint]))
            if value > maximum[0]:
                maximum = (value, step, joint)
        state['maximum_jump_deg'] = maximum[0]
        state['maximum_jump_step'] = maximum[1]
        state['maximum_jump_joint'] = ACTIVE_JOINTS[maximum[2]]
        update_robot(report['frames'][0])
        boxes = report['boxes']
        removed = removed_before(round_number)
        for box_id in range(25):
            visible = box_id not in removed and box_id not in boxes.values()
            static_boxes[box_id].visible = visible
            labels[box_id].visible = visible
        state['tool_to_box'] = {}
        for side, box_id in boxes.items():
            tool = urdf.get_transform(f'{side}_tool0', urdf.base_link)
            box = np.eye(4)
            box[:3, 3] = wall_center(
                summary['chassis_front_x_m'], summary['wall_distance_m'], box_id)
            state['tool_to_box'][side] = np.linalg.inv(tool) @ box
        for line in state['lines']:
            line.remove()
        state['lines'] = []
        for side in boxes:
            start = urdf.get_transform(f'{side}_tool0', urdf.base_link)[:3, 3]
            end = start.copy(); end[0] -= 0.35
            state['lines'].append(server.scene.add_line_segments(
                f'/extract_line/{side}', points=np.asarray([[start, end]]),
                colors=(44, 170, 95), thickness=0.008))
        frame_slider.max = len(report['frames']) - 1
        frame_slider.value = 0
        show(0)

    @selector.on_update
    def on_select(event):
        load(options[selector.value])

    @frame_slider.on_update
    def on_frame(event):
        show(int(frame_slider.value))

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-3.5, 3.7, 2.65)
        client.camera.look_at = (0.55, 0.0, 1.25)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    load(options[selector.value])
    print(json.dumps({'url': f'http://localhost:{args.port}', 'rounds': list(options.values())}), flush=True)
    while True:
        if play.value:
            value = int(frame_slider.value) + 1
            if value > frame_slider.max:
                value = 0
            frame_slider.value = value
            time.sleep(0.08 / speed.value)
        else:
            time.sleep(0.05)


if __name__ == '__main__':
    main()
