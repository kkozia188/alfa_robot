#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf
import yaml
import yourdfpy

sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_full_plan')
from v3_round1_full_plan import ACTIVE_JOINTS  # noqa: E402


LOCKED = {
    'head_joint': 0.0, 'head_pitch_joint': 0.0, 'active_suspension_joint': 0.0,
    'caster01_joint': 0.0, 'wheel01_joint': 0.0,
    'caster02_joint': 0.0, 'wheel02_joint': 0.0,
    'caster03_joint': 0.0, 'wheel03_joint': 0.0,
    'caster04_joint': 0.0, 'wheel04_joint': 0.0,
}


def loaded_reference(source):
    attach = int(source['attach_index'])
    release = int(source['release_index'])
    return np.asarray(source['frames'][attach:release + 1], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--ik-summary', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8084)
    args = parser.parse_args()

    batch = json.loads((args.results_root / 'summary.json').read_text())
    entries = {entry['round']: entry for entry in batch['entries']}
    summary = json.loads(args.ik_summary.read_text())
    rounds = {item['round']: item for item in summary['rounds']}
    robot_cfg = yaml.safe_load(Path(summary['robot_config']).read_text())['kinematics']
    urdf = yourdfpy.URDF.load(
        Path(robot_cfg['urdf_path']), load_meshes=True, build_scene_graph=True)

    server = viser.ViserServer(
        host='127.0.0.1', port=args.port,
        label='ALFA V3 · 11组负载联合优化总览')
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/floor_grid', width=6.0, height=5.0)
    server.scene.add_frame('/robot', show_axes=False)
    robot = ViserUrdf(
        server, urdf, root_node_name='/robot',
        mesh_color_override=(0.64, 0.70, 0.77, 0.72))
    actuated = list(robot.get_actuated_joint_names())
    front = summary['chassis_front_x_m'] + summary['wall_distance_m']

    boxes = {}
    for box_id in range(25):
        center = (front + 0.15, (box_id % 5 - 2) * 0.41, 0.20 + (box_id // 5) * 0.41)
        boxes[box_id] = server.scene.add_box(
            f'/boxes/box_{box_id:02}', position=center, dimensions=(0.30, 0.40, 0.40),
            color=(122, 141, 168), opacity=0.18)

    wall_back = front + 0.30 + 1e-6
    for name, center, dimensions in (
        ('ground', (wall_back - 2.0, 0.0, -0.05), (4.2, 2.6, 0.1)),
        ('left_wall', (wall_back - 2.0, -1.25, 1.2), (4.0, 0.1, 2.4)),
        ('right_wall', (wall_back - 2.0, 1.25, 1.2), (4.0, 0.1, 2.4)),
        ('front_wall', (wall_back + 0.05, 0.0, 1.2), (0.1, 2.6, 2.4)),
        ('ceiling', (wall_back - 2.0, 0.0, 2.45), (4.2, 2.6, 0.1)),
    ):
        server.scene.add_box(
            f'/container/{name}', position=center, dimensions=dimensions,
            color=(180, 150, 102), opacity=0.06 if name != 'ground' else 0.12)

    carried = {
        side: server.scene.add_box(
            f'/carried/{side}', dimensions=(0.30, 0.40, 0.40),
            color=(236, 80, 45) if side == 'left' else (36, 126, 211),
            opacity=0.82, visible=False)
        for side in ('left', 'right')
    }

    options = []
    for round_number in range(1, 12):
        item = rounds[round_number]
        targets = []
        if item['left_box'] is not None:
            targets.append(f'L{item["left_box"]}')
        if item['right_box'] is not None:
            targets.append(f'R{item["right_box"]}')
        options.append(f'{round_number:02d} · {" / ".join(targets)}')

    with server.gui.add_folder('11组联合优化结果'):
        round_select = server.gui.add_dropdown('任务组', options=options, initial_value=options[0])
        mode = server.gui.add_dropdown('轨迹', options=('优化后', '原参考'), initial_value='优化后')
        progress = server.gui.add_slider('进度', min=0, max=1000, step=1, initial_value=0)
        play = server.gui.add_checkbox('播放', initial_value=True)
        speed = server.gui.add_slider('速度', min=0.1, max=3.0, step=0.1, initial_value=1.0)
        status = server.gui.add_markdown('')
        server.gui.add_markdown(
            f'有完整初值 `{batch["optimized_count"]}/11` · '
            f'联合优化后精确通过 `{batch["validated_count"]}/11` · '
            f'缺失完整初值 `{batch["missing_reference_count"]}/11`')

    state = {'frames': np.zeros((1, len(ACTIVE_JOINTS))), 'tool_to_box': {}, 'boxes': {}}
    lock = threading.Lock()

    def update_robot(row):
        values = dict(zip(ACTIVE_JOINTS, row.tolist()))
        robot.update_cfg(np.asarray([
            values.get(name, LOCKED.get(name, 0.0)) for name in actuated]))

    def load_round():
        round_number = int(round_select.value.split(' ')[0])
        entry = entries[round_number]
        item = rounds[round_number]
        target_boxes = {}
        if item['left_box'] is not None:
            target_boxes['left'] = int(item['left_box'])
        if item['right_box'] is not None:
            target_boxes['right'] = int(item['right_box'])
        removed_before = set()
        for previous in summary['rounds']:
            if previous['round'] >= round_number:
                break
            for key in ('left_box', 'right_box'):
                if previous[key] is not None:
                    removed_before.add(int(previous[key]))

        optimized_path = args.results_root / f'round_{round_number:02d}.json'
        source_path = args.source_root / f'round_{round_number:02d}.json'
        if entry['status'] == 'optimized':
            result = json.loads(optimized_path.read_text())
            if mode.value == '优化后':
                frames = np.asarray(result['dense_frames'], dtype=float)
            else:
                frames = loaded_reference(json.loads(source_path.read_text()))
            validated = result['validated_success']
            reason = result.get('payload_reason', '')
            avoidance = result.get('home_avoidance') or {}
            reference_length = avoidance.get('reference_path_l2')
            optimized_length = avoidance.get('optimized_path_l2')
            shortened = (
                None if not reference_length else
                100.0 * (reference_length - optimized_length) / reference_length)
            summary_text = (
                f'优化 `{result["optimization_ms"]:.1f} ms` · '
                f'门禁 `{ "通过" if validated else "失败" }`'
                + ('' if avoidance.get('optimized_minimum_l2_rad') is None else
                   f' · 距第一姿态最近 `{avoidance["optimized_minimum_l2_rad"]:.3f} rad`')
                + ('' if shortened is None else f' · 路径缩短 `{shortened:.1f}%`'))
        else:
            frames = np.asarray([item['collision_free']['joints']], dtype=float)
            validated = False
            reason = entry.get('source_failure_stage') or 'missing_complete_reference'
            summary_text = '前置规划未形成完整吸附到放置轨迹，未进入联合优化'

        update_robot(frames[0])
        tool_to_box = {}
        for side, box_id in target_boxes.items():
            tool = urdf.get_transform(f'{side}_tool0', urdf.base_link)
            box = np.eye(4)
            box[:3, 3] = np.asarray([
                front + 0.15,
                (box_id % 5 - 2) * 0.41,
                0.20 + (box_id // 5) * 0.41,
            ])
            tool_to_box[side] = np.linalg.inv(tool) @ box

        state.update({
            'round': round_number,
            'frames': frames,
            'boxes': target_boxes,
            'removed_before': removed_before,
            'tool_to_box': tool_to_box,
            'validated': validated,
            'reason': reason,
            'summary': summary_text,
        })
        progress.value = 0

    def show():
        frames = state['frames']
        index = min(len(frames) - 1, int(round(progress.value / 1000.0 * (len(frames) - 1))))
        with lock, server.atomic():
            update_robot(frames[index])
            hidden = state['removed_before'] | set(state['boxes'].values())
            for box_id, handle in boxes.items():
                handle.visible = box_id not in hidden
            for side, handle in carried.items():
                handle.visible = side in state['boxes']
                if handle.visible:
                    transform = urdf.get_transform(
                        f'{side}_tool0', urdf.base_link) @ state['tool_to_box'][side]
                    handle.position = transform[:3, 3]
                    handle.wxyz = np.roll(
                        Rotation.from_matrix(transform[:3, :3]).as_quat(), 1)
                    handle.color = (
                        (236, 80, 45) if state['validated'] else (220, 35, 35))
            color = '#248a3d' if state['validated'] else '#c9362b'
            reason = '' if not state['reason'] else f' · 原因 `{state["reason"]}`'
            status.content = (
                f'<span style="color:{color}">**第{state["round"]}组 · '
                f'{state["summary"]}**</span>{reason}\n\n'
                f'第 `{index + 1}/{len(frames)}` 帧')

    @round_select.on_update
    def on_round(_):
        load_round()
        show()

    @mode.on_update
    def on_mode(_):
        load_round()
        show()

    @progress.on_update
    def on_progress(_):
        show()

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-3.4, 3.6, 2.65)
        client.camera.look_at = (0.55, 0.0, 1.25)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    load_round()
    show()
    print(json.dumps({'url': f'http://localhost:{args.port}', **{
        key: batch[key] for key in (
            'optimized_count', 'validated_count', 'missing_reference_count')}}, indent=2))
    try:
        while True:
            if play.value and len(state['frames']) > 1:
                progress.value = (progress.value + max(1, int(4 * speed.value))) % 1001
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
