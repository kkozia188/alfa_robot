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


def loaded_reference(source):
    stages = {stage['name']: stage for stage in source['stages']}
    frames = []
    for name in (
        'cartesian_extract_35cm',
        'loaded_extract_to_first_home',
        'loaded_first_home_to_first_unloading',
    ):
        if frames:
            frames.extend(stages[name]['frames'][1:])
        else:
            frames.extend(stages[name]['frames'])
    return np.asarray(frames, dtype=float), len(stages['cartesian_extract_35cm']['frames']) - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--ik-summary', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8083)
    args = parser.parse_args()

    report = json.loads(args.result.read_text())
    if not report['validated_success']:
        raise RuntimeError('viewer requires a validated result')
    summary = json.loads(args.ik_summary.read_text())
    source = json.loads(Path(report['source']).read_text())
    reference, reference_extract_end = loaded_reference(source)
    optimized = np.asarray(report['dense_frames'], dtype=float)
    trajectories = {'优化后': optimized, '原分段参考': reference}
    extraction_ends = {
        '优化后': report['dense_extract_end'],
        '原分段参考': reference_extract_end,
    }
    joint_names = report.get('joint_names', ACTIVE_JOINTS)
    robot_cfg = yaml.safe_load(Path(summary['robot_config']).read_text())['kinematics']
    urdf_path = Path(robot_cfg['urdf_path'])

    server = viser.ViserServer(
        host='127.0.0.1', port=args.port,
        label='ALFA V3 · 吸附到放置联合优化')
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/floor_grid', width=6.0, height=5.0)
    server.scene.add_frame('/robot', show_axes=False)
    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=True, build_scene_graph=True)
    robot = ViserUrdf(
        server, urdf, root_node_name='/robot',
        mesh_color_override=(0.64, 0.70, 0.77, 0.72))
    actuated = list(robot.get_actuated_joint_names())

    def update_robot(row):
        values = dict(zip(joint_names, row.tolist()))
        robot.update_cfg(np.asarray([
            values.get(name, LOCKED.get(name, 0.0)) for name in actuated]))

    front = summary['chassis_front_x_m'] + summary['wall_distance_m']
    boxes = {
        side: int(box_id)
        for side, box_id in report.get('boxes', {'left': 24, 'right': 20}).items()
    }
    pair = tuple(boxes.values())
    removed = set(report.get('removed_before', ())) | set(pair)
    for box_id in range(25):
        if box_id in removed:
            continue
        center = (front + 0.15, (box_id % 5 - 2) * 0.41, 0.20 + (box_id // 5) * 0.41)
        server.scene.add_box(
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

    update_robot(reference[0])
    carried = {}
    tool_to_box = {}
    for side, box_id in boxes.items():
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
            color=(236, 80, 45) if side == 'left' else (36, 126, 211), opacity=0.82)

    with server.gui.add_folder('吸附 → 抽离 → 放置'):
        mode = server.gui.add_dropdown('轨迹', options=('优化后', '原分段参考'), initial_value='优化后')
        progress = server.gui.add_slider('进度', min=0, max=1000, step=1, initial_value=0)
        play = server.gui.add_checkbox('播放', initial_value=True)
        speed = server.gui.add_slider('速度', min=0.1, max=3.0, step=0.1, initial_value=1.0)
        status = server.gui.add_markdown('')
        server.gui.add_markdown(
            f'联合优化 `{report["optimization_ms"]:.1f} ms` · '
            f'碰撞门禁 `通过` · 优化节点 `{report["nodes"]}` · '
            f'加密帧 `{report["dense_frame_count"]}`')

    lock = threading.Lock()

    def show():
        frames = trajectories[mode.value]
        index = min(len(frames) - 1, int(round(progress.value / 1000.0 * (len(frames) - 1))))
        with lock, server.atomic():
            update_robot(frames[index])
            for side, handle in carried.items():
                transform = urdf.get_transform(f'{side}_tool0', urdf.base_link) @ tool_to_box[side]
                handle.position = transform[:3, 3]
                handle.wxyz = np.roll(Rotation.from_matrix(transform[:3, :3]).as_quat(), 1)
            phase = '35 cm 约束抽离' if index <= extraction_ends[mode.value] else '负载全局运动至放置位'
            status.content = f'**{mode.value}** · 第 `{index + 1}/{len(frames)}` 帧 · **{phase}**'

    @mode.on_update
    def on_mode(_):
        show()

    @progress.on_update
    def on_progress(_):
        show()

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-3.4, 3.6, 2.65)
        client.camera.look_at = (0.55, 0.0, 1.25)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    show()
    print(json.dumps({
        'url': f'http://localhost:{args.port}',
        'optimized_frames': len(optimized),
        'reference_frames': len(reference),
        'optimization_ms': report['optimization_ms'],
    }, indent=2))
    try:
        while True:
            if play.value:
                progress.value = (progress.value + max(1, int(4 * speed.value))) % 1001
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
