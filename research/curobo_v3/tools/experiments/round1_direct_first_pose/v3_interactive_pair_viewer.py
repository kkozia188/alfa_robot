#!/usr/bin/env python3

import argparse
import copy
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

from v3_interactive_pair_core import ACTIVE_JOINTS, InteractivePairPlanner, SUPPORTED_ROUNDS


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


def failure_detail(result):
    stage = result.get('failure_stage', 'unknown')
    first_extract = (result.get('timings') or {}).get('extract_first_payload_failure')
    if first_extract is not None:
        return f'{stage}: {first_extract}'
    validation = result.get('validation') or {}
    for key in (
        'first_payload_failure', 'first_loaded_failure',
        'first_free_failure', 'first_empty_failure'):
        value = validation.get(key)
        if value is not None:
            return f'{stage}: {value}'
    return stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--named-poses', type=Path, required=True)
    parser.add_argument('--placement-reference', type=Path, required=True)
    parser.add_argument('--payload-fit', type=Path, required=True)
    parser.add_argument('--defer-payload-collision-until-extracted', action='store_true')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    planner = InteractivePairPlanner(
        args.summary, args.named_poses, args.placement_reference,
        payload_fit_path=args.payload_fit,
        defer_payload_collision_until_extracted=(
            args.defer_payload_collision_until_extracted))
    summary = planner.summary
    robot_config = yaml.safe_load(Path(summary['robot_config']).read_text())['kinematics']
    urdf_path = Path(robot_config['urdf_path'])
    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=True, build_scene_graph=True)
    server = viser.ViserServer(
        host='127.0.0.1', port=args.port,
        label='ALFA V3 · cuRobo实时箱对规划')
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/floor_grid', width=6.0, height=5.0)
    server.scene.add_frame('/robot', show_axes=False)
    robot = ViserUrdf(
        server, urdf, root_node_name='/robot',
        mesh_color_override=(0.64, 0.70, 0.77, 0.80))
    actuated = list(robot.get_actuated_joint_names())

    home_values = planner.home.position.reshape(-1).detach().cpu().numpy()
    current = {
        'result': None,
        'frames': np.asarray([home_values]),
        'playing': False,
        'planning': False,
        'frame': 0,
        'tool_to_box': {},
    }
    lock = threading.Lock()

    def update_robot(values):
        mapping = dict(zip(ACTIVE_JOINTS, values.tolist()))
        robot.update_cfg(np.asarray([
            mapping.get(name, LOCKED.get(name, 0.0)) for name in actuated]))

    front = summary['chassis_front_x_m'] + summary['wall_distance_m']
    static_boxes = {}
    box_labels = {}
    for box_id in range(25):
        center = (front + 0.15, (box_id % 5 - 2) * 0.41, 0.20 + (box_id // 5) * 0.41)
        static_boxes[box_id] = server.scene.add_box(
            f'/boxes/box_{box_id:02}', position=center, dimensions=(0.30, 0.40, 0.40),
            color=(122, 141, 168), opacity=0.25)
        box_labels[box_id] = server.scene.add_label(
            f'/boxes/box_{box_id:02}/label', text=str(box_id), position=(-0.16, 0.0, 0.24))
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

    options = planner.options()
    option_to_round = dict(zip(options, SUPPORTED_ROUNDS))
    with server.gui.add_folder('实时规划与仿真执行'):
        selector = server.gui.add_dropdown(
            '箱子对', options=options, initial_value=options[0])
        calculate = server.gui.add_button('计算并执行', color='green')
        progress = server.gui.add_progress_bar(0.0, animated=False)
        payload_mode = (
            '抽离阶段关闭负载球并做OBB后验；抽离后启用MorphIt 46球。'
            if args.defer_payload_collision_until_extracted
            else '附着箱：官方 VOXEL 48球凸体内接拟合。')
        status = server.gui.add_markdown(
            f'**{payload_mode}**\n\n'
            f'规划器已加载，初始化 `{planner.initialization_ms / 1000.0:.3f} s`。\n\n'
            '场景按所选轮次移除此前箱子；规划失败时不会播放未验证轨迹。')
        timing = server.gui.add_markdown('尚未计算。')
        frame_slider = server.gui.add_slider('执行帧', min=1, max=1, step=1, initial_value=1)
        play = server.gui.add_checkbox('播放', initial_value=False)
        speed = server.gui.add_slider('播放速度', min=0.1, max=3.0, step=0.1, initial_value=1.0)
        with server.gui.add_folder('阶段定位'):
            stage_buttons = {
                'home_to_pregrasp': server.gui.add_button('预抓取'),
                'cartesian_approach_5cm': server.gui.add_button('吸附'),
                'cartesian_extract_35cm': server.gui.add_button('抽离完成'),
                'loaded_return_home': server.gui.add_button('携箱回初始位'),
                'loaded_placement': server.gui.add_button('放置/释放'),
                'empty_return_home': server.gui.add_button('最终回初始位'),
            }

    stage_progress = {
        'pregrasp': 0.12,
        'extract': 0.38,
        'loaded_return': 0.55,
        'placement': 0.72,
        'validate': 0.88,
    }

    def apply_scene(index):
        result = current['result']
        if result is None:
            for box_id in range(25):
                static_boxes[box_id].visible = True
                box_labels[box_id].visible = True
            for handle in carried.values():
                handle.visible = False
            return
        pair = set(result['boxes'].values())
        removed = set(result.get('removed_before', []))
        attached = result['attach_index'] <= index < result['release_index']
        released = index >= result['release_index']
        for box_id in range(25):
            visible = box_id not in removed
            if box_id in pair:
                visible = visible and not attached and not released
            static_boxes[box_id].visible = visible
            box_labels[box_id].visible = visible
        for side, box_id in result['boxes'].items():
            handle = carried[side]
            handle.visible = attached
            if attached:
                transform = (
                    urdf.get_transform(f'{side}_tool0', urdf.base_link)
                    @ np.asarray(result['tool_to_box'][side]))
                handle.position = transform[:3, 3]
                handle.wxyz = np.roll(
                    Rotation.from_matrix(transform[:3, :3]).as_quat(), 1)

    def show_frame(index):
        with lock, server.atomic():
            index = max(0, min(index, len(current['frames']) - 1))
            current['frame'] = index
            update_robot(current['frames'][index])
            apply_scene(index)
            result = current['result']
            if result is not None:
                phase = next(
                    (stage['name'] for stage in result['stages'] if index <= stage['end']),
                    'complete')
                status.content = (
                    f'**执行中** · 第 `{index + 1}/{len(current["frames"])}` 帧 · '
                    f'阶段 `{phase}`')

    @frame_slider.on_update
    def on_frame(event):
        if not current['planning']:
            show_frame(int(frame_slider.value) - 1)

    @play.on_update
    def on_play(event):
        current['playing'] = bool(play.value)

    for stage_name, button in stage_buttons.items():
        @button.on_click
        def on_stage(event, stage_name=stage_name):
            result = current['result']
            if result is None or stage_name not in result['boundaries']:
                return
            play.value = False
            frame_slider.value = result['boundaries'][stage_name] + 1

    def progress_callback(stage, message):
        progress.value = stage_progress.get(stage, progress.value)
        status.content = f'**计算中** · {message}'

    def planning_worker(round_number):
        try:
            result = planner.plan_round(round_number, progress_callback, sequence_prefix=True)
            output = args.output_root / f'round_{round_number:02}_{int(time.time())}.json'
            output.write_text(json.dumps(result, indent=2) + '\n')
            if not result['success']:
                with lock:
                    current['result'] = None
                    current['frames'] = np.asarray([home_values])
                    current['frame'] = 0
                    current['playing'] = False
                frame_slider.max = 1
                frame_slider.value = 1
                play.value = False
                update_robot(home_values)
                apply_scene(0)
                progress.value = 1.0
                status.content = (
                    f'**规划失败** · `{failure_detail(result)}`\n\n'
                    f'计算耗时 `{result.get("timings", {}).get("total_ms", 0.0) / 1000.0:.3f} s`。'
                    '机器人保持第一初始位，未播放失败轨迹。')
                timing.content = f'结果：`{output}`'
                return
            with lock:
                current['result'] = result
                current['frames'] = np.asarray(result['frames'], dtype=float)
                current['tool_to_box'] = result['tool_to_box']
                current['frame'] = 0
            frame_slider.max = len(current['frames'])
            frame_slider.value = 1
            times = result['timings']
            line_error = max(
                result['constraint_metrics'][frame]['maximum_yz_error_mm']
                for frame in result['active_tool_frames'])
            orientation_error = max(
                result['constraint_metrics'][frame]['maximum_orientation_error_deg']
                for frame in result['active_tool_frames'])
            box_label = ' / '.join(
                f'{side[0].upper()}{box_id}'
                for side, box_id in result['boxes'].items())
            timing.content = (
                f'**计算成功 · 第{round_number}组 {box_label}**\n\n'
                f'- 预抓取约束：`{times["approach_constraint_ms"] / 1000.0:.3f} s`\n'
                f'- 初始位到预抓取：`{times["home_to_pregrasp_ms"] / 1000.0:.3f} s`\n'
                f'- 35cm约束抽离：`{times["extract_ms"] / 1000.0:.3f} s`\n'
                f'- 携箱回初始位：`{times["loaded_return_ms"] / 1000.0:.3f} s`\n'
                f'- 放置规划：`{times["loaded_placement_ms"] / 1000.0:.3f} s`\n'
                f'- 全轨迹复核：`{times["validation_ms"] / 1000.0:.3f} s`\n'
                f'- 总计：`{times["total_ms"] / 1000.0:.3f} s`\n\n'
                f'抽离直线偏差 `{line_error:.3f} mm`，姿态偏差 `{orientation_error:.3f}°`。\n\n'
                f'轨迹 `{len(result["frames"])}` 帧，最终释放并返回第一初始位。')
            progress.value = 1.0
            show_frame(0)
            play.value = True
            status.content = '**规划完成，开始仿真执行。**'
        except Exception as error:
            progress.value = 1.0
            status.content = f'**计算异常** · `{type(error).__name__}: {error}`'
        finally:
            current['planning'] = False
            selector.disabled = False
            calculate.disabled = False

    @calculate.on_click
    def on_calculate(event):
        if current['planning']:
            return
        current['planning'] = True
        current['playing'] = False
        play.value = False
        selector.disabled = True
        calculate.disabled = True
        progress.value = 0.02
        round_number = option_to_round[selector.value]
        status.content = f'**计算中** · 第{round_number}组规划准备'
        threading.Thread(
            target=planning_worker, args=(round_number,), daemon=True).start()

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-3.4, 3.6, 2.65)
        client.camera.look_at = (0.55, 0.0, 1.25)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    update_robot(home_values)
    apply_scene(0)
    print(json.dumps({
        'url': f'http://localhost:{args.port}',
        'options': options,
        'planner_initialization_ms': planner.initialization_ms,
    }, ensure_ascii=False), flush=True)
    while True:
        if current['playing'] and not current['planning'] and len(current['frames']) > 1:
            next_frame = current['frame'] + 1
            if next_frame >= len(current['frames']):
                next_frame = len(current['frames']) - 1
                play.value = False
            frame_slider.value = next_frame + 1
            time.sleep(0.02 / speed.value)
        else:
            time.sleep(0.05)


if __name__ == '__main__':
    main()
