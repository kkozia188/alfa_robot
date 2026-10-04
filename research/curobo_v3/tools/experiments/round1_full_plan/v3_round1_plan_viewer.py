import argparse
import json
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser
from viser.extras import ViserUrdf
import yaml
import yourdfpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    report = json.loads(args.result.read_text())
    config = yaml.safe_load(Path(report['robot_config']).read_text())['kinematics']
    urdf_path = Path(config['urdf_path'])
    frames = np.asarray(report['stages'][0]['rejected_frames'])
    names = report['joint_names']
    sample_dt = float(np.asarray(report['stages'][0]['joint_state']['dt']).reshape(-1)[0])
    limits = {}
    for joint in ET.parse(urdf_path).getroot().findall('joint'):
        limit = joint.find('limit')
        if limit is not None and 'lower' in limit.attrib:
            limits[joint.attrib['name']] = (float(limit.attrib['lower']), float(limit.attrib['upper']))
    updown_index = names.index('updown')
    worst_index = int(frames[:, updown_index].argmax())
    server = viser.ViserServer(host='127.0.0.1', port=args.port, label='ALFA · 第一组规划诊断')
    server.scene.set_up_direction('+z')
    server.scene.add_frame('/robot', show_axes=False)
    server.scene.add_grid('/floor_grid', width=5, height=4)
    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=True, build_scene_graph=True)
    robot = ViserUrdf(server, urdf, root_node_name='/robot', mesh_color_override=(0.72, 0.77, 0.83, 0.8))
    actuated = list(robot.get_actuated_joint_names())
    scene_root = server.scene.add_frame('/container', show_axes=False)
    boxes_root = server.scene.add_frame('/boxes', show_axes=False)
    sphere_root = server.scene.add_frame('/collision_spheres', show_axes=False, visible=False)
    frame_nodes = {}
    sphere_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1)
    for link, spheres in config['collision_spheres'].items():
        frame_nodes[link] = server.scene.add_frame(f'/collision_spheres/{link}', show_axes=False)
        radii = np.array([sphere['radius'] for sphere in spheres])
        positions = np.array([sphere['center'] for sphere in spheres])
        quaternions = np.zeros((len(spheres), 4))
        quaternions[:, 0] = 1
        server.scene.add_batched_meshes_simple(
            f'/collision_spheres/{link}/spheres', vertices=sphere_mesh.vertices, faces=sphere_mesh.faces,
            batched_positions=positions, batched_wxyzs=quaternions, batched_scales=radii,
            batched_colors=(80, 174, 209), opacity=0.55,
        )
    front = report['chassis_front_x_m'] + report['wall_distance_m']
    box_handles = {}
    for box_id in range(25):
        center = (front + 0.15, (box_id % 5 - 2) * 0.41, 0.2 + (box_id // 5) * 0.41)
        target = box_id in report['pair']
        box_handles[box_id] = server.scene.add_box(
            f'/boxes/box_{box_id:02}', position=center, dimensions=(0.3, 0.4, 0.4),
            color=(234, 136, 60) if target else (121, 139, 164), opacity=0.75 if target else 0.2,
        )
        if target:
            server.scene.add_label(f'/boxes/box_{box_id:02}/label', text=f'目标 {box_id}', position=(-0.16, 0, 0.24))
    wall_back = front + 0.3 + 1e-6
    objects = [
        ('ground', (wall_back-2, 0, -0.05), (4.2, 2.6, 0.1)),
        ('left_wall', (wall_back-2, -1.25, 1.2), (4, 0.1, 2.4)),
        ('right_wall', (wall_back-2, 1.25, 1.2), (4, 0.1, 2.4)),
        ('front_wall', (wall_back+0.05, 0, 1.2), (0.1, 2.6, 2.4)),
        ('ceiling', (wall_back-2, 0, 2.45), (4.2, 2.6, 0.1)),
    ]
    for name, center, dimensions in objects:
        server.scene.add_box(f'/container/{name}', position=center, dimensions=dimensions,
                             color=(185, 155, 106), opacity=0.07 if name != 'ground' else 0.12)
    tcp_nodes = {side: server.scene.add_frame(f'/tcp/{side}', axes_length=0.08, axes_radius=0.003)
                 for side in ('left', 'right')}
    lift_label = server.scene.add_label('/updown_status', text='')
    with server.gui.add_folder('第一组 L24 / R20'):
        server.gui.add_markdown('**规划未通过：仅回放被拒候选**\n\n当前停在“初始 → 吸附”。抽离与放置未执行。')
        frame_slider = server.gui.add_slider('帧', min=1, max=len(frames), step=1, initial_value=worst_index+1)
        play = server.gui.add_checkbox('播放候选', initial_value=False)
        speed = server.gui.add_slider('播放速度', min=0.1, max=2.0, step=0.1, initial_value=0.5)
        with server.gui.add_folder('定位与显示'):
            beginning = server.gui.add_button('回到起点')
            failure = server.gui.add_button('查看最大越界')
            show_boxes = server.gui.add_checkbox('显示箱墙', initial_value=True)
            show_container = server.gui.add_checkbox('显示集装箱', initial_value=True)
            show_spheres = server.gui.add_checkbox('显示规划碰撞球', initial_value=False)
        status = server.gui.add_markdown('')
        server.gui.add_markdown('初始 Updown −0.300 m；上限 0 m。红字表示越界。\n\n碰撞球仍为预览，立柱覆盖不足。画面中的箱子未吸附。')
    update_lock = threading.Lock()

    def show_frame(index):
        with update_lock, server.atomic():
            values = dict(zip(names, frames[index].tolist()))
            robot.update_cfg(np.array([values.get(name, 0.0) for name in actuated]))
            for link, node in frame_nodes.items():
                matrix = urdf.get_transform(link, urdf.base_link)
                node.position = matrix[:3, 3]
                node.wxyz = np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)
            for side, node in tcp_nodes.items():
                matrix = urdf.get_transform(f'{side}_tool0', urdf.base_link)
                node.position = matrix[:3, 3]
                node.wxyz = np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)
            updown = values['updown']
            violation = max(0.0, updown-limits['updown'][1], limits['updown'][0]-updown)
            carriage = urdf.get_transform('arm_carriage', urdf.base_link)
            lift_label.position = carriage[:3, 3] + np.array([0.0, 0.0, 0.25])
            lift_label.text = f'Updown {updown:+.6f} m' + (f' · 越界 {violation*1000:.2f} mm' if violation > 1e-6 else '')
            message = f'**Updown 越界 {violation*1000:.2f} mm**' if violation > 1e-6 else '当前帧 Updown 在范围内'
            color = '#d33b40' if violation > 1e-6 else '#23805d'
            status.content = (f'第 **{index+1}/{len(frames)}** 帧 · 候选时间 **{index*sample_dt:.3f} s**\n\n'
                              f'<span style="color:{color}">{message}</span>\n\n'
                              f'Updown：**{updown:+.6f} m**；范围：**[-1, 0] m**')

    @frame_slider.on_update
    def on_frame(event):
        show_frame(int(frame_slider.value)-1)

    @beginning.on_click
    def on_beginning(event):
        play.value = False
        frame_slider.value = 1

    @failure.on_click
    def on_failure(event):
        play.value = False
        frame_slider.value = worst_index+1

    @show_boxes.on_update
    def on_boxes(event):
        boxes_root.visible = show_boxes.value

    @show_container.on_update
    def on_container(event):
        scene_root.visible = show_container.value

    @show_spheres.on_update
    def on_spheres(event):
        sphere_root.visible = show_spheres.value

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-3.6, 3.8, 2.6)
        client.camera.look_at = (0.55, 0.0, 1.15)
        client.camera.up_direction = (0, 0, 1)

    show_frame(worst_index)
    print(f'Ready at http://localhost:{args.port}; rejected candidate, {len(frames)} frames', flush=True)
    while True:
        if play.value:
            frame_slider.value = int(frame_slider.value) % len(frames) + 1
            time.sleep(sample_dt / speed.value)
        else:
            time.sleep(0.05)


if __name__ == '__main__':
    main()
