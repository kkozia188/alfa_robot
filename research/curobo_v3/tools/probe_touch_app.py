import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import queue
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf
import yourdfpy

from probe_touch_ik import ROOT, ProbeTouchIK, nearby_targets
from probe_touch_motion import ProbeCollision, interpolate_motion, motion_payload
from ipc_left_prefix.smooth_timing import rt_curves, validate_preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8162)
    parser.add_argument('--backend', default='http://127.0.0.1:8155')
    args = parser.parse_args()
    server = viser.ViserServer(host='127.0.0.1', port=args.port, label='测试杆打表 · 解析IK与左臂运动')
    server.gui.configure_theme(control_width='large', show_share_button=False)
    model = yourdfpy.URDF.load(ROOT / 'generated/zero_model_candidate_20261018/robot_rezero_candidate.urdf', load_meshes=True)
    home = json.loads((ROOT / 'config/rezero_home.json').read_text())
    robot = ViserUrdf(server, model, root_node_name='/preview_robot', mesh_color_override=(140, 185, 220, .65))
    server.scene.add_grid('/ground', width=3, height=3)
    target = server.scene.add_transform_controls('/target', scale=.15, depth_test=False)
    tip = server.scene.add_icosphere('/probe_tip', radius=.009, color=(230, 40, 220))
    rod = server.scene.add_line_segments('/probe', points=np.zeros((1, 2, 3)), colors=(230, 40, 220), line_width=5)
    server.gui.add_markdown('**左7轴 · Updown固定−0.3m · Joint7＋Z 70mm测试TCP**\n拖动坐标轴/旋转环，或输入XYZ/RPY。求解→选构型→计算轨迹→播放预览→确认→获取控制权→手动发送。不会自动使能或发送。\n**碰撞仅模型参考**：右臂按Home、底盘cuboid；未建模真实工作台、接触物和杆体。请先在空旷处低速测试。')
    sync = server.gui.add_button('目标对齐实时杆尖／清除旧预览')
    position = server.gui.add_vector3('目标XYZ / m', initial_value=(.4, .4, 1.2), step=.001)
    orientation = server.gui.add_vector3('目标RPY / 度', initial_value=(0., 0., 0.), step=1.)
    tilt = server.gui.add_slider('附近朝向偏转 / 度', min=0., max=30., step=5., initial_value=25.)
    solve = server.gui.add_button('1 求解解析IK候选')
    family = server.gui.add_dropdown('候选朝向分组', options=['原朝向'])
    selector = server.gui.add_dropdown('候选构型（按关节距离排序）', options=['尚未求解'])
    detail = server.gui.add_markdown('等待求解。')
    speed = server.gui.add_dropdown('速度上限', options=['5°/s', '10°/s', '15°/s'], initial_value='5°/s')
    plan = server.gui.add_button('2 从实时姿态生成关节插值轨迹')
    play = server.gui.add_checkbox('播放轨迹预览', initial_value=False)
    timeline = server.gui.add_slider('轨迹时间 / s', min=0., max=1., step=.01, initial_value=0.)
    confirm = server.gui.add_checkbox('已看碰撞提示并预览；现场确认可运动，仅发送左7轴', initial_value=False)
    lease = server.gui.add_button('获取18121控制权')
    send = server.gui.add_button('发送已预览轨迹', disabled=True)
    cancel = server.gui.add_button('取消本后端Action', disabled=True)
    report = server.gui.add_markdown('未发送实机运动。')
    status_text = server.gui.add_markdown('读取反馈中…')
    state = dict(status={}, rows=[], all_rows=[], payload=None, curves=None, future=None, seen=False, initialized=False, revision=0)
    events = queue.Queue()
    worker = ThreadPoolExecutor(max_workers=1)
    poller = ThreadPoolExecutor(max_workers=1)
    resources = dict(ik=None, collision=None)
    evidence = ROOT / 'generated/probe_touch_runs'
    evidence.mkdir(parents=True, exist_ok=True)
    pending_status = None

    def request(path, data=None):
        body = None if data is None else json.dumps(data).encode()
        query = Request(args.backend + path, data=body, headers={'Content-Type': 'application/json', 'X-Alfa-Request': 'left-prefix'})
        try:
            with urlopen(query, timeout=8) as response:
                return json.load(response)
        except HTTPError as error:
            raise ValueError(json.load(error).get('error', str(error))) from None

    def feedback():
        status = request('/api/status')
        age = status.get('feedback_age_s')
        if age is None or age > .5 or status.get('feedback_deg') is None:
            raise ValueError('左臂反馈过期，不能使用旧状态')
        if status.get('busy'):
            raise ValueError('已有动作运行，等待停稳')
        return np.deg2rad(status['feedback_deg'])

    def pose_matrix():
        pose = np.eye(4)
        pose[:3, 3] = target.position
        pose[:3, :3] = Rotation.from_quat(np.roll(target.wxyz, -1)).as_matrix()
        return pose

    def clear():
        state['revision'] += 1
        state['payload'] = state['curves'] = None
        state['seen'] = False
        play.value = confirm.value = False

    @target.on_update
    def dragged(_event):
        events.put('drag')

    @position.on_update
    @orientation.on_update
    def typed(_event):
        events.put('typed')

    @sync.on_click
    def synchronize(_event):
        events.put('sync')

    @tilt.on_update
    def tilt_changed(_event):
        events.put('invalidate')

    @speed.on_update
    def speed_changed(_event):
        events.put('clear')

    @selector.on_update
    def selected(_event):
        events.put('clear')

    @family.on_update
    def family_changed(_event):
        events.put('family')

    def solve_work(pose, angle):
        seed = feedback()
        if resources['ik'] is None:
            resources['ik'] = ProbeTouchIK(-.3)
        rows = []
        for label, desired, deviation in nearby_targets(pose, angle):
            for row in resources['ik'].solve(desired, seed):
                rows.append(dict(row, orientation_label=label, target=desired, deviation_deg=deviation))
        rows.sort(key=lambda row: (row['deviation_deg'] != 0, row['joint_distance_rad']))
        return rows

    @solve.on_click
    def solve_clicked(_event):
        events.put('solve')

    def plan_work(row, velocity):
        start = feedback()
        points, timing = interpolate_motion(start, row['joints'], velocity)
        if resources['collision'] is None:
            resources['collision'] = ProbeCollision()
        collision = resources['collision'].inspect_motion(points)
        payload = motion_payload(points, timing, collision['collision_samples'], row['target'], collision)
        validate_preview(payload)
        return payload

    @plan.on_click
    def plan_clicked(_event):
        events.put('plan')

    @lease.on_click
    def lease_clicked(_event):
        events.put('lease')

    @send.on_click
    def send_clicked(_event):
        events.put('send')

    @cancel.on_click
    def cancel_clicked(_event):
        events.put('cancel')

    @server.on_client_connect
    def connected(client):
        client.camera.position = (2.3, 2.7, 2.0)
        client.camera.look_at = (.35, .2, 1.1)
        client.camera.up_direction = (0., 0., 1.)

    last_poll = 0.
    clock = time.monotonic()
    print(f'READY http://127.0.0.1:{args.port}/', flush=True)
    while True:
        now = time.monotonic()
        elapsed = now - clock
        clock = now
        if pending_status is not None and pending_status.done():
            try:
                state['status'] = pending_status.result()
                status_text.content = '**RT状态：** ' + (state['status'].get('reason') or '就绪')
                if not state['initialized'] and state['status'].get('feedback_deg') is not None:
                    events.put('sync')
                    state['initialized'] = True
            except Exception as error:
                state['status'] = {}
                status_text.content = str(error)
            pending_status = None
        if now - last_poll > .5 and pending_status is None:
            last_poll = now
            pending_status = poller.submit(request, '/api/status')
        while not events.empty():
            event = events.get()
            try:
                if event in ('drag', 'typed', 'sync', 'invalidate'):
                    clear()
                    state['rows'] = []
                    state['all_rows'] = []
                    selector.options = ['尚未求解']
                    selector.value = '尚未求解'
                    if event == 'typed':
                        target.position = position.value
                        target.wxyz = np.roll(Rotation.from_euler('xyz', orientation.value, degrees=True).as_quat(), 1)
                    elif event == 'sync':
                        pose = ProbeTouchIK(-.3).forward(feedback())
                        target.position = pose[:3, 3]
                        target.wxyz = np.roll(Rotation.from_matrix(pose[:3, :3]).as_quat(), 1)
                    if event != 'typed':
                        position.value = tuple(target.position)
                        orientation.value = tuple(Rotation.from_quat(np.roll(target.wxyz, -1)).as_euler('xyz', degrees=True))
                    detail.content = '目标已更新，请重新求解；没有自动发送。'
                elif event == 'clear':
                    clear()
                elif event == 'family':
                    clear()
                    state['rows'] = [row for row in state['all_rows'] if row['orientation_label'] == family.value]
                    selector.options = [f"{index + 1} · {row['orientation_label']} · 关节距离{np.rad2deg(row['joint_distance_rad']):.1f}°" for index, row in enumerate(state['rows'])] or ['无IK解']
                    selector.value = selector.options[0]
                elif event in ('solve', 'plan'):
                    if state['future'] is not None:
                        continue
                    clear()
                    revision = state['revision']
                    if event == 'solve':
                        state['future'] = (worker.submit(solve_work, pose_matrix(), tilt.value), event, revision)
                        report.content = '解析IK枚举中…'
                    else:
                        if not state['rows']:
                            raise ValueError('先求解并选择IK')
                        row = state['rows'][selector.options.index(selector.value)]
                        state['future'] = (worker.submit(plan_work, row, float(speed.value.split('°')[0])), event, revision)
                        report.content = '读取当前姿态、生成插值并参考验碰…'
                elif event == 'lease':
                    report.content = str(request('/api/release' if state['status'].get('lease') == 'you' else '/api/acquire', {}))
                elif event == 'send':
                    payload = state['payload']
                    if not payload or not confirm.value or not state['seen'] or state['future'] is not None:
                        raise ValueError('请先计算、播放预览并确认')
                    validate_preview(payload)
                    request('/api/prepare', payload)
                    report.content = str(request('/api/move', dict(direction='forward', preview_id=payload['preview_id'])))
                    confirm.value = False
                elif event == 'cancel':
                    report.content = str(request('/api/stop', {}))
            except Exception as error:
                report.content = '**未发送／操作失败：** ' + str(error)
        if state['future'] is not None:
            future, kind, revision = state['future']
            if future.done():
                state['future'] = None
                try:
                    result = future.result()
                    if revision != state['revision']:
                        raise ValueError('目标已变化，丢弃旧计算结果')
                    if kind == 'solve':
                        state['all_rows'] = result
                        family.options = list(dict.fromkeys(row['orientation_label'] for row in result)) or ['无IK解']
                        family.value = family.options[0]
                        events.put('family')
                        (evidence / 'latest_ik.json').write_text(json.dumps([
                            {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in row.items()}
                            for row in result], ensure_ascii=False, indent=2))
                        report.content = f'找到{len(result)}个解析IK解。候选是几何可达解，运动需再验碰。'
                    else:
                        state['payload'] = result
                        state['curves'] = rt_curves(result['points'])
                        (evidence / ('preview_' + result['preview_id'][:16] + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
                        timeline.max = result['report']['duration_s']
                        timeline.value = 0.
                        checked = result['report']
                        warning = '参考模型未检出碰撞。' if checked['collision_passed'] else f"**参考碰撞警告：{checked['collision_frames']}帧不合法，首次{checked['first_collision_s']:.3f}秒。按用户要求仅提示，不作为发送门禁；不代表实机无碰撞。**"
                        report.content = f"**轨迹准备好：{len(result['points'])}点／{timeline.max:.1f}秒／10Hz**\n{warning}\n先播放预览，再确认发送。"
                except Exception as error:
                    report.content = '**计算未通过：** ' + str(error)
        if play.value and state['curves']:
            timeline.value = min(timeline.max, timeline.value + elapsed)
            if timeline.value >= timeline.max:
                play.value = False
                state['seen'] = True
        values = {name: 0. for name in model.actuated_joint_names}
        values['updown'] = -.3
        values.update({f'right_joint{index}': float(np.deg2rad(angle)) for index, angle in enumerate(home['right_deg'], 1)})
        left = np.deg2rad(state['status'].get('feedback_deg') or home['left_deg'])
        if state['curves']:
            left = np.array([curve(timeline.value) for curve in state['curves']])
        elif state['rows']:
            row = state['rows'][selector.options.index(selector.value)]
            left = row['joints']
            detail.content = '**候选角度（度）**：`' + str(np.round(np.rad2deg(left), 3).tolist()) + f"`\n\n{row['orientation_label']}；FK位置误差{row['position_error_m'] * 1000:.6f}mm；朝向误差{np.rad2deg(row['rotation_error_rad']):.6f}°。"
        values.update({f'left_joint{index}': float(angle) for index, angle in enumerate(left, 1)})
        robot.update_cfg(np.array([values[name] for name in model.actuated_joint_names]))
        frame = model.get_transform('left_link7', model.base_link)
        tip.position = frame[:3, 3] + frame[:3, :3] @ [0., 0., .070]
        rod.points = np.array([[frame[:3, 3], tip.position]])
        send.disabled = not bool(state['payload'] and state['seen'] and confirm.value and state['status'].get('ready') and state['future'] is None)
        cancel.disabled = not bool(state['status'].get('busy'))
        time.sleep(.03)


if __name__ == '__main__':
    main()
