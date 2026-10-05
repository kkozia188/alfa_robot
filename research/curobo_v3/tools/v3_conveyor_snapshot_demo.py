import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy

from curobo_core.conveyor import DEFAULT_CONVEYOR_MESH, conveyor_pair
from curobo_core.contracts import PlannerAssets
from curobo_core.inspection import SnapshotInspector
from curobo_core.instances import LiveScene
from curobo_core.scene import Pose


def planar_pose(position_x, position_y, yaw_deg):
    angle = math.radians(yaw_deg) / 2
    return Pose((position_x, position_y, 0), (math.cos(angle), 0, 0, math.sin(angle)))


def main():
    parser = argparse.ArgumentParser(description="Live conveyor instances and frozen cuRobo collision snapshot")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_CONVEYOR_MESH)
    parser.add_argument("--gap-m", type=float, default=0.05)
    parser.add_argument("--follow-distance-m", type=float, default=1.9)
    parser.add_argument("--snapshot-output", type=Path, default=Path("/tmp/motion246_snapshot.json"))
    args = parser.parse_args()
    assets = PlannerAssets()
    inspector = SnapshotInspector(assets)
    template = inspector.backend.snapshot
    live = LiveScene(template)
    pair = conveyor_pair(args.mesh, gap_m=args.gap_m)
    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="龙头车实时实例与规划快照")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=12, height=5)
    world = server.scene.add_frame("/world", show_axes=True)
    live_root = server.scene.add_frame("/live", show_axes=False)
    frozen_root = server.scene.add_frame("/snapshot", show_axes=False, visible=False)
    robots = {}
    for layer, color in (("live", (0.60, 0.70, 0.82, 0.85)), ("snapshot", (0.25, 0.70, 0.60, 0.85))):
        root = server.scene.add_frame(f"/{layer}/robot", show_axes=True)
        urdf = yourdfpy.URDF.load(assets.urdf, load_meshes=True)
        robots[layer] = (root, ViserUrdf(server, urdf, root_node_name=f"/{layer}/robot", mesh_color_override=color))
    for item in template.objects:
        if item.object_id != "ground":
            server.scene.add_box(f"/world/{item.object_id}", dimensions=item.dimensions_m,
                                 position=item.pose.position, wxyz=item.pose.quaternion_wxyz,
                                 color=(140, 155, 175), opacity=0.18)
    handles = {}
    for layer in ("live", "snapshot"):
        for part in pair.parts:
            handles[layer, part.object_id] = server.scene.add_mesh_simple(
                f"/{layer}/conveyor/{part.object_id}", vertices=np.asarray(part.vertices_m),
                faces=np.asarray(part.faces), color=(230, 155, 55) if layer == "live" else (65, 190, 130),
                opacity=0.85)
    with server.gui.add_folder("实时实例状态"):
        robot_x = server.gui.add_number("机器人 x (m)", initial_value=0.0, step=0.05)
        robot_y = server.gui.add_number("机器人 y (m)", initial_value=0.0, step=0.05)
        robot_yaw = server.gui.add_number("机器人 yaw (deg)", initial_value=0.0, step=5.0)
        pair_x = server.gui.add_number("龙头车组合 x (m)", initial_value=-args.follow_distance_m, step=0.05)
        pair_y = server.gui.add_number("龙头车组合 y (m)", initial_value=0.0, step=0.05)
        pair_yaw = server.gui.add_number("龙头车组合 yaw (deg)", initial_value=0.0, step=5.0)
        collision = server.gui.add_checkbox("龙头车参与下次规划碰撞", initial_value=True)
        follow = server.gui.add_button("播放驶入与后方跟随")
        stop = server.gui.add_button("停止播放")
    with server.gui.add_folder("规划快照"):
        mode = server.gui.add_dropdown("显示内容", options=("实时场景", "本次规划快照"), initial_value="实时场景")
        capture = server.gui.add_button("冻结当前状态")
        check = server.gui.add_button("检测冻结快照碰撞")
        status = server.gui.add_markdown("先冻结，再检测。数字编辑仅改变实时实例，旧快照不随之变化。")
    current = {"snapshot": None, "playing": False, "started": 0.0, "busy": False}
    lock = threading.RLock()

    def update_live():
        stamp = time.time_ns()
        state = replace(template.state, base_pose=planar_pose(robot_x.value, robot_y.value, robot_yaw.value), stamp_ns=stamp)
        instance = replace(pair, pose=planar_pose(pair_x.value, pair_y.value, pair_yaw.value), stamp_ns=stamp,
                           collision_enabled=collision.value)
        live.update_state(state)
        live.upsert(instance)
        return state, instance

    def draw(layer, state, parts):
        root, robot = robots[layer]
        root.position = state.base_pose.position
        root.wxyz = state.base_pose.quaternion_wxyz
        positions = dict(zip(state.joint_names, state.positions))
        robot.update_cfg(np.asarray([positions.get(name, 0.0) for name in robot.get_actuated_joint_names()]))
        available = {part.object_id.rsplit("/", 1)[-1]: part for part in parts}
        for part in pair.parts:
            handle = handles[layer, part.object_id]
            handle.visible = part.object_id in available
            if handle.visible:
                geometry = available[part.object_id]
                handle.position = geometry.pose.position
                handle.wxyz = geometry.pose.quaternion_wxyz

    @follow.on_click
    def on_follow(_event):
        with lock:
            current.update(playing=True, started=time.monotonic())

    @stop.on_click
    def on_stop(_event):
        with lock:
            current["playing"] = False

    @capture.on_click
    def on_capture(_event):
        with lock:
            state, instance = update_live()
            snapshot = live.capture(time.time_ns(), 1.0)
            current["snapshot"] = snapshot
            draw("snapshot", state, snapshot.meshes)
            args.snapshot_output.parent.mkdir(parents=True, exist_ok=True)
            args.snapshot_output.write_text(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2) + "\n")
            status.content = f"已冻结 revision={snapshot.revision}；包含 {len(snapshot.meshes)} 个龙头车网格。\n\n实时更新不会修改这份快照。"

    @check.on_click
    def on_check(_event):
        with lock:
            snapshot = current["snapshot"]
            if snapshot is None or current["busy"]:
                return
            current["busy"] = True
            check.disabled = True

        def compute():
            try:
                result = inspector.check(snapshot)
                verdict = "通过：未检测到碰撞或越限" if result["valid"] else "不通过：碰撞或越限"
                status.content = f"冻结 revision={snapshot.revision}：{verdict}\n\n环境碰撞={result['environment_collision']}；参与检测龙头车网格数={len(snapshot.meshes)}。"
                args.snapshot_output.with_suffix(".collision.json").write_text(json.dumps(result, indent=2) + "\n")
            except Exception as error:
                status.content = f"检测失败：{type(error).__name__}: {error}"
            finally:
                with lock:
                    current["busy"] = False
                    check.disabled = False
        threading.Thread(target=compute, daemon=True).start()

    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (-5, -5, 3.5)
        client.camera.look_at = (-0.8, 0, 1)
        client.camera.up_direction = (0, 0, 1)

    print(f"Ready: http://127.0.0.1:{args.port}", flush=True)
    while True:
        with lock:
            if current["playing"]:
                progress = min(1.0, (time.monotonic() - current["started"]) / 5.0)
                robot_x.value = -3.0 + 3.0 * progress
                pair_x.value = robot_x.value - args.follow_distance_m
                if progress == 1.0:
                    current["playing"] = False
            state, instance = update_live()
            draw("live", state, instance.world_parts())
            live_root.visible = mode.value == "实时场景"
            frozen_root.visible = mode.value == "本次规划快照" and current["snapshot"] is not None
        time.sleep(0.05)


if __name__ == "__main__":
    main()
