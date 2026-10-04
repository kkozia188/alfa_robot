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


class InteractivePlanner:
    def __init__(self, args):
        self.args = args
        self.robot = yaml.safe_load(args.robot_config.read_text())
        self.mobile_robot = yaml.safe_load(args.mobile_robot_config.read_text())
        self.box_fit = json.loads(args.box_fit.read_text())
        self.home = yaml.safe_load(args.named_poses.read_text())["named_poses"]["home"]
        self.front = chassis_front_x(args.urdf, self.home)
        self.home_values = np.asarray(
            [self.home[name] for name in ACTIVE_JOINTS], dtype=np.float32
        )
        self.target_values = np.zeros(15, dtype=np.float32)
        self.target_values[0] = 0.0
        self.target_values[1:] = np.deg2rad([
            130, -105, 10, 90, -90, -40, 0,
            -130, 105, -10, -90, 90, 40, 0,
        ])
        self.task_options = tasks()
        self.task_map = dict(self.task_options)
        self.selected_task = self.task_options[0][1]
        self.random_start = None
        self.result = None
        self.busy = False
        self.random_counter = 0
        self.planner_kind = "informed_rrt"
        self._cached_task = None
        self._cached_solver = None
        self._cached_checkers = {}

    def search_path(self, start, goals, lower, upper, validity, budget, seed):
        if self.planner_kind == "rrtconnect":
            return batched_rrt_connect_multi_goal(start, goals, lower, upper, validity, budget, seed)
        if self.planner_kind == "prm":
            return batched_prm_multi_goal(start, goals, lower, upper, validity, budget, seed)
        return batched_rrt_multi_goal(
            start, goals, lower, upper, validity, budget, seed,
            apply_shortcut=False, informed_sampling=self.planner_kind == "informed_rrt",
        )

    def prepare_task(self, boxes):
        key = tuple(sorted(boxes.items()))
        if key != self._cached_task:
            self._cached_solver = None
            self._cached_checkers.clear()
            self._cached_task = key
            gc.collect()

    def ik_solver(self, boxes):
        self.prepare_task(boxes)
        if self._cached_solver is None:
            self._cached_solver = InverseKinematics(InverseKinematicsCfg.create(
                robot=copy.deepcopy(self.robot), scene_model=self.scene(boxes),
                collision_cache={"cuboid": 40}, num_seeds=512,
                self_collision_check=True, use_cuda_graph=False,
                position_tolerance=0.002, orientation_tolerance=math.radians(1),
                override_iters_for_multi_link_ik=500,
                optimizer_collision_activation_distance=0.005,
            ))
        return self._cached_solver

    def collision_checker(self, boxes, payload=False, mobile=False):
        self.prepare_task(boxes)
        key = (payload, mobile)
        if key not in self._cached_checkers:
            robot = self.mobile_robot if mobile else self.robot
            configured = loaded_robot(robot, self.box_fit) if payload else copy.deepcopy(robot)
            self._cached_checkers[key] = RobotCollisionChecker(
                RobotCollisionCheckerCfg.load_from_config(
                    robot_config=configured, scene_model=self.scene(boxes),
                    n_cuboids=40, n_meshes=0, collision_activation_distance=0.0,
                ))
        return self._cached_checkers[key]

    def scene(self, boxes):
        scene = make_scene(self.front, 0.9, set(boxes.values()))
        scene.cuboid = [item for item in scene.cuboid if item.name != "ground"]
        return scene

    def random_extract_start(self, boxes):
        started_all = time.perf_counter()
        scene = self.scene(boxes)
        solver = InverseKinematics(InverseKinematicsCfg.create(
            robot=copy.deepcopy(self.robot), scene_model=scene,
            collision_cache={"cuboid": 40}, num_seeds=512,
            self_collision_check=True, use_cuda_graph=False,
            position_tolerance=0.002, orientation_tolerance=math.radians(1.0),
            override_iters_for_multi_link_ik=500,
            optimizer_collision_activation_distance=0.005,
        ))
        home_state = JointState.from_position(
            torch.tensor(self.home_values, device="cuda").reshape(1, -1),
            joint_names=ACTIVE_JOINTS,
        )
        contact_poses = {}
        for side, box_id in boxes.items():
            position = wall_center(self.front, 0.9, box_id)
            position[0] -= 0.150001
            contact_poses[f"{side}_tool0"] = {
                "position": position,
                "quaternion": canonical_side_suction_quaternion_wxyz(side),
            }
        contact = solver.solve_pose(goal(contact_poses), home_state, return_seeds=32)
        names = list(contact.js_solution.joint_names)
        values = contact.js_solution.position.reshape(-1, len(names))
        success = torch.nonzero(contact.success.reshape(-1).bool(), as_tuple=False).reshape(-1)
        if not len(success):
            raise RuntimeError("接触IK无解")
        indices = torch.tensor([names.index(name) for name in ACTIVE_JOINTS], device="cuda")
        contact_candidates = values[success][:, indices]
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20261003 + self.random_counter)
        order = torch.randperm(len(contact_candidates), generator=generator, device="cuda")
        contact_candidates = contact_candidates[order]
        loaded = loaded_robot(self.robot, self.box_fit, active_sides=("left", "right"))
        checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=copy.deepcopy(loaded), scene_model=scene,
            n_cuboids=40, n_meshes=0, collision_activation_distance=0.0,
        ))
        validity = GpuValidity(
            checker, active_sides=("left", "right"), max_box_tilt_deg=89.0,
            stability_weight=10.0, check_ground=True, ground_z=0.0,
            joint_names=ACTIVE_JOINTS,
        )
        candidates = []
        for contact_values in contact_candidates:
            contact_state = JointState.from_position(
                contact_values.reshape(1, -1), joint_names=ACTIVE_JOINTS
            )
            actual = solver.compute_kinematics(contact_state).tool_poses.to_dict()
            extract_poses = {frame: pose_dict(actual[frame]) for frame in contact_poses}
            for frame in extract_poses:
                extract_poses[frame]["position"] = extract_poses[frame]["position"].copy()
                extract_poses[frame]["position"][0] -= 0.35
            extracted = solver.solve_pose(
                goal(extract_poses), contact_state, return_seeds=16
            )
            extracted_names = list(extracted.js_solution.joint_names)
            extracted_values = extracted.js_solution.position.reshape(-1, len(extracted_names))
            extracted_success = torch.nonzero(
                extracted.success.reshape(-1).bool(), as_tuple=False
            ).reshape(-1)
            if not len(extracted_success):
                continue
            extracted_indices = torch.tensor(
                [extracted_names.index(name) for name in ACTIVE_JOINTS], device="cuda"
            )
            candidates.extend(
                extracted_values[extracted_success][:, extracted_indices]
            )
        if not candidates:
            raise RuntimeError("抽离终点IK无解")
        candidate_tensor = torch.stack(candidates)
        valid = validity.mask(candidate_tensor)
        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if not len(valid_indices):
            raise RuntimeError("抽离终点全部碰撞")
        selected = candidate_tensor[valid_indices[0]].detach().cpu().numpy()
        self.random_counter += 1
        del solver, checker, validity
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "values": selected.tolist(),
            "generation_ms": (time.perf_counter() - started_all) * 1000.0,
            "candidate_count": int(len(valid_indices)),
        }

    def plan(self, boxes, start_values):
        started_all = time.perf_counter()
        timings = {}
        scene = self.scene(boxes)
        checker = self.collision_checker(boxes, payload=True)
        validity = GpuValidity(
            checker, active_sides=("left", "right"), max_box_tilt_deg=89.0,
            stability_weight=10.0, check_ground=True, ground_z=0.0,
            joint_names=ACTIVE_JOINTS,
        )
        start = torch.tensor(start_values, device="cuda", dtype=torch.float32)
        target_solver = self.ik_solver(boxes)
        target_poses = json.loads(DEFAULT_TARGET_POSES.read_text())
        torch.cuda.synchronize()
        timings["setup"] = (time.perf_counter() - started_all) * 1000
        started_ik = time.perf_counter()
        target_result = target_solver.solve_pose(
            goal(target_poses), JointState.from_position(
                start.reshape(1, -1), joint_names=ACTIVE_JOINTS
            ), return_seeds=64,
        )
        torch.cuda.synchronize()
        target_ik_ms = (time.perf_counter() - started_ik) * 1000.0
        timings["target_ik"] = target_ik_ms
        started_filter = time.perf_counter()
        names = list(target_result.js_solution.joint_names)
        values = target_result.js_solution.position.reshape(-1, len(names))
        successful = torch.nonzero(
            target_result.success.reshape(-1).bool(), as_tuple=False
        ).reshape(-1)
        indices = torch.tensor([names.index(name) for name in ACTIVE_JOINTS], device="cuda")
        goals = values[successful][:, indices]
        if not len(goals):
            raise RuntimeError("目标IK成功0个")
        full_valid, stability = validity.evaluate(goals)
        if not bool(full_valid.any().item()):
            collision_only = GpuValidity(
                checker,
                active_sides=(),
                check_ground=True,
                ground_z=0.0,
                joint_names=ACTIVE_JOINTS,
            )
            collision_valid = collision_only.mask(goals)
            stability_valid = stability <= 1.0 - validity.minimum_box_up_z
            raise RuntimeError(
                f"目标IK成功{len(goals)}个；碰撞通过{int(collision_valid.sum().item())}个；"
                f"箱体稳定性通过{int(stability_valid.sum().item())}个；联合合法0个"
            )
        goals = goals[full_valid]
        metric = torch.tensor([5.0] + [1.0] * 14, device="cuda")
        goals = goals[torch.argsort(torch.sum(((goals - start) * metric) ** 2, dim=1))[:16]]
        torch.cuda.synchronize()
        timings["target_filter"] = (time.perf_counter() - started_filter) * 1000
        started_search = time.perf_counter()
        path, stats = self.search_path(
            start, goals,
            checker.kinematics.get_joint_limits().position[0] + 1e-5,
            checker.kinematics.get_joint_limits().position[1] - 1e-5,
            validity, 2.0, 20261003 + self.random_counter,
        )
        search_ms = (time.perf_counter() - started_search) * 1000.0
        timings["rrt"] = search_ms
        started_assembly = time.perf_counter()
        if path is None:
            raise RuntimeError("2秒内未找到搬运路径")
        transport = densify(path, [5.0] + [1.0] * 14)
        mobile_names = list(
            self.mobile_robot.get("robot_cfg", self.mobile_robot)["kinematics"]["cspace"]["joint_names"]
        )
        full_frames = []
        for values15 in transport:
            mapping = dict(zip(ACTIVE_JOINTS, values15))
            full_frames.append([mapping.get(name, 0.0) for name in mobile_names])
        endpoint = full_frames[-1]
        yaw_index = mobile_names.index("base_yaw")
        for step in range(1, 361):
            frame = endpoint.copy()
            frame[yaw_index] = math.pi * step / 360.0
            full_frames.append(frame)
        timings["frame_assembly"] = (time.perf_counter() - started_assembly) * 1000
        started_mobile_setup = time.perf_counter()
        mobile_checker = self.collision_checker(boxes, payload=True, mobile=True)
        mobile_validity = GpuValidity(
            mobile_checker, active_sides=("left", "right"), max_box_tilt_deg=89.0,
            stability_weight=10.0, check_ground=True, ground_z=0.0,
            joint_names=mobile_names,
            weights=[2.0, 2.0, 1.0, 5.0] + [1.0] * 14,
        )
        torch.cuda.synchronize()
        timings["mobile_checker_setup"] = (time.perf_counter() - started_mobile_setup) * 1000
        started_validation = time.perf_counter()
        combined_valid, combined_stability = mobile_validity.evaluate(torch.tensor(
            full_frames, device="cuda", dtype=torch.float32
        ))
        invalid = torch.nonzero(~combined_valid, as_tuple=False).reshape(-1)
        torch.cuda.synchronize()
        timings["transport_and_turn_validation"] = (time.perf_counter() - started_validation) * 1000
        failure = None
        if len(invalid):
            index = int(invalid[0].item())
            stage = "搬运" if index < len(transport) else "Yaw180"
            failure_state = JointState.from_position(
                torch.tensor([full_frames[index]], device="cuda", dtype=torch.float32),
                joint_names=mobile_names,
            )
            failure = {
                "frame_index": index,
                "stage": stage,
                "yaw_deg": math.degrees(full_frames[index][yaw_index]),
                "audit": audit_state(mobile_checker.kinematics, failure_state, scene),
            }
            failure_fk = mobile_checker.kinematics.compute_kinematics(failure_state)
            failure["spheres"] = failure_fk.robot_spheres.detach().cpu().reshape(-1, 4).tolist()
        result = {
            "success": failure is None,
            "failure": failure,
            "valid_frames": combined_valid.cpu().tolist(),
            "joint_names": mobile_names,
            "frames": full_frames,
            "transport_frames": len(transport),
            "yaw_frames": 360,
            "target_ik_ms": target_ik_ms,
            "search_ms": search_ms,
            "total_ms": (time.perf_counter() - started_all) * 1000.0,
            "rrt_stats": stats,
            "max_box_tilt_deg": math.degrees(math.acos(max(
                -1.0, min(1.0, 1.0 - float(combined_stability.max().item()))
            ))),
        }
        started_cleanup = time.perf_counter()
        del target_solver, checker, validity, mobile_checker, mobile_validity
        torch.cuda.synchronize()
        timings["cleanup"] = (time.perf_counter() - started_cleanup) * 1000
        result["timing_ms"] = timings
        return result


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
