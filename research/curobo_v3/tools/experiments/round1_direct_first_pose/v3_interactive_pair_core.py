#!/usr/bin/env python3

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState

import sys
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools')
sys.path.insert(0, '/mnt/mydisk/ALFA/curobo_v2_ws/tools/experiments/round1_full_plan')
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_constrained_extract import constraint_metrics  # noqa: E402
from direct_first_pose_plan import goal  # noqa: E402
from finalize_smooth_validate import resample_segment  # noqa: E402
from v3_round1_full_plan import (  # noqa: E402
    ACTIVE_JOINTS, BOX_HALF, add_payload_links, obb_overlap, payload_grid,
    result_trajectory, state_rows, tensor_state, update_payloads,
)
from v3_wall_ik_benchmark import (  # noqa: E402
    canonical_side_tool_to_box, make_scene, wall_center,
)


SUPPORTED_ROUNDS = tuple(range(1, 12))
CONSTRAINT_SCALE = 100.0
LIMIT_MARGIN_RAD = 1e-5


def make_planner_cfg(robot, scene):
    return MotionPlannerCfg.create(
        robot=robot,
        scene_model=scene,
        collision_cache={'cuboid': 40},
        num_ik_seeds=4,
        num_trajopt_seeds=2,
        self_collision_check=True,
        use_cuda_graph=True,
        position_tolerance=0.002,
        orientation_tolerance=math.radians(1.0),
    )


def pose_json(pose):
    return {
        'position': pose.position.reshape(-1, 3)[0].detach().cpu().tolist(),
        'quaternion_wxyz': pose.quaternion.reshape(-1, 4)[0].detach().cpu().tolist(),
    }


def append_segment(output, segment):
    if not segment:
        return
    if output:
        delta = max(abs(a - b) for a, b in zip(output[-1], segment[0]))
        if delta > 2e-5:
            raise ValueError(f'trajectory boundary mismatch: {delta}')
        output.extend(segment[1:])
    else:
        output.extend(segment)


def batch_robot_validation(checker, frames):
    if not frames:
        return [], None
    q = torch.tensor(frames, device='cuda', dtype=torch.float32).unsqueeze(0)
    horizon = q.shape[1]
    checker.setup_batch_tensors(1, horizon)
    state = checker.kinematics.compute_kinematics(
        JointState.from_position(q, joint_names=ACTIVE_JOINTS))
    spheres = state.robot_spheres.view(1, horizon, -1, 4)
    checker.collision_constraint.update_num_spheres(
        spheres.shape[2], batch_size=1, horizon=horizon)
    distance = (
        checker.get_self_collision(spheres).reshape(1, horizon, -1).sum(-1)
        + checker.collision_constraint.forward(state).reshape(1, horizon, -1).sum(-1)
        + checker.get_bound(q).reshape(1, horizon, -1).sum(-1)
    )
    valid = (distance == 0)[0].detach().cpu().numpy()
    return np.flatnonzero(~valid).tolist(), state


def exact_obstacles(summary, removed_boxes):
    obstacles = []
    for box_id in range(25):
        if box_id not in removed_boxes:
            obstacles.append((
                f'wall_box_{box_id}',
                wall_center(summary['chassis_front_x_m'], summary['wall_distance_m'], box_id),
                np.eye(3), BOX_HALF,
            ))
    wall_back = summary['chassis_front_x_m'] + summary['wall_distance_m'] + 0.30 + 1e-6
    obstacles.extend([
        ('ground', np.array([wall_back - 2.0, 0.0, -0.05]), np.eye(3), np.array([2.1, 1.3, 0.05])),
        ('left_wall', np.array([wall_back - 2.0, -1.25, 1.2]), np.eye(3), np.array([2.0, 0.05, 1.2])),
        ('right_wall', np.array([wall_back - 2.0, 1.25, 1.2]), np.eye(3), np.array([2.0, 0.05, 1.2])),
        ('front_wall', np.array([wall_back + 0.05, 0.0, 1.2]), np.eye(3), np.array([0.05, 1.3, 1.2])),
        ('ceiling', np.array([wall_back - 2.0, 0.0, 2.45]), np.eye(3), np.array([2.1, 1.3, 0.05])),
    ])
    return obstacles


def exact_payload_failures(state, tool_to_box, obstacles):
    tool_poses = state.tool_poses.to_dict()
    box_poses = {}
    for side in tool_to_box:
        positions = tool_poses[f'{side}_tool0'].position.reshape(-1, 3).detach().cpu().numpy()
        quaternions = tool_poses[f'{side}_tool0'].quaternion.reshape(-1, 4).detach().cpu().numpy()
        centers = []
        rotations = []
        for position, quaternion in zip(positions, quaternions):
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()
            transform[:3, 3] = position
            box = transform @ tool_to_box[side]
            centers.append(box[:3, 3])
            rotations.append(box[:3, :3])
        box_poses[side] = (centers, rotations)
    failures = {}
    first_side = next(iter(box_poses))
    for index in range(len(box_poses[first_side][0])):
        for side in tool_to_box:
            center = box_poses[side][0][index]
            rotation = box_poses[side][1][index]
            for name, obstacle_center, obstacle_rotation, half in obstacles:
                if obb_overlap(center, rotation, BOX_HALF, obstacle_center, obstacle_rotation, half):
                    failures.setdefault(index, f'payload_{side}_collision:{name}')
                    break
        if len(box_poses) == 2 and obb_overlap(
                box_poses['left'][0][index], box_poses['left'][1][index], BOX_HALF,
                box_poses['right'][0][index], box_poses['right'][1][index], BOX_HALF):
            failures.setdefault(index, 'payload_payload_collision')
    return failures


class InteractivePairPlanner:
    def __init__(
        self, summary_path, named_poses_path, placement_reference_path,
        stage_cache_path=None, payload_fit_path=None,
        defer_payload_collision_until_extracted=False,
    ):
        self.summary_path = Path(summary_path)
        self.summary = json.loads(self.summary_path.read_text())
        self.rounds = {item['round']: item for item in self.summary['rounds']}
        self.named = yaml.safe_load(Path(named_poses_path).read_text())['named_poses']
        self.placement_reference = json.loads(
            Path(placement_reference_path).read_text())['frames']
        self.stage_cache = {}
        self.stage_cache_joint_names = []
        if stage_cache_path is not None:
            cache = json.loads(Path(stage_cache_path).read_text())
            self.stage_cache_joint_names = cache['joint_names']
            self.stage_cache = {entry['round']: entry for entry in cache['entries']}
        self.payload_fit = None
        self.payload_fit_kind = 'regular_grid_48'
        if payload_fit_path is not None:
            comparison = json.loads(Path(payload_fit_path).read_text())
            self.payload_fit = comparison['official_48']
            self.payload_fit_kind = comparison.get('kind', 'official_sphere_fit')
        self.defer_payload_collision_until_extracted = defer_payload_collision_until_extracted
        self.robot = yaml.safe_load(Path(self.summary['robot_config']).read_text())
        self.loaded_robot = copy.deepcopy(self.robot)
        add_payload_links(self.loaded_robot)
        first_pair = tuple(self.boxes_for_round(SUPPORTED_ROUNDS[0]).values())
        first_scene = make_scene(
            self.summary['chassis_front_x_m'], self.summary['wall_distance_m'], set(first_pair))
        started = time.perf_counter()
        self.free_pose_planner = MotionPlanner(make_planner_cfg(self.robot, first_scene))
        self.free_cspace_planner = MotionPlanner(make_planner_cfg(self.robot, first_scene))
        self.loaded_pose_planner = MotionPlanner(make_planner_cfg(self.loaded_robot, first_scene))
        self.loaded_cspace_planner = MotionPlanner(make_planner_cfg(self.loaded_robot, first_scene))
        self.free_checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=self.robot, scene_model=first_scene,
            n_cuboids=40, n_meshes=0, collision_activation_distance=0.0))
        self.loaded_checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=self.loaded_robot, scene_model=first_scene,
            n_cuboids=40, n_meshes=0, collision_activation_distance=0.0))
        self.initialization_ms = (time.perf_counter() - started) * 1000.0
        home = tensor_state([self.named['home'][name] for name in ACTIVE_JOINTS])
        lower, upper = self.free_cspace_planner.kinematics.get_joint_limits().position
        home.position = torch.clamp(
            home.position, min=lower + LIMIT_MARGIN_RAD, max=upper - LIMIT_MARGIN_RAD)
        self.home = home
        unloading = tensor_state([self.named['unloading'][name] for name in ACTIVE_JOINTS])
        loaded_lower, loaded_upper = self.loaded_cspace_planner.kinematics.get_joint_limits().position
        unloading.position = torch.clamp(
            unloading.position,
            min=loaded_lower + LIMIT_MARGIN_RAD,
            max=loaded_upper - LIMIT_MARGIN_RAD,
        )
        self.unloading = unloading

    def boxes_for_round(self, round_number):
        item = self.rounds[round_number]
        boxes = {}
        if item['left_box'] is not None:
            boxes['left'] = item['left_box']
        if item['right_box'] is not None:
            boxes['right'] = item['right_box']
        if not boxes:
            raise ValueError(f'round {round_number} has no box target')
        return boxes

    def options(self):
        output = []
        for round_number in SUPPORTED_ROUNDS:
            boxes = self.boxes_for_round(round_number)
            label = ' / '.join(f'{side[0].upper()}{box_id}' for side, box_id in boxes.items())
            output.append(f'第{round_number}组 · {label}')
        return output

    def contact_for_round(self, round_number):
        if round_number in self.stage_cache:
            entry = self.stage_cache[round_number]
            values = dict(zip(
                self.stage_cache_joint_names,
                entry['approach'][-1]['joints'],
            ))
            return tensor_state([values[name] for name in ACTIVE_JOINTS])
        return tensor_state(self.rounds[round_number]['collision_free']['joints'])

    def payload_spheres(self, tool_position, tool_quaternion, box_center):
        if self.payload_fit is None:
            return payload_grid(tool_position, tool_quaternion, box_center)
        tool_rotation = Rotation.from_quat(np.roll(tool_quaternion, -1)).as_matrix()
        rows = []
        for center, radius in zip(
            self.payload_fit['centers'], self.payload_fit['radii']):
            world = box_center + np.asarray(center, dtype=float)
            local = tool_rotation.T @ (world - tool_position)
            rows.append([*local.tolist(), float(radius)])
        while len(rows) < 48:
            rows.append([0.0, 0.0, 0.0, -100.0])
        if len(rows) > 48:
            raise ValueError(f'payload fit has {len(rows)} spheres, capacity is 48')
        return np.asarray(rows, dtype=np.float32)

    def removed_before_round(self, round_number):
        removed = set()
        for item in self.summary['rounds']:
            if item['round'] >= round_number:
                break
            if item['left_box'] is not None:
                removed.add(item['left_box'])
            if item['right_box'] is not None:
                removed.add(item['right_box'])
        return removed

    def _set_scene(self, removed_boxes):
        scene = make_scene(
            self.summary['chassis_front_x_m'], self.summary['wall_distance_m'], set(removed_boxes))
        self.free_pose_planner.update_world(scene)
        self.free_cspace_planner.update_world(scene)
        self.loaded_pose_planner.update_world(scene)
        self.loaded_cspace_planner.update_world(scene)
        self.free_checker.update_world(scene)
        self.loaded_checker.update_world(scene)
        return scene

    def _plan_pose_constrained(
        self, planner, start, targets, axis, project_to_goal, active_sides,
    ):
        criterion = ToolPoseCriteria.linear_motion(
            axis=axis,
            non_terminal_scale=CONSTRAINT_SCALE,
            project_distance_to_goal=project_to_goal,
        )
        hold = ToolPoseCriteria.track_position_and_orientation(
            non_terminal_scale=CONSTRAINT_SCALE)
        planner.update_tool_pose_criteria({
            f'{side}_tool0': criterion if side in active_sides else hold
            for side in ('left', 'right')
        })
        started = time.perf_counter()
        result = planner.plan_pose(goal(targets), start, max_attempts=3, enable_graph_attempt=1)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        standard = ToolPoseCriteria()
        planner.update_tool_pose_criteria({
            'left_tool0': standard,
            'right_tool0': standard,
        })
        trajectory = result_trajectory(result)
        return ([] if trajectory is None else state_rows(trajectory)), elapsed_ms

    def _plan_cspace(self, planner, start, finish):
        started = time.perf_counter()
        result = planner.plan_cspace(finish, start, max_attempts=3, enable_graph_attempt=1)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        trajectory = result_trajectory(result)
        return ([] if trajectory is None else state_rows(trajectory)), elapsed_ms

    def _validate_loaded_path(self, frames, tool_to_box, removed_boxes):
        failed, state = batch_robot_validation(self.loaded_checker, frames)
        payload_failed = exact_payload_failures(
            state, tool_to_box, exact_obstacles(self.summary, removed_boxes))
        return failed, payload_failed

    def plan_round(
        self, round_number, progress=None, sequence_prefix=True,
        direct_loaded_to_placement=False,
    ):
        if round_number not in SUPPORTED_ROUNDS:
            return {'success': False, 'failure_stage': 'unsupported_round'}
        notify = progress or (lambda stage, message: None)
        started_total = time.perf_counter()
        item = self.rounds[round_number]
        boxes = self.boxes_for_round(round_number)
        active_sides = tuple(boxes)
        pair = tuple(boxes.values())
        removed_before = self.removed_before_round(round_number) if sequence_prefix else set()
        removed_for_planning = removed_before | set(pair)
        if not item['collision_free']['success']:
            return {'success': False, 'failure_stage': 'contact_ik_unavailable'}
        self._set_scene(removed_for_planning)
        contact = self.contact_for_round(round_number)
        timings = {}

        notify('pregrasp', '计算5cm预抓取约束轨迹')
        contact_fk = self.free_pose_planner.compute_kinematics(contact).tool_poses.to_dict()
        contact_poses = {
            frame: pose_json(contact_fk[frame])
            for frame in ('left_tool0', 'right_tool0')
        }
        pregrasp_targets = copy.deepcopy(contact_poses)
        for side in active_sides:
            frame = f'{side}_tool0'
            rotation = Rotation.from_quat(
                np.roll(pregrasp_targets[frame]['quaternion_wxyz'], -1)).as_matrix()
            pregrasp_targets[frame]['position'] = (
                np.asarray(pregrasp_targets[frame]['position']) - rotation[:, 2] * 0.05
            ).tolist()
        retreat_frames, timings['approach_constraint_ms'] = self._plan_pose_constrained(
            self.free_pose_planner, contact, pregrasp_targets, 'z', True, active_sides)
        if not retreat_frames:
            return self._failure('pregrasp_constraint', timings, started_total)
        approach_frames = list(reversed(retreat_frames))
        pregrasp = tensor_state(retreat_frames[-1])

        notify('pregrasp', '规划第一初始位到预抓取位')
        home_to_pregrasp, timings['home_to_pregrasp_ms'] = self._plan_cspace(
            self.free_cspace_planner, self.home, pregrasp)
        if not home_to_pregrasp:
            return self._failure('home_to_pregrasp', timings, started_total)

        notify('extract', '构建双负载并执行35cm约束抽离')
        contact = tensor_state(approach_frames[-1])
        contact_fk = self.loaded_pose_planner.compute_kinematics(contact).tool_poses.to_dict()
        contact_poses = {
            frame: pose_json(contact_fk[frame])
            for frame in ('left_tool0', 'right_tool0')
        }
        grids = {}
        tool_to_box = {}
        disabled_grid = np.zeros((48, 4), dtype=np.float32)
        disabled_grid[:, 3] = -100.0
        for side in ('left', 'right'):
            if side not in boxes:
                grids[side] = disabled_grid.copy()
                continue
            box_id = boxes[side]
            pose = contact_fk[f'{side}_tool0']
            position = pose.position.reshape(-1, 3)[0].detach().cpu().numpy()
            quaternion = pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy()
            center = wall_center(
                self.summary['chassis_front_x_m'], self.summary['wall_distance_m'], box_id)
            grids[side] = self.payload_spheres(position, quaternion, center)
            tool_to_box[side] = canonical_side_tool_to_box(side)
        if self.defer_payload_collision_until_extracted:
            extract_grids = {
                'left': disabled_grid.copy(),
                'right': disabled_grid.copy(),
            }
            update_payloads(self.loaded_pose_planner, extract_grids)
        else:
            update_payloads(self.loaded_pose_planner, grids)
        update_payloads(self.loaded_cspace_planner, grids)
        update_payloads(self.loaded_checker, grids)
        extract_targets = copy.deepcopy(contact_poses)
        for side in active_sides:
            frame = f'{side}_tool0'
            extract_targets[frame]['position'][0] -= 0.35
        extract_frames, timings['extract_ms'] = self._plan_pose_constrained(
            self.loaded_pose_planner, contact, extract_targets, 'x', False, active_sides)
        if not extract_frames:
            return self._failure('constrained_extract', timings, started_total)
        if self.defer_payload_collision_until_extracted:
            extract_robot_failed, extract_state = batch_robot_validation(
                self.free_checker, extract_frames)
            extract_payload_failed = exact_payload_failures(
                extract_state, tool_to_box,
                exact_obstacles(self.summary, removed_for_planning))
            if extract_robot_failed or extract_payload_failed:
                timings['extract_robot_failures'] = len(extract_robot_failed)
                timings['extract_payload_failures'] = len(extract_payload_failed)
                if extract_payload_failed:
                    first_index = min(extract_payload_failed)
                    timings['extract_first_payload_failure'] = {
                        'frame': first_index,
                        'reason': extract_payload_failed[first_index],
                    }
                return self._failure('extract_exact_validation', timings, started_total)
        extract_metrics = constraint_metrics(
            self.loaded_pose_planner.kinematics, extract_frames, contact_poses, 0.35)
        if len(active_sides) == 1:
            active_frame = f'{active_sides[0]}_tool0'
            extract_metrics = {
                active_frame: extract_metrics[active_frame],
                'maximum_dual_progress_difference_mm': None,
            }

        if direct_loaded_to_placement:
            notify('placement', '抽离终点直接规划携箱到第一放置位')
            placement, timings['loaded_direct_placement_ms'] = self._plan_cspace(
                self.loaded_cspace_planner,
                tensor_state(extract_frames[-1]),
                self.unloading,
            )
            if not placement:
                return self._failure('loaded_direct_placement', timings, started_total)
            placement_robot_failed, placement_payload_failed = self._validate_loaded_path(
                placement, tool_to_box, removed_for_planning)
            if placement_robot_failed or placement_payload_failed:
                timings['loaded_placement_robot_failures'] = len(placement_robot_failed)
                timings['loaded_placement_payload_failures'] = len(placement_payload_failed)
                return self._failure(
                    'loaded_direct_placement_validation', timings, started_total)
            placement_method = 'curobo_motion_planner_direct_from_extracted'
            notify('placement', '规划释放后的空载放置位到第一初始位')
            empty_return, timings['empty_return_home_ms'] = self._plan_cspace(
                self.free_cspace_planner, self.unloading, self.home)
            if not empty_return:
                return self._failure('empty_return_home', timings, started_total)
            loaded_stages = [('loaded_direct_placement', placement, True)]
        else:
            notify('loaded_return', '规划携箱回第一初始位')
            loaded_return, timings['loaded_return_ms'] = self._plan_cspace(
                self.loaded_cspace_planner, tensor_state(extract_frames[-1]), self.home)
            if not loaded_return:
                return self._failure('loaded_return', timings, started_total)

            notify('placement', '验证第一放置参考轨迹')
            placement = copy.deepcopy(self.placement_reference)
            placement[0] = list(loaded_return[-1])
            placement_robot_failed, placement_payload_failed = self._validate_loaded_path(
                placement, tool_to_box, removed_for_planning)
            if placement_robot_failed or placement_payload_failed:
                notify('placement', '参考轨迹不适用，cuRobo重新规划携箱放置')
                placement, timings['loaded_placement_ms'] = self._plan_cspace(
                    self.loaded_cspace_planner, self.home, self.unloading)
                if not placement:
                    return self._failure('loaded_placement', timings, started_total)
                placement_robot_failed, placement_payload_failed = self._validate_loaded_path(
                    placement, tool_to_box, removed_for_planning)
                if placement_robot_failed or placement_payload_failed:
                    timings['loaded_placement_robot_failures'] = len(placement_robot_failed)
                    timings['loaded_placement_payload_failures'] = len(placement_payload_failed)
                    return self._failure('loaded_placement_validation', timings, started_total)
                placement_method = 'curobo_motion_planner'
            else:
                timings['loaded_placement_ms'] = 0.0
                placement_method = 'validated_reference'
            notify('placement', '生成释放后的空载返程')
            empty_return = [list(frame) for frame in reversed(placement)]
            loaded_stages = [
                ('loaded_return_home', loaded_return, True),
                ('loaded_placement', placement, True),
            ]
        raw_stages = [
            ('home_to_pregrasp', home_to_pregrasp, False),
            ('cartesian_approach_5cm', approach_frames, False),
            ('cartesian_extract_35cm', extract_frames, True),
            *loaded_stages,
            ('empty_return_home', empty_return, False),
        ]
        stages = []
        frames = []
        boundaries = {}
        for name, raw_frames, attached in raw_stages:
            segment = resample_segment(raw_frames)
            append_segment(frames, segment)
            boundaries[name] = len(frames) - 1
            stages.append({
                'name': name,
                'attached': attached,
                'frames': len(segment),
                'end': len(frames) - 1,
            })
        attach_index = boundaries['cartesian_approach_5cm']
        release_index = boundaries[
            'loaded_direct_placement' if direct_loaded_to_placement else 'loaded_placement']

        notify('validate', 'GPU批量复核机器人和真实双箱碰撞')
        validation_started = time.perf_counter()
        extract_end_index = boundaries['cartesian_extract_35cm']
        loaded_begin_index = (
            extract_end_index if self.defer_payload_collision_until_extracted
            else attach_index)
        free_prefix = frames[:loaded_begin_index + 1]
        loaded_segment = frames[loaded_begin_index:release_index + 1]
        empty_segment = frames[release_index:]
        free_failed, _ = batch_robot_validation(self.free_checker, free_prefix)
        loaded_failed, loaded_state = batch_robot_validation(self.loaded_checker, loaded_segment)
        empty_failed, _ = batch_robot_validation(self.free_checker, empty_segment)
        _, attached_state = batch_robot_validation(
            self.free_checker, frames[attach_index:release_index + 1])
        payload_failed = exact_payload_failures(
            attached_state, tool_to_box, exact_obstacles(self.summary, removed_for_planning))
        timings['validation_ms'] = (time.perf_counter() - validation_started) * 1000.0
        validation = {
            'free_failure_count': len(free_failed),
            'loaded_failure_count': len(loaded_failed),
            'empty_failure_count': len(empty_failed),
            'payload_failure_count': len(payload_failed),
            'first_free_failure': free_failed[0] if free_failed else None,
            'first_loaded_failure': loaded_failed[0] if loaded_failed else None,
            'first_empty_failure': empty_failed[0] if empty_failed else None,
            'first_payload_failure': (
                {'frame': min(payload_failed), 'reason': payload_failed[min(payload_failed)]}
                if payload_failed else None),
        }
        success = not free_failed and not loaded_failed and not empty_failed and not payload_failed
        timings['total_ms'] = (time.perf_counter() - started_total) * 1000.0
        return {
            'success': success,
            'failure_stage': None if success else 'final_validation',
            'round': round_number,
            'pair': [item['left_box'], item['right_box']],
            'boxes': boxes,
            'active_tool_frames': [f'{side}_tool0' for side in active_sides],
            'scene_mode': 'sequence_prefix' if sequence_prefix else 'full_wall',
            'removed_before': sorted(removed_before),
            'joint_names': ACTIVE_JOINTS,
            'frames': frames,
            'stages': stages,
            'boundaries': boundaries,
            'attach_index': attach_index,
            'release_index': release_index,
            'timings': timings,
            'constraint_metrics': extract_metrics,
            'placement_method': placement_method,
            'direct_loaded_to_placement': direct_loaded_to_placement,
            'validation': validation,
            'tool_to_box': {side: transform.tolist() for side, transform in tool_to_box.items()},
            'wall_distance_m': self.summary['wall_distance_m'],
            'chassis_front_x_m': self.summary['chassis_front_x_m'],
            'robot_config': self.summary['robot_config'],
            'payload_collision_model': self.payload_fit_kind,
            'payload_collision_enabled_from': (
                'extract_end' if self.defer_payload_collision_until_extracted
                else 'attachment'),
        }

    @staticmethod
    def _failure(stage, timings, started_total):
        timings['total_ms'] = (time.perf_counter() - started_total) * 1000.0
        return {
            'success': False,
            'failure_stage': stage,
            'timings': timings,
            'frames': [],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--named-poses', type=Path, required=True)
    parser.add_argument('--placement-reference', type=Path, required=True)
    parser.add_argument('--stage-cache', type=Path)
    parser.add_argument('--payload-fit', type=Path)
    parser.add_argument('--defer-payload-collision-until-extracted', action='store_true')
    parser.add_argument('--direct-loaded-to-placement', action='store_true')
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    planner = InteractivePairPlanner(
        args.summary, args.named_poses, args.placement_reference,
        args.stage_cache, args.payload_fit,
        args.defer_payload_collision_until_extracted)
    result = planner.plan_round(
        args.round,
        lambda stage, message: print(f'[{stage}] {message}', flush=True),
        direct_loaded_to_placement=args.direct_loaded_to_placement)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    summary = copy.deepcopy(result)
    summary.pop('frames', None)
    summary.pop('tool_to_box', None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
