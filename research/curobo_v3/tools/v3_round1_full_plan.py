#!/usr/bin/env python3

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose

sys.path.insert(0, str(Path(__file__).parent))
from v3_wall_ik_benchmark import (  # noqa: E402
    ACTIVE_JOINTS,
    contact_position,
    make_scene,
    wall_center,
)


PAIR = (24, 20)
BOX_HALF = np.array([0.15, 0.20, 0.20])


def load_robot(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def tensor_state(values: list[float]) -> JointState:
    return JointState.from_position(
        torch.tensor([values], device="cuda", dtype=torch.float32),
        joint_names=ACTIVE_JOINTS,
    )


def result_trajectory(result) -> JointState | None:
    if result is None or result.success is None or not bool(result.success.reshape(-1)[0].item()):
        return None
    return result.get_interpolated_plan()


def state_rows(state: JointState) -> list[list[float]]:
    values = state.position.detach().cpu().numpy()
    return values.reshape(-1, values.shape[-1]).tolist()


def payload_grid(tool_position: np.ndarray, tool_quaternion_wxyz: np.ndarray, center: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(np.roll(tool_quaternion_wxyz, -1)).as_matrix()
    rows = []
    for x in (-0.10, 0.0, 0.10):
        for y in (-0.15, -0.05, 0.05, 0.15):
            for z in (-0.15, -0.05, 0.05, 0.15):
                world = center + np.array([x, y, z])
                local = rotation.T @ (world - tool_position)
                rows.append([*local.tolist(), 0.05])
    return np.asarray(rows, dtype=np.float32)


def add_payload_links(robot_data: dict) -> None:
    kin = robot_data.get("robot_cfg", robot_data)["kinematics"]
    kin["extra_links"] = copy.deepcopy(kin.get("extra_links") or {})
    kin["extra_collision_spheres"] = copy.deepcopy(kin.get("extra_collision_spheres") or {})
    kin["grasp_contact_link_names"] = list(kin.get("grasp_contact_link_names") or [])
    for side in ("left", "right"):
        link = f"{side}_payload"
        kin["extra_links"][link] = {
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_name": f"{side}_payload_fixed",
            "joint_type": "FIXED",
            "link_name": link,
            "parent_link_name": f"{side}_tool0",
        }
        kin["extra_collision_spheres"][link] = 48
        kin.setdefault("self_collision_buffer", {})[link] = 0.0
        kin.setdefault("self_collision_ignore", {}).setdefault(f"{side}_link7", []).append(link)
        kin["self_collision_ignore"].setdefault(link, []).append(f"{side}_link7")
        kin["grasp_contact_link_names"].append(link)


def update_payloads(planner_or_ik, grids: dict[str, np.ndarray]) -> None:
    if hasattr(planner_or_ik, "ik_solver"):
        kinematics = planner_or_ik.ik_solver.kinematics
    else:
        kinematics = planner_or_ik.kinematics
    params = kinematics.config.kinematics_config
    for side, grid in grids.items():
        params.update_link_spheres(
            f"{side}_payload",
            torch.tensor(grid, device="cuda", dtype=torch.float32),
        )


def pose_goal(poses: dict[str, dict]) -> GoalToolPose:
    return GoalToolPose.from_poses(
        {
            frame: Pose(
                position=torch.tensor(data["position"], device="cuda").reshape(1, 3),
                quaternion=torch.tensor(data["quaternion_wxyz"], device="cuda").reshape(1, 4),
            )
            for frame, data in poses.items()
        },
        ordered_tool_frames=["left_tool0", "right_tool0"],
    )


def solve_extraction(
    ik: InverseKinematics,
    start: JointState,
    contact_fk: dict[str, dict],
    distance: float = 0.35,
    step: float = 0.01,
) -> tuple[list[list[float]], str]:
    frames = [start.position.reshape(-1).detach().cpu().tolist()]
    current = start
    count = int(round(distance / step))
    for index in range(1, count + 1):
        fraction = index / count
        poses = copy.deepcopy(contact_fk)
        for frame in ("left_tool0", "right_tool0"):
            poses[frame]["position"][0] -= distance * fraction
        result = ik.solve_pose(pose_goal(poses), current_state=current, return_seeds=4)
        if not bool(result.success.reshape(-1)[0].item()):
            return frames, f"extraction_ik_failed@{index}/{count}"
        solution = result.js_solution.position.reshape(-1, len(ACTIVE_JOINTS))[0]
        current = JointState.from_position(solution.reshape(1, -1), joint_names=ACTIVE_JOINTS)
        frames.append(solution.detach().cpu().tolist())
    return frames, ""


def obb_overlap(
    center_a: np.ndarray,
    rotation_a: np.ndarray,
    half_a: np.ndarray,
    center_b: np.ndarray,
    rotation_b: np.ndarray,
    half_b: np.ndarray,
    epsilon: float = 1e-8,
) -> bool:
    r = rotation_a.T @ rotation_b
    abs_r = np.abs(r) + epsilon
    t = rotation_a.T @ (center_b - center_a)
    for i in range(3):
        if abs(t[i]) > half_a[i] + np.dot(half_b, abs_r[i, :]):
            return False
    for j in range(3):
        if abs(np.dot(t, r[:, j])) > half_b[j] + np.dot(half_a, abs_r[:, j]):
            return False
    for i in range(3):
        for j in range(3):
            ra = half_a[(i + 1) % 3] * abs_r[(i + 2) % 3, j] + half_a[(i + 2) % 3] * abs_r[(i + 1) % 3, j]
            rb = half_b[(j + 1) % 3] * abs_r[i, (j + 2) % 3] + half_b[(j + 2) % 3] * abs_r[i, (j + 1) % 3]
            value = abs(t[(i + 2) % 3] * r[(i + 1) % 3, j] - t[(i + 1) % 3] * r[(i + 2) % 3, j])
            if value > ra + rb:
                return False
    return True


def payload_transform(tool_pose, tool_to_box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = tool_pose.position.reshape(-1, 3)[0].detach().cpu().numpy()
    quaternion = tool_pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy()
    tool_rotation = Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = tool_rotation
    transform[:3, 3] = position
    box = transform @ tool_to_box
    return box[:3, 3], box[:3, :3]


def validate_payload_path(
    kinematics,
    frames: list[list[float]],
    tool_to_box: dict[str, np.ndarray],
    front_x: float,
    wall_distance: float,
) -> tuple[bool, str]:
    obstacles = []
    for box_id in range(25):
        if box_id in PAIR:
            continue
        obstacles.append((f"wall_box_{box_id}", wall_center(front_x, wall_distance, box_id), np.eye(3), BOX_HALF))
    wall_back = front_x + wall_distance + 0.30 + 1e-6
    obstacles.extend([
        ("ground", np.array([wall_back - 2.0, 0.0, -0.05]), np.eye(3), np.array([2.1, 1.3, 0.05])),
        ("left_wall", np.array([wall_back - 2.0, -1.25, 1.2]), np.eye(3), np.array([2.0, 0.05, 1.2])),
        ("right_wall", np.array([wall_back - 2.0, 1.25, 1.2]), np.eye(3), np.array([2.0, 0.05, 1.2])),
        ("front_wall", np.array([wall_back + 0.05, 0.0, 1.2]), np.eye(3), np.array([0.05, 1.3, 1.2])),
        ("ceiling", np.array([wall_back - 2.0, 0.0, 2.45]), np.eye(3), np.array([2.1, 1.3, 0.05])),
    ])
    for frame_index, row in enumerate(frames):
        state = tensor_state(row)
        poses = kinematics.compute_kinematics(state).tool_poses.to_dict()
        boxes = {}
        for side in ("left", "right"):
            boxes[side] = payload_transform(poses[f"{side}_tool0"], tool_to_box[side])
            center, rotation = boxes[side]
            for name, obstacle_center, obstacle_rotation, obstacle_half in obstacles:
                if obb_overlap(center, rotation, BOX_HALF, obstacle_center, obstacle_rotation, obstacle_half):
                    return False, f"payload_{side}_collision:{name}@{frame_index}"
        if obb_overlap(boxes["left"][0], boxes["left"][1], BOX_HALF, boxes["right"][0], boxes["right"][1], BOX_HALF):
            return False, f"payload_payload_collision@{frame_index}"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ik-summary", type=Path, required=True)
    parser.add_argument("--named-poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.ik_summary.read_text())
    round_one = summary["rounds"][0]
    contact = round_one["collision_free"]
    if not contact["success"]:
        raise RuntimeError("round one contact IK is not available")
    robot_data = load_robot(Path(summary["robot_config"]))
    scene = make_scene(summary["chassis_front_x_m"], summary["wall_distance_m"], set(PAIR))
    named = yaml.safe_load(args.named_poses.read_text())["named_poses"]
    home_state = tensor_state([named["home"][name] for name in ACTIVE_JOINTS])
    contact_state = tensor_state(contact["joints"])
    unloading_state = tensor_state([named["unloading"][name] for name in ACTIVE_JOINTS])

    report = {"success": False, "stages": [], "joint_names": ACTIVE_JOINTS}
    started_total = time.perf_counter()

    free_cfg = MotionPlannerCfg.create(
        robot=robot_data,
        scene_model=scene,
        collision_cache={"cuboid": 40},
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        self_collision_check=True,
        use_cuda_graph=False,
        position_tolerance=0.002,
        orientation_tolerance=math.radians(1.0),
    )
    free_planner = MotionPlanner(free_cfg)
    started = time.perf_counter()
    approach_result = free_planner.plan_cspace(contact_state, home_state, max_attempts=3)
    approach = result_trajectory(approach_result)
    report["stages"].append({
        "name": "home_to_contact",
        "success": approach is not None,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "frames": [] if approach is None else state_rows(approach),
    })
    if approach is None:
        report["failure_stage"] = "home_to_contact"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return

    contact_fk = free_planner.compute_kinematics(contact_state).tool_poses.to_dict()
    contact_pose_json = {}
    payload_grids = {}
    tool_to_box = {}
    for side, box_id in zip(("left", "right"), PAIR):
        frame = f"{side}_tool0"
        pose = contact_fk[frame]
        tool_position = pose.position.reshape(-1, 3)[0].detach().cpu().numpy()
        tool_quaternion = pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy()
        box_center = wall_center(summary["chassis_front_x_m"], summary["wall_distance_m"], box_id)
        payload_grids[side] = payload_grid(tool_position, tool_quaternion, box_center)
        tool_rotation = Rotation.from_quat(np.roll(tool_quaternion, -1)).as_matrix()
        tool_transform = np.eye(4)
        tool_transform[:3, :3] = tool_rotation
        tool_transform[:3, 3] = tool_position
        box_transform = np.eye(4)
        box_transform[:3, 3] = box_center
        tool_to_box[side] = np.linalg.inv(tool_transform) @ box_transform
        contact_pose_json[frame] = {
            "position": tool_position.tolist(),
            "quaternion_wxyz": tool_quaternion.tolist(),
        }

    loaded_robot = copy.deepcopy(robot_data)
    add_payload_links(loaded_robot)
    loaded_ik_cfg = InverseKinematicsCfg.create(
        robot=loaded_robot,
        scene_model=scene,
        collision_cache={"cuboid": 40},
        num_seeds=64,
        self_collision_check=True,
        use_cuda_graph=False,
        position_tolerance=0.002,
        orientation_tolerance=math.radians(1.0),
        override_iters_for_multi_link_ik=300,
    )
    loaded_ik = InverseKinematics(loaded_ik_cfg)
    update_payloads(loaded_ik, payload_grids)
    started = time.perf_counter()
    extraction_frames, extraction_reason = solve_extraction(
        loaded_ik, contact_state, contact_pose_json
    )
    extraction_ok = not extraction_reason
    exact_ok, exact_reason = validate_payload_path(
        loaded_ik.kinematics,
        extraction_frames,
        tool_to_box,
        summary["chassis_front_x_m"],
        summary["wall_distance_m"],
    )
    report["stages"].append({
        "name": "cartesian_extract_35cm",
        "success": extraction_ok and exact_ok,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "reason": extraction_reason or exact_reason,
        "frames": extraction_frames,
    })
    if not extraction_ok or not exact_ok:
        report["failure_stage"] = "cartesian_extract_35cm"
        report["total_ms"] = (time.perf_counter() - started_total) * 1000.0
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"success": False, "failure_stage": report["failure_stage"], "reason": extraction_reason or exact_reason}, indent=2))
        return

    extracted_state = tensor_state(extraction_frames[-1])
    loaded_cfg = MotionPlannerCfg.create(
        robot=loaded_robot,
        scene_model=scene,
        collision_cache={"cuboid": 40},
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        self_collision_check=True,
        use_cuda_graph=False,
        position_tolerance=0.002,
        orientation_tolerance=math.radians(1.0),
    )
    loaded_planner = MotionPlanner(loaded_cfg)
    update_payloads(loaded_planner, payload_grids)
    started = time.perf_counter()
    placement_result = loaded_planner.plan_cspace(
        unloading_state, extracted_state, max_attempts=5
    )
    placement = result_trajectory(placement_result)
    placement_frames = [] if placement is None else state_rows(placement)
    exact_ok, exact_reason = validate_payload_path(
        loaded_planner.kinematics,
        placement_frames,
        tool_to_box,
        summary["chassis_front_x_m"],
        summary["wall_distance_m"],
    ) if placement is not None else (False, "")
    report["stages"].append({
        "name": "loaded_to_unloading",
        "success": placement is not None and exact_ok,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "reason": exact_reason,
        "frames": placement_frames,
    })
    if placement is None or not exact_ok:
        report["failure_stage"] = "loaded_to_unloading"
    else:
        report["success"] = True
    report["total_ms"] = (time.perf_counter() - started_total) * 1000.0
    report["payload_spheres_per_box"] = 48
    report["exact_payload_validation"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "success": report["success"],
        "failure_stage": report.get("failure_stage"),
        "total_ms": report["total_ms"],
        "stages": [
            {key: stage.get(key) for key in ("name", "success", "wall_ms", "reason")}
            | {"points": len(stage["frames"])}
            for stage in report["stages"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
