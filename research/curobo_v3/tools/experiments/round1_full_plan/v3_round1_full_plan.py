#!/usr/bin/env python3

import argparse
import copy
import gc
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


def guard_limit_roundoff(state: JointState, kinematics) -> tuple[JointState, list[float]]:
    lower, upper = kinematics.get_joint_limits().position
    guarded = state.clone()
    guarded.position = torch.clamp(state.position, min=lower + 1e-6, max=upper - 1e-6)
    delta = (guarded.position - state.position).reshape(-1).detach().cpu().tolist()
    if max(abs(value) for value in delta) > 2e-6:
        raise ValueError("endpoint exceeds numerical joint-limit guard")
    return guarded, delta


def result_trajectory(result) -> JointState | None:
    if result is None or result.success is None or not bool(result.success.reshape(-1)[0].item()):
        return None
    return result.get_interpolated_plan()


def state_rows(state: JointState) -> list[list[float]]:
    values = state.position.detach().cpu().numpy()
    return values.reshape(-1, values.shape[-1]).tolist()


def linear_joint_path(start: JointState, goal: JointState, maximum_step: float = math.radians(1.0)) -> list[list[float]]:
    start_values = start.position.reshape(-1)
    goal_values = goal.position.reshape(-1)
    count = max(1, int(torch.ceil(torch.max(torch.abs(goal_values-start_values)) / maximum_step).item()))
    return [((1.0-fraction)*start_values + fraction*goal_values).detach().cpu().tolist()
            for fraction in torch.linspace(0.0, 1.0, count+1, device=start_values.device)]


def audit_state(kinematics, state: JointState, scene: Scene) -> dict:
    config = kinematics.config.kinematics_config
    fk = kinematics.compute_kinematics(state)
    spheres = fk.robot_spheres.detach().cpu().numpy().reshape(-1, 4)
    sphere_links = [""] * len(spheres)
    for link_name in config.link_name_to_idx_map:
        for sphere_index in config.get_sphere_index_from_link_name(link_name).cpu().tolist():
            sphere_links[sphere_index] = link_name
    valid = spheres[:, 3] > 0.0
    collision_pairs = kinematics.get_self_collision_config().collision_pairs.cpu().numpy().astype(int)
    padding = kinematics.get_self_collision_config().sphere_padding.cpu().numpy()
    penetrations = {}
    for first_index, second_index in collision_pairs:
        if not valid[first_index] or not valid[second_index]:
            continue
        clearance = float(np.linalg.norm(spheres[first_index, :3] - spheres[second_index, :3])
            - spheres[first_index, 3] - spheres[second_index, 3]
            - padding[first_index] - padding[second_index])
        if clearance < -1e-6:
            pair = tuple(sorted((sphere_links[first_index], sphere_links[second_index])))
            candidate = {
                "links": list(pair), "penetration_mm": -clearance * 1000.0,
                "sphere_indices": [int(first_index), int(second_index)],
                "sphere_centers": [spheres[first_index, :3].tolist(), spheres[second_index, :3].tolist()],
            }
            if pair not in penetrations or candidate["penetration_mm"] > penetrations[pair]["penetration_mm"]:
                penetrations[pair] = candidate
    world_collisions = {}
    for obstacle in scene.cuboid or []:
        world_rotation = Rotation.from_quat(np.roll(obstacle.pose[3:7], -1)).as_matrix()
        local_centers = (spheres[:, :3] - obstacle.pose[:3]) @ world_rotation
        distances = np.abs(local_centers) - np.asarray(obstacle.dims) / 2.0
        signed_distance = np.linalg.norm(np.maximum(distances, 0.0), axis=1) + np.minimum(distances.max(axis=1), 0.0)
        clearances = signed_distance - spheres[:, 3]
        for sphere_index in np.flatnonzero(valid & (clearances < -1e-6)):
            pair = (sphere_links[sphere_index], obstacle.name)
            candidate = {
                "link": pair[0], "obstacle": pair[1],
                "penetration_mm": float(-clearances[sphere_index] * 1000.0),
                "sphere_index": int(sphere_index), "sphere_center": spheres[sphere_index, :3].tolist(),
            }
            if pair not in world_collisions or candidate["penetration_mm"] > world_collisions[pair]["penetration_mm"]:
                world_collisions[pair] = candidate
    limits = kinematics.get_joint_limits()
    values = state.position.detach().cpu().numpy().reshape(-1)
    lower, upper = limits.position.detach().cpu().numpy()
    violations = [name for index, name in enumerate(kinematics.joint_names)
                  if values[index] < lower[index] - 1e-6 or values[index] > upper[index] + 1e-6]
    return {
        "valid": not penetrations and not world_collisions and not violations,
        "sphere_count": int(valid.sum()),
        "self_collisions": sorted(penetrations.values(), key=lambda item: -item["penetration_mm"]),
        "world_collisions": sorted(world_collisions.values(), key=lambda item: -item["penetration_mm"]),
        "joint_limit_violations": violations,
    }


def summarize_result(result) -> dict:
    if result is None:
        return {"returned_none": True}
    report = {}
    for name in ("success", "feasible", "cspace_error", "position_error", "rotation_error", "solve_time", "total_time"):
        value = getattr(result, name, None)
        report[name] = value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
    for metric_name in ("metrics", "interpolated_metrics"):
        metric = getattr(result, metric_name, None)
        if metric is None:
            continue
        collection = metric.costs_and_constraints
        report[metric_name] = {}
        for field_name in ("costs", "constraints", "hybrid_costs_constraints"):
            field = getattr(collection, field_name, None)
            if field is not None:
                report[metric_name][field_name] = {
                    name: {"shape": list(value.shape), "max": float(value.max().item()), "min": float(value.min().item())}
                    for name, value in zip(field.names, field.values)
                }
    return report


def capture_trajectory_metrics(planner) -> list[dict]:
    captured = []
    original = planner.trajopt_solver._get_best_result

    def capture(all_seeds_result, return_seeds):
        captured.append(summarize_result(all_seeds_result))
        return original(all_seeds_result, return_seeds)

    planner.trajopt_solver._get_best_result = capture
    return captured


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
        if link not in kin["collision_link_names"]:
            kin["collision_link_names"].append(link)
        kin["collision_spheres"][link] = []
        kin.setdefault("self_collision_buffer", {})[link] = 0.0
        kin.setdefault("self_collision_ignore", {}).setdefault(f"{side}_link7", []).append(link)
        kin["self_collision_ignore"].setdefault(link, []).append(f"{side}_link7")
        kin["grasp_contact_link_names"] = list(kin.get("grasp_contact_link_names") or [])
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
    parser.add_argument("--guard-limit-roundoff", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

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

    report = {"success": False, "stages": [], "joint_names": ACTIVE_JOINTS,
              "source_summary": str(args.ik_summary), "wall_distance_m": summary["wall_distance_m"],
              "chassis_front_x_m": summary["chassis_front_x_m"], "pair": list(PAIR),
              "robot_config": summary["robot_config"], "diagnostic_only": True}
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
    report["trajectory_metrics"] = capture_trajectory_metrics(free_planner)
    if args.guard_limit_roundoff:
        home_state, home_delta = guard_limit_roundoff(home_state, free_planner.kinematics)
        contact_state, contact_delta = guard_limit_roundoff(contact_state, free_planner.kinematics)
        unloading_state, unloading_delta = guard_limit_roundoff(unloading_state, free_planner.kinematics)
        report["endpoint_float32_roundoff_guard"] = {
            "maximum_allowed_adjustment_rad_or_m": 2e-6,
            "home_delta": home_delta, "contact_delta": contact_delta, "unloading_delta": unloading_delta,
        }
    report["endpoint_audit"] = {
        name: audit_state(free_planner.kinematics, state, scene)
        for name, state in (("home", home_state), ("contact", contact_state), ("unloading_empty", unloading_state))
    }
    report["endpoint_joints"] = {
        "home": state_rows(home_state)[0], "contact": state_rows(contact_state)[0],
        "unloading_empty": state_rows(unloading_state)[0],
    }
    print(json.dumps({"endpoint_audit": report["endpoint_audit"]}, indent=2), flush=True)
    started = time.perf_counter()
    approach_result = free_planner.plan_cspace(contact_state, home_state, max_attempts=3)
    approach = result_trajectory(approach_result)
    report["stages"].append({
        "name": "home_to_contact",
        "success": approach is not None,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "frames": [] if approach is None else state_rows(approach),
        "planner_result": summarize_result(approach_result),
    })
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if approach is None:
        projected_frames = []
        projected_audits = []
        projected_bad = []
        if approach_result is not None and approach_result.interpolated_trajectory is not None:
            rejected = approach_result.get_interpolated_plan()
            rejected_tensor = rejected.position.reshape(-1, len(ACTIVE_JOINTS))
            lower, upper = free_planner.kinematics.get_joint_limits().position
            projected_tensor = torch.clamp(rejected_tensor, min=lower+1e-6, max=upper-1e-6)
            projected_frames = projected_tensor.detach().cpu().tolist()
            projected_audits = [audit_state(free_planner.kinematics, tensor_state(row), scene)
                                for row in projected_frames]
            projected_bad = [index for index, audit in enumerate(projected_audits) if not audit["valid"]]
            report["stages"][-1]["bounded_projection_repair"] = {
                "checked": True, "success": not projected_bad,
                "failed_frame_indices": projected_bad, "frames": projected_frames,
                "collision_audit": projected_audits,
                "maximum_projection_rad_or_m": float(torch.max(torch.abs(projected_tensor-rejected_tensor)).item()),
            }
        shortcut_frames = linear_joint_path(home_state, contact_state)
        shortcut_audits = [audit_state(free_planner.kinematics, tensor_state(row), scene) for row in shortcut_frames]
        shortcut_bad = [index for index, audit in enumerate(shortcut_audits) if not audit["valid"]]
        report["stages"][-1]["motion_planner_rejected_reason"] = "bspline_cspace_bound_violation"
        report["stages"][-1]["shortcut_fallback"] = {
            "checked": True, "success": not shortcut_bad, "failed_frame_indices": shortcut_bad,
            "frames": shortcut_frames, "collision_audit": shortcut_audits,
        }
        if approach_result is not None and approach_result.interpolated_trajectory is not None:
            rejected = approach_result.get_interpolated_plan()
            report["stages"][-1]["rejected_frames"] = state_rows(rejected)
            report["stages"][-1]["joint_state"] = {
                name: (getattr(rejected, name).detach().cpu().tolist() if isinstance(getattr(rejected, name, None), torch.Tensor)
                       else getattr(rejected, name, None))
                for name in ("velocity", "acceleration", "jerk", "dt")
            }
        if projected_frames and not projected_bad:
            approach = tensor_state(projected_frames[-1])
            approach.position = torch.tensor(projected_frames, device="cuda", dtype=torch.float32).unsqueeze(0)
            report["stages"][-1]["success"] = True
            report["stages"][-1]["method"] = "curobo_bspline_bounded_projection_repair"
            report["stages"][-1]["frames"] = projected_frames
        elif shortcut_bad:
            report["failure_stage"] = "home_to_contact"
            report["total_ms"] = (time.perf_counter() - started_total) * 1000.0
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"failure_stage": report["failure_stage"], "total_ms": report["total_ms"],
                              "planner_result": report["stages"][-1]["planner_result"]}, indent=2))
            return
        else:
            approach = tensor_state(shortcut_frames[-1])
            approach.position = torch.tensor(shortcut_frames, device="cuda", dtype=torch.float32).unsqueeze(0)
            report["stages"][-1]["success"] = True
            report["stages"][-1]["method"] = "joint_shortcut_after_curobo_bspline_rejection"
            report["stages"][-1]["frames"] = shortcut_frames

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

    approach_audits = [audit_state(free_planner.kinematics, tensor_state(row), scene) for row in state_rows(approach)]
    bad_indices = [index for index, audit in enumerate(approach_audits) if not audit["valid"]]
    report["stages"][0]["collision_audit"] = {"frames": approach_audits, "failed_frame_indices": bad_indices}
    if bad_indices:
        report["failure_stage"] = "home_to_contact_collision_audit"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("Stopped: independent sphere audit rejected free-space trajectory", flush=True)
        return
    report["tool_to_box"] = {side: matrix.tolist() for side, matrix in tool_to_box.items()}
    print("home_to_contact accepted", flush=True)
    free_planner.destroy()
    del free_planner, free_cfg, approach_result
    gc.collect()
    torch.cuda.empty_cache()

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
    report["loaded_contact_audit"] = audit_state(loaded_ik.kinematics, contact_state, scene)
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

    print("cartesian_extract_35cm accepted", flush=True)
    loaded_ik.destroy()
    del loaded_ik, loaded_ik_cfg
    gc.collect()
    torch.cuda.empty_cache()

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
    report["loaded_trajectory_metrics"] = capture_trajectory_metrics(loaded_planner)
    report["loaded_endpoint_audit"] = {
        "extracted": audit_state(loaded_planner.kinematics, extracted_state, scene),
        "unloading": audit_state(loaded_planner.kinematics, unloading_state, scene),
    }
    print(json.dumps({"loaded_endpoint_audit": report["loaded_endpoint_audit"]}, indent=2), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
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
        "planner_result": summarize_result(placement_result),
    })
    if placement_result is not None and placement_result.interpolated_trajectory is not None:
        report["stages"][-1]["rejected_frames"] = state_rows(placement_result.get_interpolated_plan())
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
