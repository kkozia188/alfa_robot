from __future__ import annotations

import math

import numpy as np

from scipy.spatial.transform import Rotation

def graph_config(args):
    return {
        'graph_planner': {
            'max_nodes': args.max_nodes,
            'steer_buffer_size': args.steer_buffer_size,
            'cspace_similarity_threshold': args.cspace_resolution,
            'sample_rejection_ratio': 10,
            'neighbors_per_node': args.neighbors,
            'feasibility_buffer_size': args.feasibility_buffer_size,
            'new_nodes_per_iteration': args.nodes_per_iteration,
            'max_path_finding_iterations': args.graph_iterations,
            'min_finetune_iterations': 2,
            'use_default_position_heuristic': args.use_default_position_heuristic,
            'exploration_radius': args.exploration_radius,
            'exploration_radius_growth_factor': 1.15,
            'sampler_seed': args.seed,
            'sampler_buffer_size': args.sampler_buffer_size,
            'connect_terminal_nodes_with_nearest': args.connect_terminal_nodes_with_nearest,
            'ellipsoid_projection_method': 'householder',
            'neighbors_per_node_growth_factor': 1.1,
            'new_nodes_per_iteration_growth_factor': 1.05,
        }
    }

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

def validate_payload_path(*args, **kwargs):
    from experiments.round1_full_plan.v3_round1_full_plan import validate_payload_path as validate
    return validate(*args, **kwargs)
