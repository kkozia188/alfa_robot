import math
import threading

import numpy as np
import torch

from .backend import CuroboBackend
from .contracts import PlannerAssets
from v3_batched_loaded_search import GpuValidity


class SnapshotInspector:
    def __init__(self, assets=PlannerAssets()):
        self.backend = CuroboBackend(assets)
        self._lock = threading.Lock()

    def check(self, snapshot):
        with self._lock:
            self.backend.set_snapshot(snapshot)
            checker = self.backend.collision_checker({}, mobile=True)
            names = self.backend.mobile_robot["kinematics"]["cspace"]["joint_names"]
            state = dict(zip(snapshot.state.joint_names, snapshot.state.positions))
            position, quaternion = snapshot.state.base_pose.position, snapshot.state.base_pose.quaternion_wxyz
            if abs(position[2]) > 1e-9 or max(abs(quaternion[1]), abs(quaternion[2])) > 1e-9:
                raise ValueError("inspection requires a planar robot base pose")
            state.update(base_x=position[0], base_y=position[1],
                         base_yaw=2 * math.atan2(quaternion[3], quaternion[0]))
            missing = set(names) - state.keys()
            if missing:
                raise ValueError(f"missing inspection joints: {sorted(missing)}")
            values = torch.tensor([[state[name] for name in names]], device="cuda", dtype=torch.float32)
            validity = GpuValidity(checker, joint_names=names, check_ground=True,
                                   ground_z=snapshot.policy.ground_z_m)
            valid = bool(validity.mask(values).item())
            robot_state = checker.get_kinematics(values[:, None, :])
            costs = checker.collision_constraint.forward(robot_state).reshape(-1)
            spheres = robot_state.robot_spheres.detach().cpu().reshape(-1, 4).numpy()
            collided = costs.detach().cpu().numpy() > 0
            return {"snapshot_id": snapshot.identity, "revision": snapshot.revision,
                    "valid": valid, "environment_collision": bool(np.any(collided)),
                    "environment_colliding_sphere_indices": np.flatnonzero(collided).tolist(),
                    "robot_spheres": spheres.tolist(), "mesh_object_ids": [item.object_id for item in snapshot.meshes]}
