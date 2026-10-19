from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yourdfpy

from v3_rezero_analytic import RezeroAnalytic


ROOT = Path(__file__).resolve().parents[1]
PROBE_OFFSET_M = .070


def nearby_targets(target, tilt_deg=25.):
    if not 0 <= tilt_deg <= 30:
        raise ValueError('朝向偏转必须在0～30度')
    output = [('原朝向', target.copy(), 0.)]
    for axis, vector in [('X', np.array([1., 0., 0.])), ('Y', np.array([0., 1., 0.])), ('Z', np.array([0., 0., 1.]))]:
        for sign in (-1, 1):
            pose = target.copy()
            pose[:3, :3] = target[:3, :3] @ Rotation.from_rotvec(vector * np.deg2rad(tilt_deg * sign)).as_matrix()
            output.append((f'局部{axis}{tilt_deg * sign:+g}°', pose, tilt_deg))
    return output


class ProbeTouchIK:
    def __init__(self, updown_m):
        self.updown_m = float(updown_m)
        self.model = yourdfpy.URDF.load(
            ROOT / 'generated/zero_model_candidate_20261018/robot_rezero_candidate.urdf',
            load_meshes=False)
        joint = self.model.joint_map['updown']
        if not joint.limit.lower <= self.updown_m <= joint.limit.upper:
            raise ValueError('固定Updown超出模型限位')
        self.analytic = RezeroAnalytic()

    def state(self, left):
        values = {name: 0. for name in self.model.actuated_joint_names}
        values['updown'] = self.updown_m
        values.update({f'left_joint{index}': angle for index, angle in enumerate(left, 1)})
        self.model.update_cfg(values)

    def forward(self, left):
        self.state(left)
        pose = self.model.get_transform('left_link7', self.model.base_link).copy()
        pose[:3, 3] += pose[:3, :3] @ np.array([0., 0., PROBE_OFFSET_M])
        return pose

    def solve(self, target, seed, step_deg=3.):
        target = np.asarray(target, dtype=float)
        seed = np.asarray(seed, dtype=float)
        if (target.shape != (4, 4) or seed.shape != (7,)
                or not np.isfinite(target).all() or not np.isfinite(seed).all()
                or not np.allclose(target[3], [0, 0, 0, 1])
                or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(target[:3, :3]), 1.)
                or not 0.1 <= step_deg <= 30.):
            raise ValueError('目标必须为合法SE(3)，种子为七轴有限弧度，步长0.1～30度')
        self.state(seed)
        carriage = self.model.get_transform('arm_carriage', self.model.base_link)
        link_to_old_tool = self.model.get_transform('left_tool0', 'left_link7')
        probe_to_link = np.eye(4)
        probe_to_link[2, 3] = -PROBE_OFFSET_M
        old_target = np.linalg.inv(carriage) @ target @ probe_to_link @ link_to_old_tool
        swivels = np.r_[self.analytic.swivel(0, seed), np.deg2rad(np.arange(-180., 180., step_deg))]
        solutions = []
        for swivel in swivels:
            for angles in self.analytic.solve(0, old_target, float(swivel), seed):
                if any(np.max(np.abs(angles - row['joints'])) < 1e-6 for row in solutions):
                    continue
                actual = self.forward(angles)
                position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
                rotation_error = float(Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude())
                if position_error > 2e-6 or rotation_error > 2e-6:
                    continue
                solutions.append(dict(joints=angles.copy(), swivel_rad=float(swivel),
                    joint_distance_rad=float(np.linalg.norm(angles - seed)),
                    position_error_m=position_error, rotation_error_rad=rotation_error))
        solutions.sort(key=lambda row: row['joint_distance_rad'])
        self.state(seed)
        return solutions
