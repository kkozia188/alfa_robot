import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from probe_touch_ik import ProbeTouchIK, nearby_targets
from probe_touch_motion import interpolate_motion


class ProbeTouchIKTest(unittest.TestCase):
    def test_fixed_lift_probe_forward_inverse(self):
        solver = ProbeTouchIK(-.3)
        for angles in ([65, -75, 20, -90, 0, 40, 90], [40, -50, 15, -60, 20, 30, 60]):
            seed = np.deg2rad(angles)
            target = solver.forward(seed)
            solutions = solver.solve(target, seed)
            self.assertGreater(len(solutions), 1)
            self.assertLess(solutions[0]['joint_distance_rad'], 1e-4)
            for row in solutions:
                self.assertLess(row['position_error_m'], 2e-6)
                self.assertLess(row['rotation_error_rad'], 2e-6)
                self.assertAlmostEqual(solver.model.cfg[solver.model.actuated_joint_names.index('updown')], -.3)

    def test_reject_invalid_pose(self):
        solver = ProbeTouchIK(-.3)
        with self.assertRaises(ValueError):
            solver.solve(np.zeros((4, 4)), np.zeros(7))

    def test_nearby_orientations_keep_position(self):
        from scipy.spatial.transform import Rotation
        target = np.eye(4)
        target[:3, 3] = [.3, .4, .5]
        poses = nearby_targets(target)
        self.assertEqual(len(poses), 7)
        for label, pose, angle in poses:
            np.testing.assert_allclose(pose[:3, 3], target[:3, 3])
            self.assertAlmostEqual(np.rad2deg(Rotation.from_matrix(pose[:3, :3]).magnitude()), angle)

    def test_monotonic_quintic_joint_interpolation(self):
        from ipc_left_prefix.smooth_timing import rt_curves
        start = np.deg2rad([0, -10, 20, -30, 40, -50, 60])
        goal = start + np.deg2rad([25, 10, -10, 5, 0, -30, -20])
        points, report = interpolate_motion(start, goal)
        self.assertLessEqual(report['peak_velocity_deg_s'], 10.00001)
        self.assertLessEqual(report['peak_acceleration_deg_s2'], 30.00001)
        self.assertLessEqual(report['peak_jerk_deg_s3'], 150.00001)
        stamps = np.linspace(0, report['duration_s'], 2001)
        actual = np.column_stack([curve(stamps) for curve in rt_curves(points)])
        self.assertTrue(np.all(actual >= np.minimum(start, goal) - 1e-10))
        self.assertTrue(np.all(actual <= np.maximum(start, goal) + 1e-10))
        self.assertEqual(len(points), round(report['duration_s'] * 10) + 1)


if __name__ == '__main__':
    unittest.main()
