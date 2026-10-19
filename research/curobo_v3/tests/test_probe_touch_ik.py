import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from probe_touch_ik import ProbeTouchIK


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


if __name__ == '__main__':
    unittest.main()
