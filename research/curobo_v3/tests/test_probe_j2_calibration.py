from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ipc_left_prefix.coordinate_mapping import motor_to_model, model_to_motor, EXECUTION_REVISION
from ipc_left_prefix.smooth_timing import rt_curves, validate_preview
from probe_touch_motion import interpolate_motion, motion_payload, selected_endpoint


class J2CalibrationTest(unittest.TestCase):
    def test_motor_one_degree_is_model_zero_and_other_axes_unchanged(self):
        motor = np.deg2rad([10., 1., 20., 30., 40., 50., 60.])
        model = motor_to_model(motor)
        self.assertAlmostEqual(model[1], 0.)
        np.testing.assert_array_equal(model[[0,2,3,4,5,6]], motor[[0,2,3,4,5,6]])
        np.testing.assert_allclose(model_to_motor(model), motor, atol=1e-15)

    def test_payload_motor_coordinates_match_model_preview(self):
        start = motor_to_model(np.deg2rad([0., 1., 0., 0., 0., 0., 0.]))
        goal = np.deg2rad([2., 5., -4., 1., 0., 3., 2.])
        points, timing = interpolate_motion(start, goal)
        payload = motion_payload(points, timing, 10, np.eye(4))
        validate_preview(payload)
        self.assertEqual(payload['coordinate_revision'], EXECUTION_REVISION)
        self.assertAlmostEqual(np.rad2deg(payload['points'][0]['positions'][1]), 1.)
        self.assertAlmostEqual(np.rad2deg(payload['points'][-1]['positions'][1]), 6.)
        original = rt_curves(points)
        sent = rt_curves(payload['points'])
        stamps = np.linspace(0, timing['duration_s'], 101)
        for order in (0, 1, 2):
            expected = np.column_stack([curve.derivative(order)(stamps) for curve in original])
            actual = np.column_stack([curve.derivative(order)(stamps) for curve in sent])
            if order == 0:
                actual = motor_to_model(actual)
            np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_model_zero_and_fixed_axis_use_same_mapping(self):
        locked = dict(joint_index=1, angle_rad=float(motor_to_model(np.deg2rad([0, 21, 0, 0, 0, 0, 0]))[1]))
        zero, context = selected_endpoint(None, locked)
        self.assertAlmostEqual(np.rad2deg(model_to_motor(zero)[1]), 1.)
        self.assertAlmostEqual(np.rad2deg(context['saved_lock']['angle_rad']), 20.)
        goal = np.zeros(7)
        goal[1] = locked['angle_rad']
        selected, _context = selected_endpoint(dict(joints=goal), locked)
        self.assertAlmostEqual(np.rad2deg(model_to_motor(selected)[1]), 21.)


if __name__ == '__main__':
    unittest.main()
