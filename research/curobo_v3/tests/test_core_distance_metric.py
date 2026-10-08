import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from curobo_core.distance_metric import UPDOWN_DISTANCE_WEIGHT, joint_distance_weights


class DistanceMetricTest(unittest.TestCase):
    def test_ten_centimeters_equal_fifteen_degrees(self):
        lift_distance = 0.1 * UPDOWN_DISTANCE_WEIGHT
        joint_distance = math.radians(15)
        self.assertAlmostEqual(lift_distance, joint_distance)
        self.assertAlmostEqual(lift_distance ** 2, joint_distance ** 2)

    def test_named_mapping_preserves_base_weights_in_mobile_group(self):
        names = ("base_x", "base_y", "base_yaw", "updown", "left_joint1")
        original = [2, 2, 1, 5, 1]
        result = joint_distance_weights(names, original)
        self.assertEqual(result[:3], original[:3])
        self.assertEqual(result[4], 1)
        self.assertEqual(result[3], UPDOWN_DISTANCE_WEIGHT)
        self.assertEqual(original, [2, 2, 1, 5, 1])

    def test_non_robot_search_dimensions_remain_unweighted(self):
        self.assertEqual(joint_distance_weights(("joint_a", "joint_b")), [1, 1])
        with self.assertRaises(ValueError):
            joint_distance_weights(("updown", "left_joint1"), [1])


if __name__ == "__main__":
    unittest.main()
