import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from curobo_core.distance_metric import UPDOWN_DISTANCE_WEIGHT, joint_distance_weights


class DistanceMetricTest(unittest.TestCase):
    def test_ten_centimeters_equal_fifteen_degrees(self):
        self.assertAlmostEqual(.1 * UPDOWN_DISTANCE_WEIGHT, math.radians(15))

    def test_named_mapping_preserves_base_weights(self):
        original = [2, 2, 1, 5, 1]
        result = joint_distance_weights(("base_x", "base_y", "base_yaw", "updown", "joint"), original)
        self.assertEqual(result[:3], original[:3])
        self.assertEqual(result[3], UPDOWN_DISTANCE_WEIGHT)
        self.assertEqual(original, [2, 2, 1, 5, 1])

    def test_invalid_length_rejected(self):
        with self.assertRaises(ValueError):
            joint_distance_weights(("updown", "joint"), [1])


if __name__ == "__main__":
    unittest.main()
