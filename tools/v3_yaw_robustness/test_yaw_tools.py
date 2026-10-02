#!/usr/bin/env python3
"""Small deterministic tests for the yaw planning orchestration helpers."""

from __future__ import annotations

import math
import unittest

from plan_yaw_pickups import arm_seed_degrees, unique
from run_yaw_case import yaw_slug
from run_yaw_sweep import parse_yaws
from summarize_yaw_sweep import operation_metrics


class YawToolTests(unittest.TestCase):
    def test_yaw_slug_and_list(self) -> None:
        self.assertEqual(yaw_slug(-5), "yaw-m05")
        self.assertEqual(yaw_slug(0), "yaw-p00")
        self.assertEqual(yaw_slug(5), "yaw-p05")
        self.assertEqual(parse_yaws("-5,-4,0,5"), [-5, -4, 0, 5])

    def test_seed_extracts_only_active_arm(self) -> None:
        payload = {"frames": [{"joints": list(range(16))}]}
        left = [float(value) for value in arm_seed_degrees(payload, "left").split(",")]
        right = [float(value) for value in arm_seed_degrees(payload, "right").split(",")]
        for actual, expected in zip(left, [math.degrees(value) for value in range(2, 9)]):
            self.assertAlmostEqual(actual, expected, places=9)
        for actual, expected in zip(right, [math.degrees(value) for value in range(9, 16)]):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_unique_preserves_candidate_order(self) -> None:
        self.assertEqual(unique([0.0, -0.25, 0.0, -0.5]), [0.0, -0.25, -0.5])

    def test_metrics_include_joint_and_base_motion(self) -> None:
        names = ["updown", "head_joint", "left_joint1", "right_joint1"]
        operation = {
            "base_pose_map": [0.0, 0.0, 0.0],
            "frames": [
                {"joints": [0.0, 0.0, 0.0, 0.0]},
                {
                    "joints": [0.0, 0.0, math.radians(2.0), math.radians(1.0)],
                    "base_pose_map": [0.03, 0.04, 0.0],
                },
            ],
        }
        metrics = operation_metrics(operation, names)
        self.assertAlmostEqual(metrics["maximum_joint_step_deg"], 2.0)
        self.assertEqual(metrics["joint_flip_events"], 0)
        self.assertAlmostEqual(metrics["base_travel_m"], 0.05)


if __name__ == "__main__":
    unittest.main()
