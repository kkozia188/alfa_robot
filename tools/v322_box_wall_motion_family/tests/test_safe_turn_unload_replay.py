#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_safe_turn_unload_replay import transform_operation  # noqa: E402


def frame(
    stage: str,
    pose: list[float],
    attached: bool,
    joint_value: float = 0.0,
) -> dict[str, object]:
    return {
        "stage": stage,
        "joints": [joint_value] * 17,
        "left_attached": attached,
        "right_attached": attached,
        "base_pose_map": pose,
    }


class SafeTurnUnloadReplayTest(unittest.TestCase):
    def operation(self) -> dict[str, object]:
        return {
            "kind": "dual_conveyor_cycle",
            "base_pose_map": [-0.6, 0.0, 0.0],
            "backoff_distance_m": 2.35,
            "right_shuttle_distance_m": 1.5,
            "conveyor_pose_map": [-2.95, -1.5, 0.0],
            "frames": [
                frame("group_start", [-0.6, 0.0, 0.0], False),
                frame("attach_box", [-0.6, 0.0, 0.0], True),
                frame("base_backoff_before_conveyor", [-2.95, 0.0, 0.0], True),
                frame("base_right_to_conveyor", [-2.95, -1.5, 0.0], True),
                frame("release_at_right_conveyor", [-2.95, -1.5, 0.0], False),
                frame(
                    "post_release_stow_outside_warehouse",
                    [-2.95, -1.5, 0.0],
                    False,
                    joint_value=1.0,
                ),
                frame("base_left_from_conveyor", [-2.95, 0.0, 0.0], False, 1.0),
                frame("base_forward_to_pickup", [-0.6, 0.0, 0.0], False, 1.0),
            ],
        }

    def test_replaces_lateral_shuttle_with_loaded_and_empty_turns(self) -> None:
        result = transform_operation(self.operation(), base_speed_m_s=0.25)
        stages = [item["stage"] for item in result["frames"]]
        self.assertEqual(result["kind"], "dual_safe_turn_unload_cycle")
        self.assertNotIn("base_right_to_conveyor", stages)
        self.assertNotIn("base_left_from_conveyor", stages)
        self.assertNotIn("release_at_right_conveyor", stages)
        self.assertIn("release_after_turn_at_safe_exit", stages)
        self.assertEqual(stages.count("base_turn_180_loaded_at_safe_exit"), 180)
        self.assertEqual(
            stages.count("base_turn_180_empty_to_pickup_heading"), 180
        )
        release = next(
            item
            for item in result["frames"]
            if item["stage"] == "release_after_turn_at_safe_exit"
        )
        self.assertEqual(release["base_pose_map"][:2], [-2.95, 0.0])
        self.assertAlmostEqual(release["base_pose_map"][2], math.pi)
        self.assertEqual(result["frames"][-1]["base_pose_map"], [-0.6, 0.0, 0.0])
        self.assertEqual(result["base_speed_m_s"], 0.25)
        self.assertNotIn("conveyor_pose_map", result)

    def test_clockwise_turn_has_negative_yaw(self) -> None:
        result = transform_operation(self.operation(), turn_direction="clockwise")
        release = next(
            item
            for item in result["frames"]
            if item["stage"] == "release_after_turn_at_safe_exit"
        )
        self.assertAlmostEqual(release["base_pose_map"][2], -math.pi)

    def test_auto_turn_keeps_positive_initial_yaw_within_bounds(self) -> None:
        operation = self.operation()
        for item in operation["frames"]:
            item["base_pose_map"][2] = math.radians(15.0)
        result = transform_operation(operation)
        self.assertEqual(result["turn_direction"], "clockwise")
        self.assertTrue(
            all(
                -math.pi <= item["base_pose_map"][2] <= math.pi
                for item in result["frames"]
            )
        )

    def test_base_transition_only_receives_faster_speed(self) -> None:
        operation = {"kind": "base_transition", "frames": []}
        result = transform_operation(operation, base_speed_m_s=0.25)
        self.assertEqual(result["base_speed_m_s"], 0.25)
        self.assertEqual(result["frames"], [])


if __name__ == "__main__":
    unittest.main()
