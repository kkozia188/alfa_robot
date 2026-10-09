#!/usr/bin/python3
"""Contract checks for the generated chest-carry replay and report."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
RESULTS = WORKSPACE / "results/v322_chest_carry_fast_base"


class ChestCarryContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.replay = json.loads(
            (RESULTS / "v322-chest-carry-fast-base-optimized-replay.json").read_text()
        )
        cls.validation = json.loads(
            (RESULTS / "v322-chest-carry-fast-base-optimized-validation.json").read_text()
        )
        cls.report = json.loads(
            (RESULTS / "optimized-acceptance-report.json").read_text()
        )

    def test_acceptance(self) -> None:
        self.assertTrue(self.validation["success"])
        self.assertTrue(self.report["success"])
        self.assertEqual(self.report["completed_boxes"], 25)
        self.assertEqual(self.report["group_count"], 15)
        self.assertEqual(self.report["optimization"]["boxes_within_3s"], 25)
        self.assertLess(
            self.report["optimization"]["maximum_task_core_ms"], 3000.0
        )

    def test_chest_precedes_base_motion(self) -> None:
        cycles = [
            operation for operation in self.replay["operations"]
            if operation["kind"] != "base_transition"
        ]
        self.assertEqual(len(cycles), 15)
        for operation in cycles:
            stages = [frame["stage"] for frame in operation["frames"]]
            chest = stages.index("chest_carry_ready")
            base = stages.index("base_backoff_after_chest_carry")
            self.assertLess(chest, base, operation["label"])
            self.assertEqual(operation["base_speed_m_s"], 0.35)
            chest_joints = operation["frames"][chest]["joints"]
            for frame in operation["frames"][base:]:
                if frame["stage"] == "release_at_right_conveyor":
                    break
                self.assertLessEqual(
                    max(
                        abs(float(actual) - float(expected))
                        for actual, expected in zip(frame["joints"], chest_joints)
                    ),
                    1e-12,
                    operation["label"],
                )

    def test_dual_attachment_is_simultaneous(self) -> None:
        dual_cycles = [
            operation for operation in self.replay["operations"]
            if operation["kind"] == "dual_chest_carry_conveyor_cycle"
        ]
        self.assertEqual(len(dual_cycles), 10)
        for operation in dual_cycles:
            one_sided = sum(
                bool(frame["left_attached"]) != bool(frame["right_attached"])
                for frame in operation["frames"]
            )
            self.assertEqual(one_sided, 0, operation["label"])

    def test_geometric_margins(self) -> None:
        self.assertLessEqual(self.report["maximum_chest_center_error_m"], 1e-4)
        self.assertGreater(
            self.report["minimum_vehicle_front_to_box_clearance_m"], 0.09
        )
        self.assertGreater(self.report["minimum_dual_box_gap_m"], 0.20)
        self.assertAlmostEqual(
            self.report["base"]["speed_multiplier"], 3.5, places=12
        )


if __name__ == "__main__":
    unittest.main()
