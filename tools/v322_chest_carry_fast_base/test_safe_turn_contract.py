#!/usr/bin/python3
"""Contract checks for the chest-first safe-turn unload result."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
LOCAL_RESULTS = WORKSPACE / "results/v322_chest_turn_place"
SOURCE_RESULTS = Path(
    "/home/tim/alfa_robot-v322-chest-turn-place/results/v322_chest_turn_place"
)
RESULTS = (
    LOCAL_RESULTS
    if (LOCAL_RESULTS / "v322-chest-pull-safe-turn-unload-replay.json").is_file()
    else SOURCE_RESULTS
)


class SafeTurnContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.replay = json.loads(
            (RESULTS / "v322-chest-pull-safe-turn-unload-replay.json").read_text()
        )
        cls.validation = json.loads(
            (RESULTS / "v322-chest-pull-safe-turn-unload-validation.json").read_text()
        )
        cls.contract = json.loads(
            (RESULTS / "v322-chest-pull-safe-turn-contract.json").read_text()
        )

    def test_acceptance(self) -> None:
        self.assertTrue(self.validation["success"])
        self.assertTrue(self.contract["success"])
        self.assertEqual(self.contract["box_count"], 25)
        self.assertEqual(self.contract["cycle_count"], 15)
        self.assertEqual(self.contract["boxes_within_3s"], 25)
        self.assertLess(self.contract["maximum_task_core_ms"], 3000.0)

    def test_chest_backoff_turn_release_order(self) -> None:
        for operation in self.replay["operations"]:
            if operation["kind"] == "base_transition":
                continue
            stages = [frame["stage"] for frame in operation["frames"]]
            chest = stages.index("chest_carry_ready")
            backoff = stages.index("base_backoff_after_chest_carry")
            turn = stages.index("base_turn_180_loaded_at_safe_exit")
            release = stages.index("release_after_turn_at_safe_exit")
            self.assertLess(chest, backoff, operation["label"])
            self.assertLess(backoff, turn, operation["label"])
            self.assertLess(turn, release, operation["label"])

    def test_route_geometry(self) -> None:
        for case in self.contract["cases"]:
            self.assertAlmostEqual(case["backoff_m"], 2.35, places=9)
            self.assertAlmostEqual(case["lateral_shuttle_m"], 0.0, places=12)
            self.assertAlmostEqual(case["turn_deg"], 180.0, places=9)

    def test_dual_release_is_equal_height(self) -> None:
        dual = [case for case in self.contract["cases"] if len(case["boxes"]) == 2]
        self.assertEqual(len(dual), 10)
        self.assertLessEqual(
            self.contract["maximum_dual_release_height_error_m"], 1e-6
        )
        self.assertTrue(all(
            math.isfinite(case["dual_release_height_error_m"]) for case in dual
        ))


if __name__ == "__main__":
    unittest.main()
