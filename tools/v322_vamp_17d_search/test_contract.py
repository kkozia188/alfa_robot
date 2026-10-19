#!/usr/bin/python3
"""Contract checks for the mandatory contract-17 VAMP search result."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
RESULTS = WORKSPACE / "results/v322_vamp_17d_search"
SEEDS = (169343505, 271828183, 314159265)
SOURCE = (
    WORKSPACE
    / "ros2_ws/src/alfa_robot_moveit_config/src/"
    / "v3_chest_carry_shortcut_optimizer.cpp"
)


class MandatorySearchContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reports = [
            json.loads((RESULTS / f"seed-{seed}-report.json").read_text())
            for seed in SEEDS
        ]
        cls.summary = json.loads(
            (RESULTS / "contract17-unbiased-search-benchmark.json").read_text()
        )

    def test_search_interface_only_receives_endpoints(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        signature = source.split(
            "nlohmann::json optimizeOperationWithMandatorySearch(", 1
        )[1].split(") const {", 1)[0]
        self.assertIn("search_endpoints", signature)
        self.assertNotIn("golden_reference", signature)
        self.assertIn(
            'search_endpoints.push_back(golden.front())', source
        )
        self.assertIn(
            'search_endpoints.push_back(golden.back())', source
        )

    def test_all_runs_use_search_tree_without_fallback(self) -> None:
        for report in self.reports:
            self.assertTrue(report["success"])
            self.assertEqual(len(report["cycles"]), 15)
            for cycle in report["cycles"]:
                self.assertEqual(cycle["path_source"], "vamp_rrtc_search_tree")
                self.assertTrue(cycle["search_executed"])
                self.assertTrue(cycle["search_result_required"])
                self.assertFalse(cycle["fallback_used"])
                self.assertFalse(cycle["golden_intermediate_frames_used_for_search"])
                self.assertEqual(cycle["search_input_frame_count"], 2)
                self.assertEqual(cycle["local_rrt_state_dimension"], 17)
                self.assertFalse(cycle["local_rrt_pruned"])
                self.assertTrue(cycle["local_candidate_fcl_valid"])
                self.assertEqual(cycle["visibility_checks"], 0)
                self.assertEqual(len(cycle["local_search_bounds"]), 17)
                self.assertEqual(len(cycle["local_search_worker_trials"]), 4)

    def test_multiple_seeds_expand_the_hard_cycles(self) -> None:
        self.assertEqual(len({item["vamp_seed"] for item in self.reports}), 3)
        for report in self.reports:
            expanded = {
                cycle["label"]
                for cycle in report["cycles"]
                if cycle["local_search_iterations"] > 0
            }
            self.assertEqual(
                expanded, {"Box 3", "Box 16 + Box 20", "Box 21 + Box 25"}
            )

    def test_validation_and_rerun(self) -> None:
        self.assertTrue(self.summary["success"])
        self.assertEqual(
            self.summary["three_seed_acceptance"]["searched_cycles"], 45
        )
        self.assertEqual(
            self.summary["three_seed_acceptance"]["fallback_cycles"], 0
        )
        self.assertTrue((RESULTS / "search-production-safe-turn.rrd").is_file())


if __name__ == "__main__":
    unittest.main()
