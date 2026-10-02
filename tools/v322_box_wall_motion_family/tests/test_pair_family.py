#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pair_family import compile_pair_contract  # noqa: E402


class PairFamilyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(
            (ROOT / "examples/cross_row_pair_requests.json").read_text(encoding="utf-8")
        )
        cls.family = json.loads(
            (ROOT / "config/action_family.json").read_text(encoding="utf-8")
        )

    def test_cross_row_requests_compile_to_two_arms(self) -> None:
        plan = compile_pair_contract(self.contract, self.family)
        self.assertEqual(plan["summary"], {"requested": 3, "compiled": 3, "failed": 0})
        for request in plan["requests"]:
            self.assertTrue(request["different_rows"])
            selected = request["selected_candidate"]
            self.assertNotEqual(selected["left_box_id"], selected["right_box_id"])
            self.assertEqual(set(request["box_ids"]), {
                selected["left_box_id"], selected["right_box_id"]
            })

    def test_ids_map_to_expected_rows_and_columns(self) -> None:
        plan = compile_pair_contract(self.contract, self.family)
        first = plan["requests"][0]
        self.assertEqual(
            [(target["box_id"], target["row_from_top"], target["column_from_left"])
             for target in first["targets"]],
            [(1, 1, 1), (10, 2, 5)],
        )

    def test_pair_candidates_share_mode_and_lift(self) -> None:
        plan = compile_pair_contract(self.contract, self.family)
        for request in plan["requests"]:
            for candidate in request["candidates"]:
                self.assertIn(candidate["grasp_mode"], {"front", "top_suction"})
                self.assertGreaterEqual(candidate["common_updown_m"], -1.0)
                self.assertLessEqual(candidate["common_updown_m"], 0.0)
                self.assertGreaterEqual(candidate["retreat_distance_m"], 0.20)

    def test_same_side_pair_adds_lateral_base_candidate(self) -> None:
        contract = json.loads(json.dumps(self.contract))
        contract["requests"] = [{
            "request_id": "same-side-01-02", "box_ids": [1, 2],
            "removed_box_ids": [], "candidate_limit": 200,
        }]
        plan = compile_pair_contract(contract, self.family)
        base_y_values = {
            round(float(candidate["base_pose_map"][1]), 6)
            for candidate in plan["requests"][0]["candidates"]
        }
        self.assertIn(0.6, base_y_values)
        self.assertIn(0.45, base_y_values)
        self.assertIn(0.0, base_y_values)
        yaw_values = {
            round(float(candidate["base_pose_map"][2]), 6)
            for candidate in plan["requests"][0]["candidates"]
        }
        self.assertIn(round(math.radians(15.0), 6), yaw_values)


if __name__ == "__main__":
    unittest.main()
