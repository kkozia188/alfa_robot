#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motion_family import ContractError, compile_contract  # noqa: E402


class MotionFamilyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(
            (ROOT / "examples/representative_tasks.json").read_text(encoding="utf-8")
        )
        cls.family = json.loads(
            (ROOT / "config/action_family.json").read_text(encoding="utf-8")
        )

    def test_five_rows_compile_without_public_box_ids(self) -> None:
        plan = compile_contract(self.contract, self.family)
        self.assertEqual(plan["summary"]["compiled"], 5)
        self.assertEqual(plan["compile_failures"], [])
        self.assertTrue(all("box_id" not in task for task in self.contract["tasks"]))
        self.assertEqual(
            [task["derived_geometry"]["row_from_top"] for task in plan["tasks"]],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            [task["selected_candidate"]["grasp_mode"] for task in plan["tasks"]],
            ["front", "front", "front", "front", "top_suction"],
        )
        self.assertEqual(
            [task["selected_candidate"]["updown_m"] for task in plan["tasks"]],
            [0.0, 0.0, -0.5, -0.5, -0.75],
        )
        self.assertEqual(
            plan["tasks"][-1]["derived_geometry"]["clearance_scene_slots"],
            [
                1, 2, 3, 4, 5,
                6, 7, 8, 9, 10,
                11, 12, 13, 14, 15,
                16, 17, 18, 19, 20,
                21, 22, 24, 25,
            ],
        )
        self.assertEqual(
            plan["tasks"][2]["derived_geometry"]["clearance_scene_slots"],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        )
        self.assertEqual(
            plan["tasks"][2]["selected_candidate"]["opposite_arm_policy"],
            "auto_safe",
        )
        self.assertEqual(
            plan["tasks"][0]["selected_candidate"]["opposite_arm_policy"],
            "hold_current",
        )
        self.assertEqual(
            plan["tasks"][3]["selected_candidate"]["opposite_arm_policy"],
            "auto_safe",
        )

    def test_pose_drives_arm_selection(self) -> None:
        plan = compile_contract(self.contract, self.family)
        self.assertEqual(plan["tasks"][0]["selected_candidate"]["arm"], "left")
        self.assertEqual(plan["tasks"][1]["selected_candidate"]["arm"], "right")
        self.assertEqual(plan["tasks"][2]["selected_candidate"]["arm"], "right")
        self.assertEqual(plan["tasks"][4]["selected_candidate"]["arm"], "left")

    def test_bottom_front_blockage_selects_top_template(self) -> None:
        plan = compile_contract(self.contract, self.family)
        bottom = plan["tasks"][-1]
        self.assertEqual(bottom["selected_candidate"]["template"], "top_face_extract")
        self.assertIn(
            "front clearance",
            " ".join(item["reason"] for item in bottom["rejected_templates"]),
        )

    def test_orientation_limit_is_reported_per_request(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["tasks"] = [copy.deepcopy(contract["tasks"][0])]
        contract["tasks"][0]["box_pose"]["orientation_xyzw"] = [0, 0, 0.173648, 0.984808]
        plan = compile_contract(contract, self.family)
        self.assertEqual(plan["summary"]["compiled"], 0)
        self.assertIn("orientation differs", plan["compile_failures"][0]["reason"])

    def test_no_clearance_produces_actionable_failure(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["tasks"] = [copy.deepcopy(contract["tasks"][0])]
        contract["tasks"][0]["available_space"].update(
            front_clearance_m=0.1, top_clearance_m=0.1
        )
        plan = compile_contract(contract, self.family)
        self.assertEqual(plan["summary"]["failed"], 1)
        self.assertIn("no feasible action template", plan["compile_failures"][0]["reason"])

    def test_invalid_current_state_fails_contract(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["current_state"]["joint_positions"] = [0.0] * 16
        with self.assertRaisesRegex(ContractError, "17 numbers"):
            compile_contract(contract, self.family)


if __name__ == "__main__":
    unittest.main()
