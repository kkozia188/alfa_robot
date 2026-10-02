#!/usr/bin/env python3
"""Unit tests for benchmark design, classification, and report math."""

from __future__ import annotations

import unittest

from docking_error_common import (
    build_summary,
    classify_failure,
    cycle_catalog,
    infer_planning_phases,
    matrix_samples,
    sample_id,
)


CONFIG = {
    "schema": "alfa.v3_docking_error_benchmark_config.v1",
    "experiment_label": "test",
    "strategy_label": "before",
    "model_revision": "robot_v3.2.2-suction",
    "tool0_offset_local_z_m": 0.151,
    "upstream_base_commit": "d9c330cef72981390d81ac2b1cd5a6eb9e892195",
    "error_matrix": {"x_m": [-0.05, 0.0], "y_m": [0.0], "yaw_deg": [0.0]},
    "cycle_ids": [1],
    "nominal_base_pose_map": {
        "upper_rows": [-0.6, 0.0, 0.0],
        "lower_rows": [-0.35, 0.0, 0.0],
    },
    "wall_pose_map": {"near_face_x_m": 0.75},
}


class CommonTests(unittest.TestCase):
    def test_catalog_matches_motion_261_group_and_arm_contract(self) -> None:
        catalog = cycle_catalog()
        self.assertEqual(len(catalog), 15)
        self.assertEqual(catalog[0]["motion_box_ids"], [1, 5])
        self.assertEqual(catalog[0]["left_catalog_box_id"], 24)
        self.assertEqual(catalog[0]["right_catalog_box_id"], 20)
        self.assertEqual(catalog[13]["grasp_mode"], "top_suction")
        self.assertEqual(catalog[14]["left_motion_box_id"], 23)
        self.assertIsNone(catalog[14]["right_motion_box_id"])

    def test_matrix_and_stable_ids(self) -> None:
        self.assertEqual(len(matrix_samples(CONFIG)), 2)
        self.assertEqual(sample_id(0.0, 0.0, 0.0), "dx_p000_dy_p000_yaw_p0000")
        self.assertEqual(sample_id(-0.05, 0.025, -3.0), "dx_m050_dy_p025_yaw_m3000")

    def test_failure_classification_and_phase_inference(self) -> None:
        stage = {
            "stage": "PREGRASP",
            "ok": False,
            "failure_stage": "cartesian_retreat",
            "error_detail": "cartesian_retreat: collision:base_link<->left_link5",
        }
        self.assertEqual(classify_failure(stage), "collision")
        phases = infer_planning_phases({"stages": [stage]})
        self.assertEqual(
            phases,
            {"IK": True, "PREGRASP": True, "EXTRACT": False, "RETURN": None},
        )

    def test_summary_keeps_all_cycles_in_denominator(self) -> None:
        success_stages = [{"stage": name, "ok": True} for name in ("PREGRASP", "APPROACH", "PLACE", "HOME")]
        failure_stage = {
            "stage": "PREGRASP",
            "ok": False,
            "failure_stage": "precontact_ik",
            "error_detail": "precontact_ik: candidates=0",
        }
        records = [
            {
                "sample_id": "a",
                "error": {"x_m": -0.05, "y_m": 0.0, "yaw_deg": 0.0},
                "cycles": [{"cycle_id": 1, "motion_box_ids": [1, 5], "grasp_mode": "side_suction", "ok": False, "stages": [failure_stage]}],
            },
            {
                "sample_id": "b",
                "error": {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0},
                "cycles": [{"cycle_id": 1, "motion_box_ids": [1, 5], "grasp_mode": "side_suction", "ok": True, "stages": success_stages}],
            },
        ]
        summary = build_summary(CONFIG, records)
        self.assertEqual(summary["cycle_success_rate"], 0.5)
        self.assertEqual(summary["failure_categories"], {"ik_unreachable": 1})
        self.assertEqual(summary["planning_phase_success"]["IK"]["all_cycle_success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
