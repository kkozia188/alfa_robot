#!/usr/bin/env python3

from __future__ import annotations

import math
import unittest
from pathlib import Path

from generate_matrix import build_matrix, build_x_clearance_matrix
from matrix_common import (
    base_x_for_front_clearance,
    case_slug,
    classify_failure,
    clearance_slug,
    evaluate_quality,
    local_shuttle_poses,
    quantize_front_clearance,
    is_single_axis_error,
    read_json,
    x_clearance_variable,
)


ROOT = Path(__file__).resolve().parent


class MatrixToolTests(unittest.TestCase):
    def test_case_slug_is_stable_and_sign_preserving(self) -> None:
        self.assertEqual(
            case_slug(-0.05, 0.1, -2.5),
            "dx-m050mm_dy-p100mm_yaw-m025d10",
        )

    def test_x_clearance_contract_uses_vehicle_front_plane(self) -> None:
        self.assertAlmostEqual(
            base_x_for_front_clearance(0.85), -0.600000002779484, places=12
        )
        self.assertEqual(clearance_slug(0.79, 0.60), "front-u0790mm_l0600mm")
        self.assertEqual(x_clearance_variable(0.79, 0.60), "upper_rows_1_to_3")
        self.assertEqual(x_clearance_variable(0.85, 0.66), "lower_rows_4_to_5")
        with self.assertRaises(ValueError):
            x_clearance_variable(0.79, 0.66)
        self.assertEqual(quantize_front_clearance(0.854), 0.85)
        self.assertEqual(quantize_front_clearance(0.855), 0.86)

    def test_failure_classification(self) -> None:
        self.assertEqual(classify_failure("precontact_ik", "no solution"), "ik_or_reachability")
        self.assertEqual(
            classify_failure(
                "precontact_ik", "candidates=0 bounds=0 collision=0 edge=0"
            ),
            "ik_or_reachability",
        )
        self.assertEqual(classify_failure("validation", "collision:a<->b"), "collision")
        self.assertEqual(classify_failure("planner", "timeout"), "planning_timeout")
        self.assertEqual(
            classify_failure("cartesian_approach", "no continuous solution"),
            "approach",
        )
        self.assertEqual(classify_failure("retreat", "carried box tilt"), "carried_box_tilt")

    def test_quality_gate_distinguishes_warning_and_failure(self) -> None:
        thresholds = read_json(ROOT / "default_matrix.json")["thresholds"]
        status, findings = evaluate_quality(
            task_core_ms=[3200.0], maximum_bridge_ms=100.0,
            maximum_tilt_deg=1.0, maximum_joint_step_deg=2.0,
            joint_flip_events=0, thresholds=thresholds,
        )
        self.assertEqual(status, "degraded")
        self.assertEqual(findings, ["task_core_over_target"])
        status, findings = evaluate_quality(
            task_core_ms=[100.0], maximum_bridge_ms=100.0,
            maximum_tilt_deg=20.0, maximum_joint_step_deg=2.0,
            joint_flip_events=0, thresholds=thresholds,
        )
        self.assertEqual(status, "failed_quality_gate")
        self.assertIn("carried_box_tilt_failure", findings)

    def test_default_matrix_is_strictly_single_axis(self) -> None:
        spec = read_json(ROOT / "default_matrix.json")
        cases = build_matrix(spec)
        self.assertEqual(len(cases), 49)
        self.assertTrue(all(
            is_single_axis_error(case["dx_m"], case["dy_m"], case["yaw_deg"])
            for case in cases
        ))
        self.assertTrue(any("y_baseline" in case["phases"] for case in cases))
        self.assertTrue(any("boundary_probe_y" in case["phases"] for case in cases))
        x_cases = build_x_clearance_matrix(spec)
        self.assertEqual(len(x_cases), 77)
        self.assertTrue(all(
            case["upper_front_clearance_m"] == 0.85
            or case["lower_front_clearance_m"] == 0.60
            for case in x_cases
        ))

    def test_shuttle_route_is_relative_to_robot_yaw(self) -> None:
        backed, conveyor = local_shuttle_poses(
            [1.0, 2.0, math.pi / 2.0], 2.35, 1.5
        )
        self.assertAlmostEqual(backed[0], 1.0)
        self.assertAlmostEqual(backed[1], -0.35)
        self.assertAlmostEqual(conveyor[0], 2.5)
        self.assertAlmostEqual(conveyor[1], -0.35)


if __name__ == "__main__":
    unittest.main()
