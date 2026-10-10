#!/usr/bin/env python3

from __future__ import annotations

import math
import tempfile
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
from open_single_axis_rerun import (
    certified_integer_yaw,
    exact_y_case,
    round_y,
    round_yaw,
    selected_axis,
    x_case_mode,
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

    def test_unified_rerun_rejects_combined_errors(self) -> None:
        self.assertEqual(selected_axis(0.86, 0.60, 0.0, 0.0), "x")
        self.assertEqual(selected_axis(0.85, 0.60, -0.1, 0.0), "y")
        self.assertEqual(selected_axis(0.85, 0.60, 0.0, 3.0), "yaw")
        with self.assertRaises(ValueError):
            selected_axis(0.86, 0.60, 0.01, 0.0)
        with self.assertRaises(ValueError):
            selected_axis(0.85, 0.60, 0.01, 1.0)

    def test_unified_rerun_requires_complete_discrete_y_and_integer_yaw(self) -> None:
        expected = Path(tempfile.gettempdir()) / "case-result.json"
        value, path = exact_y_case(0.0875, {0.0875: expected})
        self.assertEqual(value, 0.0875)
        self.assertEqual(path, expected)
        with self.assertRaises(ValueError):
            exact_y_case(0.08, {0.0875: expected})
        self.assertEqual(certified_integer_yaw(-5.0), -5)
        self.assertIsNone(certified_integer_yaw(2.5))

    def test_unified_rerun_accepts_maximum_independent_ranges(self) -> None:
        self.assertEqual(x_case_mode(0.80, 0.55), "strong_grid")
        self.assertEqual(x_case_mode(0.74, 0.60), "upper_extension")
        self.assertEqual(x_case_mode(1.02, 0.60), "upper_extension")
        self.assertEqual(x_case_mode(0.85, 0.36), "lower_extension")
        self.assertEqual(x_case_mode(0.85, 0.79), "lower_extension")
        with self.assertRaises(ValueError):
            x_case_mode(0.74, 0.36)
        with self.assertRaises(ValueError):
            x_case_mode(0.73, 0.60)
        with self.assertRaises(ValueError):
            x_case_mode(0.85, 0.80)

    def test_unified_rerun_rounds_y_and_yaw_to_stable_case_ids(self) -> None:
        self.assertEqual(round_y(0.08749), 0.0875)
        self.assertEqual(round_y(0.0866), 0.087)
        self.assertEqual(round_y(0.09), 0.09)
        self.assertEqual(round_yaw(2.46), 2.5)
        self.assertEqual(round_yaw(-2.45), -2.5)


if __name__ == "__main__":
    unittest.main()
