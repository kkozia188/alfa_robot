#!/usr/bin/env python3

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v3_warehouse_return_demo.py"
SPEC = importlib.util.spec_from_file_location("v3_warehouse_return_demo", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def geometry():
    return MODULE.load_model_geometry()


def test_certified_v322_home_shell_dimensions(geometry):
    assert geometry.urdf_path.name == "alfa-robot-v322-tool0151.urdf"
    assert np.allclose(
        geometry.local_min,
        [-0.50830927, -0.54284679, 0.015],
        atol=1e-7,
    )
    assert np.allclose(
        geometry.local_max,
        [0.61188312, 0.54284685, 2.02251868],
        atol=1e-7,
    )
    assert geometry.rotation_radius == pytest.approx(0.7334765584, abs=1e-9)
    assert geometry.z_min >= 0.0
    assert geometry.z_max + MODULE.SAFETY_CLEARANCE_M < MODULE.WAREHOUSE_HEIGHT_M
    assert len(geometry.footprint) >= 4


def test_uniform_input_range_is_collision_free_for_every_yaw(geometry):
    bounds = MODULE.safe_input_bounds(geometry)
    assert bounds.x_min == pytest.approx(-0.5299999972, abs=1e-9)
    assert bounds.x_max == pytest.approx(0.6165234444, abs=1e-9)
    assert bounds.y_min == pytest.approx(-0.4065234416, abs=1e-9)
    assert bounds.y_max == pytest.approx(0.4065234416, abs=1e-9)
    for x in (bounds.x_min, 0.0, bounds.x_max):
        for y in (bounds.y_min, 0.0, bounds.y_max):
            for yaw_deg in range(-180, 181, 5):
                pose = MODULE.validate_input_pose(geometry, x, y, yaw_deg)
                assert not MODULE.collision_reason(geometry, pose)


def test_pose_outside_safe_input_contract_is_rejected(geometry):
    bounds = MODULE.safe_input_bounds(geometry)
    with pytest.raises(ValueError, match="x must be"):
        MODULE.validate_input_pose(geometry, bounds.x_max + 0.001, 0.0, 0.0)
    with pytest.raises(ValueError, match="y must be"):
        MODULE.validate_input_pose(geometry, 0.0, bounds.y_max + 0.001, 0.0)
    with pytest.raises(ValueError, match="Yaw must be"):
        MODULE.validate_input_pose(geometry, 0.0, 0.0, 180.001)
    rounded_boundary = MODULE.validate_input_pose(geometry, -0.530000, 0.0, 0.0)
    assert rounded_boundary.x == pytest.approx(bounds.x_min)


@pytest.mark.parametrize(
    "x,y,yaw_deg",
    [
        (0.0, 0.0, 0.0),
        (0.35, 0.20, 135.0),
        (-0.40, -0.30, -170.0),
        (0.60, -0.38, 42.0),
        (-0.50, 0.38, -83.0),
    ],
)
def test_closed_loop_return_is_bounded_and_collision_free(
    geometry, x, y, yaw_deg
):
    start = MODULE.validate_input_pose(geometry, x, y, yaw_deg)
    plan = MODULE.plan_return(geometry, start)
    final = plan.samples[-1]
    assert math.hypot(final.x, final.y) < 1e-3
    assert abs(MODULE.wrap_angle(final.yaw)) < math.radians(0.1)
    assert plan.translation_length_m == pytest.approx(math.hypot(x, y), abs=0.01)
    assert max(abs(sample.linear_velocity_mps) for sample in plan.samples) <= (
        MODULE.MAX_LINEAR_SPEED_MPS + 1e-9
    )
    assert max(abs(sample.angular_velocity_rad_s) for sample in plan.samples) <= (
        MODULE.MAX_ANGULAR_SPEED_RAD_S + 1e-9
    )
    for left, right in zip(plan.samples, plan.samples[1:]):
        dt = right.time_s - left.time_s
        assert abs(right.linear_velocity_mps - left.linear_velocity_mps) <= (
            MODULE.MAX_LINEAR_ACCEL_MPS2 * dt + 1e-9
        )
        assert abs(right.angular_velocity_rad_s - left.angular_velocity_rad_s) <= (
            MODULE.MAX_ANGULAR_ACCEL_RAD_S2 * dt + 1e-9
        )
        assert not MODULE.collision_reason(
            geometry, MODULE.Pose2D(right.x, right.y, right.yaw)
        )


def test_forward_reverse_choice_minimizes_total_rotation():
    start = MODULE.Pose2D(-0.4, 0.0, math.radians(170.0))
    _, signed_distance, direction, initial, final = MODULE.select_straight_motion(start)
    assert direction == "forward"
    assert signed_distance > 0.0
    chosen_rotation = abs(initial) + abs(final)
    reverse_heading = math.pi
    alternate_rotation = (
        abs(MODULE.wrap_angle(reverse_heading - start.yaw))
        + abs(MODULE.wrap_angle(-reverse_heading))
    )
    assert chosen_rotation <= alternate_rotation


def test_boundary_matrix_returns_collision_free(geometry):
    bounds = MODULE.safe_input_bounds(geometry)
    x_values = (bounds.x_min, 0.5 * (bounds.x_min + bounds.x_max), bounds.x_max)
    y_values = (bounds.y_min, 0.0, bounds.y_max)
    checked = 0
    for x in x_values:
        for y in y_values:
            for yaw_deg in range(-180, 181, 15):
                start = MODULE.validate_input_pose(geometry, x, y, yaw_deg)
                plan = MODULE.plan_return(geometry, start)
                final = plan.samples[-1]
                assert math.hypot(final.x, final.y) < 1e-3
                assert abs(MODULE.wrap_angle(final.yaw)) < math.radians(0.1)
                checked += 1
    assert checked == 225
