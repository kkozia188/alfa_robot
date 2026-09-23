#!/usr/bin/env python3

import copy
import importlib.util
import math
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE / "scripts"
sys.path.insert(0, str(PACKAGE.parent / "alfa_robot_rerun"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load("scan_v3_single_arm_box_wall")
SEQUENCE = load("v3_5x5_grasp_sequence_rerun")
SCOOP = load("v3_scoop_5x5_grasp_sequence_rerun")


def generalized_config():
    config = copy.deepcopy(SEQUENCE.station_contract(
        SCOOP.STATION_CONFIG_NAME, SCOOP.STATION_SCHEMA
    ))
    config["layout"] = {
        "rows": 3,
        "columns": 4,
        "active_box_ids": [1, 3, 5, 8, 11],
        "contact_x_m": 0.79,
        "center_y_m": 0.06,
        "bottom_z_m": 0.02,
        "box_depth_m": 0.32,
        "box_width_m": 0.36,
        "box_height_m": 0.38,
    }
    config["planning"].update({
        "top_down_box_ids": [8, 11],
        "left_arm_columns": [2, 4],
        "right_arm_columns": [1, 3],
        "box_order": [4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9],
        "low_transfer_tcp_box_ids": [],
        "box_overrides": {},
    })
    return config


def test_default_5x5_layout_and_selection_contract_are_unchanged():
    config = SEQUENCE.station_contract(SCOOP.STATION_CONFIG_NAME, SCOOP.STATION_SCHEMA)
    layout = SCOOP.layout_profile(config)
    assert layout == SCOOP.DEFAULT_LAYOUT
    specs = list(SCAN.box_specs(0.75, 0.30, 0.40, 0.40))
    legacy = []
    box_id = 0
    for row in range(5):
        for column in range(5):
            box_id += 1
            legacy.append((
                box_id, row + 1, column + 1, 0.90,
                (2.0 - column) * 0.40, (4.5 - row) * 0.40,
            ))
    assert specs == legacy
    top_ids, right_ids, _, _, order = SCOOP.planning_profile(config, layout)
    assert top_ids == set(range(16, 26))
    assert right_ids == {
        box_id for box_id in range(1, 26) if (box_id - 1) % 5 + 1 >= 3
    }
    assert order == config["planning"]["box_order"]
    assert SEQUENCE.ordered_candidates(1, 1, 0.8, 1.8, "front")[0] == ("right", 0.0)


def test_layout_contract_covers_non_5x5_missing_boxes_pose_dimensions_and_arm_order():
    config = generalized_config()
    layout = SCOOP.layout_profile(config)
    specs = list(SCAN.box_specs(
        layout["contact_x_m"],
        layout["box_depth_m"],
        layout["box_width_m"],
        layout["box_height_m"],
        layout["rows"],
        layout["columns"],
        layout["center_y_m"],
        layout["bottom_z_m"],
    ))
    assert len(specs) == 12
    assert specs[0][:3] == (1, 1, 1)
    assert all(math.isclose(value, expected) for value, expected in zip(
        specs[0][3:], (0.95, 0.6, 0.97)
    ))
    assert specs[-1][:3] == (12, 3, 4)
    assert all(math.isclose(value, expected) for value, expected in zip(
        specs[-1][3:], (0.95, -0.48, 0.21)
    ))

    top_ids, right_ids, overrides, box_overrides, order = SCOOP.planning_profile(
        config, layout
    )
    assert top_ids == {8, 11}
    assert right_ids == {1, 3, 5, 11}
    assert order == [3, 1, 8, 5, 11]
    assert box_overrides == {
        8: {"cartesian_transfer_search_enabled": False},
        11: {"cartesian_transfer_search_enabled": False},
    }
    assert overrides["continuous_sequence"] is True


def test_default_geometry_keeps_the_c_call_chain_and_perturbations_are_explicit():
    default = type("Args", (), {"box_depth": 0.30, "box_width": 0.40, "box_height": 0.40})()
    assert SCAN.box_geometry_launch_arguments(default) == []
    changed = type("Args", (), {"box_depth": 0.31, "box_width": 0.38, "box_height": 0.39})()
    assert SCAN.box_geometry_launch_arguments(changed) == [
        "box_depth:=0.310000", "box_width:=0.380000", "box_height:=0.390000"
    ]


def test_initial_state_seed_variation_is_reproducible_and_default_is_unchanged():
    config = generalized_config()
    assert SCOOP.jittered_initial_states(config, 0.0, 7) == {}
    first = SCOOP.jittered_initial_states(config, 0.5, 7)
    assert first == SCOOP.jittered_initial_states(config, 0.5, 7)
    assert first != SCOOP.jittered_initial_states(config, 0.5, 8)


def test_plan_sequence_uses_active_order_missing_scene_and_configured_arm_assignment(
    monkeypatch, tmp_path
):
    config = generalized_config()
    layout = SCOOP.layout_profile(config)
    specs = list(SCAN.box_specs(
        layout["contact_x_m"], layout["box_depth_m"], layout["box_width_m"],
        layout["box_height_m"], layout["rows"], layout["columns"],
        layout["center_y_m"], layout["bottom_z_m"],
    ))
    order = [8, 3]
    allowed = {8: {"left"}, 3: {"right"}}
    names = ["updown", "head_joint"] + [
        f"{side}_joint{index}" for side in ("left", "right") for index in range(1, 8)
    ]
    attempts = []

    def attempt(args, box_id, mode, removed, side, updown, center, **kwargs):
        if "transition_from_joints" in kwargs:
            start = list(kwargs["transition_from_joints"])
            goal = list(kwargs["transition_to_joints"])
            return {"success": True, "trajectory_result": {
                "frames": [
                    {"stage": "between_boxes", "joints": values, "box_attached": False}
                    for values in (start, goal)
                ],
                "transition_group": f"{side}_arm_with_updown",
            }}
        attempts.append((box_id, set(removed), side, tuple(center), args.box_grid_rows,
                         args.box_grid_columns, args.box_width))
        joints = [updown, 0.0] + [0.01 * box_id] * 14
        frames = [
            {"stage": stage, "joints": list(joints), "box_attached": attached,
             **({"carried_box_tilt_deg": 10.0} if attached else {})}
            for stage, attached in (
                ("precontact", False),
                ("cartesian_approach", False),
                ("attach_box", True),
                ("rrt_to_place", True),
                ("release_at_place", False),
            )
        ]
        return {"success": True, "trajectory_result": {
            "success": True,
            "task_mode": "full_extract",
            "return_to_ready": False,
            "joint_names": names,
            "frames": frames,
            "tool_to_box_center": [0.0, 0.0, 0.15],
            "achieved_place_tcp_pose": [-0.5, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0],
        }}

    monkeypatch.setattr(SEQUENCE.scan, "run_attempt", attempt)
    tasks = SEQUENCE.plan_sequence(
        5.0,
        0,
        tmp_path,
        limit_boxes=2,
        config=config,
        verified_selection={},
        top_suction_box_ids=set(),
        planning_overrides={
            "box_depth": layout["box_depth_m"],
            "box_width": layout["box_width_m"],
            "box_height": layout["box_height_m"],
            "box_grid_rows": layout["rows"],
            "box_grid_columns": layout["columns"],
            "box_grid_center_y": layout["center_y_m"],
            "box_grid_bottom_z": layout["bottom_z_m"],
        },
        allowed_sides_by_box=allowed,
        place_joint_resolver=SCOOP.unloading_joint_degrees,
        box_order=order,
        active_box_ids=order,
        wall_specs=specs,
    )
    missing = set(range(1, 13)) - set(order)
    assert [task["box_id"] for task in tasks] == order
    assert attempts[0][:3] == (8, missing, "left")
    assert attempts[1][:3] == (3, missing | {8}, "right")
    assert attempts[0][4:] == (3, 4, 0.36)
    assert math.isclose(attempts[0][3][0], 0.95)


if __name__ == "__main__":
    test_default_5x5_layout_and_selection_contract_are_unchanged()
    test_layout_contract_covers_non_5x5_missing_boxes_pose_dimensions_and_arm_order()
    test_initial_state_seed_variation_is_reproducible_and_default_is_unchanged()
    print("v3 scoop layout generalization gate: PASS")
