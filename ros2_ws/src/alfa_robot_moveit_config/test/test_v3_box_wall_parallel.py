#!/usr/bin/env python3

import importlib.util
import json
import math
import sys
from pathlib import Path

import yaml

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "v3_box_wall_parallel_common.py"
)
SPEC = importlib.util.spec_from_file_location("v3_box_wall_parallel_common", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

GPU_SCRIPT = SCRIPT.with_name("v3_curobo_gpu_prefilter.py")
GPU_SPEC = importlib.util.spec_from_file_location("v3_curobo_gpu_prefilter", GPU_SCRIPT)
assert GPU_SPEC and GPU_SPEC.loader
GPU_MODULE = importlib.util.module_from_spec(GPU_SPEC)
GPU_SPEC.loader.exec_module(GPU_MODULE)

SCAN_SCRIPT = SCRIPT.with_name("scan_v3_single_arm_box_wall.py")
SCAN_SPEC = importlib.util.spec_from_file_location("scan_v3_single_arm_box_wall", SCAN_SCRIPT)
assert SCAN_SPEC and SCAN_SPEC.loader
SCAN_MODULE = importlib.util.module_from_spec(SCAN_SPEC)
SCAN_SPEC.loader.exec_module(SCAN_MODULE)

WORKSPACE_SRC = SCRIPT.parents[2]
sys.path.insert(0, str(WORKSPACE_SRC / "alfa_robot_rerun"))
SEQUENCE_SCRIPT = SCRIPT.with_name("v3_5x5_grasp_sequence_rerun.py")
SEQUENCE_SPEC = importlib.util.spec_from_file_location(
    "v3_5x5_grasp_sequence_rerun", SEQUENCE_SCRIPT
)
assert SEQUENCE_SPEC and SEQUENCE_SPEC.loader
SEQUENCE_MODULE = importlib.util.module_from_spec(SEQUENCE_SPEC)
sys.modules[SEQUENCE_SPEC.name] = SEQUENCE_MODULE
SEQUENCE_SPEC.loader.exec_module(SEQUENCE_MODULE)

SCOOP_SCRIPT = SCRIPT.with_name("v3_scoop_5x5_grasp_sequence_rerun.py")
SCOOP_SPEC = importlib.util.spec_from_file_location(
    "v3_scoop_5x5_grasp_sequence_rerun", SCOOP_SCRIPT
)
assert SCOOP_SPEC and SCOOP_SPEC.loader
SCOOP_MODULE = importlib.util.module_from_spec(SCOOP_SPEC)
SCOOP_SPEC.loader.exec_module(SCOOP_MODULE)


def test_box_wall_has_expected_25_contact_points():
    boxes = MODULE.box_specs(0.65)
    assert len(boxes) == 25
    assert boxes[0].box_id == 1
    assert boxes[0].center_x_m == 0.8
    assert boxes[0].center_y_m == 0.8
    assert boxes[0].center_z_m == 1.8
    assert boxes[12].box_id == 13
    assert boxes[12].center_y_m == 0.0
    assert boxes[12].center_z_m == 1.0
    assert boxes[-1].box_id == 25
    assert boxes[-1].center_y_m == -0.8
    assert boxes[-1].center_z_m == 0.2


def test_random_wall_distances_are_reproducible_and_shared_by_each_wall():
    first = MODULE.random_contact_xs(0.60, 0.80, 8, 309)
    second = MODULE.random_contact_xs(0.60, 0.80, 8, 309)
    assert first == second
    assert len(first) == 8
    assert all(0.60 <= value <= 0.80 for value in first)
    candidates = MODULE.build_candidates(
        first[:1], ("left", "right"), MODULE.DEFAULT_UPDOWN, set(), 0.30, 0.40, 0.40
    )
    assert len(candidates) == 25 * 2 * 5
    assert {item.box.contact_x_m for item in candidates} == {first[0]}


def test_x_grid_085_to_099_has_15_one_centimeter_points():
    values = MODULE.grid_contact_xs(0.85, 0.99, 0.01)
    assert len(values) == 15
    assert values[0] == 0.85
    assert values[-1] == 0.99
    assert all(abs((right - left) - 0.01) < 1e-12 for left, right in zip(values, values[1:]))


def test_gpu_results_only_change_cpu_candidate_order():
    box = MODULE.box_specs(0.65)[0]
    group = [
        MODULE.Candidate(0, box, "left", updown)
        for updown in MODULE.DEFAULT_UPDOWN
    ]
    gpu = {
        group[-1].key: {"success": True, "score": 0.2},
        group[1].key: {"success": True, "score": 0.1},
    }
    ordered = sorted(group, key=lambda item: MODULE._candidate_priority(item, gpu))
    assert [item.updown_m for item in ordered[:2]] == [-0.25, -1.0]
    assert {item.updown_m for item in ordered} == set(MODULE.DEFAULT_UPDOWN)


def test_unreachable_arm_is_not_scheduled_for_extraction():
    box = MODULE.box_specs(0.65)[0]
    reachability = {
        "walls": [
            {
                "wall_index": 0,
                "cells": [
                    {
                        **MODULE.asdict(box),
                        "arms": {
                            "left": {
                                "reachable": True,
                                "reachable_updowns_m": [0.0, -0.5],
                            },
                            "right": {
                                "reachable": False,
                                "reachable_updowns_m": [],
                            },
                        },
                    }
                ],
            }
        ]
    }
    args = type("Args", (), {"sides": ("left", "right")})()
    candidates, skipped = MODULE._reachable_candidates(reachability, args)
    assert [(item.side, item.updown_m) for item in candidates] == [
        ("left", 0.0),
        ("left", -0.5),
    ]
    assert (0, 1, "left") not in skipped
    assert skipped[(0, 1, "right")]["status"] == "skipped_unreachable"


def test_updown_parser_rejects_values_outside_physical_range():
    assert MODULE.parse_float_list("0,-0.25,-0.5,-0.75,-1") == MODULE.DEFAULT_UPDOWN
    try:
        MODULE.parse_float_list("-1.01")
    except Exception as error:
        assert "[-1.0, 0.0]" in str(error)
    else:
        raise AssertionError("out-of-range updown should fail")


def test_gpu_world_contains_24_neighbors_and_uses_carriage_frame():
    candidate = MODULE.candidate_payload(
        MODULE.Candidate(0, MODULE.box_specs(0.65)[12], "left", -0.5)
    )
    box = {
        "depth_m": 0.30,
        "width_m": 0.40,
        "height_m": 0.40,
        "collision_inset_m": 0.002,
    }
    cuboids = GPU_MODULE.wall_world(candidate, box)["cuboid"]
    assert len(cuboids) == 24
    assert "neighbor_box_13" not in cuboids
    assert all(value["dims"] == [0.296, 0.396, 0.396] for value in cuboids.values())
    assert GPU_MODULE.goal_pose(candidate, box, "contact")[:3] == [0.65, 0.0, 1.5]
    assert GPU_MODULE.goal_pose(candidate, box, "retreat")[0] == 0.30000000000000004


def test_x075_mixed_grasp_strategy_top_suctions_box18_and_complete_bottom_row():
    modes = {
        box_id: SCAN_MODULE.grasp_mode_for_box(
            box_id, SCAN_MODULE.DEFAULT_TOP_SUCTION_BOX_IDS
        )
        for box_id in range(1, 26)
    }
    assert [box_id for box_id, mode in modes.items() if mode == "top_suction"] == [
        18, 21, 22, 23, 24, 25,
    ]
    assert sum(mode == "front" for mode in modes.values()) == 19


def test_mixed_strategy_prioritizes_high_platform_then_lower_rows():
    top_row = SCAN_MODULE.candidate_order(1, 0.8, 1.8, 0.25, "front")
    lower_side = SCAN_MODULE.candidate_order(4, 0.8, 0.6, 0.25, "front")
    lower_center = SCAN_MODULE.candidate_order(5, 0.0, 0.2, 0.25, "top_suction")
    assert top_row[0] == ("left", 0.0)
    assert lower_side[0] == ("left", -0.75)
    assert lower_center[0] == ("right", -1.0)
    assert {side for side, _ in lower_center} == {"left", "right"}


def test_mixed_strategy_keeps_front_suction_points_inside_exposed_faces():
    assert SCAN_MODULE.front_suction_offsets(
        8, 0.0, "left", "front", 0.08, -0.05, 0.12
    ) == (0.08, -0.05)
    assert SCAN_MODULE.front_suction_offsets(
        8, 0.0, "right", "front", 0.08, -0.05, 0.12
    ) == (-0.08, -0.05)
    assert SCAN_MODULE.front_suction_offsets(
        21, 0.8, "right", "front", 0.08, -0.05, 0.12
    ) == (0.0, 0.12)
    assert SCAN_MODULE.front_suction_offsets(
        23, 0.0, "left", "top_suction", 0.08, -0.05, 0.12
    ) == (0.0, 0.0)


def test_rerun_sequence_has_a_verified_selection_for_every_box():
    selection = SEQUENCE_MODULE.VERIFIED_SELECTION
    assert set(selection) == set(range(1, 26))
    assert all(side in {"left", "right"} for side, _ in selection.values())
    assert all(-1.0 <= updown <= 0.0 for _, updown in selection.values())
    specs = list(
        SCAN_MODULE.box_specs(
            SEQUENCE_MODULE.CONTACT_X,
            SEQUENCE_MODULE.BOX_DEPTH,
            SEQUENCE_MODULE.BOX_WIDTH,
            SEQUENCE_MODULE.BOX_HEIGHT,
        )
    )
    for box_id, row, _, _, y, z in specs:
        mode = SCAN_MODULE.grasp_mode_for_box(
            box_id, SCAN_MODULE.DEFAULT_TOP_SUCTION_BOX_IDS
        )
        assert SEQUENCE_MODULE.ordered_candidates(box_id, row, y, z, mode)[0] == selection[box_id]


def test_rerun_warehouse_has_requested_clear_dimensions_and_open_front():
    panels = {panel["id"]: panel for panel in SEQUENCE_MODULE.warehouse_panels()}
    assert set(panels) == {
        "warehouse_left_wall",
        "warehouse_right_wall",
        "warehouse_ceiling",
        "warehouse_rear_wall",
    }
    assert SEQUENCE_MODULE.WAREHOUSE_LENGTH == 2.38
    assert SEQUENCE_MODULE.WAREHOUSE_WIDTH == 2.38
    assert SEQUENCE_MODULE.WAREHOUSE_HEIGHT == 2.35
    assert SEQUENCE_MODULE.WAREHOUSE_OPENING_X == -1.18
    assert panels["warehouse_left_wall"]["size"] == [2.38, 0.05, 2.35]
    assert math.isclose(panels["warehouse_right_wall"]["center"][1], -1.215)
    assert math.isclose(panels["warehouse_left_wall"]["center"][1], 1.215)
    assert math.isclose(panels["warehouse_ceiling"]["center"][2], 2.375)
    assert math.isclose(panels["warehouse_rear_wall"]["center"][0], 1.225)
    args = SEQUENCE_MODULE.planning_args(35.0)
    assert args.initial_left_arm_joints_deg == "0,90,-120,75,0,0,0"
    assert args.initial_right_arm_joints_deg == "-45,-90,120,-75,0,0,0"
    assert args.top_initial_right_arm_joints_deg == "0,-45,120,-75,0,0,0"
    assert args.top_suction_x_offset == -0.10
    assert args.front_retreat_distance_m == 0.35
    assert args.top_retreat_distance_m == 0.35
    assert args.natural_seed_swivel_sampling is False
    assert args.natural_joint_acceleration_weight == 0.0
    assert args.natural_joint_wrap_weight == 0.0
    assert args.natural_place_return_weight == 0.0
    assert args.natural_cartesian_replay_step_deg == 0.0
    assert args.natural_rrt_shortcut_enabled is False
    assert args.natural_rrt_shortcut_max_nodes == 0
    assert args.upper_front_success_trials == 1
    assert args.upper_front_max_success_trials == 1
    assert args.conveyor_success_trials == 1
    assert args.conveyor_max_success_trials == 1
    assert args.place_updown_enabled is False
    assert args.place_updown_m == 0.0
    assert args.top_loaded_transfer_direct_only is False
    assert args.natural_max_proximal_step_deg == 12.0
    assert args.natural_max_wrist_step_deg == 8.0


def test_rerun_sequence_only_retries_stochastic_planning_failures():
    assert SEQUENCE_MODULE.retryable_failure({"failure_stage": "rrt_return"})
    assert SEQUENCE_MODULE.retryable_failure({"failure_stage": "rrt_to_precontact"})
    assert SEQUENCE_MODULE.retryable_failure({"failure_stage": "rrt_to_updown_safe"})
    assert SEQUENCE_MODULE.retryable_failure({"failure_stage": "timeout"})
    assert SEQUENCE_MODULE.retryable_failure({"failure_stage": "transition_path_collision"})
    assert not SEQUENCE_MODULE.retryable_failure({"failure_stage": "contact_ik"})
    assert not SEQUENCE_MODULE.retryable_failure({"failure_stage": "cartesian_retreat"})


def test_complete_motion_score_prefers_less_travel_and_fewer_reversals():
    names = ["updown", "head_joint"] + [
        f"{side}_joint{index}"
        for side in ("left", "right")
        for index in range(1, 8)
    ]

    def payload(joint1_degrees):
        frames = []
        for index, degrees in enumerate(joint1_degrees):
            joints = [0.0] * len(names)
            joints[names.index("left_joint1")] = math.radians(degrees)
            frames.append({
                "stage": "rrt_to_precontact" if index < 3 else "rrt_to_place",
                "joints": joints,
            })
        return {"joint_names": names, "frames": frames}

    compact = SEQUENCE_MODULE.active_arm_motion_metrics(
        payload([0.0, 1.0, 2.0, 3.0, 4.0]), "left"
    )
    winding = SEQUENCE_MODULE.active_arm_motion_metrics(
        payload([0.0, 2.0, 0.0, 4.0, 1.0]), "left"
    )
    assert compact["total_joint_travel_deg"] < winding["total_joint_travel_deg"]
    assert compact["direction_reversals"] < winding["direction_reversals"]
    assert compact["selection_score"] < winding["selection_score"]


def test_upper_front_motion_quality_contract_is_strict():
    contract = {
        "max_upper_front_total_joint_travel_deg": 1900.0,
        "max_upper_front_stage_excess_travel_deg": 500.0,
        "max_upper_front_direction_reversals": 15,
        "max_upper_front_joint_range_deg": 310.0,
        "max_upper_front_joint_step_deg": 3.0,
    }
    accepted = {
        "total_joint_travel_deg": 1900.0,
        "stage_excess_joint_travel_deg": 500.0,
        "direction_reversals": 15,
        "max_joint_range_deg": 310.0,
        "max_joint_step_deg": 3.0,
    }
    assert SEQUENCE_MODULE.upper_front_motion_accepted(accepted, contract)
    for name in accepted:
        rejected = dict(accepted)
        rejected[name] = rejected[name] + 1
        assert not SEQUENCE_MODULE.upper_front_motion_accepted(rejected, contract)


def test_named_station_handoff_tcp_and_stow_are_explicit():
    config = SEQUENCE_MODULE.station_contract()
    assert config["schema"] == "alfa.v3_5x5_station.v1"
    folded = SEQUENCE_MODULE.folded_start_joints(config)
    assert len(folded) == 16
    assert math.isclose(math.degrees(folded[9]), -25.0)
    args = SEQUENCE_MODULE.planning_args(35.0, config)
    assert args.initial_left_arm_joints_deg == "0,90,-120,75,0,0,0"
    assert args.initial_right_arm_joints_deg == "-45,-90,120,-75,0,0,0"
    assert args.top_initial_right_arm_joints_deg == "0,-45,120,-75,0,0,0"
    for side, mode in (("left", "front"), ("right", "front"), ("right", "top_suction")):
        high = SEQUENCE_MODULE.handoff_tcp_pose(config, side, mode, 0.0)
        low = SEQUENCE_MODULE.handoff_tcp_pose(config, side, mode, -0.75)
        assert high[:2] == low[:2]
        assert math.isclose(high[2] - low[2], 0.75)
        assert high[3:] == low[3:]
        assert math.isclose(sum(value * value for value in high[3:]), 1.0, abs_tol=1e-6)


def test_handoff_requires_fixed_tcp_and_complete_release_cycle():
    pose = SEQUENCE_MODULE.handoff_tcp_pose(
        SEQUENCE_MODULE.station_contract(), "right", "front", -0.5
    )
    payload = {
        "success": True,
        "task_mode": "full_extract",
        "achieved_place_tcp_pose": pose,
        "frames": [
            {"stage": "rrt_to_place"},
            {"stage": "release_at_place"},
            {"stage": "rrt_to_ready"},
        ],
    }
    SEQUENCE_MODULE.check_handoff_result(payload, pose)
    for invalid in (
        {**payload, "frames": payload["frames"][:2]},
        {**payload, "achieved_place_tcp_pose": [pose[0] + 0.01, *pose[1:]]},
        {**payload, "frames": list(reversed(payload["frames"]))},
    ):
        try:
            SEQUENCE_MODULE.check_handoff_result(invalid, pose)
        except ValueError:
            pass
        else:
            raise AssertionError("an invalid handoff must not count as success")


def test_continuous_handoff_ends_at_release_and_rejects_hidden_home_return():
    pose = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    payload = {
        "success": True,
        "task_mode": "full_extract",
        "return_to_ready": False,
        "achieved_place_tcp_pose": pose,
        "frames": [{"stage": "rrt_to_place"}, {"stage": "release_at_place"}],
    }
    SEQUENCE_MODULE.check_handoff_result(payload, pose)
    for forbidden in ("rrt_to_ready", "updown_to_task"):
        invalid = {**payload, "frames": [*payload["frames"], {"stage": forbidden}]}
        try:
            SEQUENCE_MODULE.check_handoff_result(invalid, pose)
        except ValueError:
            pass
        else:
            raise AssertionError("continuous release must not include a home return")


def test_carried_orientation_metrics_only_measure_loaded_frames():
    metrics = SEQUENCE_MODULE.carried_box_orientation_metrics({"frames": [
        {"box_attached": False, "carried_box_tilt_deg": 180.0},
        {"box_attached": True, "carried_box_tilt_deg": 0.0},
        {"box_attached": True, "carried_box_tilt_deg": 90.9},
    ]})
    assert metrics["max_tilt_deg"] == 90.9
    assert metrics["inverted_frames"] == 0
    assert metrics["loaded_frames"] == 2
    inverted = SEQUENCE_MODULE.carried_box_orientation_metrics({"frames": [
        {"box_attached": True, "carried_box_tilt_deg": 180.0},
    ]})
    assert inverted["inverted_frames"] == 1


def test_transition_quality_rejects_long_or_oscillating_paths():
    contract = SEQUENCE_MODULE.station_contract(
        SCOOP_MODULE.STATION_CONFIG_NAME, SCOOP_MODULE.STATION_SCHEMA
    )["motion_acceptance"]
    accepted = {
        "frame_count": 450,
        "total_joint_travel_deg": 3500.0,
        "stage_excess_joint_travel_deg": 2200.0,
        "direction_reversals": 30,
        "max_joint_range_deg": 360.0,
        "simultaneous_dual_arm_motion": False,
    }
    assert SEQUENCE_MODULE.transition_motion_accepted(accepted, contract)
    for name in accepted:
        if name == "simultaneous_dual_arm_motion":
            continue
        rejected = dict(accepted)
        rejected[name] = rejected[name] + 1
        assert not SEQUENCE_MODULE.transition_motion_accepted(rejected, contract)
    simultaneous = dict(accepted)
    simultaneous["simultaneous_dual_arm_motion"] = True
    assert not SEQUENCE_MODULE.transition_motion_accepted(simultaneous, contract)


def test_transition_metrics_detect_joint_limit_winding():
    names = ["updown", "head_joint"] + [
        f"{side}_joint{index}"
        for side in ("left", "right") for index in range(1, 8)
    ]
    frames = []
    for joint1_deg in (155.0, 0.0, -151.0):
        joints = [0.0] * len(names)
        joints[names.index("left_joint1")] = math.radians(joint1_deg)
        frames.append({"stage": "between_boxes", "joints": joints})
    metrics = SEQUENCE_MODULE.dual_arm_motion_metrics({
        "joint_names": names,
        "frames": frames,
    })
    assert metrics["max_joint_winding_excess_deg"] > 250.0
    assert metrics["joint_winding_excess_deg"]["left_joint1"] > 250.0


def test_continuous_sequence_connects_exact_release_to_next_precontact(monkeypatch, tmp_path):
    config = SEQUENCE_MODULE.station_contract(
        SCOOP_MODULE.STATION_CONFIG_NAME, SCOOP_MODULE.STATION_SCHEMA
    )
    names = ["updown", "head_joint"] + [
        f"{side}_joint{index}" for side in ("left", "right") for index in range(1, 8)
    ]
    attempts = []
    transitions = []

    def attempt(args, box_id, mode, removed, side, updown, center, **kwargs):
        if "transition_from_joints" in kwargs:
            start = kwargs["transition_from_joints"]
            goal = kwargs["transition_to_joints"]
            transitions.append((side, list(start), list(goal)))
            frames = [
                {"stage": "between_boxes", "joints": list(values), "box_attached": False}
                for values in (start, goal)
            ]
            return {"success": True, "trajectory_result": {
                "frames": frames,
                "transition_group": f"{side}_arm_with_updown",
            }}
        assert args.continuous_sequence is True
        attempts.append((box_id, set(removed)))
        precontact = [updown, 0.0] + [0.01 * box_id] * 14
        release = list(precontact)
        release[0] = -0.3
        for index, value in enumerate(config["unloading_joint_degrees"][side], 1):
            release[names.index(f"{side}_joint{index}")] = math.radians(value)
        frames = [
            {"stage": stage, "joints": values, "box_attached": attached,
             **({"carried_box_tilt_deg": 30.0} if attached else {})}
            for stage, values, attached in (
                ("precontact", precontact, False),
                ("cartesian_approach", precontact, False),
                ("attach_box", precontact, True),
                ("rrt_to_place", release, True),
                ("release_at_place", release, False),
            )
        ]
        return {"success": True, "trajectory_result": {
            "success": True, "task_mode": "full_extract", "return_to_ready": False,
            "joint_names": names, "frames": frames,
            "tool_to_box_center": [0.0, 0.0, 0.15],
            "achieved_place_tcp_pose": [-0.5, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0],
        }}

    monkeypatch.setattr(SEQUENCE_MODULE.scan, "run_attempt", attempt)
    tasks = SEQUENCE_MODULE.plan_sequence(
        35.0, 0, tmp_path, limit_boxes=2, config=config,
        verified_selection=SCOOP_MODULE.VERIFIED_SELECTION,
        place_joint_resolver=SCOOP_MODULE.unloading_joint_degrees,
        planning_overrides={"continuous_sequence": True},
    )
    assert attempts == [(1, set()), (2, {1})]
    assert [item[0] for item in transitions] == ["right", "left", "right", "left"]
    assert transitions[0][1] == SEQUENCE_MODULE.folded_start_joints(config)
    assert transitions[2][1] == tasks[0]["payload"]["frames"][-1]["joints"]
    assert transitions[-1][2] == tasks[1]["payload"]["frames"][0]["joints"]
    assert all(task["payload"]["frames"][-1]["stage"] == "release_at_place" for task in tasks)


def test_station_summary_records_selected_tcp_and_validated_transitions(tmp_path):
    pose = SEQUENCE_MODULE.handoff_tcp_pose(
        SEQUENCE_MODULE.station_contract(), "left", "front", 0.0
    )
    tasks = [
        {
            "box_id": 1,
            "side": "left",
            "mode": "front",
            "updown": 0.0,
            "handoff_tcp_pose": pose,
                "transition_frames": [{"stage": "between_boxes", "joints": [0.0]}],
                "payload": {
                    "joint_names": ["left_joint1"],
                    "achieved_place_tcp_pose": pose,
                    "frames": [{"stage": "release_at_place", "joints": [0.1]}],
            },
        }
    ]
    recording = tmp_path / "sequence.rrd"
    path = SEQUENCE_MODULE.write_sequence_summary(tasks, recording)
    summary = json.loads(path.read_text())
    assert summary["completed_boxes"] == 1
    assert summary["validated_transitions"] == 1
    assert summary["boxes"][0]["handoff_tcp_target"] == pose
    assert summary["boxes"][0]["handoff_tcp_actual"] == pose


def test_scoop_5x5_profile_is_independent_from_legacy_suction():
    config = SEQUENCE_MODULE.station_contract(
        SCOOP_MODULE.STATION_CONFIG_NAME, SCOOP_MODULE.STATION_SCHEMA
    )
    assert config["end_effector"] == "scoop"
    assert config["source"]["branch"] == "alfa_v3_dev"
    assert config["source"]["commit"] == "eb898a95f966fb87fd25717464341b7365c7b133"
    assert config["source"]["model_revision"] == "robot_v3.1.1-hybrid"
    top_down_ids, right_arm_ids, overrides, box_overrides, box_order = (
        SCOOP_MODULE.planning_profile(config)
    )
    assert top_down_ids == set(range(16, 26))
    assert right_arm_ids == {
        box_id for box_id in range(1, 26) if ((box_id - 1) % 5) + 1 >= 3
    }
    assert config["planning"]["left_arm_columns"] == [1, 2]
    assert config["planning"]["right_arm_columns"] == [3, 4, 5]
    assert box_order == [
        1, 2, 5, 4, 3, 6, 7, 10, 9, 8, 11, 12, 15, 14, 13,
        16, 17, 20, 19, 18, 21, 22, 25, 24, 23,
    ]
    assert config["planning"]["low_transfer_tcp_box_ids"] == [6]
    assert overrides["center_front_suction_z_offset"] == -0.05
    assert overrides["front_retreat_distance_m"] == 0.35
    assert overrides["top_retreat_distance_m"] == 0.20
    assert overrides["natural_seed_swivel_sampling"] is True
    assert overrides["natural_seed_swivel_step_deg"] == 1.0
    assert overrides["natural_seed_swivel_neighbor_steps"] == 2
    assert overrides["natural_joint_acceleration_weight"] == 10.0
    assert overrides["natural_place_return_weight"] == 20.0
    assert overrides["natural_cartesian_replay_step_deg"] == 1.0
    assert overrides["natural_rrt_shortcut_enabled"] is True
    assert overrides["natural_rrt_shortcut_max_nodes"] == 64
    assert overrides["cartesian_transfer_search_enabled"] is True
    assert all(
        box_overrides[box_id]["cartesian_transfer_search_enabled"] is False
        for box_id in top_down_ids
    )
    assert overrides["cartesian_transfer_translation_step"] == 0.02
    assert overrides["cartesian_transfer_rotation_step_deg"] == 2.0
    assert overrides["cartesian_transfer_max_search_attempts"] == 128
    assert overrides["shortcut_repair_rrt_enabled"] is True
    assert overrides["shortcut_repair_rrt_budget_ms"] == 1800.0
    assert overrides["shortcut_repair_rrt_max_samples"] == 800
    assert overrides["shortcut_repair_max_joint_offset_deg"] == 35.0
    assert overrides["contact_tool_roll_deg"] == 90.0
    assert overrides["upper_front_success_trials"] == 2
    assert overrides["upper_front_max_success_trials"] == 12
    assert overrides["conveyor_success_trials"] == 2
    assert overrides["conveyor_max_success_trials"] == 12
    assert overrides["ignore_opposite_arm"] is False
    assert overrides["continuous_sequence"] is True
    assert overrides["continuous_seed_previous"] is True
    assert overrides["maximum_carried_box_tilt_deg"] == 95.0
    assert overrides["place_updown_enabled"] is True
    assert overrides["place_updown_m"] == -0.30
    assert overrides["top_loaded_transfer_direct_only"] is False
    assert overrides["natural_max_proximal_step_deg"] == 3.0
    assert overrides["natural_max_wrist_step_deg"] == 3.0
    non_top_transfer_overrides = {
        box_id: {
            name: value for name, value in values.items()
            if name != "cartesian_transfer_search_enabled"
        }
        for box_id, values in box_overrides.items()
        if any(name != "cartesian_transfer_search_enabled" for name in values)
    }
    assert non_top_transfer_overrides == {
        1: {"natural_joint_wrap_weight": 1.0},
        2: {"conveyor_max_success_trials": 24.0},
        3: {
            "front_retreat_distance_m": 0.10,
            "conveyor_max_success_trials": 24.0,
            "continuous_seed_previous": 0.0,
        },
        4: {
            "conveyor_max_success_trials": 24.0,
            "continuous_seed_previous": 0.0,
            "natural_joint_wrap_weight": 1.0,
        },
        6: {"natural_joint_wrap_weight": 1.0},
        7: {"cartesian_transfer_max_search_attempts": "2"},
        8: {
            "front_retreat_distance_m": 0.10,
        },
        20: {
            "conveyor_success_trials": 1.0,
            "loaded_transfer_waypoint_start_deg": (
                "-94.318614684,-67.295795782,-169.712051978,-101.995385230,"
                "128.787173925,101.569630092,-1.758071295"
            ),
            "loaded_transfer_joint_waypoints_deg": config["planning"]["box_overrides"]
            ["20"]["loaded_transfer_joint_waypoints_deg"],
        },
        13: {
            "center_front_suction_z_offset": -0.10,
            "front_retreat_distance_m": 0.10,
        },
        23: {
            "top_suction_x_offset": -0.13,
            "initial_left_arm_joints_deg": (
                "78.75491519,-77.90398167,48.50317505,81.27547476,"
                "-117.95526955,69.92472942,48.32198580"
            ),
        },
    }
    assert {
        box_id: SCOOP_MODULE.VERIFIED_SELECTION[box_id]
        for box_id in range(16, 26)
    } == {
        16: ("left", -0.25),
        17: ("left", -0.25),
        18: ("right", -0.25),
        19: ("right", -0.25),
        20: ("right", -0.25),
        21: ("left", -0.50),
        22: ("left", -0.50),
        23: ("right", -0.62),
        24: ("right", -0.50),
        25: ("right", -0.50),
    }
    assert set(SCOOP_MODULE.VERIFIED_SELECTION) == set(range(1, 26))
    assert SCOOP_MODULE.VERIFIED_SELECTION[1] == ("left", -0.25)
    assert SCOOP_MODULE.VERIFIED_SELECTION[3] == ("right", -0.50)
    assert SCOOP_MODULE.VERIFIED_SELECTION[7] == ("left", 0.0)
    assert SCOOP_MODULE.VERIFIED_SELECTION[1] != SEQUENCE_MODULE.VERIFIED_SELECTION[1]
    assert SCOOP_MODULE.VERIFIED_SELECTION is not SEQUENCE_MODULE.VERIFIED_SELECTION
    assert config["ready_joint_degrees"]["right_top_suction"] == (
        config["ready_joint_degrees"]["right_front"]
    )
    assert config["motion_acceptance"] == {
        "direction_change_threshold_deg": 0.10,
        "max_retreat_joint_step_deg": 3.0,
        "max_retreat_joint_delta_step_deg": 2.0,
        "max_retreat_direction_reversals": 20,
        "max_retreat_excess_travel_deg": 120.0,
        "max_upper_front_total_joint_travel_deg": 3000.0,
        "max_upper_front_stage_excess_travel_deg": 1000.0,
        "max_upper_front_direction_reversals": 20,
        "max_upper_front_joint_range_deg": 360.0,
        "max_upper_front_joint_step_deg": 3.0,
        "max_conveyor_total_joint_travel_deg": 3000.0,
        "max_conveyor_stage_excess_travel_deg": 1000.0,
        "max_conveyor_direction_reversals": 20,
        "max_conveyor_joint_range_deg": 360.0,
        "max_conveyor_joint_step_deg": 3.0,
        "max_conveyor_updown_travel_m": 1.0,
        "max_conveyor_updown_step_m": 0.02,
        "max_conveyor_named_pose_error_deg": 0.01,
        "low_transfer_box_ids": [4, 6],
        "max_low_transfer_frames": 140,
        "max_low_transfer_joint_travel_deg": 850.0,
        "max_low_transfer_excess_travel_deg": 250.0,
        "max_low_transfer_direction_reversals": 20,
        "max_low_transfer_joint_step_deg": 3.0,
        "max_transition_frames": 450,
        "max_transition_total_joint_travel_deg": 3500.0,
        "max_transition_excess_joint_travel_deg": 2200.0,
        "max_transition_direction_reversals": 30,
        "max_transition_joint_range_deg": 360.0,
        "max_frozen_arm_transition_travel_deg": 0.01,
        "max_first_transition_joint_winding_excess_deg": 100.0,
        "max_first_transition_tcp_path_m": 3.6,
        "max_first_transition_tcp_line_deviation_m": 1.10,
        "max_first_transition_tcp_orientation_travel_deg": 550.0,
        "max_top_loaded_transfer_frames": 220,
        "max_top_complete_total_joint_travel_deg": 3000.0,
        "max_top_complete_direction_reversals": 25,
        "max_top_loaded_joint_travel_deg": 1650.0,
        "max_top_loaded_excess_travel_deg": 850.0,
        "max_top_loaded_direction_reversals": 20,
        "max_top_loaded_joint_step_deg": 3.0,
        "max_contact_face_normal_error_deg": 0.1,
        "max_scoop_horizontal_error_deg": 0.1,
    }
    conveyor = SCOOP_MODULE.rear_conveyor_contract(config)
    assert conveyor["modeled"] is False
    assert conveyor["side_semantics"] == "left_is_positive_y"
    assert conveyor["named_pose_file"] == "named_poses_suction.yaml"
    assert conveyor["named_pose"] == "unloading"
    assert config["folded_start_updown_m"] == -0.30
    assert config["folded_start_joint_degrees"] == {
        "left": [155, -105, 20, 90, -90, -40, 0],
        "right": [-155, -105, -20, 90, 90, 40, 0],
    }
    assert SCOOP_MODULE.unloading_joint_degrees(config, "left") == [
        130, -105, -180, 20, -90, -30, 0
    ]
    assert SCOOP_MODULE.unloading_joint_degrees(config, "right") == [
        -130, -105, 180, 20, 90, 30, 0
    ]
    for side in ("left", "right"):
        tcp_pose = SCOOP_MODULE.unloading_tcp_pose(
            config, 4, side, "front", -0.30, (0.0, 0.0, 0.0)
        )
        assert len(tcp_pose) == 7
        assert math.isclose(
            math.sqrt(sum(value * value for value in tcp_pose[3:])),
            1.0,
            abs_tol=1e-6,
        )
    named_poses_path = (
        WORKSPACE_SRC / "alfa_robot_description" / "config"
        / conveyor["named_pose_file"]
    )
    with named_poses_path.open(encoding="utf-8") as stream:
        named_poses = yaml.safe_load(stream)["named_poses"]
    for side in ("left", "right"):
        for pose_name, station_degrees in (
            ("home", config["folded_start_joint_degrees"][side]),
            ("unloading", config["unloading_joint_degrees"][side]),
        ):
            for index, expected_degrees in enumerate(station_degrees, 1):
                assert math.isclose(
                    named_poses[pose_name][f"{side}_joint{index}"],
                    math.radians(expected_degrees),
                        abs_tol=1e-8,
                )
    assert math.isclose(named_poses["home"]["updown"], -0.3)
    assert math.isclose(named_poses["unloading"]["updown"], -0.3)
    folded = SEQUENCE_MODULE.folded_start_joints(config)
    assert math.isclose(folded[0], -0.30)
    assert math.isclose(math.degrees(folded[2]), 155.0)
    assert math.isclose(math.degrees(folded[9]), -155.0)
    assert SCOOP_MODULE.default_recording_path() != SEQUENCE_MODULE.default_recording_path()
    args = SEQUENCE_MODULE.planning_args(35.0, config, end_effector="scoop")
    assert args.end_effector == "scoop"


def test_scoop_plan_cache_round_trip(tmp_path):
    config = SEQUENCE_MODULE.station_contract(
        SCOOP_MODULE.STATION_CONFIG_NAME, SCOOP_MODULE.STATION_SCHEMA
    )
    cache = tmp_path / "sequence-plan-cache.json"
    tasks = [{"box_id": 1, "payload": {"frames": [{"joints": [0.0]}]}}]
    SCOOP_MODULE.write_plan_cache(cache, config, tasks)
    assert SCOOP_MODULE.load_plan_cache(cache, config, 25) == tasks
    assert SCOOP_MODULE.load_plan_cache(tmp_path / "missing.json", config, 25) == []


def test_scoop_trajectory_report_has_tcp_path_and_joint_reversals(tmp_path):
    import numpy as np

    class Robot:
        def fk(self, joints):
            transform = np.eye(4)
            transform[0, 3] = joints["left_joint1"]
            return {"left_tool0": transform}

    names = ["updown", "head_joint"] + [
        f"{side}_joint{index}"
        for side in ("left", "right") for index in range(1, 8)
    ]
    frames = []
    for degrees in (0.0, 2.0, 0.0):
        joints = [0.0] * len(names)
        joints[names.index("left_joint1")] = math.radians(degrees)
        frames.append({"stage": "cartesian_retreat", "joints": joints})
    task = {
        "side": "left",
        "payload": {"joint_names": names, "tool_link": "left_tool0", "frames": frames},
        "planning_search": {
            "task": {"process_wall_ms": 2100.0},
            "transition": {"process_wall_ms": 300.0},
            "selected_task": {"process_wall_ms": 900.0},
        },
    }
    analysis = SCOOP_MODULE.analyze_task_trajectory(task, Robot())
    assert math.isclose(analysis["tcp_path_length_m"], 2 * math.radians(2))
    assert analysis["joint_reversals_by_axis"]["left_joint1"] == 1
    assert analysis["joint_flip_events"] == 0
    assert analysis["search_wall_ms"] == 2400.0
    assert len(analysis["tcp_stage_positions"]["cartesian_retreat"]) == 3
    assert analysis["max_shortcut_tcp_position_deviation_m"] > 0.0

    report = {"boxes": [{
        "box_id": 1, "side": "left", "mode": "front",
        "trajectory_analysis": analysis, "planning_search": task["planning_search"],
        "motion_selection": {"total_joint_travel_deg": 4.0},
        "carried_box_orientation": {"max_tilt_deg": 80.0},
        "task_frames": 3, "validated_transition_frames": 1,
    }]}
    csv_path = SCOOP_MODULE.write_metrics_csv(report, tmp_path / "sample-summary.json")
    assert "2.4" in csv_path.read_text()


def test_scoop_summary_has_a_distinct_contract(tmp_path):
    config = SEQUENCE_MODULE.station_contract(
        SCOOP_MODULE.STATION_CONFIG_NAME, SCOOP_MODULE.STATION_SCHEMA
    )
    pose = [
        -0.5637086821, -0.3420342055, 0.9594514873,
        0.5195131498, -0.4857615399, -0.4879138958, 0.5060452982,
    ]
    names = ["updown", "head_joint"] + [
        f"{side}_joint{index}"
        for side in ("left", "right")
        for index in range(1, 8)
    ]
    unloading = SCOOP_MODULE.unloading_joint_degrees(config, "right")
    release_joints = SEQUENCE_MODULE.folded_start_joints(config)
    for index, value in enumerate(unloading, start=1):
        release_joints[names.index(f"right_joint{index}")] = math.radians(value)
    payload = {
        "joint_names": names,
        "achieved_place_tcp_pose": pose,
        "tool_to_box_center": [0.0, 0.0, 0.15],
        "frames": [{
            "stage": "release_at_place",
            "joints": release_joints,
            "box_attached": False,
        }],
    }
    placement_center = SEQUENCE_MODULE.placed_box_center(payload)
    tasks = [{
        "box_id": 1,
        "side": "right",
        "mode": "front",
        "updown": 0.0,
        "handoff_tcp_pose": pose,
        "place_arm_joints_deg": unloading,
        "placement_box_center_target": placement_center,
        "transition_frames": [{"stage": "between_boxes", "joints": release_joints}],
        "payload": payload,
    }]
    path = SEQUENCE_MODULE.write_sequence_summary(
        tasks,
        tmp_path / "scoop.rrd",
        station_config_name=SCOOP_MODULE.STATION_CONFIG_NAME,
        result_schema=SCOOP_MODULE.RESULT_SCHEMA,
        task_name="v3_scoop_5x5",
        end_effector=SCOOP_MODULE.END_EFFECTOR,
    )
    summary = json.loads(path.read_text())
    assert summary["schema"] == SCOOP_MODULE.RESULT_SCHEMA
    assert summary["task_name"] == "v3_scoop_5x5"
    assert summary["end_effector"] == "scoop"
    assert summary["station_config"] == SCOOP_MODULE.STATION_CONFIG_NAME
    assert summary["boxes"][0]["placement_box_center_target"] == placement_center
    assert summary["boxes"][0]["place_arm_joints_deg"] == unloading
    assert summary["boxes"][0]["place_arm_joint_error_deg"] < 1e-9
    assert all(
        math.isclose(actual, expected, abs_tol=1e-9)
        for actual, expected in zip(
            summary["boxes"][0]["placement_box_center_actual"], placement_center
        )
    )


def _scoop_motion_task(joint1_degrees):
    names = ["updown", "head_joint"] + [
        f"{side}_joint{index}"
        for side in ("left", "right")
        for index in range(1, 8)
    ]
    frames = []
    for frame_index, degrees in enumerate(joint1_degrees):
        joints = [0.0] * len(names)
        joints[names.index("left_joint1")] = math.radians(degrees)
        frames.append({
            "stage": "attach_box" if frame_index == 0 else "cartesian_retreat",
            "joints": joints,
            "box_attached": True,
        })
    return {
        "box_id": 1,
        "side": "left",
        "payload": {
            "joint_names": names,
            "frames": frames,
            "contact_tool_metrics": {
                "face_normal_error_deg": 0.0,
                "scoop_horizontal_error_deg": 0.0,
                "joint7_deg": 90.0,
            },
            "metrics": {
                "cartesian_transfer_attempts": 1,
                "cartesian_guided_segments": 1,
                "joint_fallback_segments": 0,
                "cartesian_transfer_budget_exhaustions": 0,
            },
        },
    }


def test_scoop_retreat_motion_acceptance_rejects_oscillation():
    acceptance = dict(SEQUENCE_MODULE.station_contract(
        SCOOP_MODULE.STATION_CONFIG_NAME, SCOOP_MODULE.STATION_SCHEMA
    )["motion_acceptance"])
    acceptance.update({
        "max_retreat_direction_reversals": 1,
        "max_retreat_excess_travel_deg": 4.0,
        "max_top_loaded_direction_reversals": 2,
    })
    smooth = _scoop_motion_task([0.0, 1.0, 2.0, 3.0])
    aggregate = SCOOP_MODULE.validate_retreat_motion([smooth], acceptance)
    assert math.isclose(aggregate["max_joint_step_deg"], 1.0, abs_tol=1e-9)
    assert aggregate["max_direction_reversals"] == 0

    oscillatory = _scoop_motion_task([0.0, 2.0, 0.0, 2.0])
    try:
        SCOOP_MODULE.validate_retreat_motion([oscillatory], acceptance)
    except ValueError as error:
        assert "reversals=2" in str(error)
    else:
        raise AssertionError("oscillatory retreat must fail the scoop motion contract")

    loaded_oscillation = _scoop_motion_task([0.0, 1.0, 2.0, 3.0])
    loaded_oscillation["mode"] = "top_suction"
    loaded_oscillation["motion_selection"] = {
        "total_joint_travel_deg": 3.0,
        "direction_reversals": 0,
    }
    names = loaded_oscillation["payload"]["joint_names"]
    for degrees in (3.0, 5.0, 3.0, 5.0):
        joints = [0.0] * len(names)
        joints[names.index("left_joint1")] = math.radians(degrees)
        loaded_oscillation["payload"]["frames"].append({
            "stage": "rrt_to_place",
            "joints": joints,
            "box_attached": True,
        })
    loaded_acceptance = dict(acceptance)
    loaded_acceptance["max_top_loaded_direction_reversals"] = 1
    try:
        SCOOP_MODULE.validate_retreat_motion([loaded_oscillation], loaded_acceptance)
    except ValueError as error:
        assert "loaded-reversals=2" in str(error)
    else:
        raise AssertionError("oscillatory loaded transfer must fail the scoop motion contract")
