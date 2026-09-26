#!/usr/bin/env python3
from pathlib import Path
SOURCE = (Path(__file__).resolve().parents[1] / "src/v3_single_arm_box_extract_demo.cpp").read_text()
SEQUENCE = (Path(__file__).resolve().parents[1] / "scripts/v3_5x5_grasp_sequence_rerun.py").read_text()
for value in (
    'all_joint_names_.reserve(17)', 'all_joint_names_.push_back("head_pitch_joint")',
    'payload["solve_time_ms"] = result.total_ms', 'payload["path_duration"]',
    'selected_replay_velocity_bound_duration_excludes_dwell',
    'selected_replay_waypoints_and_interpolated_edges_full_robot_phase_aware_attached_box',
    'request.distance = true', 'state_context_joint_count',
): assert value in SOURCE
for value in ('"solve_contract"', 'failed_attempts', 'all_timed_task_planner_attempts_including_failures',
              '"minimum_clearance_scope"', '"state_context_joint_count"'):
    assert value in SEQUENCE
assert SOURCE.index('if (frame.box_attached) attachCarriedBox(state)') < SOURCE.index('sample_clearance(state)')
assert 'sample_clearance(probe)' in SOURCE
assert 'if motion_accepted(motion):' in SEQUENCE
assert 'required_success_target = 1' in SEQUENCE
assert 'if not eligible_first:' in SEQUENCE
assert 'validated_transition_waypoints' in SEQUENCE
print("v3 scoop current safety/timing/telemetry contracts: PASS")
