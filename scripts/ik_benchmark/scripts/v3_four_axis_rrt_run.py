#!/usr/bin/env python3
"""Plan an offline four-axis V3 transition with analytic wrist compensation."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-s", type=float, default=6.0)
    parser.add_argument("--fcl-budget", type=int, default=6000)
    parser.add_argument("--optimize-input", type=Path)
    parser.add_argument("--short-wrist-goal", action="store_true")
    parser.add_argument("--keep-wrists-fixed", action="store_true")
    parser.add_argument("--include-updown", action="store_true")
    parser.add_argument("--synchronized-updown-input", type=Path)
    parser.add_argument("--synchronized-updown-goal", type=float, default=-0.6)
    parser.add_argument("--srdf", type=Path)
    parser.add_argument("--named-shortcut-from")
    parser.add_argument("--monotone-home", action="store_true")
    parser.add_argument("--smooth-wrist-input", type=Path)
    parser.add_argument("--farthest-shortcut-input", type=Path)
    parser.add_argument("--named-shortcut-to")
    parser.add_argument("--named-shortcut-attached", action="store_true")
    parser.add_argument("--collision-benchmark-samples", type=int, default=0)
    parser.add_argument("--scan-chassis-casters", action="store_true")
    parser.add_argument("--loaded-orientation-context", type=Path)
    parser.add_argument("--manual-four-axis-context", type=Path)
    parser.add_argument("--wrist-preview-recorded", type=Path)
    parser.add_argument("--dual-wrist-preview-segment", type=Path)
    parser.add_argument("--dual-face-ik-request", type=Path)
    parser.add_argument("--dual-pregrasp-candidates", type=Path)
    parser.add_argument("--dual-cartesian-candidates", type=Path)
    parser.add_argument("--dual-cartesian-pregrasp-plan", type=Path)
    parser.add_argument("--dual-cartesian-approach", type=Path)
    parser.add_argument("--placement-reference", type=Path)
    parser.add_argument("--unloaded-retract-cm", type=int, default=5)
    parser.add_argument("--side-up-back-wrist-rrt", action="store_true")
    parser.add_argument("--side-up-back-fixed-orientation", action="store_true")
    parser.add_argument("--stop-after-extraction", action="store_true")
    parser.add_argument("--loaded-joint-return", action="store_true")
    parser.add_argument("--box-id", type=int)
    parser.add_argument("--shortcut-loaded-input", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.scene.read_text())
    if data.get("moving_joints") != "both_arms_j1_to_j4":
        raise ValueError("expected first-four transition scene")
    snapshot = {
        "kind": data["kind"],
        "moving_joints": data["moving_joints"],
        "joint_names": data["joint_names"],
        "home_joints": data["frames"][0]["joints"],
        "goal_joints": data["frames"][-1]["joints"],
        "environment": data["environment"],
        "wall_boxes": data["wall_boxes"],
    }
    parameters = {"v3_wrist_orientation_probe": {"ros__parameters": {
        "robot_description": (args.scene.parent / "robot.urdf").read_text(),
        "robot_description_semantic": (args.srdf or args.scene.parent / "robot.srdf").read_text(),
        "scene_json": json.dumps(snapshot, separators=(",", ":")),
    }}}
    settings = parameters["v3_wrist_orientation_probe"]["ros__parameters"]
    if args.dual_cartesian_approach:
        if not args.dual_cartesian_candidates or not args.dual_cartesian_pregrasp_plan:
            parser.error("retreat requires --dual-cartesian-candidates and --dual-cartesian-pregrasp-plan")
        settings["dual_cartesian_candidates_json"] = args.dual_cartesian_candidates.read_text()
        settings["dual_cartesian_pregrasp_plan_json"] = args.dual_cartesian_pregrasp_plan.read_text()
        settings["dual_cartesian_approach_json"] = args.dual_cartesian_approach.read_text()
        settings["dual_attached_retreat_output_path"] = str(args.output.resolve())
        if args.placement_reference:
            settings["dual_placement_reference_path"] = str(args.placement_reference.resolve())
        settings["dual_unloaded_retract_steps"] = args.unloaded_retract_cm
        settings["dual_side_up_back_wrist_rrt"] = args.side_up_back_wrist_rrt
        settings["dual_side_up_back_fixed_orientation"] = args.side_up_back_fixed_orientation
        settings["dual_stop_after_extraction"] = args.stop_after_extraction
        settings["dual_loaded_joint_return"] = args.loaded_joint_return
    elif args.dual_cartesian_candidates:
        if not args.dual_cartesian_pregrasp_plan:
            parser.error("--dual-cartesian-pregrasp-plan is required")
        settings["dual_cartesian_candidates_json"] = args.dual_cartesian_candidates.read_text()
        settings["dual_cartesian_pregrasp_plan_json"] = args.dual_cartesian_pregrasp_plan.read_text()
        settings["dual_cartesian_approach_output_path"] = str(args.output.resolve())
    elif args.dual_pregrasp_candidates:
        settings["dual_pregrasp_candidates_json"] = args.dual_pregrasp_candidates.read_text()
        settings["dual_pregrasp_budget_s"] = args.time_s
        settings["dual_pregrasp_plan_output_path"] = str(args.output.resolve())
    elif args.dual_face_ik_request:
        settings["dual_face_ik_request_json"] = args.dual_face_ik_request.read_text()
        settings["dual_face_ik_output_path"] = str(args.output.resolve())
    elif args.dual_wrist_preview_segment:
        settings["dual_wrist_preview_segment_json"] = args.dual_wrist_preview_segment.read_text()
        settings["dual_wrist_preview_output_path"] = str(args.output.resolve())
    elif args.wrist_preview_recorded:
        if args.box_id is None:
            parser.error("--box-id is required for a wrist preview")
        settings["wrist_preview_recorded_json"] = args.wrist_preview_recorded.read_text()
        settings["wrist_preview_box_id"] = args.box_id
        settings["wrist_preview_output_path"] = str(args.output.resolve())
    elif args.shortcut_loaded_input:
        if not args.manual_four_axis_context:
            parser.error("--manual-four-axis-context is required for loaded shortcut")
        settings["manual_loaded_four_axis_context_json"] = args.manual_four_axis_context.read_text()
        settings["manual_loaded_shortcut_input_path"] = str(args.shortcut_loaded_input.resolve())
        settings["manual_loaded_shortcut_output_path"] = str(args.output.resolve())
    elif args.manual_four_axis_context:
        settings["manual_loaded_four_axis_context_json"] = args.manual_four_axis_context.read_text()
        settings["manual_loaded_four_axis_output_path"] = str(args.output.resolve())
        settings["manual_loaded_four_axis_budget_s"] = args.time_s
    elif args.loaded_orientation_context:
        settings["loaded_orientation_context_json"] = args.loaded_orientation_context.read_text()
        settings["loaded_orientation_output_path"] = str(args.output.resolve())
        settings["loaded_orientation_budget_s"] = args.time_s
    elif args.scan_chassis_casters:
        settings["caster_scan_output_path"] = str(args.output.resolve())
    elif args.collision_benchmark_samples:
        settings["collision_benchmark_samples"] = args.collision_benchmark_samples
        settings["collision_benchmark_output_path"] = str(args.output.resolve())
    elif args.farthest_shortcut_input:
        settings["farthest_shortcut_input_path"] = str(args.farthest_shortcut_input.resolve())
        settings["farthest_shortcut_output_path"] = str(args.output.resolve())
    elif args.smooth_wrist_input:
        settings["smooth_wrist_input_path"] = str(args.smooth_wrist_input.resolve())
        settings["smooth_wrist_output_path"] = str(args.output.resolve())
    elif args.monotone_home:
        settings["monotone_home_output_path"] = str(args.output.resolve())
    elif args.named_shortcut_from:
        if not args.named_shortcut_to:
            parser.error("--named-shortcut-to is required")
        settings["named_shortcut_from"] = args.named_shortcut_from
        settings["named_shortcut_to"] = args.named_shortcut_to
        settings["named_shortcut_output_path"] = str(args.output.resolve())
        settings["named_shortcut_attached"] = args.named_shortcut_attached
    elif args.synchronized_updown_input:
        settings["synchronized_updown_input_path"] = str(args.synchronized_updown_input.resolve())
        settings["synchronized_updown_output_path"] = str(args.output.resolve())
        settings["synchronized_updown_goal"] = args.synchronized_updown_goal
    elif args.optimize_input:
        settings["optimize_input_path"] = str(args.optimize_input.resolve())
        settings["optimize_output_path"] = str(args.output.resolve())
        settings["optimize_short_wrist_goal"] = args.short_wrist_goal
    else:
        settings["plan_output_path"] = str(args.output.resolve())
        settings["plan_time_s"] = args.time_s
        settings["plan_fcl_budget"] = args.fcl_budget
        settings["plan_keep_wrists_fixed"] = args.keep_wrists_fixed
        settings["plan_include_updown"] = args.include_updown
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as temporary:
        yaml.safe_dump(parameters, temporary, allow_unicode=True)
        temporary.flush()
        process = subprocess.run(
            [str(args.backend.resolve()), "--ros-args", "--params-file", temporary.name],
            capture_output=True, text=True,
            timeout=300 if args.optimize_input or args.synchronized_updown_input
                    or args.named_shortcut_from or args.monotone_home or args.smooth_wrist_input or
                    args.farthest_shortcut_input or
                    args.scan_chassis_casters or
                    args.loaded_orientation_context or args.manual_four_axis_context or
                    args.shortcut_loaded_input or args.wrist_preview_recorded or
                    args.dual_wrist_preview_segment or args.dual_face_ik_request or
                    args.dual_pregrasp_candidates or args.dual_cartesian_candidates or
                    args.dual_cartesian_approach else args.time_s + 30,
        )
    if process.returncode:
        raise RuntimeError(process.stdout + process.stderr)
    result = json.loads(args.output.read_text())
    print(json.dumps({key: value for key, value in result.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
