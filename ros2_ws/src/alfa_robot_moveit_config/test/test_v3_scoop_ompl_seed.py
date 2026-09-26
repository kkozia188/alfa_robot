#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parents[1]
SEEDS = json.loads((ROOT / "tools/v3_scoop_golden_20260921/OMPL_SEEDS.json").read_text())
SOURCE = (PACKAGE / "src/v3_single_arm_box_extract_demo.cpp").read_text()
LAUNCH = (PACKAGE / "launch/v3_single_arm_box_extract_demo.launch.py").read_text()
SCAN = (PACKAGE / "scripts/scan_v3_single_arm_box_wall.py").read_text()
SCOOP = (PACKAGE / "scripts/v3_scoop_5x5_grasp_sequence_rerun.py").read_text()
assert len(SEEDS["seeds"]) == 8 and len(set(SEEDS["seeds"])) == 8
assert SEEDS["acceptance"]["all_seeds_must_pass"] is True
assert SOURCE.index("ompl::RNG::setSeed") < SOURCE.index("RobotModelLoader")
assert "deterministicOmplSeed(" in SOURCE
assert 'std::getenv("V3_OMPL_SEED")' in SOURCE
assert 'environment["V3_OMPL_SEED"]' in SCAN
assert SOURCE.count("seedOmplRequest(") == 3
assert "loaded_transfer_joint_waypoints_alt_deg" in SOURCE
assert "validated_transition_waypoints" in SOURCE
assert "validated_waypoint_profiles_path" in SOURCE
assert "task_waypoint_profiles_" in SOURCE
assert "validated_task_waypoints" in SOURCE
assert "result.achieved_place_tcp = previous.getGlobalLinkTransform(tool_link_)" in SOURCE
assert "fullCollisionReason(scene, probe" in SOURCE
PROFILES = json.loads((PACKAGE / "config/v3_scoop_validated_waypoint_profiles.json").read_text())
assert len(PROFILES["loaded_profiles"]) == len(PROFILES["task_profiles"]) == 25
assert len(PROFILES["transition_profiles"]) >= 25
assert all("box_id" not in profile for group in ("loaded_profiles", "transition_profiles", "task_profiles") for profile in PROFILES[group])
assert SOURCE.count("request.num_planning_attempts = rrt_planning_attempts_") == 2
assert 'getParameter<int>("planning_seed", 0)' in SOURCE
assert 'output["planning_seed"] = planning_seed_' in SOURCE
assert 'DeclareLaunchArgument("planning_seed", default_value="0")' in LAUNCH
assert "planning_seed:={getattr(args, 'planning_seed', 0)}" in SCAN
assert 'parser.add_argument("--ompl-seed", type=int, default=0)' in SCOOP
assert 'overrides["planning_seed"] = args.ompl_seed' in SCOOP
SEQUENCE = (PACKAGE / "scripts/v3_5x5_grasp_sequence_rerun.py").read_text()
assert SEQUENCE.count('0x9E3779B9') == 2
print("v3 scoop fixed OMPL seed contract: PASS")
