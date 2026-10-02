#!/usr/bin/env python3

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
WORKSPACE_SRC = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(WORKSPACE_SRC / "alfa_robot_rerun"))

SCRIPT = SCRIPT_DIR / "v3_scoop_5x5_conveyor_demo.py"
SPEC = importlib.util.spec_from_file_location("v3_scoop_5x5_conveyor_demo", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_clearance_input_rounds_half_up_to_certified_centimeter():
    assert MODULE.round_to_certified_centimeter(0.804) == 0.80
    assert MODULE.round_to_certified_centimeter(0.805) == 0.81
    assert MODULE.round_to_certified_centimeter(0.895) == 0.90
    assert MODULE.round_to_certified_centimeter(0.554) == 0.55
    assert MODULE.round_to_certified_centimeter(0.555) == 0.56


def test_interactive_clearance_prompt_rejects_invalid_values():
    responses = iter(["not-a-number", "0.79", "0.86"])
    messages = []
    selected = MODULE.prompt_front_clearance(
        None,
        "前三排",
        MODULE.UPPER_RANGE_M,
        MODULE.DEFAULT_UPPER_M,
        input_fn=lambda _prompt: next(responses),
        output_fn=messages.append,
    )
    assert selected == 0.86
    assert len(messages) == 2
    assert MODULE.prompt_front_clearance(
        None,
        "后两排",
        MODULE.LOWER_RANGE_M,
        MODULE.DEFAULT_LOWER_M,
        input_fn=lambda _prompt: "",
    ) == 0.60


def certificate_root(workspace: Path) -> Path:
    return (
        workspace
        / "data/ik_benchmark/v3_scoop_5x5/range_certification_v322_tool0151"
    )


def write_certificate(workspace: Path, model_revision: str = "robot_v3.2.2-suction"):
    root = certificate_root(workspace)
    pair = root / "pairs/front-u085-l060"
    pair.mkdir(parents=True)
    replay = pair / "replay.json"
    validation = pair / "validation.json"
    replay.write_text(json.dumps({"schema": "replay"}), encoding="utf-8")
    validation.write_text(json.dumps({"success": True}), encoding="utf-8")
    entry = {
        "success": True,
        "upper_front_clearance_m": 0.85,
        "lower_front_clearance_m": 0.60,
        "replay": str(replay),
        "replay_sha256": MODULE.sha256(replay),
        "validation": str(validation),
        "validation_sha256": MODULE.sha256(validation),
    }
    manifest = {
        "schema": "alfa.v322_tool0151_conveyor_clearance_grid.v1",
        "success": True,
        "certified_pair_count": 121,
        "model_revision": model_revision,
        "tool0_offset_local_z_m": 0.151,
        "upstream_base_commit": "d9c330cef72981390d81ac2b1cd5a6eb9e892195",
        "input_resolution_m": 0.01,
        "pairs": [entry],
    }
    (root / "clearance-grid-certificate.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return entry


def test_v322_certificate_entry_verifies_model_and_hashes(tmp_path):
    expected = write_certificate(tmp_path)
    actual = MODULE.certified_pair_entry(tmp_path, 0.85, 0.60)
    assert actual == expected

    replay = Path(expected["replay"])
    replay.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        MODULE.certified_pair_entry(tmp_path, 0.85, 0.60)


def test_legacy_or_wrong_model_certificate_is_rejected(tmp_path):
    write_certificate(tmp_path, model_revision="robot_v3.1.1-hybrid")
    with pytest.raises(ValueError, match="invalid V3.2.2"):
        MODULE.certified_pair_entry(tmp_path, 0.85, 0.60)


def test_planar_base_pose_is_supported_by_validator_and_recorder():
    package = SCRIPT_DIR.parent
    validator = (package / "src" / "v3_dual_arm_5x5_replay_validator.cpp").read_text()
    recorder = (SCRIPT_DIR / "v3_scoop_5x5_dual_rerun.py").read_text()
    assert '"base_pose_map"' in validator
    assert 'getJointModel("map_to_base_footprint")' in validator
    assert "frame_base_transform" in recorder
    assert "right conveyor" in recorder
