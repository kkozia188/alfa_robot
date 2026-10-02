#!/usr/bin/env python3
"""Keep every V3 stage-planning solver selection on the V3.2.2 model."""

from pathlib import Path


SOURCE = Path(__file__).parents[1] / "src" / "v3_single_arm_box_extract_demo.cpp"


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    select_arm = text.split("void selectArm(const std::string& side)", 1)[1].split(
        "void createBoxMarker()", 1
    )[0]
    shoulder_setup = text.split("const Eigen::Vector3d shoulder_midpoint", 1)[1].split(
        "initial_shoulder_z_", 1
    )[0]
    for block in (select_arm, shoulder_setup):
        assert "V322Left" in block
        assert "V322Right" in block
        assert "V311Left" not in block
        assert "V311Right" not in block
    chassis_bounds = text.split("double modelChassisFrontX", 1)[1].split(
        "Eigen::Vector3d wallBoxCenter", 1
    )[0]
    assert 'hasLinkModel("model_base")' in chassis_bounds
    assert 'hasLinkModel("base_link")' in chassis_bounds
    assert 'chassis_links = {"base_link"}' in chassis_bounds
    assert 'getParameter<bool>("retain_placed_boxes", true)' in text
    assert "retain_placed_boxes_ && trajectory_cache_.is_null()" in text
    assert "enable_stage_action_ && retain_placed_boxes_" in text
    assert 'post_extract_policy_ != "external_handoff"' in text
    assert 'post_extract_policy_ == "external_handoff"' in text
    assert 'getParameter<bool>("dual_entry_repair_enabled", false)' in text
    assert 'getJointModelGroup("dual_arm_with_updown")' in text


if __name__ == "__main__":
    main()
