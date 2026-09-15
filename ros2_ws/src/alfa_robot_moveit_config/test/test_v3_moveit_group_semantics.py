#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from pathlib import Path
import json

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GROUPS = {
    "left_arm",
    "right_arm",
    "dual_arm",
    "dual_arm_with_updown",
    "whole",
}


def main() -> None:
    srdf = ET.parse(PACKAGE_ROOT / "config" / "alfa_robot.srdf").getroot()
    groups = {group.attrib["name"]: group for group in srdf.findall("group")}
    assert set(groups) == EXPECTED_GROUPS
    assert groups["left_arm"].find("chain").attrib["tip_link"] == "left_joint7"
    assert groups["right_arm"].find("chain").attrib["tip_link"] == "right_joint7"
    whole_joints = {joint.attrib["name"] for joint in groups["whole"].findall("joint")}
    assert whole_joints == {
        "rear_suspension_joint",
        "front_left_steering_joint",
        "front_left_wheel_joint",
        "front_right_steering_joint",
        "front_right_wheel_joint",
        "rear_left_steering_joint",
        "rear_left_wheel_joint",
        "rear_right_steering_joint",
        "rear_right_wheel_joint",
        "updown",
        "head_joint",
        "left_hand_joint",
        "right_hand_joint",
    }
    assert {group.attrib["name"] for group in groups["whole"].findall("group")} == {"dual_arm"}

    lock = json.loads(
        (PACKAGE_ROOT.parent / "alfa_robot_description/config/upstream_description.lock.json").read_text()
    )
    assert lock["model_revision"] == "robot_v3.1.1"
    assert len(lock["movable_joint_names"]) == 27

    ompl = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "ompl_planning.yaml").read_text(encoding="utf-8")
    )
    configured_groups = {
        name for name in EXPECTED_GROUPS if name in ompl
    }
    assert configured_groups == EXPECTED_GROUPS
    for removed in {
        "head",
        "left_arm_with_updown",
        "right_arm_with_updown",
        "whole_body",
    }:
        assert removed not in ompl

if __name__ == "__main__":
    main()
