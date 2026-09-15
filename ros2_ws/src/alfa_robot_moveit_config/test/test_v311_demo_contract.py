#!/usr/bin/env python3

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


MOVEIT_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION_ROOT = MOVEIT_ROOT.parent / "alfa_robot_description"


def controller_joints(path):
    payload = yaml.safe_load(path.read_text())
    if "controller_manager" in payload:
        return payload["dual_arm_controller"]["ros__parameters"]["joints"]
    return payload["moveit_simple_controller_manager"]["dual_arm_controller"]["joints"]


def main():
    lock = json.loads((DESCRIPTION_ROOT / "config/upstream_description.lock.json").read_text())
    expected = set(lock["movable_joint_names"])
    assert lock["model_revision"] == "robot_v3.1.1"
    assert lock["upstream_commit"] == "b0c53aa8ca0bba1701d2f6680115012ed38079c0"

    initial = yaml.safe_load((MOVEIT_ROOT / "config/initial_positions.yaml").read_text())["initial_positions"]
    limits = yaml.safe_load((MOVEIT_ROOT / "config/joint_limits.yaml").read_text())["joint_limits"]
    assert set(initial) == expected
    assert set(limits) == expected
    assert initial["rear_suspension_joint"] == 0.085
    assert initial["updown"] == -0.5
    assert limits["updown"]["min_position"] == -1.0
    assert limits["updown"]["max_position"] == 0.0
    assert limits["left_hand_joint"]["max_position"] == 0.1
    assert limits["right_hand_joint"]["max_position"] == 0.1

    ros2_control = controller_joints(MOVEIT_ROOT / "config/ros2_controllers.yaml")
    moveit = controller_joints(MOVEIT_ROOT / "config/moveit_controllers.yaml")
    assert len(ros2_control) == len(set(ros2_control)) == 27
    assert len(moveit) == len(set(moveit)) == 27
    assert set(ros2_control) == set(moveit) == expected

    srdf = ET.parse(MOVEIT_ROOT / "config/alfa_robot.srdf").getroot()
    groups = {group.get("name"): group for group in srdf.findall("group")}
    assert set(groups) == {"left_arm", "right_arm", "dual_arm", "dual_arm_with_updown", "whole"}
    assert groups["left_arm"].find("chain").get("tip_link") == "left_joint7"
    assert groups["right_arm"].find("chain").get("tip_link") == "right_joint7"
    srdf_text = (MOVEIT_ROOT / "config/alfa_robot.srdf").read_text()
    assert "tool0" not in srdf_text
    assert "arm_base" not in srdf_text


if __name__ == "__main__":
    main()
