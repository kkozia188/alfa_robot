import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def render():
    result = subprocess.run(
        [
            "xacro",
            str(PACKAGE_ROOT / "urdf/alfa_robot.urdf.xacro"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return ET.fromstring(result.stdout)


def test_demo_consumes_pinned_v322_suction_profile():
    lock = json.loads((PACKAGE_ROOT / "config/upstream_description.lock.json").read_text())
    assert lock["model_revision"] == "robot_v3.2.2-suction"
    assert lock["profile"] == "suction"
    assert lock["upstream_branch"] == "robot_v3_suction_chassis"
    assert lock["upstream_commit"] == "542da7f3398c29d0a531668038bc91f341d5c857"
    assert lock["upstream_ref_used_for_sync"] == "origin/robot_v3_suction_chassis"
    for profile in ("", "_suction"):
        assert f"config/named_poses{profile}.yaml" in lock["managed_destination_files"]

    for relative_path, expected_hash in lock["managed_destination_files"].items():
        path = PACKAGE_ROOT / relative_path
        assert path.is_file(), relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash

    entrypoint = (PACKAGE_ROOT / "urdf/alfa_robot.urdf.xacro").read_text()
    assert "robot_v3_2_2.xacro" in entrypoint
    assert "robot_v3_0_9.xacro" not in entrypoint


def test_suction_profile_matches_control_inventory():
    for profile, expected_count in (("suction", 17),):
        robot = render()
        joints = {joint.get("name"): joint for joint in robot.findall("joint")}
        movable = {name for name, joint in joints.items() if joint.get("type") != "fixed"}
        control = robot.find("ros2_control")
        control_joints = {joint.get("name") for joint in control.findall("joint")}
        expected_initial = yaml.safe_load(
            (PACKAGE_ROOT / f"config/initial_positions_{profile}.yaml").read_text()
        )["initial_positions"]
        assert len(movable) == expected_count
        assert movable == control_joints
        assert "left_moving_jaw_joint" not in movable
        for joint in control.findall("joint"):
            position_state = next(
                state for state in joint.findall("state_interface")
                if state.get("name") == "position"
            )
            initial_value = position_state.find("param[@name='initial_value']")
            assert float(initial_value.text) == expected_initial[joint.get("name")]

        updown = joints["updown"].find("limit")
        assert float(updown.get("lower")) == -1.0
        assert float(updown.get("upper")) == 0.0
        assert joints["left_tool0_fixed"].find("parent").get("link") == "left_suction"
        assert joints["right_tool0_fixed"].find("parent").get("link") == "right_suction"
        assert joints["base_footprint_to_base_link"].find("origin").get("xyz") == "0 0 0.335"
        child_links = {
            joint.find("child").get("link") for joint in robot.findall("joint")
        }
        assert {link.get("name") for link in robot.findall("link")} - child_links == {
            "base_footprint"
        }
        assert "world_to_base" not in joints
