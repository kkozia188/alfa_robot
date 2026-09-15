import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory


EXPECTED_ARM_LIMITS = {
    1: (-3.14159265, 3.14159265),
    2: (-1.8325957145940461, 1.8325957145940461),
    3: (-3.14159265, 3.14159265),
    4: (-2.53072742, 2.53072742),
    5: (-3.14159265, 3.14159265),
    6: (-1.91986218, 1.91986218),
    7: (-3.14159265, 3.14159265),
}


def test_urdf_xacro():
    package_path = Path(get_package_share_directory("alfa_robot_description"))
    description_path = package_path / "urdf/alfa_robot.urdf.xacro"
    lock = json.loads((package_path / "config/upstream_description.lock.json").read_text())
    file_descriptor, output_path = tempfile.mkstemp(suffix=".urdf")
    os.close(file_descriptor)

    try:
        with open(output_path, "w", encoding="utf-8") as output:
            xacro_process = subprocess.run(
                [shutil.which("xacro"), str(description_path)],
                stdout=output,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        assert xacro_process.returncode == 0, xacro_process.stderr
        checked = subprocess.run(
            [shutil.which("check_urdf"), output_path],
            capture_output=True,
            text=True,
            check=False,
        )
        assert checked.returncode == 0, checked.stderr

        robot = ET.parse(output_path).getroot()
        links = {link.get("name"): link for link in robot.findall("link")}
        joints = {joint.get("name"): joint for joint in robot.findall("joint")}
        movable = {name: joint for name, joint in joints.items() if joint.get("type") != "fixed"}
        assert robot.get("name") == "alfa_robot"
        assert len(links) == 30
        assert len(joints) == 29
        assert len(movable) == 27
        assert set(movable) == set(lock["movable_joint_names"])
        assert {"left_tool0", "right_tool0", "left_arm_base", "right_arm_base"}.isdisjoint(links)

        visuals = robot.findall(".//visual")
        collisions = robot.findall(".//collision")
        assert len(visuals) == len(collisions) == lock["mesh_count"] == 50
        for mesh in robot.findall(".//mesh"):
            filename = mesh.get("filename")
            assert "/meshes/robot_v3_1_1/" in filename
            assert Path(filename.replace("package://alfa_robot_description", str(package_path))).is_file()

        child_links = {joint.find("child").get("link") for joint in joints.values()}
        assert set(links) - child_links == {"base_footprint"}
        assert joints["base_footprint_to_base_link"].find("origin").get("xyz") == "0.195 0.015 0.400"
        model_origin = joints["base_to_model"].find("origin")
        assert model_origin.get("xyz") == "-0.174000020720362 6.50500007062811e-09 0.802299964909"
        assert model_origin.get("rpy") == "1.5707963 0 1.57079632679"

        updown = joints["updown"]
        assert updown.find("parent").get("link") == "model_base"
        assert updown.find("child").get("link") == "arm_carriage"
        assert updown.find("origin").get("xyz") == "-6.505e-09 -0.80229997 0.174"
        assert float(updown.find("limit").get("lower")) == -1.0
        assert float(updown.find("limit").get("upper")) == 0.0

        for side in ("left", "right"):
            for index, bounds in EXPECTED_ARM_LIMITS.items():
                name = f"{side}_joint{index}"
                joint = joints[name]
                assert joint.find("child").get("link") == name
                expected_parent = "arm_carriage" if index == 1 else f"{side}_joint{index - 1}"
                assert joint.find("parent").get("link") == expected_parent
                limit = joint.find("limit")
                assert abs(float(limit.get("lower")) - bounds[0]) < 1e-8
                assert abs(float(limit.get("upper")) - bounds[1]) < 1e-8
            hand = joints[f"{side}_hand_joint"]
            assert hand.find("parent").get("link") == f"{side}_joint7"
            assert hand.find("child").get("link") == f"{side}_hand"
            assert hand.find("origin").get("xyz") == "-0.030000001 0.015 0.12275"
            assert float(hand.find("limit").get("lower")) == 0.0
            assert float(hand.find("limit").get("upper")) == 0.1

        control = robot.find("ros2_control")
        control_joints = {joint.get("name"): joint for joint in control.findall("joint")}
        assert set(control_joints) == set(movable)
        expected_initial = {name: 0.0 for name in movable}
        expected_initial.update({
            "rear_suspension_joint": 0.085,
            "updown": -0.5,
            "left_joint1": 2.61799387799,
            "left_joint2": 1.57079632679,
            "left_joint3": -0.0872664625997,
            "left_joint4": 2.09439510239,
            "right_joint1": -2.61799387799,
            "right_joint2": -1.57079632679,
            "right_joint3": 0.0872664625997,
            "right_joint4": -2.09439510239,
        })
        for name, joint in control_joints.items():
            initial = joint.find("state_interface[@name='position']/param[@name='initial_value']")
            assert initial is not None
            assert abs(float(initial.text) - expected_initial[name]) < 1e-8
            command = joint.find("command_interface[@name='position']")
            parameters = {parameter.get("name"): float(parameter.text) for parameter in command.findall("param")}
            urdf_limit = joints[name].find("limit")
            assert parameters == {
                "min": float(urdf_limit.get("lower")),
                "max": float(urdf_limit.get("upper")),
            }
    finally:
        os.remove(output_path)
