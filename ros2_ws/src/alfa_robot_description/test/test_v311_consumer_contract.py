import json
import hashlib
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def render_urdf():
    result = subprocess.run(
        ["xacro", str(PACKAGE_ROOT / "urdf/alfa_robot.urdf.xacro")],
        capture_output=True,
        text=True,
        check=True,
    )
    return ET.fromstring(result.stdout)


def test_demo_consumes_pinned_v311_description_contract():
    lock = json.loads((PACKAGE_ROOT / "config/upstream_description.lock.json").read_text())
    assert lock["model_revision"] == "robot_v3.1.1"
    assert lock["upstream_branch"] == "robot_v3"
    assert lock["upstream_commit"] == "b0c53aa8ca0bba1701d2f6680115012ed38079c0"

    entrypoint = (PACKAGE_ROOT / "urdf/alfa_robot.urdf.xacro").read_text()
    assert "robot_v3_1_1.xacro" in entrypoint
    assert "robot_v3_0_9.xacro" not in entrypoint
    active_xacro = PACKAGE_ROOT / "urdf/alfa_robot/robot_v3_1_1.xacro"
    assert hashlib.sha256(active_xacro.read_bytes()).hexdigest() == lock["active_xacro_sha256"]
    upstream_manifest_path = PACKAGE_ROOT / "config/upstream_description_manifest.json"
    assert hashlib.sha256(upstream_manifest_path.read_bytes()).hexdigest() == lock["upstream_manifest_sha256"]
    upstream_manifest = json.loads(upstream_manifest_path.read_text())
    for relative_path, metadata in upstream_manifest["meshes"].items():
        local_path = PACKAGE_ROOT / "meshes/robot_v3_1_1" / Path(relative_path).name
        assert local_path.stat().st_size == metadata["bytes"]
        assert hashlib.sha256(local_path.read_bytes()).hexdigest() == metadata["sha256"]

    robot = render_urdf()
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    movable = {name: joint for name, joint in joints.items() if joint.get("type") != "fixed"}
    assert len(movable) == 27
    assert set(lock["movable_joint_names"]) == set(movable)

    updown_limit = joints["updown"].find("limit")
    assert float(updown_limit.get("lower")) == -1.0
    assert float(updown_limit.get("upper")) == 0.0
    for side in ("left", "right"):
        hand_limit = joints[f"{side}_hand_joint"].find("limit")
        assert float(hand_limit.get("lower")) == 0.0
        assert float(hand_limit.get("upper")) == 0.1
        assert joints[f"{side}_joint7"].find("child").get("link") == f"{side}_joint7"

    control = robot.find("ros2_control")
    assert control is not None
    assert {joint.get("name") for joint in control.findall("joint")} == set(movable)

    with tempfile.NamedTemporaryFile("w", suffix=".urdf") as output:
        output.write(ET.tostring(robot, encoding="unicode"))
        output.flush()
        checked = subprocess.run(
            ["check_urdf", output.name], capture_output=True, text=True, check=False
        )
    assert checked.returncode == 0, checked.stderr
