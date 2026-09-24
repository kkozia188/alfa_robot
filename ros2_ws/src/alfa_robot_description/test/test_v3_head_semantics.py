import xml.etree.ElementTree as ET

import pytest

from test_v3_chassis_semantics import (
    joint_transform,
    link_origin_in_footprint,
    mesh_points_in_footprint,
    transform_point,
)
from test_v3_description_semantics import PACKAGE_ROOT, render_urdf


HEAD_VISUALS = {
    "part_046_solid_046.stl",
    "part_048_surface_shell_048.stl",
    "part_049_surface_shell_049.stl",
    "part_050_surface_shell_050.stl",
    "part_051_surface_shell_051.stl",
    "part_052_surface_shell_052.stl",
    "part_058_surface_shell_058.stl",
    "part_059_surface_shell_059.stl",
    "part_060_surface_shell_060.stl",
    "part_061_surface_shell_061.stl",
    "part_062_surface_shell_062.stl",
    "part_068_solid_068.stl",
}
CHEST_VISUALS = {
    "part_070_solid_070.stl",
    "part_072_surface_shell_072.stl",
    "part_073_surface_shell_073.stl",
    "part_074_surface_shell_074.stl",
    "part_075_surface_shell_075.stl",
    "part_076_surface_shell_076.stl",
    "part_082_surface_shell_082.stl",
    "part_083_surface_shell_083.stl",
    "part_084_surface_shell_084.stl",
    "part_085_surface_shell_085.stl",
    "part_086_surface_shell_086.stl",
}
EXCLUDED_FIELD_GEOMETRY = {
    "part_047_surface_shell_047.stl",
    "part_057_surface_shell_057.stl",
    "part_067_solid_067.stl",
    "part_071_surface_shell_071.stl",
    "part_081_surface_shell_081.stl",
}


def visual_names(link):
    return {
        visual.find("geometry/mesh").attrib["filename"].rsplit("/", 1)[-1]
        for visual in link.findall("visual")
    }


@pytest.mark.parametrize("variant", ("gripper", "suction"))
def test_new_head_and_chest_camera_preserve_the_source_motion_chain(variant):
    robot = ET.fromstring(render_urdf(variant))
    joints = {joint.attrib["name"]: joint for joint in robot.findall("joint")}
    links = {link.attrib["name"]: link for link in robot.findall("link")}

    mount = joints["head_mount_fixed"]
    yaw = joints["head_joint"]
    pitch = joints["head_pitch_joint"]
    chest = joints["chest_camera_fixed"]
    assert mount.find("parent").attrib["link"] == "arm_carriage"
    assert mount.find("child").attrib["link"] == "head_mount"
    assert mount.find("origin").attrib == {
        "xyz": "0.267 0 1.4228", "rpy": "0 0 1.57079632679"
    }
    assert yaw.find("parent").attrib["link"] == "head_mount"
    assert yaw.find("child").attrib["link"] == "head_yaw"
    assert yaw.find("origin").attrib == {"xyz": "0 0 0", "rpy": "0 0 0"}
    assert pitch.find("parent").attrib["link"] == "head_yaw"
    assert pitch.find("child").attrib["link"] == "head"
    assert pitch.find("origin").attrib == {
        "xyz": "0.01393364 0.022 0.044", "rpy": "0 -1.5707963 0"
    }
    for joint in (yaw, pitch):
        assert joint.attrib["type"] == "revolute"
        assert joint.find("axis").attrib["xyz"] == "0 0 1"
        assert joint.find("limit").attrib == {
            "lower": "-1.57", "upper": "1.57", "effort": "10", "velocity": "5"
        }

    assert chest.attrib["type"] == "fixed"
    assert chest.find("parent").attrib["link"] == "arm_carriage"
    assert chest.find("child").attrib["link"] == "chest_camera"
    assert chest.find("origin").attrib == {
        "xyz": "0.366 -3.92e-09 1.11530000583", "rpy": "0 0 0"
    }

    assert visual_names(links["head_yaw"]) == {"part_069_solid_069.stl"}
    assert visual_names(links["head"]) == HEAD_VISUALS
    assert visual_names(links["chest_camera"]) == CHEST_VISUALS
    assert float(links["head_yaw"].find("inertial/mass").attrib["value"]) == pytest.approx(0.07210205)
    assert float(links["head"].find("inertial/mass").attrib["value"]) == pytest.approx(1.00116361294)
    assert float(links["chest_camera"].find("inertial/mass").attrib["value"]) == pytest.approx(0.137417570647)

    all_meshes = {mesh.attrib["filename"] for mesh in robot.findall(".//mesh")}
    assert all("/meshes/head_chest_camera/" in filename for link_name in ("head_yaw", "head", "chest_camera")
               for filename in (mesh.attrib["filename"] for mesh in links[link_name].findall(".//mesh")))
    assert not any(filename.rsplit("/", 1)[-1] in EXCLUDED_FIELD_GEOMETRY for filename in all_meshes)
    assert not any("/meshes/head/" in filename for filename in all_meshes)
    for filename in all_meshes:
        path = PACKAGE_ROOT / filename.removeprefix("package://alfa_robot_description/")
        assert path.is_file(), filename


def test_head_axes_are_orthogonal_and_camera_payload_follows_both_axes():
    robot = ET.fromstring(render_urdf())
    yaw = robot.find("joint[@name='head_joint']")
    pitch = robot.find("joint[@name='head_pitch_joint']")
    yaw_rotation, _ = joint_transform(yaw)
    pitch_rotation, _ = joint_transform(pitch)
    assert transform_point((0, 0, 1), yaw_rotation, (0, 0, 0)) == pytest.approx((0, 0, 1), abs=1e-7)
    assert transform_point((0, 0, 1), pitch_rotation, (0, 0, 0)) == pytest.approx((-1, 0, 0), abs=1e-7)

    camera = "part_046_solid_046.stl"
    zero = next(mesh_points_in_footprint(robot, "head", camera))
    yawed = next(mesh_points_in_footprint(robot, "head", camera, {"head_joint": 0.5}))
    pitched = next(mesh_points_in_footprint(robot, "head", camera, {"head_pitch_joint": 0.5}))
    assert yawed != pytest.approx(zero)
    assert pitched != pytest.approx(zero)

    yaw_bracket_zero = next(mesh_points_in_footprint(robot, "head_yaw", "part_069_solid_069.stl"))
    yaw_bracket_pitch = next(mesh_points_in_footprint(
        robot, "head_yaw", "part_069_solid_069.stl", {"head_pitch_joint": 0.5}
    ))
    assert yaw_bracket_zero == pytest.approx(yaw_bracket_pitch)


def test_chest_camera_is_body_centred_and_moves_only_with_updown():
    robot = ET.fromstring(render_urdf())
    mesh = "part_070_solid_070.stl"
    zero_points = list(mesh_points_in_footprint(robot, "chest_camera", mesh))
    body_center = tuple(
        (min(point[axis] for point in zero_points) + max(point[axis] for point in zero_points)) / 2
        for axis in range(3)
    )
    assert body_center == pytest.approx(link_origin_in_footprint(robot, "chest_camera"), abs=2e-8)
    body_size = tuple(
        max(point[axis] for point in zero_points) - min(point[axis] for point in zero_points)
        for axis in range(3)
    )
    assert body_size == pytest.approx((0.04, 0.17, 0.045), abs=2e-7)

    head_motion = list(mesh_points_in_footprint(
        robot, "chest_camera", mesh, {"head_joint": 0.6, "head_pitch_joint": -0.4}
    ))
    assert head_motion == pytest.approx(zero_points)
    lowered = list(mesh_points_in_footprint(robot, "chest_camera", mesh, {"updown": -0.4}))
    for before, after in zip(zero_points, lowered):
        assert after[0] == pytest.approx(before[0], abs=2e-8)
        assert after[1] == pytest.approx(before[1], abs=2e-8)
        assert after[2] == pytest.approx(before[2] - 0.4, abs=2e-8)
