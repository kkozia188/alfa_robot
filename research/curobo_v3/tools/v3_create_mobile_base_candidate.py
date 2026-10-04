#!/usr/bin/env python3

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


MOBILE_JOINTS = ["base_x", "base_y", "base_yaw"]


def add_mobile_chain(urdf_source: Path, urdf_output: Path, yaw_limit_deg: float):
    tree = ET.parse(urdf_source)
    root = tree.getroot()
    chain = [
        ("world", None),
        ("base_x_link", ("base_x", "prismatic", "world", "base_x_link", "1 0 0", "-0.5", "0.5", "0.6")),
        ("base_y_link", ("base_y", "prismatic", "base_x_link", "base_y_link", "0 1 0", "-0.5", "0.5", "0.6")),
        (None, ("base_yaw", "revolute", "base_y_link", "base_footprint", "0 0 1", str(-yaw_limit_deg * 3.141592653589793 / 180.0), str(yaw_limit_deg * 3.141592653589793 / 180.0), "0.5")),
    ]
    insert_index = 0
    for link_name, joint_data in chain:
        if link_name is not None:
            root.insert(insert_index, ET.Element("link", {"name": link_name}))
            insert_index += 1
        if joint_data is None:
            continue
        name, joint_type, parent, child, axis, lower, upper, velocity = joint_data
        joint = ET.Element("joint", {"name": name, "type": joint_type})
        ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": child})
        ET.SubElement(joint, "axis", {"xyz": axis})
        ET.SubElement(joint, "limit", {
            "lower": lower, "upper": upper, "effort": "10000", "velocity": velocity,
        })
        root.insert(insert_index, joint)
        insert_index += 1
    ET.indent(tree, space="  ")
    tree.write(urdf_output, encoding="utf-8", xml_declaration=True)


def add_mobile_cspace(config_source: Path, config_output: Path, urdf_output: Path,
                      yaw_limit_deg: float):
    data = yaml.safe_load(config_source.read_text())
    kinematics = data.get("robot_cfg", data)["kinematics"]
    kinematics["base_link"] = "world"
    kinematics["urdf_path"] = str(urdf_output)
    kinematics["asset_root_path"] = str(urdf_output.parent)
    cspace = kinematics["cspace"]
    old_count = len(cspace["joint_names"])
    prepend = {
        "joint_names": MOBILE_JOINTS,
        "default_joint_position": [0.0, 0.0, 0.0],
        "cspace_distance_weight": [2.0, 2.0, 1.0],
        "null_space_weight": [1.0, 1.0, 1.0],
        "null_space_maximum_distance": [
            0.5, 0.5, yaw_limit_deg * 3.141592653589793 / 180.0
        ],
        "max_acceleration": [1.0, 1.0, 1.0],
        "max_jerk": [10.0, 10.0, 10.0],
        "velocity_scale": [1.0, 1.0, 1.0],
        "acceleration_scale": [1.0, 1.0, 1.0],
        "jerk_scale": [1.0, 1.0, 1.0],
    }
    for key, value in list(cspace.items()):
        if isinstance(value, list) and len(value) == old_count:
            cspace[key] = prepend.get(key, [1.0, 1.0, 1.0]) + value
    config_output.write_text(yaml.safe_dump(data, sort_keys=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yaw-limit-deg", type=float, default=30.0)
    args = parser.parse_args()
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    source_urdf = args.source_dir / "model/robot_meter_meshes.urdf"
    output_urdf = model_dir / "robot_meter_meshes_mobile18.urdf"
    source_config = (
        args.source_config.resolve()
        if args.source_config is not None
        else args.source_dir / "alfa_v322_suction_final.yml"
    )
    output_config = args.output_dir / "alfa_v322_suction_mobile18.yml"
    add_mobile_chain(source_urdf, output_urdf, args.yaw_limit_deg)
    add_mobile_cspace(source_config, output_config, output_urdf, args.yaw_limit_deg)
    (args.output_dir / "README.md").write_text(
        "Experimental 18-DoF cuRobo candidate: base_x/base_y +/-0.5m, "
        f"base_yaw +/-{args.yaw_limit_deg:g}deg, plus existing 15-DoF planning group.\n"
    )
    print(output_config)


if __name__ == "__main__":
    main()
