#!/usr/bin/env python3
"""Generate an offline wall fixture using the installed V3 model's named poses."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    root = ET.parse(args.srdf).getroot()
    names = source["joint_names"]

    def named_pose(name):
        group = next(item for item in root.findall("group_state")
                     if item.get("group") == "whole_body" and item.get("name") == name)
        values = {joint.get("name"): float(joint.get("value")) for joint in group}
        return [values[joint] for joint in names]

    output = {
        "kind": "v3_initial_transition_direct_shortcut_diagnostic",
        "moving_joints": "both_arms_j1_to_j4",
        "joint_names": names,
        "environment": source["environment"],
        "wall_boxes": source["wall_boxes"],
        "frames": [{"joints": named_pose("home")},
                   {"joints": named_pose("second_home")}],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
