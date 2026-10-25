#!/usr/bin/env python3
"""Inject conservative oriented boxes for selected fixed links."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from trimesh.transformations import euler_from_matrix

from spherize_model import split_faces, transformed_mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_urdf", type=Path)
    parser.add_argument("sphere_urdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--link", action="append", required=True)
    parser.add_argument("--margin", type=float, default=0.002)
    parser.add_argument("--boxes-per-mesh", type=int, default=1)
    args = parser.parse_args()
    mesh_root = ET.parse(args.mesh_urdf).getroot()
    output_tree = ET.parse(args.sphere_urdf)
    output_root = output_tree.getroot()
    report = []
    for link_name in args.link:
        mesh_link = mesh_root.find(f"link[@name='{link_name}']")
        output_link = output_root.find(f"link[@name='{link_name}']")
        if mesh_link is None or output_link is None:
            raise ValueError(f"missing link {link_name}")
        boxes = []
        for collision in mesh_link.findall("collision"):
            mesh = transformed_mesh(collision)
            for face_indices in split_faces(mesh, args.boxes_per_mesh):
                points = mesh.vertices[np.unique(mesh.faces[face_indices].reshape(-1))]
                oriented = trimesh.points.PointCloud(points).bounding_box_oriented
                transform = np.asarray(oriented.primitive.transform, dtype=float)
                extents = np.asarray(oriented.primitive.extents, dtype=float) + 2.0 * args.margin
                boxes.append((transform, extents))
        for collision in list(output_link.findall("collision")):
            output_link.remove(collision)
        for transform, extents in boxes:
            collision = ET.SubElement(output_link, "collision")
            ET.SubElement(
                collision, "origin",
                xyz=" ".join(f"{value:.12g}" for value in transform[:3, 3]),
                rpy=" ".join(f"{value:.12g}" for value in euler_from_matrix(transform)),
            )
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "box", size=" ".join(f"{value:.12g}" for value in extents))
        report.append({"link": link_name, "box_count": len(boxes)})
    ET.indent(output_tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_tree.write(args.output, encoding="utf-8", xml_declaration=True)
    args.output.with_suffix(".json").write_text(json.dumps({
        "schema": "alfa.v322.injected_link_boxes.v1",
        "mesh_urdf": str(args.mesh_urdf.resolve()),
        "base_sphere_urdf": str(args.sphere_urdf.resolve()),
        "output": str(args.output.resolve()),
        "margin_m": args.margin,
        "boxes_per_mesh": args.boxes_per_mesh,
        "links": report,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
