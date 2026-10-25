#!/usr/bin/env python3
"""Inject conservative spheres for selected links into an existing sphere URDF."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from spherize_model import mesh_spheres, transformed_mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_urdf", type=Path)
    parser.add_argument("sphere_urdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--link", action="append", required=True)
    parser.add_argument("--spheres-per-component", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.002)
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
        generated = []
        for collision in mesh_link.findall("collision"):
            generated.extend(mesh_spheres(
                transformed_mesh(collision), args.spheres_per_component, args.margin
            ))
        for collision in list(output_link.findall("collision")):
            output_link.remove(collision)
        for center, radius in generated:
            collision = ET.SubElement(output_link, "collision")
            ET.SubElement(
                collision, "origin",
                xyz=" ".join(f"{value:.12g}" for value in center), rpy="0 0 0",
            )
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "sphere", radius=f"{radius:.12g}")
        report.append({"link": link_name, "sphere_count": len(generated)})
    ET.indent(output_tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_tree.write(args.output, encoding="utf-8", xml_declaration=True)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps({
        "schema": "alfa.v322.injected_link_spheres.v1",
        "mesh_urdf": str(args.mesh_urdf.resolve()),
        "base_sphere_urdf": str(args.sphere_urdf.resolve()),
        "output": str(args.output.resolve()),
        "spheres_per_component": args.spheres_per_component,
        "margin_m": args.margin,
        "links": report,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
