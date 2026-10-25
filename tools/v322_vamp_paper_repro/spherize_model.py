#!/usr/bin/env python3
"""Create a deterministic sphere-only collision URDF for the V3.2.2 robot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from trimesh.nsphere import minimum_nsphere
from trimesh.transformations import euler_matrix


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "results/v322_vamp_paper_repro/model/alfa-v322-15d-mesh.urdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    return np.fromiter((float(value) for value in text.split()), dtype=float)


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"not a triangle mesh: {path}")
    if len(loaded.faces) == 0:
        raise ValueError(f"mesh has no faces: {path}")
    return loaded


def split_faces(mesh: trimesh.Trimesh, count: int) -> list[np.ndarray]:
    groups = [np.arange(len(mesh.faces), dtype=np.int64)]
    while len(groups) < count:
        candidates = [(len(group), index) for index, group in enumerate(groups) if len(group) > 1]
        if not candidates:
            break
        _, index = max(candidates)
        group = groups.pop(index)
        centroids = mesh.triangles_center[group]
        centered = centroids - centroids.mean(axis=0)
        covariance = centered.T @ centered
        axis = np.linalg.eigh(covariance)[1][:, -1]
        order = np.argsort(centroids @ axis, kind="stable")
        middle = max(1, len(order) // 2)
        left = group[order[:middle]]
        right = group[order[middle:]]
        groups.extend((left, right))
    return groups


def component_spheres(
    mesh: trimesh.Trimesh,
    count: int,
    margin: float,
) -> list[tuple[np.ndarray, float]]:
    spheres = []
    for face_indices in split_faces(mesh, count):
        points = mesh.vertices[np.unique(mesh.faces[face_indices].reshape(-1))]
        try:
            center, radius = minimum_nsphere(points)
        except Exception:
            center = points.mean(axis=0)
            radius = np.linalg.norm(points - center, axis=1).max()
        spheres.append((np.asarray(center), float(radius) + margin))
    return spheres


def mesh_spheres(
    mesh: trimesh.Trimesh,
    count: int,
    margin: float,
) -> list[tuple[np.ndarray, float]]:
    components = list(mesh.split(only_watertight=False))
    if not components:
        components = [mesh]
    spheres = []
    for component in components:
        spheres.extend(component_spheres(component, count, margin))
    return spheres


def transformed_mesh(collision: ET.Element) -> trimesh.Trimesh:
    geometry = collision.find("geometry")
    if geometry is None:
        raise ValueError("collision geometry missing")
    mesh_element = geometry.find("mesh")
    if mesh_element is None:
        raise ValueError("only mesh collision geometry is supported in the certified model")
    mesh = load_mesh(Path(mesh_element.attrib["filename"]))
    # Binary STL commonly duplicates all three vertices per face. Weld spatially
    # identical vertices before connected-component analysis.
    mesh.merge_vertices(digits_vertex=6)
    scale = vector(mesh_element.attrib.get("scale"), (1.0, 1.0, 1.0))
    mesh.vertices *= scale
    origin = collision.find("origin")
    xyz = vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    rpy = vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
    transform = euler_matrix(*rpy, axes="sxyz")
    transform[:3, 3] = xyz
    mesh.apply_transform(transform)
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spheres-per-mesh", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.002)
    parser.add_argument("--exclude-link", action="append", default=["base_link"])
    parser.add_argument(
        "--link-spheres", action="append", default=[], metavar="LINK=COUNT",
        help="Override the sphere count per connected component for one link",
    )
    args = parser.parse_args()
    if args.spheres_per_mesh < 1:
        parser.error("--spheres-per-mesh must be positive")
    if args.margin < 0.0:
        parser.error("--margin must be non-negative")
    link_spheres = {}
    for item in args.link_spheres:
        try:
            name, raw_count = item.rsplit("=", maxsplit=1)
            link_spheres[name] = int(raw_count)
        except ValueError:
            parser.error(f"invalid --link-spheres value: {item}")
        if not name or link_spheres[name] < 1:
            parser.error(f"invalid --link-spheres value: {item}")

    source = args.input.resolve()
    output = (
        args.output.resolve()
        if args.output
        else source.with_name(f"alfa-v322-15d-spheres-{args.spheres_per_mesh}.urdf")
    )
    tree = ET.parse(source)
    root = tree.getroot()
    report_links = []
    total_spheres = 0
    maximum_radius = 0.0
    minimum_radius = math.inf
    excluded_links = set(args.exclude_link)

    for link in root.findall("link"):
        collisions = list(link.findall("collision"))
        if not collisions:
            continue
        if link.attrib["name"] in excluded_links:
            for collision in collisions:
                link.remove(collision)
            report_links.append(
                {
                    "link": link.attrib["name"],
                    "source_geometries": len(collisions),
                    "spheres": 0,
                    "excluded_static_link": True,
                }
            )
            continue
        generated = []
        link_count = link_spheres.get(link.attrib["name"], args.spheres_per_mesh)
        for collision in collisions:
            mesh = transformed_mesh(collision)
            generated.extend(mesh_spheres(mesh, link_count, args.margin))
        for collision in collisions:
            link.remove(collision)
        for center, radius in generated:
            collision = ET.SubElement(link, "collision")
            ET.SubElement(
                collision,
                "origin",
                xyz=" ".join(f"{value:.12g}" for value in center),
                rpy="0 0 0",
            )
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "sphere", radius=f"{radius:.12g}")
            minimum_radius = min(minimum_radius, radius)
            maximum_radius = max(maximum_radius, radius)
        total_spheres += len(generated)
        report_links.append(
            {
                "link": link.attrib["name"],
                "source_geometries": len(collisions),
                "spheres": len(generated),
                "spheres_per_component": link_count,
            }
        )

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    report = {
        "schema": "alfa.v322.vamp_sphere_model.v1",
        "method": "recursive_triangle_centroid_pca_minimum_bounding_sphere",
        "connected_components_split_before_partition": True,
        "vertex_weld_decimal_digits": 6,
        "input": str(source),
        "input_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "spheres_per_source_mesh": args.spheres_per_mesh,
        "link_sphere_overrides": link_spheres,
        "inflation_margin_m": args.margin,
        "excluded_static_links": sorted(excluded_links),
        "sphere_count": total_spheres,
        "minimum_radius_m": minimum_radius,
        "maximum_radius_m": maximum_radius,
        "links": report_links,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
