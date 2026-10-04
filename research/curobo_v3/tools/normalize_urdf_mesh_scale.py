#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def resolve_mesh(filename: str, package_root: Path) -> Path:
    prefix = "package://"
    if filename.startswith(prefix):
        package_and_path = filename.removeprefix(prefix)
        if "/" not in package_and_path:
            raise ValueError(f"Invalid package mesh URI: {filename}")
        _, relative_path = package_and_path.split("/", 1)
        return package_root / relative_path
    path = Path(filename)
    if path.is_absolute():
        return path
    return package_root / path


def scale_key(source: Path, scale: np.ndarray) -> str:
    value = f"{source.resolve()}:{scale.tolist()}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bake URDF mesh scales into derived mesh files."
    )
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    mesh_root = output_root / "meshes"
    mesh_root.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(args.urdf)
    cache: dict[tuple[str, tuple[float, float, float]], Path] = {}
    converted = 0

    for mesh_element in tree.getroot().findall(".//mesh"):
        filename = mesh_element.attrib["filename"]
        scale = np.fromstring(mesh_element.attrib.get("scale", "1 1 1"), sep=" ")
        if scale.shape != (3,):
            raise ValueError(f"Invalid scale for {filename}: {scale}")

        source = resolve_mesh(filename, args.package_root).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)

        key = (str(source), tuple(float(value) for value in scale))
        destination = cache.get(key)
        if destination is None:
            destination = mesh_root / f"{source.stem}-{scale_key(source, scale)}.stl"
            geometry = trimesh.load(source, force="mesh", process=False)
            if isinstance(geometry, trimesh.Scene):
                geometry = geometry.to_geometry()
            geometry.vertices = np.asarray(geometry.vertices) * scale
            geometry.export(destination)
            cache[key] = destination

        mesh_element.set("filename", str(destination))
        mesh_element.set("scale", "1 1 1")
        converted += 1

    output_urdf = output_root / "robot_meter_meshes.urdf"
    ET.indent(tree, space="  ")
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"output_urdf={output_urdf}")
    print(f"mesh_references={converted}")
    print(f"derived_meshes={len(cache)}")


if __name__ == "__main__":
    main()
