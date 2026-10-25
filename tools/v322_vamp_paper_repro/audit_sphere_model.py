#!/usr/bin/env python3
"""Compare a Foam sphere model with the unchanged mesh collision model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pinocchio as pin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_urdf", type=Path)
    parser.add_argument("sphere_urdf", type=Path)
    parser.add_argument("srdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--endpoint-json", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry(model, urdf: Path, srdf: Path):
    result = pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.COLLISION)
    result.addAllCollisionPairs()
    pin.removeCollisionPairs(model, result, str(srdf))
    return result, pin.GeometryData(result)


def main() -> int:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")

    model = pin.buildModelFromUrdf(str(args.mesh_urdf))
    sphere_model = pin.buildModelFromUrdf(str(args.sphere_urdf))
    if list(model.names) != list(sphere_model.names) or model.nq != sphere_model.nq:
        raise RuntimeError("sphere URDF changed the robot kinematic structure")

    mesh_geometry, mesh_data = geometry(model, args.mesh_urdf, args.srdf)
    sphere_geometry, sphere_data = geometry(model, args.sphere_urdf, args.srdf)
    model_data = model.createData()

    counts = {
        "both_clear": 0,
        "both_collision": 0,
        "sphere_false_positive": 0,
        "sphere_false_negative": 0,
    }
    false_negative_examples = []
    generator = np.random.default_rng(args.seed)
    started = time.perf_counter()
    for _ in range(args.samples):
        state = generator.uniform(model.lowerPositionLimit, model.upperPositionLimit)
        mesh_collision = pin.computeCollisions(
            model, model_data, mesh_geometry, mesh_data, state, True
        )
        sphere_collision = pin.computeCollisions(
            model, model_data, sphere_geometry, sphere_data, state, True
        )
        if mesh_collision and sphere_collision:
            label = "both_collision"
        elif not mesh_collision and not sphere_collision:
            label = "both_clear"
        elif sphere_collision:
            label = "sphere_false_positive"
        else:
            label = "sphere_false_negative"
        counts[label] += 1
        if label == "sphere_false_negative" and len(false_negative_examples) < 5:
            false_negative_examples.append(state.tolist())

    endpoint_labels = {}
    if args.endpoint_json:
        payload = json.loads(args.endpoint_json.read_text())
        for name in ("start", "goal"):
            state = np.asarray(payload["paper_benchmark"][name], dtype=float)
            endpoint_labels[name] = {
                "mesh_collision": bool(pin.computeCollisions(
                    model, model_data, mesh_geometry, mesh_data, state, True
                )),
                "sphere_collision": bool(pin.computeCollisions(
                    model, model_data, sphere_geometry, sphere_data, state, True
                )),
            }

    result = {
        "schema": "alfa.v322.vamp_sphere_audit.v1",
        "kinematic_structure_unchanged": True,
        "joint_names": list(model.names[1:]),
        "samples": args.samples,
        "seed": args.seed,
        "counts": counts,
        "rates": {key: value / args.samples for key, value in counts.items()},
        "false_negative_examples": false_negative_examples,
        "endpoints": endpoint_labels,
        "elapsed_seconds": time.perf_counter() - started,
        "files": {
            "mesh_urdf": str(args.mesh_urdf.resolve()),
            "mesh_urdf_sha256": sha256(args.mesh_urdf),
            "sphere_urdf": str(args.sphere_urdf.resolve()),
            "sphere_urdf_sha256": sha256(args.sphere_urdf),
            "srdf": str(args.srdf.resolve()),
            "srdf_sha256": sha256(args.srdf),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
