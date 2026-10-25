#!/usr/bin/env python3
"""Count collision pairs responsible for sphere/mesh label disagreement."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pinocchio as pin


def geometry(model, urdf: Path, srdf: Path):
    result = pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.COLLISION)
    result.addAllCollisionPairs()
    pin.removeCollisionPairs(model, result, str(srdf))
    return result, pin.GeometryData(result)


def pair_name(geometry_model, pair) -> str:
    first_object = geometry_model.geometryObjects[pair.first].name
    second_object = geometry_model.geometryObjects[pair.second].name
    return " <-> ".join(sorted((first_object, second_object)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_urdf", type=Path)
    parser.add_argument("sphere_urdf", type=Path)
    parser.add_argument("srdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    model = pin.buildModelFromUrdf(str(args.mesh_urdf))
    mesh_model, mesh_data = geometry(model, args.mesh_urdf, args.srdf)
    sphere_model, sphere_data = geometry(model, args.sphere_urdf, args.srdf)
    model_data = model.createData()
    false_positive_pairs: collections.Counter[str] = collections.Counter()
    false_negative_pairs: collections.Counter[str] = collections.Counter()
    false_positive_states = 0
    false_negative_states = 0
    mesh_clear_states = 0
    generator = np.random.default_rng(args.seed)
    for _ in range(args.samples):
        state = generator.uniform(model.lowerPositionLimit, model.upperPositionLimit)
        mesh_collision = pin.computeCollisions(
            model, model_data, mesh_model, mesh_data, state, False
        )
        sphere_collision = pin.computeCollisions(
            model, model_data, sphere_model, sphere_data, state, False
        )
        mesh_clear_states += int(not mesh_collision)
        if sphere_collision and not mesh_collision:
            false_positive_states += 1
            for index, result in enumerate(sphere_data.collisionResults):
                if result.isCollision():
                    false_positive_pairs[pair_name(
                        sphere_model, sphere_model.collisionPairs[index]
                    )] += 1
        if mesh_collision and not sphere_collision:
            false_negative_states += 1
            for index, result in enumerate(mesh_data.collisionResults):
                if result.isCollision():
                    false_negative_pairs[pair_name(
                        mesh_model, mesh_model.collisionPairs[index]
                    )] += 1
    output = {
        "schema": "alfa.v322.sphere_false_positive_pairs.v1",
        "samples": args.samples,
        "mesh_clear_samples": mesh_clear_states,
        "sphere_false_positive_states": false_positive_states,
        "sphere_false_negative_states": false_negative_states,
        "top_false_positive_pairs": [
            {"pair": name, "count": count}
            for name, count in false_positive_pairs.most_common(50)
        ],
        "top_false_negative_pairs": [
            {"pair": name, "count": count}
            for name, count in false_negative_pairs.most_common(50)
        ],
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
