#!/usr/bin/env python3
"""Generate a deterministic Foam sphere URDF for the V3.2.2 benchmark.

Foam's stock URDF script derives the branch count from signed mesh volume.  The
V3.2.2 model contains thin shells with negative or degenerate volume, so that
heuristic can request a negative number of spheres.  This driver deliberately
uses one fixed branch count for every collision element and otherwise calls
Foam's normal repair and SphereTree pipeline unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

from foam import get_urdf_meshes, get_urdf_primitives, load_urdf
from foam import save_urdf, set_urdf_spheres, spherize_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--branch", type=int, default=3)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--manifold-leaves", type=int, default=1000)
    parser.add_argument("--simplification-ratio", type=float, default=0.2)
    parser.add_argument(
        "--exclude-link", action="append", default=[],
        help="link whose invariant geometry is checked outside the timed kernel",
    )
    parser.add_argument(
        "--branch-override", action="append", default=[], metavar="TEXT=COUNT",
        help="use COUNT branches when TEXT occurs in Foam's link::mesh key",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.branch < 1 or args.depth < 0 or args.threads < 1:
        raise ValueError("branch and threads must be positive; depth must be non-negative")

    urdf = load_urdf(args.input.resolve())
    geometries = [*get_urdf_meshes(urdf), *get_urdf_primitives(urdf)]
    geometries = [
        geometry for geometry in geometries
        if geometry.name.split("::", maxsplit=1)[0] not in args.exclude_link
    ]
    overrides = []
    for value in args.branch_override:
        selector, separator, count = value.rpartition("=")
        if not separator or not selector or int(count) < 1:
            raise ValueError(f"invalid branch override: {value!r}")
        overrides.append((selector, int(count)))

    base_settings = {
        "depth": args.depth,
        "branch": args.branch,
        "method": "medial",
        "testerLevels": 2,
        "numCover": 5000,
        "minCover": 5,
        "initSpheres": 1000,
        "minSpheres": 200,
        "erFact": 2,
        "expand": True,
        "merge": True,
        "burst": False,
        "optimise": True,
        "maxOptLevel": 1,
        "balExcess": 0.05,
        "verify": True,
        "num_samples": 500,
        "min_samples": 1,
    }
    repair = {
        "manifold_leaves": args.manifold_leaves,
        "ratio": args.simplification_ratio,
    }

    def run(geometry):
        branch = next(
            (count for selector, count in reversed(overrides) if selector in geometry.name),
            args.branch,
        )
        settings = dict(base_settings)
        settings["branch"] = branch
        # URDF applies mesh scale in mesh coordinates, then applies the collision
        # origin. Foam's helper currently performs these in the opposite order,
        # which incorrectly scales the origin for millimetre STL files.
        mesh = geometry.mesh.copy()
        mesh.apply_scale(geometry.scale)
        levels = spherize_mesh(
            geometry.name,
            mesh,
            None,
            geometry.xyz,
            geometry.rpy,
            settings,
            repair,
        )
        return geometry.name, branch, levels[args.depth]

    results = {}
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {pool.submit(run, geometry): geometry.name for geometry in geometries}
        for completed, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                key, branch, spheres = future.result()
            except Exception as exc:
                print(f"FAILED {name}: {exc}", file=sys.stderr)
                for pending in futures:
                    pending.cancel()
                return 1
            results[key] = spheres
            print(
                f"[{completed}/{len(futures)}] {key}: "
                f"branch={branch}, {len(spheres.spheres)} spheres"
            )

    set_urdf_spheres(urdf, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_urdf(urdf, args.output)
    print(f"wrote {args.output} from {len(geometries)} collision geometries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
