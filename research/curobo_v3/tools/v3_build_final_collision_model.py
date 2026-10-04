#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml

from curobo.robot_builder import RobotBuilder
from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh


def morphit_parameters(report, target):
    method = report[target]["methods"]["morphit"]
    return {
        "num_spheres": int(method["requested"]),
        "iterations": int(method["iterations"]),
        "coverage_weight": float(method["coverage_weight"]),
        "protrusion_weight": float(method["protrusion_weight"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-config", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-box", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.fit_report.read_text())
    base_params = morphit_parameters(report, "base_link")
    box_params = morphit_parameters(report, "box")
    torch.manual_seed(20260929)
    np.random.seed(20260929)

    builder = RobotBuilder.from_config(str(args.preview_config))
    for link_name in builder.collision_link_names:
        if link_name == "base_link":
            params = base_params
        else:
            params = {
                "num_spheres": None,
                "iterations": 1000,
                "coverage_weight": None,
                "protrusion_weight": None,
            }
        print(f"fitting {link_name}: {params}", flush=True)
        builder.refit_link_spheres(
            link_name,
            num_spheres=params["num_spheres"],
            sphere_density=0.1,
            fit_type=SphereFitType.MORPHIT,
            iterations=params["iterations"],
            coverage_weight=params["coverage_weight"],
            protrusion_weight=params["protrusion_weight"],
            compute_metrics=False,
        )

    builder.compute_collision_matrix(prune_collisions=False)
    final_config = builder.build()
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    builder.save(final_config, str(args.output_config))

    box_mesh = trimesh.creation.box(extents=(0.30, 0.40, 0.40))
    box_result = fit_spheres_to_mesh(
        box_mesh,
        fit_type=SphereFitType.MORPHIT,
        compute_metrics=True,
        **box_params,
    )
    box_output = {
        "description_commit": "bd1e455",
        "dimensions_m": [0.30, 0.40, 0.40],
        "parameters": box_params,
        "actual_spheres": int(box_result.num_spheres),
        "centers": box_result.centers.detach().cpu().tolist(),
        "radii": box_result.radii.detach().cpu().reshape(-1).tolist(),
        "metrics": {
            "coverage": box_result.metrics.coverage,
            "protrusion": box_result.metrics.protrusion,
            "protrusion_p95_mm": box_result.metrics.protrusion_dist_p95 * 1000.0,
            "surface_gap_p95_mm": box_result.metrics.surface_gap_p95 * 1000.0,
            "max_gap_mm": box_result.metrics.max_uncovered_gap * 1000.0,
        },
    }
    args.output_box.parent.mkdir(parents=True, exist_ok=True)
    args.output_box.write_text(json.dumps(box_output, indent=2) + "\n")
    saved_config = yaml.safe_load(args.output_config.read_text())
    spheres = saved_config.get("robot_cfg", saved_config)["kinematics"]["collision_spheres"]
    print(json.dumps({
        "robot_config": str(args.output_config),
        "robot_spheres": sum(len(rows) for rows in spheres.values()),
        "base_link_spheres": len(spheres["base_link"]),
        "box_fit": box_output,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
