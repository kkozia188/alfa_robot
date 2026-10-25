#!/usr/bin/env python3
"""Audit sphere labels on actual newly searched task-path states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin


def geometry(model, urdf: Path, srdf: Path):
    result = pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.COLLISION)
    result.addAllCollisionPairs()
    pin.removeCollisionPairs(model, result, str(srdf))
    return result, pin.GeometryData(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("mesh_urdf", type=Path)
    parser.add_argument("sphere_urdf", type=Path)
    parser.add_argument("srdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-states", type=int, default=5000)
    args = parser.parse_args()
    model = pin.buildModelFromUrdf(str(args.mesh_urdf))
    mesh_model, mesh_data = geometry(model, args.mesh_urdf, args.srdf)
    sphere_model, sphere_data = geometry(model, args.sphere_urdf, args.srdf)
    model_data = model.createData()
    trials = json.loads((args.experiment_dir / "trials.json").read_text())["trials"]
    selected = {}
    for trial in trials:
        box_id = int(trial["box_id"])
        if trial.get("success") and box_id not in selected:
            selected[box_id] = trial
    raw_states = []
    for box_id in sorted(selected):
        payload = json.loads(Path(selected[box_id]["result_json"]).read_text())
        benchmark = payload["paper_benchmark"]
        for segment in benchmark["selected_segments"]:
            raw_states.extend(segment["ompl"]["replay_states"])
            raw_states.extend(segment["vamp"]["replay_states"])
    stride = max(1, len(raw_states) // args.maximum_states)
    states = raw_states[::stride][:args.maximum_states]
    counts = {"both_clear": 0, "both_collision": 0,
              "sphere_false_positive": 0, "sphere_false_negative": 0}
    for values in states:
        state = np.asarray(values, dtype=float)
        mesh_collision = pin.computeCollisions(
            model, model_data, mesh_model, mesh_data, state, True
        )
        sphere_collision = pin.computeCollisions(
            model, model_data, sphere_model, sphere_data, state, True
        )
        if mesh_collision and sphere_collision:
            counts["both_collision"] += 1
        elif not mesh_collision and not sphere_collision:
            counts["both_clear"] += 1
        elif sphere_collision:
            counts["sphere_false_positive"] += 1
        else:
            counts["sphere_false_negative"] += 1
    result = {
        "schema": "alfa.v322.sphere_task_state_audit.v1",
        "boxes": sorted(selected), "source_state_count": len(raw_states),
        "audited_states": len(states), "stride": stride, "counts": counts,
        "rates": {key: value / len(states) for key, value in counts.items()},
        "historical_path_used": False,
        "state_source": "new OMPL and VAMP paths from shared-IK paired experiment",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
