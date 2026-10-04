"""GPU integration checks for independent snapshots and planning results."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

from curobo_core.contracts import PlannerAssets
from curobo_core.planner import FullCyclePlanner
from curobo_core.scene import Pose, SceneObject, SceneStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets = PlannerAssets()
    planner = FullCyclePlanner(assets)
    task = {"left": 24, "right": 20}
    checker = planner.checker(task)
    solver = planner.ik_solver(task)
    assert planner.checker(task) is checker
    assert planner.ik_solver(task) is solver
    store = SceneStore(planner.snapshot)
    obstacle = SceneObject("test_conveyor", (0.8, 3.0, 1.0), Pose((-8.0, 0.0, 0.5)))
    moved_scene = store.upsert(obstacle, store.snapshot().revision)
    planner.set_snapshot(moved_scene)
    changed_checker = planner.checker(task)
    changed_solver = planner.ik_solver(task)
    assert changed_checker is not checker
    assert changed_solver is not solver
    del checker, solver

    before = changed_checker
    planner.box_fit = json.loads(json.dumps(planner.box_fit))
    planner.box_fit["parameters"]["cache_test"] = True
    after = planner.checker(task)
    assert before is not after
    del before, changed_checker, changed_solver
    planner.box_fit["parameters"].pop("cache_test")
    planner.set_snapshot(replace(planner.snapshot, objects=tuple(
        item for item in planner.snapshot.objects if item.object_id != "test_conveyor")))
    initial_snapshot = planner.snapshot
    request = planner.demo_request(task)
    result = planner.plan_request(request, lambda message: print(message, file=sys.stderr, flush=True))
    assert result["success"], result.get("error")
    assert len(result["predicted_scene_events"]) == 2
    assert len(initial_snapshot.attachments) == 0
    assert result["scene_id"] == request.snapshot.identity
    frames = np.asarray(result["frames"])
    assert np.isfinite(frames).all()
    assert len(frames) == len(result["phases"]) == len(result["payload"])
    assert np.max(np.abs(frames[0] - frames[-1])) < 1e-7
    assert result["max_adjacent_joint_step_deg"] <= 0.50001
    for event in result["predicted_scene_events"]:
        state_names = set(event["snapshot"]["state"]["joint_names"])
        assert set(initial_snapshot.state.joint_names) <= state_names
        names = {item["object_id"] for item in event["snapshot"]["objects"]}
        assert not {"wall_box_24", "wall_box_20"} & names
    stale_store = SceneStore(initial_snapshot)
    targets = dict(request.targets)
    modified = False

    def invalidate(message):
        nonlocal modified
        if not modified:
            stale_store.upsert(obstacle, stale_store.snapshot().revision)
            modified = True
        print(message, file=sys.stderr, flush=True)

    stale_result = planner.plan_from_store(stale_store, task, targets, invalidate)
    assert not stale_result["success"]
    assert stale_result["error"]["code"] == "SCENE_CHANGED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "gpu_cache_reused": True, "scene_update_rebuilt_cache": True,
        "payload_update_rebuilt_cache": True, "stale_plan_rejected": True,
        "core_did_not_load_viser": "viser" not in sys.modules,
        "success_result": result, "stale_result": stale_result,
        "cuda_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"success": True, "total_ms": result["total_ms"],
                      "stale_code": stale_result["error"]["code"],
                      "viser_loaded": "viser" in sys.modules}))


if __name__ == "__main__":
    main()
