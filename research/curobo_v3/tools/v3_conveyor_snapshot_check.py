import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from curobo_core.adapter import to_curobo_scene
from curobo_core.conveyor import conveyor_pair
from curobo_core.inspection import SnapshotInspector
from curobo_core.instances import LiveScene
from curobo_core.scene import Pose, SceneSnapshot


def main():
    parser = argparse.ArgumentParser(description="cuRobo frozen conveyor mesh integration check")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspector = SnapshotInspector()
    template = inspector.backend.snapshot
    live = LiveScene(template)
    pair = conveyor_pair(stamp_ns=time.time_ns(), pose=Pose((-4, 0, 0)))
    live.update_state(replace(template.state, stamp_ns=pair.stamp_ns))
    live.upsert(pair)
    far = live.capture(pair.stamp_ns, 1)
    converted = to_curobo_scene(far, mobile=True)
    assert len(converted.mesh) == 2
    assert not {item.name for item in converted.cuboid} & {item.name for item in converted.mesh}
    far_result = inspector.check(far)
    assert far_result["valid"] and not far_result["environment_collision"]
    solver = inspector.backend.ik_solver({})
    assert inspector.backend.ik_solver({}) is solver
    cached = inspector.backend.collision_checker({}, mobile=True)
    live.upsert(replace(pair, pose=Pose((0, 0, 0))))
    near = live.capture(pair.stamp_ns, 1)
    assert near.geometry_key != far.geometry_key
    assert inspector.check(far)["valid"]
    assert inspector.backend.collision_checker({}, mobile=True) is cached
    near_result = inspector.check(near)
    assert not near_result["valid"] and near_result["environment_collision"]
    assert inspector.backend.collision_checker({}, mobile=True) is not cached
    assert inspector.backend.ik_solver({}) is not solver
    disabled = live.capture(pair.stamp_ns, 1, instance_ids=())
    disabled_result = inspector.check(disabled)
    assert disabled_result["valid"] and not disabled_result["environment_collision"]
    assert not disabled.meshes
    assert SceneSnapshot.from_dict(json.loads(json.dumps(near.to_dict()))) == near
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"success": True, "far": far_result, "near": near_result,
                                     "disabled": disabled_result, "snapshot_isolation": True,
                                     "cache_reused_for_frozen_snapshot": True,
                                     "cache_rebuilt_for_changed_mesh_pose": True,
                                     "ik_solver_accepts_mesh_snapshot": True,
                                     "mesh": {"triangle_count_each": len(pair.parts[0].faces),
                                              "units": "m", "gap_m": 0.05}}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"success": True, "far_valid": far_result["valid"],
                      "near_environment_collision": near_result["environment_collision"],
                      "disabled_valid": disabled_result["valid"]}))


if __name__ == "__main__":
    main()
