from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from curobo_core.adapter import world_meshes
from curobo_core.instances import LiveScene, SceneInstance
from curobo_core.scene import MeshObject, Pose, RobotState, SceneObject, SceneSnapshot


class InstanceTest(unittest.TestCase):
    def setUp(self):
        self.mesh = MeshObject("left", ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
                               ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)))
        self.state = RobotState(("updown",), (0.0,), 100)
        self.template = SceneSnapshot("model", self.state, ())
        self.live = LiveScene(self.template)
        self.instance = SceneInstance("pair", Pose((-2, 0, 0)), 100,
                                      (self.mesh, replace(self.mesh, object_id="right", pose=Pose((0, -1, 0)))))
        self.live.upsert(self.instance)

    def test_live_moves_without_mutating_capture(self):
        frozen = self.live.capture(100, 1)
        self.live.upsert(replace(self.instance, pose=Pose((-1, 0, 0)), stamp_ns=101))
        updated = self.live.capture(101, 1)
        self.assertEqual(frozen.meshes[0].pose.position, (-2, 0, 0))
        self.assertEqual(updated.meshes[0].pose.position, (-1, 0, 0))
        self.assertNotEqual(frozen.geometry_key, updated.geometry_key)
        self.assertEqual(len(self.template.meshes), 0)

    def test_robot_yaw_never_changes_world_instance(self):
        yaw = Pose((0, 0, 0), (math.sqrt(0.5), 0, 0, math.sqrt(0.5)))
        self.live.update_state(replace(self.state, base_pose=yaw))
        frozen = self.live.capture(100, 1)
        self.assertEqual(frozen.meshes[0].pose.position, (-2, 0, 0))
        local = world_meshes(frozen)
        self.assertTrue(np.allclose(local[0].pose.position, (0, 2, 0), atol=1e-12))
        self.assertEqual(world_meshes(frozen, mobile=True), frozen.meshes)

    def test_pair_parts_capture_atomically_and_rotate_about_own_root(self):
        rotated = replace(self.instance, pose=Pose((0, 0, 0), (math.sqrt(0.5), 0, 0, math.sqrt(0.5))))
        self.live.upsert(rotated)
        frozen = self.live.capture(100, 1)
        self.assertEqual([item.object_id for item in frozen.meshes], ["pair/left", "pair/right"])
        self.assertTrue(np.allclose(frozen.meshes[1].pose.position, (1, 0, 0), atol=1e-12))

    def test_display_only_and_disabled_instances_do_not_enter_planner(self):
        self.live.upsert(replace(self.instance, collision_enabled=False))
        self.assertEqual(len(self.live.view()[1]), 1)
        self.assertEqual(self.live.capture(100, 1).meshes, ())
        self.live.upsert(self.instance)
        self.assertEqual(self.live.capture(100, 1, instance_ids=()).meshes, ())

    def test_payload_geometry_on_instance_is_also_captured(self):
        cargo = SceneObject("cargo", (0.4, 0.4, 0.4), Pose((0, 0, 1)))
        self.live.upsert(replace(self.instance, parts=self.instance.parts + (cargo,)))
        frozen = self.live.capture(100, 1)
        self.assertEqual(frozen.objects[0].object_id, "pair/cargo")
        self.assertEqual(frozen.objects[0].pose.position, (-2, 0, 1))

    def test_missing_stale_future_or_unresolved_pose_rejects_capture(self):
        for action in (lambda: self.live.capture(2_000_000_000, 1),
                       lambda: self.live.capture(99, 1),
                       lambda: self.live.capture(100, 1, ("missing",))):
            with self.assertRaises(ValueError):
                action()
        self.live.upsert(replace(self.instance, frame_id="unknown"))
        with self.assertRaisesRegex(ValueError, "transform missing"):
            self.live.capture(100, 1)

    def test_mesh_roundtrip_freezes_geometry_and_identity(self):
        frozen = self.live.capture(100, 1)
        self.assertEqual(SceneSnapshot.from_dict(json.loads(json.dumps(frozen.to_dict()))), frozen)
        with self.assertRaises(ValueError):
            replace(self.mesh, faces=((0, 1, 8),))
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        copied = replace(self.mesh, vertices_m=vertices)
        vertices[0][0] = 5
        self.assertEqual(copied.vertices_m[0], (0, 0, 0))
        with self.assertRaises(ValueError):
            replace(frozen, objects=(SceneObject("pair/left", (1, 1, 1), Pose()),))


if __name__ == "__main__":
    unittest.main()
