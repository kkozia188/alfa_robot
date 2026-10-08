"""CPU contract/geometry tests. These do NOT substitute for nine GPU full runs."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from curobo_core.adapter import pose_matrix
from curobo_core.contracts import PlanRequest, PlannerAssets
from curobo_core.planner import CycleBlocked, FullCyclePlanner
from curobo_core.scene import AttachedObject, Pose, SceneStore, digest
from curobo_core.sequential import (
    _SequentialTask, check_frozen, completion, contact_sphere_overlaps, corners, mesh_contact_proof, overlap, pose_error, sequential_request, tool_mesh_vertices,
)
from curobo_core.sequential_validity import ReducedValidity, SequentialValidity, matmul3, obb_overlap, quaternion_matrix, world_box_overlaps, material_self_overlap
from v3_batched_loaded_search import loaded_robot
from v3_plan_cycle import summarize


class SequentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = FullCyclePlanner(PlannerAssets())
        cls.request = sequential_request(cls.planner)

    def test_frozen_support_must_leave_entire_lower_sweep_empty(self):
        from types import SimpleNamespace
        lower = self.request.snapshot.object("wall_box_17")
        center = np.array(lower.pose.position)
        center[0] -= self.request.extraction_m/2
        spheres = {"right_link4": [{"center": center.tolist(), "radius": .02}],
                   "left_link4": [{"center": center.tolist(), "radius": .02}]}
        task = object.__new__(_SequentialTask)
        task.request = self.request
        task.planner = SimpleNamespace(robot={"kinematics": {"collision_spheres": spheres}},
            fk=lambda q: None, fk_robot=SimpleNamespace(base_link="base", get_transform=lambda *args: np.eye(4)))
        hits = task.support_sweep_hits(np.zeros(15))
        self.assertEqual([hit["link"] for hit in hits], ["right_link4"])
        spheres["right_link4"][0]["center"][1] += .5
        self.assertEqual(task.support_sweep_hits(np.zeros(15)), [])

    def test_short_edges_still_check_interpolation_midpoint(self):
        from v3_batched_loaded_search import GpuValidity
        class MidpointObstacle(GpuValidity):
            def __init__(self):
                self.weights = torch.ones(1)
                self.minimum_edge_steps = 2
            def evaluate(self, values):
                return (values[:, 0]-.5).abs() > .01, torch.zeros(len(values))
        check = MidpointObstacle()
        self.assertTrue(check.mask(torch.tensor([[0.], [1.]])).all().item())
        self.assertFalse(check.edges(torch.tensor([[0.]]), torch.tensor([[1.]]), resolution=2.).item())

    def test_batched_world_boxes_match_scalar_obstacle_checks(self):
        rng = np.random.default_rng(29)
        positions = torch.tensor(rng.uniform(-1, 1, (20, 3)))
        rotations = torch.tensor(Rotation.random(20, random_state=rng).as_matrix())
        half = torch.tensor([.15, .2, .2])
        wp = torch.tensor(rng.uniform(-1, 1, (30, 3)))
        wr = torch.tensor(Rotation.random(30, random_state=rng).as_matrix())
        wh = torch.tensor(rng.uniform(.02, .5, (30, 3)))
        scalar = torch.stack([obb_overlap(positions, rotations, half, wp[i].expand_as(positions),
                            wr[i].expand_as(rotations), wh[i]) for i in range(30)]).any(0)
        torch.testing.assert_close(world_box_overlaps(positions, rotations, half, wp, wr, wh), scalar)

    def test_payload_self_roundoff_does_not_exempt_robot_or_material_overlap(self):
        spheres = torch.tensor([[[0., 0., 0., .1], [.2-5e-8, 0., 0., .1]]], dtype=torch.float64)
        pairs, padding = torch.tensor([[0, 1]]), torch.zeros(2, dtype=torch.float64)
        self.assertFalse(material_self_overlap(spheres, pairs, padding, torch.tensor([True])).item())
        self.assertTrue(material_self_overlap(spheres, pairs, padding, torch.tensor([False])).item())
        spheres[0, 1, 0] = .2-2e-6
        self.assertTrue(material_self_overlap(spheres, pairs, padding, torch.tensor([True])).item())

    def test_separated_boxes_do_not_compute_unnecessary_sat_axes(self):
        first, second = np.eye(4), np.eye(4)
        second[0, 3] = 10.
        with patch("curobo_core.sequential.np.cross", side_effect=AssertionError("unnecessary SAT cross products")):
            self.assertFalse(overlap(first, (.3, .4, .4), second, (.3, .4, .4)))
        second[0, 3] = .299
        self.assertTrue(overlap(first, (.3, .4, .4), second, (.3, .4, .4)))

    def test_small_geometry_products_preserve_batch_semantics(self):
        generator = torch.Generator().manual_seed(11)
        for shape_a, shape_b in (((3, 3), (3, 3)), ((5, 8, 3), (5, 3, 3)),
                                ((3, 3), (5, 3, 3)), ((5, 3), (3, 3))):
            a = torch.randn(shape_a, generator=generator, dtype=torch.float64)
            b = torch.randn(shape_b, generator=generator, dtype=torch.float64)
            torch.testing.assert_close(matmul3(a, b), a @ b, atol=1e-14, rtol=1e-14)

    def test_payload_tangency_tolerance_does_not_relax_robot_or_real_penetration(self):
        from types import SimpleNamespace
        from v3_batched_loaded_search import GpuValidity
        raw = torch.tensor([[[3e-8, 3e-8, 2e-6]]])
        checker = SimpleNamespace(collision_constraint=SimpleNamespace(forward=lambda state: raw.clone()))
        legacy = object.__new__(GpuValidity)
        legacy.checker = checker
        torch.testing.assert_close(legacy._world_collision_cost(None), raw, atol=0, rtol=0)
        strict = object.__new__(SequentialValidity)
        strict.geometry_tolerance = 1e-6
        strict.checker, strict.payload_indices = checker, [1, 2]
        expected = raw.clone()
        expected[..., 1] = 0.
        torch.testing.assert_close(strict._world_collision_cost(None), expected, atol=0, rtol=0)

    def test_scene_is_fixed_and_targets_are_named_pose_fk(self):
        r = self.request
        self.assertEqual(r.mode, "sequential_unload")
        self.assertEqual(len(r.snapshot.objects), 31)
        lower, upper = (r.snapshot.object(f"wall_box_{i:02d}") for i in (17, 22))
        self.assertAlmostEqual(lower.pose.position[0]-.15-self.planner.front, .75)
        self.assertEqual(upper.pose.position[1], 0.)
        self.assertAlmostEqual(upper.pose.position[2]-lower.pose.position[2], .4)
        door = r.snapshot.object("warehouse_top_door_leaf")
        self.assertAlmostEqual(door.pose.position[0]+.02, lower.pose.position[0]-.15-.015)
        self.assertAlmostEqual(door.pose.position[2]-.235, 1.88)
        for _, pose in r.targets:
            self.assertLess(pose.position[0], 0.)
        self.assertFalse(torch.cuda.is_initialized())

    def test_legacy_json_identity_and_default_mode_unchanged(self):
        old = self.planner.demo_request({"left": 24, "right": 20})
        serialized = old.to_dict()
        self.assertEqual(set(serialized), {"snapshot", "tasks", "targets"})
        self.assertEqual(PlanRequest.from_dict(serialized), old)
        self.assertEqual(old.identity, digest(serialized))
        self.assertEqual(old.mode, "dual_cycle")

    def test_sequential_elbow_preference_is_explicit_without_changing_legacy_mode(self):
        baseline = self.request.to_dict()
        self.assertEqual(baseline["support_elbow_rise_m"], .18)
        changed = replace(self.request, support_elbow_rise_m=.1)
        self.assertEqual(changed.to_dict()["support_elbow_rise_m"], .1)
        self.assertEqual(PlanRequest.from_dict(changed.to_dict()), changed)
        self.assertNotEqual(changed.identity, self.request.identity)
        self.assertNotIn("support_elbow_rise_m", self.planner.demo_request({"left":24,"right":20}).to_dict())
        for value in (-.1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                replace(self.request, support_elbow_rise_m=value)

    def test_round_trip_and_seed_identity(self):
        restored = PlanRequest.from_dict(json.loads(json.dumps(self.request.to_dict())))
        self.assertEqual(restored, self.request)
        self.assertNotEqual(replace(restored, seed=29).identity, restored.identity)
        self.assertEqual(replace(restored, seed=29).snapshot.geometry_key, restored.snapshot.geometry_key)

    def test_reject_swapped_unadjacent_or_wrong_arm_contract(self):
        for changes in ({"upper_box": 17, "lower_box": 22}, {"upper_box": 12},
                        {"support_arm": "left"}, {"seed": -1}, {"seed": True},
                        {"extraction_m": .75}, {"mode": "bad"}):
            with self.assertRaises(ValueError, msg=str(changes)):
                replace(self.request, **changes)
        with self.assertRaises(ValueError):
            replace(self.request, snapshot=replace(self.request.snapshot,
                policy=replace(self.request.snapshot.policy, exclude_task_objects_before_contact=True)))

    def test_actual_single_double_empty_attachment_models(self):
        planner, request = self.planner, self.request
        before = digest(planner.robot)
        attachments = tuple(AttachedObject(f"wall_box_{i:02d}", (.3, .4, .4), side+"_tool0", Pose((.1, .2, .3)))
                            for side, i in (("right", 22), ("left", 17)))
        original = planner.robot["kinematics"]["collision_spheres"]
        for count in range(3):
            robot = loaded_robot(planner.robot, planner.box_fit, attachments=attachments[:count])
            sphere_map = robot["kinematics"]["collision_spheres"]
            for side in ("left", "right"):
                extra = len(planner.box_fit["centers"]) if any(
                    a.parent_link == side+"_tool0" for a in attachments[:count]) else 0
                self.assertEqual(len(sphere_map[side+"_link7"]), len(original[side+"_link7"])+extra)
        self.assertEqual(digest(planner.robot), before)
        with self.assertRaises(ValueError):
            loaded_robot(planner.robot, planner.box_fit, attachments=(attachments[0], attachments[0]))

    def test_lifecycle_partial_cannot_complete_and_release_removes_one(self):
        store = SceneStore(self.request.snapshot)
        events = []
        upper, lower = "wall_box_22", "wall_box_17"
        for side, object_id in (("right", upper), ("left", lower)):
            store.attach(AttachedObject(object_id, (.3, .4, .4), side+"_tool0", Pose()), store.snapshot().revision)
            events.append({"phase": side+"_attach", "object_id": object_id, "world_pose": {}})
        store.release(lower, store.snapshot().revision)
        events.append({"phase": "left_release", "object_id": lower, "world_pose": {}})
        self.assertEqual(len(store.snapshot().attachments), 1)
        self.assertFalse(completion(events, store.snapshot(), upper, lower))
        store.release(upper, store.snapshot().revision)
        events.append({"phase": "right_release", "object_id": upper, "world_pose": {}})
        self.assertTrue(completion(events, store.snapshot(), upper, lower))
        self.assertFalse(completion(events+events[-1:], store.snapshot(), upper, lower))

    def test_support_joint_drift_rejected(self):
        start = np.zeros(15)
        rows = np.zeros((2, 15))
        rows[-1, 1] = .3
        self.assertEqual(check_frozen(rows, start, range(1, 8)), 0.)
        for index in (0, 8, 14):
            drifted = rows.copy()
            drifted[-1, index] = 2e-6
            with self.assertRaisesRegex(ValueError, "frozen joint drift"):
                check_frozen(drifted, start, range(1, 8))

    def test_door_blocks_upper_extraction_but_not_lowered_box(self):
        box = self.request.snapshot.object("wall_box_22")
        door = self.request.snapshot.object("warehouse_top_door_leaf")
        pose = pose_matrix(box.pose)
        extracted = pose.copy()
        extracted[0, 3] -= .1
        self.assertFalse(overlap(pose, box.dimensions_m, pose_matrix(door.pose), door.dimensions_m))
        self.assertTrue(overlap(extracted, box.dimensions_m, pose_matrix(door.pose), door.dimensions_m))
        extracted[2, 3] -= .4
        self.assertFalse(overlap(extracted, box.dimensions_m, pose_matrix(door.pose), door.dimensions_m))

    def test_attached_box_neighbour_collision_and_tangent_distinct(self):
        a = pose_matrix(self.request.snapshot.object("wall_box_17").pose)
        b = a.copy()
        b[1, 3] += .4
        dims = (.3, .4, .4)
        self.assertFalse(overlap(a, dims, b, dims))
        b[1, 3] -= .001
        self.assertTrue(overlap(a, dims, b, dims))

    def test_tensor_sat_agrees_with_independent_corner_projection(self):
        random = np.random.default_rng(11)
        for _ in range(100):
            a, b = np.eye(4), np.eye(4)
            a[:3, :3] = Rotation.random(random_state=random).as_matrix()
            b[:3, :3] = Rotation.random(random_state=random).as_matrix()
            b[:3, 3] = random.uniform(-.5, .5, 3)
            da, db = random.uniform(.1, .8, (2, 3))
            actual = obb_overlap(*(torch.tensor(x) for x in
                (a[:3, 3], a[:3, :3], da/2, b[:3, 3], b[:3, :3], db/2))).item()
            self.assertEqual(actual, overlap(a, da, b, db))
        q = torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0]])
        np.testing.assert_allclose(quaternion_matrix(q).numpy(), [np.eye(3), np.diag([1, -1, -1])])

    def test_reduced_search_expands_seven_and_eight_dimensions(self):
        from types import SimpleNamespace
        full = SimpleNamespace(weights=torch.tensor([5.]+[1.]*14), stability_weight=10.)
        reference = np.arange(15)/20
        original = torch.as_tensor
        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original(*args, **kwargs)
        with patch("curobo_core.sequential_validity.torch.as_tensor", side_effect=cpu_tensor):
            for active in (list(range(1, 8)), [0]+list(range(8, 15))):
                reduced = ReducedValidity(full, reference, active)
                candidates = torch.zeros((3, len(active)))
                expanded = reduced.expand(candidates).numpy()
                self.assertEqual(expanded.shape, (3, 15))
                self.assertEqual(len(reduced.joint_names), len(active))
                self.assertEqual("updown" in reduced.joint_names, 0 in active)
                self.assertLessEqual(check_frozen(expanded, reference, active), 1e-6)
                np.testing.assert_equal(expanded[:, active], 0.)
        # Exercise the batch shape used by native payload checks, not only scalar SAT.
        position = torch.tensor([[0., 0., 0.], [2., 0., 0.]])
        rotation = torch.eye(3).expand(2, 3, 3)
        result = obb_overlap(position, rotation, torch.ones(3)/2,
                             torch.zeros_like(position), rotation, torch.ones(3)/2)
        self.assertEqual(result.tolist(), [True, False])

    def test_unreachable_target_is_failure_at_place_not_completion(self):
        task = object.__new__(_SequentialTask)
        task.request = replace(self.request, targets=(("left_tool0", Pose((100., 0., 0.))), self.request.targets[1]))
        task.progress = lambda text: None
        task.Blocked = CycleBlocked
        task.report = {"success": False}
        task.ik = lambda side, target, active: np.empty((0, 15))
        with self.assertRaises(CycleBlocked) as caught:
            task.place("left", list(range(1, 8)))
        self.assertEqual(caught.exception.stage, "left_place")
        self.assertFalse(task.report["success"])

    def test_contact_permission_only_masks_certified_tool_target_pair(self):
        from types import SimpleNamespace
        from v3_batched_loaded_search import GpuValidity
        from v3_wall_ik_benchmark import canonical_side_suction_quaternion_wxyz
        item = self.request.snapshot.object("wall_box_22")
        position = np.array(item.pose.position)-np.array([.15, 0., 0.])
        tool = SimpleNamespace(position=torch.tensor(position)[None],
            quaternion=torch.tensor(canonical_side_suction_quaternion_wxyz("right"))[None])
        spheres = torch.tensor([[[*list(position-np.array([.05, 0, 0])), .075], [-10., 0., 1., .01]]])
        fk = SimpleNamespace(robot_spheres=spheres,
                             tool_poses=SimpleNamespace(to_dict=lambda: {"right_tool0": tool}))
        validity = object.__new__(SequentialValidity)
        validity.geometry_tolerance = 1e-6
        validity.support_pose = None
        def compute(q):
            from curobo_core.contracts import ACTIVE_JOINTS
            self.assertEqual(q.joint_names, list(ACTIVE_JOINTS))
            return fk
        validity.checker = SimpleNamespace(kinematics=SimpleNamespace(compute_kinematics=compute))
        validity.snapshot, validity.attachments = self.request.snapshot, ()
        validity.contact_pair, validity.collision_ms = ("right", "wall_box_22"), 0.
        validity.tool_meshes = {"right": tool_mesh_vertices(self.planner, "right")}
        validity.robot_indices, validity.link_indices = [0, 1], {"right_link7": [0], "left_link7": [1]}
        q = torch.zeros((1, 15), dtype=torch.float64)
        # Stub only the native constraints; exercise the real pair-scoped
        # supplemental check on CPU. This is not native GPU validation.
        with patch.object(GpuValidity, "evaluate", return_value=(torch.ones(1, dtype=torch.bool), torch.zeros(1))), \
             patch("torch.cuda.synchronize"):
            self.assertTrue(validity.mask(q).item())
        spheres[0, 1, :3] = torch.tensor(item.pose.position)
        with patch.object(GpuValidity, "evaluate", return_value=(torch.ones(1, dtype=torch.bool), torch.zeros(1))), \
             patch("torch.cuda.synchronize"):
            self.assertFalse(validity.mask(q).item(), "other robot parts must still hit this target")
        spheres[0, 1, 0] = -10.
        tool.position[0, 0] += .002
        with patch.object(GpuValidity, "evaluate", return_value=(torch.ones(1, dtype=torch.bool), torch.zeros(1))), \
             patch("torch.cuda.synchronize"):
            self.assertFalse(validity.mask(q).item(), "actual tool mesh penetration must not be exempted")

    def test_exact_tool_mesh_contact_proof_without_radius_changes(self):
        from v3_wall_ik_benchmark import canonical_side_suction_quaternion_wxyz
        for side in ("left", "right"):
            box_to_tool = pose_matrix(Pose((-.15, 0., 0.), tuple(canonical_side_suction_quaternion_wxyz(side))))
            vertices = tool_mesh_vertices(self.planner, side)
            proof = mesh_contact_proof(vertices, box_to_tool, (.3, .4, .4), .001)
            self.assertTrue(proof["valid"])
            self.assertLess(abs(proof["mesh_front_gap_m"]), 1e-6)
            penetrating = box_to_tool.copy()
            penetrating[0, 3] += .002
            self.assertFalse(mesh_contact_proof(vertices, penetrating, (.3, .4, .4))["valid"])
            off_face = box_to_tool.copy()
            off_face[1, 3] += .1
            self.assertFalse(mesh_contact_proof(vertices, off_face, (.3, .4, .4))["valid"])

    def test_uncertifiable_contact_fails_explicitly_without_attachment(self):
        hits = contact_sphere_overlaps(self.planner, "right", self.request.snapshot.object("wall_box_22"))
        self.assertEqual(len(hits), 8)
        self.assertGreater(max(hit["overlap_m"] for hit in hits), .025)
        task = object.__new__(_SequentialTask)
        task.planner, task.request = self.planner, self.request
        task.store, task.progress = SceneStore(self.request.snapshot), lambda text: None
        task.Blocked, task.report = CycleBlocked, {"success": False}
        from curobo_core.contracts import ACTIVE_JOINTS
        task.q = np.array(self.request.snapshot.state.ordered(ACTIVE_JOINTS))
        with patch("curobo_core.sequential.mesh_contact_proof", return_value={"valid": False}), \
             self.assertRaises(CycleBlocked) as caught:
            task.contact("right", "wall_box_22", [0]+list(range(8, 15)))
        self.assertEqual(caught.exception.stage, "right_contact_model")
        self.assertEqual(task.report["failure_code"], "CONTACT_MODEL_UNREPRESENTABLE")
        self.assertEqual(task.store.snapshot().attachments, ())
        self.assertFalse(task.report["success"])

    def test_core_dispatch_preserves_structured_contact_failure_and_empty_lifecycle(self):
        # CPU dispatch contract only: native validation is explicitly stubbed out.
        planner = FullCyclePlanner(PlannerAssets())
        request = sequential_request(planner)
        class Validity:
            def mask(self, rows):
                return torch.ones(len(rows), dtype=torch.bool)
        with patch.object(_SequentialTask, "audit"), \
             patch.object(_SequentialTask, "validity", return_value=Validity()), \
             patch.object(_SequentialTask, "tensor", side_effect=lambda value: torch.as_tensor(np.asarray(value))), \
             patch("torch.cuda.synchronize"), \
             patch("curobo_core.sequential.mesh_contact_proof", return_value={"valid": False}):
            result = planner.plan_request(request)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["code"], "CONTACT_MODEL_UNREPRESENTABLE")
        self.assertEqual(result["error"]["stage"], "right_contact_model")
        self.assertEqual(result["predicted_scene_events"], [])
        self.assertEqual(result["final_snapshot"]["attachments"], ())
        self.assertEqual(len(result["joint_names"]), 15)
        self.assertEqual(result["request_id"], request.identity)
        self.assertFalse(torch.cuda.is_initialized())

    def test_analytic_single_arm_path_preserves_other_variables(self):
        # Real upstream bridge/FK, CPU collision stub; not a GPU collision test.
        planner = self.planner
        from curobo_core.contracts import ACTIVE_JOINTS
        q = np.array([planner.home[name] for name in ACTIVE_JOINTS])
        class Validity:
            def mask(self, rows):
                return torch.ones(len(rows), dtype=torch.bool)
            def edges(self, starts, ends, resolution):
                return torch.ones(len(starts), dtype=torch.bool)
        original_tensor = torch.tensor
        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)
        with patch("curobo_core.planner.torch.tensor", side_effect=cpu_tensor):
            rows, details = planner.analytic_extract(q, Validity(), Validity(), sides=("left",),
                                                      direction=(-1., 0., 0.), distance_m=.01)
        self.assertIsNotNone(rows, details)
        self.assertLessEqual(check_frozen(rows, q, range(1, 8)), 1e-6)
        planner.fk(q)
        start = planner.fk_robot.get_transform("left_tool0", planner.fk_robot.base_link).copy()
        planner.fk(rows[-1])
        end = planner.fk_robot.get_transform("left_tool0", planner.fk_robot.base_link).copy()
        np.testing.assert_allclose(end[:3, 3]-start[:3, 3], (-.01, 0., 0.), atol=1e-6)

    def test_default_cartesian_matches_f044_bridge(self):
        import importlib.util
        reference = ROOT / "artifacts/sequential-unload/baseline-restore-20261007/f044-planner.py"
        spec = importlib.util.spec_from_file_location("curobo_core.f044_cartesian_reference", reference)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        q = self.planner.home_values.copy()
        class Validity:
            def mask(self, rows):
                return torch.ones(len(rows), dtype=torch.bool)
        original_tensor = torch.tensor
        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)
        validity = Validity()
        with patch("curobo_core.planner.torch.tensor", side_effect=cpu_tensor):
            with patch.object(self.planner.analytic, "solve", wraps=self.planner.analytic.solve) as solve:
                expected, _ = module.FullCyclePlanner.analytic_extract(self.planner, q, validity, validity)
                original_calls = list(solve.call_args_list)
                solve.reset_mock()
                actual, _ = self.planner.analytic_extract(q, validity, validity)
                restored_calls = list(solve.call_args_list)
        self.assertGreater(len(original_calls), 0)
        self.assertEqual(len(original_calls), len(restored_calls))
        for old, new in zip(original_calls, restored_calls):
            for a, b in zip(old.args, new.args):
                np.testing.assert_array_equal(a, b)
        if expected is None:
            self.assertIsNone(actual)
        else:
            np.testing.assert_array_equal(expected, actual)

    def test_right_extraction_uses_only_needed_local_swivel_continuation(self):
        recorded = ROOT / "artifacts/sequential-unload/elbow-profile-20261007/optimized-nine-runs-s11-r1.json"
        if not recorded.exists():
            self.skipTest("requires recorded successful path")
        result = json.loads(recorded.read_text())["result"]
        rows = np.asarray([q for q, phase in zip(result["frames"], result["phases"])
                           if phase == "right_extract"])
        class Validity:
            def mask(self, values):
                return torch.ones(len(values), dtype=torch.bool)
        original_tensor = torch.tensor
        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)
        self.planner.fk(rows[-1])
        end_psi = self.planner.analytic.swivel(1, rows[-1, 8:15])
        with patch("curobo_core.planner.torch.tensor", side_effect=cpu_tensor):
            fixed, _ = self.planner.analytic_extract(rows[0], Validity(), Validity(),
                sides=("right",), direction=(-1., 0., 0.), distance_m=.36)
            interpolated, details = self.planner.analytic_extract(rows[0], Validity(), Validity(),
                sides=("right",), direction=(-1., 0., 0.), distance_m=.36, swivel_continuation=True)
        self.assertIsNone(fixed)
        self.assertIsNotNone(interpolated, details)
        self.assertEqual(details["cartesian_step_m"], .01)
        self.assertIsNone(details["fixed_psi_rad"])
        self.assertGreater(details["swivel_continuation_trials"], 0)
        self.assertEqual(details["swivel_continuation_candidate_cap_per_failed_segment"], 14)
        self.assertLess(details["max_line_error_mm"], 2.)
        self.assertLess(details["max_orientation_error_deg"], 1.)

    def test_nine_run_acceptance_rejects_any_partial_or_missing_run(self):
        rows = [{"success": True, "measured_wall_ms": float(i)} for i in range(9)]
        self.assertTrue(summarize(rows, 9)["passed"])
        self.assertFalse(summarize(rows[:8], 9)["passed"])
        rows[3]["success"] = False
        self.assertFalse(summarize(rows, 9)["passed"])
        self.assertEqual(summarize(rows, 9)["median_ms"], 4.)
        self.assertEqual(summarize(rows, 9)["worst_ms"], 8.)


if __name__ == "__main__":
    unittest.main()
