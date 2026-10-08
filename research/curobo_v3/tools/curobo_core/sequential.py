"""Sequential two-box mode inside FullCyclePlanner (simulation geometry only)."""
from dataclasses import asdict, replace
import itertools
import math
import time
import weakref

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .adapter import matrix_pose, pose_matrix
from .contracts import ACTIVE_JOINTS, PlanRequest
from .scene import AttachedObject, CollisionPolicy, Pose, SceneObject, SceneStore


CORNERS = np.array(list(itertools.product((-1., 1.), repeat=3)))


def corners(matrix, dimensions):
    return (CORNERS * np.asarray(dimensions)/2) @ matrix[:3, :3].T + matrix[:3, 3]


def overlap(a, dimensions_a, b, dimensions_b, tolerance=1e-6):
    """Independent CPU SAT audit of actual boxes, not their inscribed spheres."""
    ca, cb = corners(a, dimensions_a), corners(b, dimensions_b)
    if np.any(np.minimum(ca.max(axis=0), cb.max(axis=0)) -
              np.maximum(ca.min(axis=0), cb.min(axis=0)) <= tolerance):
        return False
    # The existing exact broad-phase test already proves separation. Compute
    # SAT axes only for pairs that actually need them; keep the same gate.
    axes = [*a[:3, :3].T, *b[:3, :3].T]
    axes += [np.cross(x, y) for x in a[:3, :3].T for y in b[:3, :3].T]
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-8:
            continue
        pa, pb = ca @ (axis/length), cb @ (axis/length)
        if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= tolerance:
            return False
    return True


def pose_error(actual, target):
    return (float(np.linalg.norm(actual[:3, 3]-target[:3, 3])),
            float(Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude()))


def check_frozen(rows, reference, active):
    frozen = sorted(set(range(15))-set(active))
    error = float(np.max(np.abs(np.asarray(rows)[:, frozen]-np.asarray(reference)[frozen]))) if frozen else 0.
    if error > 1e-6:
        raise ValueError(f"frozen joint drift {error:.9g} > 1e-6")
    return error


def completion(events, snapshot, upper_id, lower_id):
    released = [event["object_id"] for event in events if event["phase"].endswith("_release")]
    attached = [event["object_id"] for event in events if event["phase"].endswith("_attach")]
    expected = [lower_id, upper_id]
    return (released == expected and attached == expected[::-1] and not snapshot.attachments
            and not {upper_id, lower_id}.intersection(item.object_id for item in snapshot.objects)
            and all("world_pose" in event for event in events))


def sequential_request(planner, seed=11):
    """Freeze original C3 5x5/door semantics at this model's chassis front + .75m."""
    from .adapter import from_curobo_scene
    from v3_wall_ik_benchmark import make_scene

    front = planner.front + 0.75
    objects = [item for item in from_curobo_scene(make_scene(planner.front, .75, set()))
               if not item.object_id.startswith("wall_box_")]
    for row in range(5):
        for column in range(5):
            objects.append(SceneObject(f"wall_box_{row*5+column:02d}", (.3, .4, .4),
                                       Pose((front+.15, (column-2)*.4, .2+row*.4))))
    objects.append(SceneObject("warehouse_top_door_leaf", (.04, 2., .47),
                               Pose((front-.035, 0., 2.115))))
    snapshot = replace(planner.snapshot, objects=tuple(objects), attachments=(), revision=0,
                       state=planner.snapshot.state.with_positions(planner.home),
                       policy=CollisionPolicy(False, False, max_box_tilt_deg=89.))
    named = yaml.safe_load(planner.args.named_poses.read_text())["named_poses"]["unloading"]
    planner.fk([named[name] for name in ACTIVE_JOINTS])
    targets = tuple((side+"_tool0", matrix_pose(planner.fk_robot.get_transform(
        side+"_tool0", planner.fk_robot.base_link))) for side in ("left", "right"))
    return PlanRequest(snapshot, (("left", 17), ("right", 22)), targets,
                       mode="sequential_unload", upper_box=22, lower_box=17, seed=seed,
                       support_elbow_rise_m=.18, cartesian_geometry_tolerance_m=.0001)


def contact_sphere_overlaps(planner, side, item):
    """Pose-independent tool/box check for this mode's centered front contact.

    Only the fixed link7-to-tool transform is used; no IK or historical path.
    This detects a model representation blocker, not physical mesh penetration.
    """
    from v3_wall_ik_benchmark import canonical_side_suction_quaternion_wxyz

    planner.fk([planner.home[name] for name in ACTIVE_JOINTS])
    fk = planner.fk_robot
    link_to_tool = np.linalg.inv(fk.get_transform(side+"_link7", fk.base_link)) @ fk.get_transform(
        side+"_tool0", fk.base_link)
    target = pose_matrix(Pose((item.pose.position[0]-item.dimensions_m[0]/2, *item.pose.position[1:]),
                             tuple(canonical_side_suction_quaternion_wxyz(side))))
    box_to_link = np.linalg.inv(pose_matrix(item.pose)) @ target @ np.linalg.inv(link_to_tool)
    overlaps = []
    kin = planner.robot.get("robot_cfg", planner.robot)["kinematics"]
    for index, sphere in enumerate(kin["collision_spheres"][side+"_link7"]):
        center = (box_to_link @ np.r_[sphere["center"], 1.])[:3]
        distance = np.linalg.norm(np.maximum(np.abs(center)-np.array(item.dimensions_m)/2, 0))
        if sphere["radius"] > 0 and distance < sphere["radius"]-1e-6:
            overlaps.append({"side": side, "object_id": item.object_id, "sphere_index": index,
                             "overlap_m": float(sphere["radius"]-distance),
                             "box_local_center": center.tolist(), "radius_m": sphere["radius"]})
    return overlaps


def tool_mesh_vertices(planner, side):
    """Original collision mesh expressed in Tool0; never refit or alter it."""
    import xml.etree.ElementTree as ET
    import trimesh
    from v3_wall_ik_benchmark import transform

    cache = getattr(planner, "_contact_mesh_vertices", {})
    if side in cache:
        return cache[side]
    planner.fk([planner.home[name] for name in ACTIVE_JOINTS])
    fk = planner.fk_robot
    tool_to_link = np.linalg.inv(fk.get_transform(side+"_tool0", fk.base_link)) @ fk.get_transform(
        side+"_link7", fk.base_link)
    link = ET.parse(planner.args.urdf).getroot().find(f"link[@name='{side}_link7']")
    output = []
    for collision in link.findall("collision"):
        mesh = collision.find("geometry/mesh")
        if mesh is None:
            raise ValueError("contact proof requires the unchanged tool collision mesh")
        origin = collision.find("origin")
        matrix = tool_to_link @ transform(origin.get("xyz") if origin is not None else None,
                                          origin.get("rpy") if origin is not None else None)
        vertices = np.asarray(trimesh.load(mesh.get("filename"), force="mesh", process=False).vertices)
        vertices = vertices * np.array([float(value) for value in mesh.get("scale", "1 1 1").split()])
        output.append((matrix @ np.c_[vertices, np.ones(len(vertices))].T).T[:, :3])
    if not output:
        raise ValueError("no tool collision mesh; cannot certify a contact exemption")
    # The convex hull vertices suffice for all plane/footprint extrema, without
    # approximating inward or changing geometry. Full vertices work here as well.
    vertices = np.unique(np.concatenate(output), axis=0)
    from scipy.spatial import ConvexHull
    vertices = vertices[ConvexHull(vertices).vertices]
    cache[side] = vertices
    planner._contact_mesh_vertices = cache
    return vertices


def mesh_contact_proof(vertices_tool, box_to_tool, dimensions, maximum_gap=.051, tolerance=1e-6):
    """Half-space proof for every triangle, plus an on-face contact footprint.

    All triangle vertices outside the front plane implies no triangle can
    penetrate the solid box. This is stronger than testing sampled mesh points.
    """
    vertices = vertices_tool @ box_to_tool[:3, :3].T + box_to_tool[:3, 3]
    half = np.asarray(dimensions)/2
    front = float(vertices[:, 0].max())
    gap = -half[0]-front
    footprint = vertices[vertices[:, 0] >= front-.001, 1:]
    return {"valid": bool(-tolerance <= gap <= maximum_gap
                           and np.all(np.abs(footprint) <= half[1:]+1e-6)),
            "mesh_front_gap_m": gap, "contact_footprint_vertices": len(footprint)}


class _SequentialTask:
    """Per-request bookkeeping, not a second planner or ROS orchestration layer."""
    def __init__(self, planner, request, progress):
        import torch
        from .planner import CycleBlocked
        self.torch, self.Blocked = torch, CycleBlocked
        self.planner, self.request, self.progress = planner, request, progress
        self.store = SceneStore(request.snapshot)
        self.q = np.array(request.snapshot.state.ordered(ACTIVE_JOINTS), dtype=float)
        self.started = time.perf_counter()
        self.counter = 0
        self.phase = "initialize"
        self.stage_started = self.started
        self.stage_collision_checkpoint = 0.
        self.cached_key, self.cached_validity = None, None
        self.validities = []
        self.support_pose = None
        self.released_offsets = {}
        self.contact_pair = None
        self.pending_extract = None
        self.report = {"success": False, "mode": request.mode, "task": dict(request.tasks),
                       "planner": "rrtconnect", "joint_names": list(ACTIVE_JOINTS),
                       "frames": [], "phases": [], "payload": [], "attachments_by_frame": [],
                       "predicted_scene_events": [], "stage_audits": [], "attempts": [],
                       "timing_ms": {"ik": 0., "search": 0., "cartesian": 0.},
                       "stage_timing_ms": {}, "stage_breakdown_ms": {}, "stage_timing_spans": [],
                       "stage_timing_schema": 1,
                       "seed": request.seed, "time_parameterized": False,
                       "support_joint_freeze_required": False,
                       "cartesian_strategy": "f044_fixed_swivel_local_continuation",
                       "cartesian_geometry_tolerance_m": request.cartesian_geometry_tolerance_m,
                       "contact_policy": "exact tool-mesh half-space proof; per-tool/per-box/per-stage only",
                       "scene_geometry_key": request.snapshot.geometry_key}
        planner.last_partial = self.report

    def fail(self, details):
        raise self.Blocked(self.phase, details, self.report)

    def seed(self, operation):
        value = (self.request.seed + self.counter) % 2**32
        self.counter += 1
        self.report["attempts"].append({"stage": self.phase, "operation": operation, "seed": value})
        return value

    def tensor(self, values):
        return self.torch.as_tensor(np.asarray(values), device="cuda", dtype=self.torch.float32)

    def tool(self, q, side):
        self.planner.fk(q)
        return self.planner.fk_robot.get_transform(side+"_tool0", self.planner.fk_robot.base_link).copy()

    def box_pose(self, q, attachment):
        return self.tool(q, attachment.parent_link.removesuffix("_tool0")) @ pose_matrix(attachment.tool_to_object)

    def close_stage_timing(self):
        self.torch.cuda.synchronize()
        now = time.perf_counter()
        elapsed = (now-self.stage_started)*1000
        totals = self.report["stage_timing_ms"]
        totals[self.phase] = totals.get(self.phase, 0.)+elapsed
        collisions = sum(item["ms"] for item in self.report.get("collision_batches", ()))
        collisions += sum(item.collision_ms for item in self.validities)
        details = self.report["stage_breakdown_ms"].setdefault(self.phase, {})
        details["collision"] = details.get("collision", 0.)+collisions-self.stage_collision_checkpoint
        self.report["stage_timing_spans"].append({"stage": self.phase,
            "start_ms": (self.stage_started-self.started)*1000, "wall_ms": elapsed})
        self.stage_started, self.stage_collision_checkpoint = now, collisions

    def record_operation_time(self, kind, started):
        self.torch.cuda.synchronize()
        elapsed = (time.perf_counter()-started)*1000
        details = self.report["stage_breakdown_ms"].setdefault(self.phase, {})
        details[kind] = details.get(kind, 0.)+elapsed
        return elapsed

    def enter(self, phase):
        if "stage_timing_ms" in self.report:
            self.close_stage_timing()
        self.phase = phase
        self.report["current_stage"] = phase
        self.progress(phase)

    def validity(self):
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        from v3_batched_loaded_search import loaded_robot
        from .adapter import to_curobo_scene
        from .sequential_validity import SequentialValidity
        snapshot = self.store.snapshot()
        support_key = None if self.support_pose is None else tuple(self.support_pose.reshape(-1))
        key = (snapshot.geometry_key, self.contact_pair, support_key, self.request.cartesian_geometry_tolerance_m)
        if self.cached_key != key:
            robot = loaded_robot(self.planner.robot, self.planner.box_fit, attachments=snapshot.attachments)
            # A permitted target is checked against ALL spheres below, with only
            # its certified tool pair delegated to the exact mesh-plane proof.
            native_snapshot = snapshot
            if self.contact_pair is not None:
                native_snapshot = replace(snapshot, objects=tuple(
                    item for item in snapshot.objects if item.object_id != self.contact_pair[1]))
            checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
                robot_config=robot, scene_model=to_curobo_scene(native_snapshot),
                n_cuboids=max(40, len(snapshot.objects)), n_meshes=0, collision_activation_distance=0.))
            self.cached_validity = SequentialValidity(checker, snapshot, self.planner.robot,
                {side: tool_mesh_vertices(self.planner, side) for side in ("left", "right")}, self.contact_pair,
                geometry_tolerance=self.request.cartesian_geometry_tolerance_m, support_pose=self.support_pose)
            self.cached_key = key
            # Retain metrics, not all GPU checkers.
            self.validities.append(self.cached_validity)
            if len(self.validities) > 1:
                old = self.validities[-2]
                self.report.setdefault("collision_batches", []).append(
                    {"states": old.states_checked, "ms": old.collision_ms})
                self.validities.remove(old)
        return self.cached_validity

    def audit(self, rows, active, reference, line=None, target=None):
        from curobo.types import JointState
        self.torch.cuda.synchronize()
        audit_started = time.perf_counter()
        rows = np.asarray(rows)
        validity = self.validity()
        good = validity.mask(self.tensor(rows))
        invalid = self.torch.nonzero(~good).reshape(-1)
        if len(invalid):
            i = int(invalid[0].item())
            self.report["failure_location"] = {"phase": self.phase, "segment_frame": i,
                                               "joint_positions": rows[i].tolist()}
            self.fail("cuRobo/actual-box collision, bound, ground or tilt check failed; all non-contact pairs remain active")
        try:
            frozen_error = check_frozen(rows, reference, active)
        except ValueError as error:
            self.fail(str(error))
        audit = {"stage": self.phase, "points": len(rows), "active_variables": len(active),
                 "contact_pair": self.contact_pair, "active_joint_names": [ACTIVE_JOINTS[i] for i in active],
                 "support_requirement": "box_pose_not_joint_freeze",
                 "frozen_joint_max_error": frozen_error, "max_box_tilt_deg": 0.,
                 "minimum_box_bottom_m": None, "support_translation_drift_m": 0.,
                 "max_fk_translation_error_m": 0., "max_fk_rotation_error_rad": 0.}
        snapshot = self.store.snapshot()
        # Independent URDF FK versus cuRobo FK at every densified path point.
        gpu_fk = validity.checker.kinematics.compute_kinematics(
            JointState.from_position(self.tensor(rows)[None], joint_names=list(ACTIVE_JOINTS))).tool_poses.to_dict()
        gpu_tools = {side: (gpu_fk[side+"_tool0"].position.reshape(-1, 3).cpu().numpy(),
                            gpu_fk[side+"_tool0"].quaternion.reshape(-1, 4).cpu().numpy())
                     for side in ("left", "right")}
        # This snapshot is immutable throughout the audit. Transform static
        # world boxes and rigid attachments once, not once per trajectory row.
        world_boxes = [(item, pose_matrix(item.pose)) for item in snapshot.objects if item.object_id != "ground"]
        attachment_offsets = {item.object_id: pose_matrix(item.tool_to_object) for item in snapshot.attachments}
        contact_geometry = None
        if self.contact_pair is not None:
            side, object_id = self.contact_pair
            item = snapshot.object(object_id)
            contact_geometry = (side, item, tool_mesh_vertices(self.planner, side), np.linalg.inv(pose_matrix(item.pose)))
        previous_progress = -1e-6
        max_line_error = 0.
        max_line_angle = 0.
        for index, q in enumerate(rows):
            self.planner.fk(q)
            frame_tools = {side: self.planner.fk_robot.get_transform(side+"_tool0", self.planner.fk_robot.base_link).copy()
                           for side in ("left", "right")}
            if contact_geometry is not None:
                side, item, vertices, world_to_box = contact_geometry
                proof = mesh_contact_proof(vertices, world_to_box @ frame_tools[side], item.dimensions_m, tolerance=self.request.cartesian_geometry_tolerance_m)
                if not proof["valid"]:
                    self.fail(f"independent tool contact proof failed at {index}")
            for side, (positions, quaternions) in gpu_tools.items():
                actual = frame_tools[side]
                translation, angle = pose_error(actual, pose_matrix(Pose(positions[index], quaternions[index])))
                audit["max_fk_translation_error_m"] = max(audit["max_fk_translation_error_m"], translation)
                audit["max_fk_rotation_error_rad"] = max(audit["max_fk_rotation_error_rad"], angle)
                if translation > .001 or angle > math.radians(.5):
                    self.fail(f"independent FK disagreement at {index}/{side}: {translation}, {angle}")
            previous_boxes = []
            for item in snapshot.attachments:
                matrix = frame_tools[item.parent_link.removesuffix("_tool0")] @ attachment_offsets[item.object_id]
                for other, other_matrix in previous_boxes:
                    if overlap(matrix, item.dimensions_m, other_matrix, other.dimensions_m, self.request.cartesian_geometry_tolerance_m):
                        self.fail(f"independent payload/payload collision at {index}")
                previous_boxes.append((item, matrix))
                bottom = float(corners(matrix, item.dimensions_m)[:, 2].min())
                tilt = math.degrees(math.acos(np.clip(matrix[2, 2], -1., 1.)))
                audit["max_box_tilt_deg"] = max(audit["max_box_tilt_deg"], tilt)
                previous_bottom = audit["minimum_box_bottom_m"]
                audit["minimum_box_bottom_m"] = bottom if previous_bottom is None else min(previous_bottom, bottom)
                if bottom < snapshot.policy.ground_z_m-1e-6 or tilt > snapshot.policy.max_box_tilt_deg+1e-6:
                    self.fail(f"independent box ground/tilt failure at {index}/{item.object_id}")
                for obstacle, obstacle_matrix in world_boxes:
                    if overlap(matrix, item.dimensions_m, obstacle_matrix, obstacle.dimensions_m, self.request.cartesian_geometry_tolerance_m):
                        self.fail(f"independent actual-box collision at {index}: {item.object_id}/{obstacle.object_id}")
                if self.support_pose is not None and item.parent_link == "right_tool0":
                    drift, angle = pose_error(matrix, self.support_pose)
                    audit["support_translation_drift_m"] = max(audit["support_translation_drift_m"], drift)
                    if drift > .001 or angle > math.radians(.5):
                        self.fail(f"support box drift at {index}: {drift}, {angle}")
            if line is not None:
                side, origin, direction, length = line
                actual = frame_tools[side]
                delta = actual[:3, 3]-origin[:3, 3]
                along = float(delta @ direction)
                error = float(np.linalg.norm(delta-direction*along))
                angle = pose_error(actual, origin)[1]
                max_line_error = max(max_line_error, error)
                max_line_angle = max(max_line_angle, angle)
                if (error > .001 or angle > math.radians(.5) or along < previous_progress-1e-5
                        or along < -1e-5 or along > length+.001):
                    self.fail(f"Cartesian line/orientation/order error at {index}")
                previous_progress = along
        if line is not None:
            audit.update(max_line_error_m=max_line_error, max_line_rotation_error_rad=max_line_angle,
                         line_distance_m=previous_progress)
            if abs(previous_progress-line[3]) > .001:
                self.fail("Cartesian endpoint distance error")
        if target is not None:
            side, matrix = target
            translation, angle = pose_error(self.tool(rows[-1], side), matrix)
            audit.update(target_translation_error_m=translation, target_rotation_error_deg=math.degrees(angle))
            if translation > .001 or angle > math.radians(.5):
                self.fail(f"target FK tolerance failure: {translation}, {angle}")
        audit["wall_ms"] = self.record_operation_time("audit", audit_started)
        self.report["stage_audits"].append(audit)

    def append(self, rows):
        attachments = [asdict(item) for item in self.store.snapshot().attachments]
        for row in rows:
            self.report["frames"].append([float(value) for value in row])
            self.report["phases"].append(self.phase)
            self.report["payload"].append(bool(attachments))
            self.report["attachments_by_frame"].append(attachments)
        self.q = np.asarray(rows[-1], dtype=float).copy()
        snapshot = self.store.snapshot()
        self.store.update_state(snapshot.state.with_positions(dict(zip(ACTIVE_JOINTS, self.q))), snapshot.revision)

    def support_sweep_hits(self, q):
        """A frozen support arm must leave the required lower-box sweep empty.

        This is a task constraint on the support IK, not an obstacle moved or
        excluded from collision checking. The translated axis-aligned box's
        swept volume is exactly this cuboid.
        """
        lower = self.request.snapshot.object(f"wall_box_{self.request.lower_box:02d}")
        center = np.array(lower.pose.position)
        center[0] -= self.request.extraction_m/2
        half = np.array(lower.dimensions_m)/2
        half[0] += self.request.extraction_m/2
        self.planner.fk(q)
        kin = self.planner.robot.get("robot_cfg", self.planner.robot)["kinematics"]
        hits = []
        for link, spheres in kin["collision_spheres"].items():
            if link.startswith("left_"):
                continue  # these seven variables are allowed to move during extraction
            transform = self.planner.fk_robot.get_transform(link, self.planner.fk_robot.base_link)
            points = np.array([sphere["center"] for sphere in spheres]) @ transform[:3, :3].T + transform[:3, 3]
            radii = np.array([sphere["radius"] for sphere in spheres])
            distance = np.linalg.norm(np.maximum(np.abs(points-center)-half, 0.), axis=1)
            penetration = radii-distance
            if np.any((radii > 0.) & (penetration > 1e-6)):
                hits.append({"link": link, "penetration_m": float(penetration.max())})
        return hits

    def ik(self, side, target, active, support_target=None):
        import copy
        from curobo.types import JointState
        from v3_batched_loaded_search import loaded_robot
        from .backend import goal
        snapshot = self.store.snapshot()
        configured = loaded_robot(self.planner.robot, self.planner.box_fit, attachments=snapshot.attachments)
        kin = configured.get("robot_cfg", configured)["kinematics"]
        names = list(kin["cspace"]["joint_names"])
        active_names = [ACTIVE_JOINTS[i] for i in active]
        kin["lock_joints"].update({name: float(self.q[i]) for i, name in enumerate(ACTIVE_JOINTS) if i not in active})
        indices = [names.index(name) for name in active_names]
        kin["cspace"] = {key: [value[i] for i in indices] if isinstance(value, list) and len(value) == len(names)
                         else copy.deepcopy(value) for key, value in kin["cspace"].items()}
        poses = {side+"_tool0": matrix_pose(target)}
        if support_target is not None:
            poses["right_tool0"] = matrix_pose(support_target)
        kin["tool_frames"] = list(poses)
        self.seed("ik")  # Keep the request's existing stage seed sequence for RRT.
        attempt = self.report["attempts"][-1]
        attempt["solver_seed"] = self.request.seed
        attempt["active_joints"] = active_names
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        solver = self.planner.ik_solver(dict(self.request.tasks), robot=configured,
                                        snapshot=snapshot, random_seed=self.request.seed)
        self.torch.cuda.synchronize()
        attempt["initialization_ms"] = (time.perf_counter()-started)*1000
        attempt["cache_hit"] = self.planner.ik_cache_hit
        # Reset the native sampler at the request boundary, not on every goal.
        seen_solver = getattr(self, "_seen_solver", None)
        if seen_solver is None or seen_solver() is not solver:
            solver.reset_seed()
            self._seen_solver = weakref.ref(solver)
        goal_poses = {name: {"position": pose.position, "quaternion": pose.quaternion_wxyz}
                      for name, pose in poses.items()}
        solve_started = time.perf_counter()
        result = solver.solve_pose(goal(goal_poses, frames=list(poses)), JointState.from_position(
            self.tensor(self.q[active])[None], joint_names=active_names), return_seeds=32)
        self.torch.cuda.synchronize()
        attempt["solve_ms"] = (time.perf_counter()-solve_started)*1000
        filter_started = time.perf_counter()
        # f044bf1: return 32, use native success, then distance-rank. Only the
        # measured held-payload/tangency failure is re-certified, on these SAME
        # 32 outputs and the original 2mm/1deg pose tolerances.
        result_names = list(result.js_solution.joint_names)
        values = result.js_solution.position.reshape(-1, len(result_names))
        successful = result.success.reshape(-1).bool()
        attempt.update(solver_successes=int(successful.sum().item()), returned_candidates=len(successful),
                       position_tolerance_m=.002, orientation_tolerance_deg=1., return_seeds=32)
        all_candidates = np.repeat(self.q[None], len(values), axis=0)
        all_candidates[:, active] = values[:, [result_names.index(name) for name in active_names]].detach().cpu().numpy()
        if not bool(successful.any().item()) and snapshot.attachments:
            validity = self.validity()
            tensors = self.tensor(all_candidates)
            recovered = validity.mask(tensors)
            fk = validity.checker.kinematics.compute_kinematics(JointState.from_position(
                tensors[None], joint_names=list(ACTIVE_JOINTS))).tool_poses.to_dict()
            for frame, pose in poses.items():
                position_error = self.torch.linalg.vector_norm(fk[frame].position.reshape(-1, 3)-self.tensor(pose.position), dim=-1)
                cosine = (fk[frame].quaternion.reshape(-1, 4)*self.tensor(pose.quaternion_wxyz)).sum(-1).abs().clamp(0., 1.)
                recovered &= (position_error <= .002) & (2*self.torch.acos(cosine) <= math.radians(1.))
            candidates = all_candidates[recovered.cpu().numpy()]
            attempt["recertification"] = {"reason": "native_all_failed_with_attached_payload; same32 full-model validation",
                                         "accepted": len(candidates), "extra_ik_samples": 0}
        else:
            candidates = all_candidates[successful.cpu().numpy()]
            if len(candidates):
                candidates = candidates[self.validity().mask(self.tensor(candidates)).cpu().numpy()]
        weights = np.array([5.]+[1.]*14)
        distance = np.sum(((candidates-self.q)*weights)**2, axis=1)
        order = np.argsort(distance)
        if side == "right" and self.phase == "right_precontact" and self.request.support_elbow_rise_m > 0 and len(candidates):
            heights = []
            for row in candidates:
                self.planner.fk(row)
                heights.append(float(self.planner.fk_robot.get_transform("right_link4", self.planner.fk_robot.base_link)[2, 3]))
            heights = np.asarray(heights)
            target_height = heights[order[0]]+self.request.support_elbow_rise_m
            order = np.lexsort((distance, np.maximum(0., target_height-heights)))
            attempt["elbow_preference"] = {"target_height_m": float(target_height),
                "requested_rise_m": self.request.support_elbow_rise_m, "first_candidate_height_m": float(heights[order[0]])}
        candidates = candidates[order]
        self.torch.cuda.synchronize()
        attempt["filter_ms"] = (time.perf_counter()-filter_started)*1000
        attempt["valid_candidates"] = len(candidates)
        attempt["wall_ms"] = self.record_operation_time("ik", started)
        self.report["timing_ms"]["ik"] += attempt["wall_ms"]
        self.progress(f"{self.phase}: f044 IK {int(successful.sum())}/32 native, {len(candidates)} validated; cache={attempt['cache_hit']}")
        return candidates

    def search(self, goals, active):
        from v3_batched_loaded_search import batched_rrt_connect_multi_goal, densify
        from .sequential_validity import ReducedValidity
        validity = self.validity()
        reduced = ReducedValidity(validity, self.q, active)
        lower, upper = validity.checker.kinematics.get_joint_limits().position
        seed = self.seed("rrtconnect")
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        path, stats = batched_rrt_connect_multi_goal(
            self.tensor(self.q[active]), self.tensor(np.asarray(goals)[:, active]),
            lower[active]+1e-5, upper[active]-1e-5, reduced, 2., seed)
        self.torch.cuda.synchronize()
        self.report["timing_ms"]["search"] += (time.perf_counter()-started)*1000
        self.report["attempts"][-1]["stats"] = stats
        self.report["attempts"][-1]["wall_ms"] = self.record_operation_time("search", started)
        if path is None:
            return None
        expanded = reduced.expand(self.tensor(path)).cpu().numpy()
        return np.asarray(densify(expanded))

    def cartesian(self, side, direction, distance, active, lift_end=None, start=None, swivel_continuation=False):
        reference = self.q if start is None else np.asarray(start)
        started = time.perf_counter()
        validity = self.validity()
        rows, details = self.planner.analytic_extract(reference, validity, validity, sides=(side,),
                          direction=direction, distance_m=distance, lift_end=lift_end,
                          swivel_continuation=swivel_continuation)
        self.torch.cuda.synchronize()
        self.report["timing_ms"]["cartesian"] += (time.perf_counter()-started)*1000
        operation_ms = self.record_operation_time("cartesian", started)
        self.report["attempts"].append({"stage": self.phase, "operation": "analytic_cartesian",
                                        "details": details, "wall_ms": operation_ms})
        self.progress(f"{self.phase}: Cartesian {'accepted' if rows is not None else details}")
        if rows is None:
            failure = getattr(self.planner, "last_cartesian_failure", None)
            if failure is not None:
                self.report["failure_location"] = {"phase": self.phase, **failure}
        else:
            self.report.pop("failure_location", None)
            if start is None:
                origin = self.tool(reference, side)
                self.audit(rows, active, reference, line=(side, origin, np.array(direction), distance))
        return rows

    def contact_target(self, side, item, roll):
        from v3_wall_ik_benchmark import canonical_side_suction_quaternion_wxyz
        target = pose_matrix(Pose((item.pose.position[0]-item.dimensions_m[0]/2,
                                  item.pose.position[1], item.pose.position[2]),
                                 tuple(canonical_side_suction_quaternion_wxyz(side))))
        target[:3, :3] = Rotation.from_rotvec((roll, 0., 0.)).as_matrix() @ target[:3, :3]
        # Select a point on the unchanged box face using the unchanged tool sphere
        # envelope and overhead door. Only the contact point moves, not the scene.
        fk = self.planner.fk_robot
        self.planner.fk(self.q)
        tool_to_link = np.linalg.inv(fk.get_transform(side+"_tool0", fk.base_link)) @ fk.get_transform(
            side+"_link7", fk.base_link)
        kin = self.planner.robot.get("robot_cfg", self.planner.robot)["kinematics"]
        tool_spheres = kin["collision_spheres"][side+"_link7"]
        extent_z = max((target[:3, :3] @ (tool_to_link @ np.r_[sphere["center"], 1.])[:3])[2]
                       + sphere["radius"] for sphere in tool_spheres if sphere["radius"] > 0)
        door = self.store.snapshot().object("warehouse_top_door_leaf")
        door_bottom = float(corners(pose_matrix(door.pose), door.dimensions_m)[:, 2].min())
        if side == "left":
            # The lower face is not constrained to its center. Use its highest
            # certified in-face contact, bounded by the real overhead support
            # box and door. This reduces wrist folding without moving any box.
            mesh_z = (tool_mesh_vertices(self.planner, side) @ target[:3, :3].T)[:, 2]
            target[2, 3] = item.pose.position[2]+item.dimensions_m[2]/2-float(mesh_z.max())-.001
            for held in self.store.snapshot().attachments:
                held_corners = corners(self.box_pose(self.q, held), held.dimensions_m)
                held_bottom = float(held_corners[:, 2].min())
                if held_bottom >= item.pose.position[2]:
                    target[2, 3] = min(target[2, 3], held_bottom-extent_z-.001)
        target[2, 3] = min(target[2, 3], door_bottom-extent_z-.001)
        proof = mesh_contact_proof(tool_mesh_vertices(self.planner, side),
                                   np.linalg.inv(pose_matrix(item.pose)) @ target, item.dimensions_m, .001,
                                   tolerance=self.request.cartesian_geometry_tolerance_m)
        return target, proof

    def contact(self, side, object_id, active):
        item = self.store.snapshot().object(object_id)
        rolls = (0., math.pi/6, -math.pi/6, math.pi/4, -math.pi/4) if side == "left" else (0.,)
        geometry_proven = False
        for roll in rolls:
            self.enter(side+"_contact_model")
            target, proof = self.contact_target(side, item, roll)
            entry = {"side": side, "object_id": item.object_id, "roll_rad": roll,
                     "target": asdict(matrix_pose(target)), "proof": proof, "selected": False}
            self.report.setdefault("contact_geometry", []).append(entry)
            if not proof["valid"]:
                continue
            geometry_proven = True
            endpoint_goals = None
            if side == "left":
                # The f044 fixed-swivel bridge is retained. The only task-specific
                # extension is selecting its analytic branch from a reachable
                # extraction endpoint instead of an unrelated precontact IK.
                attachment = AttachedObject(item.object_id, item.dimensions_m, side+"_tool0",
                    matrix_pose(np.linalg.inv(target) @ pose_matrix(item.pose)), (side+"_link7",))
                saved = self.store
                try:
                    self.store = SceneStore(saved.snapshot())
                    self.store.attach(attachment, self.store.snapshot().revision)
                    self.enter("left_extract_target")
                    endpoint = target.copy()
                    endpoint[0, 3] -= self.request.extraction_m
                    endpoint_goals = self.ik(side, endpoint, active)
                    entry["extraction_endpoint_candidates"] = len(endpoint_goals)
                finally:
                    self.store = saved
                if not len(endpoint_goals):
                    continue
            if self.try_contact(side, item, active, target, endpoint_goals, 32//len(rolls)):
                entry["selected"] = True
                return
        self.report["failure_code"] = "NO_CERTIFIED_GRASP" if geometry_proven else "CONTACT_MODEL_UNREPRESENTABLE"
        self.fail("f044 fixed-swivel endpoint-guided candidates exhausted")

    def try_contact(self, side, item, active, target, endpoint_goals, limit):
        precontact = target.copy()
        precontact[0, 3] -= .05
        self.enter(side+"_precontact")
        if side == "left":
            candidates = endpoint_goals
        elif 0 in active:
            candidates = self.ik(side, precontact, [i for i in active if i != 0])
            self.report["support_lift_policy"] = "fixed_initial_lift_acceleration"
            if not len(candidates):
                candidates = self.ik(side, precontact, active)
                self.report["support_lift_policy"] = "lift_released"
        else:
            candidates = self.ik(side, precontact, active)
        for candidate in candidates[:limit]:
            if side == "left":
                # Recover the contact on the endpoint's fixed-psi branch, then
                # validate the normal forward 1cm f044 path. No reverse path,
                # angle grid, or denser Cartesian sampling is used.
                self.planner.fk(candidate)
                carriage = np.linalg.inv(self.planner.fk_robot.get_transform(
                    "arm_carriage", self.planner.fk_robot.base_link))
                psi = self.planner.analytic.swivel(0, candidate[1:8])
                solutions = self.planner.analytic.solve(0, carriage @ target, psi, candidate[1:8])
                if not len(solutions):
                    continue
                contact_q = candidate.copy()
                contact_q[1:8] = min(solutions, key=lambda row: np.linalg.norm(row-candidate[1:8]))
                saved, saved_q = self.store, self.q
                tool = self.tool(contact_q, side)
                attachment = AttachedObject(item.object_id, item.dimensions_m, side+"_tool0",
                    matrix_pose(np.linalg.inv(tool) @ pose_matrix(item.pose)), (side+"_link7",))
                try:
                    self.store = SceneStore(saved.snapshot())
                    self.store.attach(attachment, self.store.snapshot().revision)
                    self.enter("left_extract")
                    started = time.perf_counter()
                    extracted, details = self.planner.analytic_extract(contact_q, self.validity, self.validity,
                        sides=(side,), direction=(-1., 0., 0.), distance_m=self.request.extraction_m)
                    elapsed = self.record_operation_time("cartesian", started)
                    self.report["timing_ms"]["cartesian"] += elapsed
                    self.report["attempts"].append({"stage": self.phase,
                        "operation": "endpoint_guided_forward_fixed_swivel", "candidate_only": True,
                        "details": details, "wall_ms": elapsed,
                        "failure_location": self.planner.last_cartesian_failure})
                finally:
                    self.store, self.q = saved, saved_q
                if extracted is None:
                    continue
                self.enter("left_contact")
                self.contact_pair = (side, item.object_id)
                outward = self.cartesian(side, (-1., 0., 0.), .05, active, start=contact_q)
                self.contact_pair = None
                if outward is None:
                    continue
                contact = outward[::-1].copy()
                candidate = contact[0]
                self.pending_extract = (extracted, active)
            else:
                self.enter(side+"_contact")
                self.contact_pair = (side, item.object_id)
                contact = self.cartesian(side, (1., 0., 0.), .05, active, start=candidate)
                self.contact_pair = None
                if contact is None:
                    continue
            self.enter(side+"_precontact")
            path = self.search([candidate], active)
            if path is None:
                continue
            self.audit(path, active, self.q, target=(side, precontact))
            self.append(path)
            self.enter(side+"_contact")
            self.contact_pair = (side, item.object_id)
            self.audit(contact, active, self.q, target=(side, target),
                       line=(side, self.tool(contact[0], side), np.array((1., 0., 0.)), .05))
            self.append(contact)
            self.contact_pair = None
            self.attach(side, item)
            return True
        return False

    def attach(self, side, item):
        self.enter(side+"_attach")
        tool = self.tool(self.q, side)
        world = pose_matrix(item.pose)
        # Exact face contact, rather than assuming a canonical payload offset.
        local_tip = np.linalg.inv(world) @ np.r_[tool[:3, 3], 1.]
        half = np.array(item.dimensions_m)/2
        if abs(local_tip[0]+half[0]) > .001 or np.any(np.abs(local_tip[1:3]) > half[1:]+.001):
            self.fail("tool not on the designated box front face")
        proof = mesh_contact_proof(tool_mesh_vertices(self.planner, side), np.linalg.inv(world) @ tool,
                                   item.dimensions_m, .001, tolerance=self.request.cartesian_geometry_tolerance_m)
        if not proof["valid"]:
            self.fail("attachment tool-mesh contact is not certified")
        attachment = AttachedObject(item.object_id, item.dimensions_m, side+"_tool0",
                                    matrix_pose(np.linalg.inv(tool) @ world), (side+"_link7",))
        snapshot = self.store.snapshot()
        self.store.attach(attachment, snapshot.revision)
        # Own-tool contact is certified; all other robot/world/payload pairs stay active.
        try:
            self.audit([self.q], [], self.q)
        except self.Blocked:
            self.report["rejected_attachment"] = asdict(attachment)
            self.store = SceneStore(snapshot)
            raise
        self.append([self.q])
        self.event(item.object_id, world)

    def event(self, object_id, world):
        self.report["predicted_scene_events"].append({"phase": self.phase, "object_id": object_id,
            "frame_index": len(self.report["frames"])-1, "world_pose": asdict(matrix_pose(world)),
            "snapshot": self.store.snapshot().to_dict()})

    def extract(self, side, active):
        self.enter(side+"_extract")
        pending = self.pending_extract if side == "left" else None
        rows = None
        if pending is not None:
            rows, active = pending
        self.pending_extract = None
        if rows is None:
            rows = self.cartesian(side, (-1., 0., 0.), self.request.extraction_m, active)
        if (rows is None and side == "right"
                and getattr(self.planner, "last_cartesian_failure", {}).get("reason") == "fixed_swivel_ik_no_solution"):
            endpoint = self.tool(self.q, side)
            endpoint[:3, 3] += np.asarray((-1., 0., 0.))*self.request.extraction_m
            # Upper unloading is the one task stage allowed to coordinate the
            # lift with the right arm. One baseline 8D endpoint IK supplies only
            # the lift target; the path remains the 1cm forward analytic bridge.
            for goal in self.ik(side, endpoint, active):
                rows = self.cartesian(side, (-1., 0., 0.), self.request.extraction_m,
                                      active, lift_end=float(goal[0]), swivel_continuation=True)
                if rows is not None:
                    break
        if rows is None:
            self.fail("0.36 m extraction failed; no extension or obstacle movement")
        self.audit(rows, active, self.q, line=(side, self.tool(self.q, side), np.array([-1., 0., 0.]), self.request.extraction_m))
        self.append(rows)
        item = next(item for item in self.store.snapshot().attachments if item.parent_link == side+"_tool0")
        original = self.request.snapshot.object(item.object_id)
        wall_face = original.pose.position[0]-original.dimensions_m[0]/2
        box_back = float(corners(self.box_pose(self.q, item), item.dimensions_m)[:, 0].max())
        self.report["stage_audits"][-1]["wall_clearance_m"] = wall_face-box_back
        if box_back > wall_face-1e-6:
            self.fail("actual box has not cleared the wall front plane")

    def place(self, side, active):
        self.enter(side+"_place")
        target = pose_matrix(dict(self.request.targets)[side+"_tool0"])
        candidates = self.ik(side, target, active)
        if not len(candidates):
            self.fail("rear unloading target has no valid online IK")
        path = self.search(candidates, active)
        if path is None:
            self.fail("rear unloading search exhausted its 2 s budget")
        self.audit(path, active, self.q, target=(side, target))
        self.append(path)
        item = next(item for item in self.store.snapshot().attachments if item.parent_link == side+"_tool0")
        world = self.box_pose(self.q, item)
        box_target = target @ pose_matrix(item.tool_to_object)
        box_position_error, box_angle_error = pose_error(world, box_target)
        if box_position_error > .001 or box_angle_error > math.radians(.5):
            self.fail("actual release box pose differs from unloading target")
        self.released_offsets[side] = pose_matrix(item.tool_to_object)
        self.enter(side+"_release")
        self.store.release(item.object_id, self.store.snapshot().revision)
        self.append([self.q])
        self.event(item.object_id, world)
        self.report["predicted_scene_events"][-1]["target_world_pose"] = asdict(matrix_pose(box_target))
        self.report["predicted_scene_events"][-1].update(
            translation_error_m=box_position_error, rotation_error_deg=math.degrees(box_angle_error))

    def run(self):
        left, right = list(range(1, 8)), [0]+list(range(8, 15))
        self.enter("initialize")
        if not np.allclose(self.q, [self.planner.home[name] for name in ACTIVE_JOINTS], atol=1e-9, rtol=0):
            self.fail("sequential unload must begin at the default Home")
        for _, box_id in self.request.tasks:
            item = self.request.snapshot.object(f"wall_box_{box_id:02d}")
            if abs(item.pose.position[0]-item.dimensions_m[0]/2-self.planner.front-.75) > 1e-9:
                self.fail("task wall front must remain 0.75 m from the model chassis front")
        self.audit([self.q], [], self.q)
        self.append([self.q])
        named = yaml.safe_load(self.planner.args.named_poses.read_text())["named_poses"]["unloading"]
        unloading_q = [named[name] for name in ACTIVE_JOINTS]
        self.report["initial_unloading_posture_valid"] = bool(self.validity().mask(self.tensor([unloading_q])).item())
        if not self.report["initial_unloading_posture_valid"]:
            self.fail("default unloading posture fails the unchanged model/scene validation")
        upper_id, lower_id = (f"wall_box_{value:02d}" for value in (self.request.upper_box, self.request.lower_box))
        self.report["unloading_targets"] = {key: asdict(value) for key, value in self.request.targets}
        self.contact("right", upper_id, right)
        upper = next(item for item in self.store.snapshot().attachments if item.object_id == upper_id)
        self.support_pose = self.box_pose(self.q, upper)
        self.contact("left", lower_id, left)
        self.extract("left", left)
        self.place("left", left)
        self.enter("left_retreat")
        # Retreat away from the release contact, then plan to a checked parking pose.
        direction = -(self.tool(self.q, "left")[:3, :3] @ self.released_offsets["left"][:3, 3])
        direction /= np.linalg.norm(direction)
        rows = self.cartesian("left", tuple(direction), .05, left)
        if rows is None:
            self.fail("empty-arm release retreat failed")
        self.append(rows)
        self.enter("left_park")
        distance = (self.request.snapshot.object(upper_id).pose.position[2]
                    - self.request.snapshot.object(lower_id).pose.position[2])
        poses = yaml.safe_load(self.planner.args.named_poses.read_text())["named_poses"]
        parked = False
        release_tool = pose_matrix(dict(self.request.targets)["left_tool0"])
        # This sweep predicts the post-unfreeze lowering stage. Keeping the
        # pre-unfreeze support-pose constraint here rejects every nonzero lift.
        support_pose = self.support_pose
        self.support_pose = None
        future_lowering_validity = self.validity()
        self.support_pose = support_pose
        for name in ("home", "second_home", "second_unloading", "third_unloading"):
            park = self.q.copy()
            park[left] = [poses[name][ACTIVE_JOINTS[i]] for i in left]
            clearance = float(np.linalg.norm(self.tool(park, "left")[:3, 3]-release_tool[:3, 3]))
            if clearance < .05:
                continue
            from v3_batched_loaded_search import densify
            lowered = park.copy()
            lowered[0] -= distance
            sweep = np.asarray(densify(np.array([park, lowered])))
            sweep_good = future_lowering_validity.mask(self.tensor(sweep))
            attempt = {"stage": self.phase, "operation": "named_park_lift_sweep", "name": name,
                       "release_clearance_m": clearance, "valid": bool(sweep_good.all().item())}
            self.report["attempts"].append(attempt)
            if not attempt["valid"]:
                attempt["first_invalid_index"] = int(self.torch.nonzero(~sweep_good)[0].item())
                continue
            path = self.search([park], left)
            if path is None:
                continue
            self.audit(path, left, self.q)
            self.append(path)
            self.report["parking_pose"] = name
            parked = True
            break
        if not parked:
            self.fail("no named left-arm parking pose supports the complete future lift sweep")
        self.support_pose = None
        self.enter("right_lower")
        upper_source = self.request.snapshot.object(upper_id)
        lower_source = self.request.snapshot.object(lower_id)
        distance = upper_source.pose.position[2]-lower_source.pose.position[2]
        rows = self.cartesian("right", (0., 0., -1.), distance, right, lift_end=float(self.q[0]-distance))
        if rows is None:
            self.fail("baseline vertical Cartesian path failed")
        self.append(rows)
        self.extract("right", right)
        self.place("right", right)
        if not completion(self.report["predicted_scene_events"], self.store.snapshot(), upper_id, lower_id):
            self.fail("incomplete attachment/release lifecycle")
        self.report["success"] = True
        self.report["final_snapshot"] = self.store.snapshot().to_dict()
        return self.report


def plan_sequential(planner, request, progress):
    planner.prepare_task(dict(request.tasks))
    task = _SequentialTask(planner, request, progress)
    try:
        return task.run()
    finally:
        task.close_stage_timing()
        task.report["total_ms"] = (task.stage_started-task.started)*1000
        task.report["stage_timing_sum_ms"] = sum(task.report["stage_timing_ms"].values())
        task.report["final_snapshot"] = task.store.snapshot().to_dict()
        timing = task.report["timing_ms"]
        timing["collision"] = sum(item["ms"] for item in task.report.get("collision_batches", ()))
        timing["collision"] += sum(item.collision_ms for item in task.validities)
        task.report["timing_categories_overlap"] = True
        rows = np.asarray(task.report["frames"])
        task.report["joint_travel_rad_m"] = (np.abs(np.diff(rows, axis=0)).sum(axis=0).tolist()
                                              if len(rows) > 1 else [0.]*15)
        from collections import Counter
        searches = Counter(item["stage"] for item in task.report["attempts"] if item["operation"] == "rrtconnect")
        task.report["retry_count"] = sum(max(0, count-1) for count in searches.values())
        audits = [item for item in task.report["stage_audits"] if not item.get("candidate_only")]
        task.report["max_box_tilt_deg"] = max((item["max_box_tilt_deg"] for item in audits
            if item["minimum_box_bottom_m"] is not None), default=None)
        bottoms = [item["minimum_box_bottom_m"] for item in audits if item["minimum_box_bottom_m"] is not None]
        task.report["minimum_box_bottom_m"] = min(bottoms) if bottoms else None
