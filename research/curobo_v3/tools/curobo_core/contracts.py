from dataclasses import dataclass
from pathlib import Path
import math

from .scene import Pose, SceneSnapshot, digest


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_JOINTS = ("updown", *(f"{side}_joint{index}" for side in ("left", "right") for index in range(1, 8)))


@dataclass(frozen=True)
class PlannerAssets:
    robot_config: Path = WORKSPACE_ROOT / "generated/v3_analytic_071cb95/alfa_v322_suction_final.yml"
    mobile_robot_config: Path = WORKSPACE_ROOT / "generated/v3_analytic_071cb95/mobile18/alfa_v322_suction_mobile18.yml"
    urdf: Path = WORKSPACE_ROOT / "generated/v3_analytic_071cb95/model/robot_meter_meshes.urdf"
    named_poses: Path = WORKSPACE_ROOT / "models/robot_description_analytic_071cb95/config/named_poses_analytic_proxy.yaml"
    box_fit: Path = WORKSPACE_ROOT / "generated/v3_suction_v322_6bb184b/frozen_collision_model/box_voxel_edge_corner_final.json"
    target_poses: Path = WORKSPACE_ROOT / "generated/current_carry_target_6d.json"


@dataclass(frozen=True)
class PlanRequest:
    snapshot: SceneSnapshot
    tasks: tuple
    targets: tuple
    mode: str = "dual_cycle"
    upper_box: int | None = None
    lower_box: int | None = None
    support_arm: str = "right"
    transport_arm: str = "left"
    extraction_m: float = 0.36
    seed: int = 11
    support_elbow_rise_m: float = 0.0
    cartesian_geometry_tolerance_m: float = 1e-6

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(tuple(item) for item in self.tasks))
        object.__setattr__(self, "targets", tuple(tuple(item) for item in self.targets))
        if {side for side, _ in self.tasks} != {"left", "right"} or len(self.tasks) != 2:
            raise ValueError("current full-cycle planner requires one task per arm")
        box_ids = [box_id for _, box_id in self.tasks]
        if any(not isinstance(box_id, int) or box_id < 0 for box_id in box_ids) or len(set(box_ids)) != 2:
            raise ValueError("invalid task box IDs")
        if len(self.targets) != 2 or {name for name, _ in self.targets} != {"left_tool0", "right_tool0"}:
            raise ValueError("both complete Tool0 targets are required")
        if not all(isinstance(pose, Pose) for _, pose in self.targets):
            raise ValueError("target must be a Pose")
        self.snapshot.state.ordered(ACTIVE_JOINTS)
        for box_id in box_ids:
            self.snapshot.object(f"wall_box_{box_id:02d}")
        if self.mode not in ("dual_cycle", "sequential_unload"):
            raise ValueError("unknown task mode")
        if (type(self.support_elbow_rise_m) not in (int, float)
                or not math.isfinite(self.support_elbow_rise_m) or self.support_elbow_rise_m < 0.):
            raise ValueError("support elbow rise must be finite and nonnegative")
        if (type(self.cartesian_geometry_tolerance_m) not in (int, float)
                or not math.isfinite(self.cartesian_geometry_tolerance_m)
                or not 0. <= self.cartesian_geometry_tolerance_m <= .001):
            raise ValueError("Cartesian geometry tolerance must be within 0..1 mm")
        if self.mode != "sequential_unload" and self.support_elbow_rise_m != 0.:
            raise ValueError("elbow preference requires sequential_unload")
        if self.mode == "sequential_unload":
            if type(self.seed) is not int or not 0 <= self.seed < 2**32:
                raise ValueError("seed must be an unsigned 32-bit integer")
            if (self.support_arm, self.transport_arm) != ("right", "left"):
                raise ValueError("sequential unload requires right support and left transport")
            if (type(self.upper_box) is not int or type(self.lower_box) is not int
                    or dict(self.tasks) != {"right": self.upper_box, "left": self.lower_box}):
                raise ValueError("upper/lower identities disagree with arm tasks")
            if self.extraction_m != 0.36:
                raise ValueError("sequential unload uses a fixed 0.36 m extraction")
            if (self.snapshot.policy.exclude_task_objects_before_contact
                    or self.snapshot.policy.defer_payload_until_extract_end):
                raise ValueError("sequential unload forbids task and payload collision exemptions")
            extras = set(self.snapshot.state.joint_names) - set(ACTIVE_JOINTS) - {"head_joint", "head_pitch_joint"}
            if extras:
                raise ValueError("sequential fixed model cannot contain extra/mobile joints")
            if self.snapshot.attachments or self.snapshot.state.base_pose != Pose():
                raise ValueError("sequential unload must start empty at a fixed origin")
            if self.snapshot.frame_id != "map" or self.snapshot.meshes:
                raise ValueError("sequential unload currently requires map-frame cuboid geometry")
            upper = self.snapshot.object(f"wall_box_{self.upper_box:02d}")
            lower = self.snapshot.object(f"wall_box_{self.lower_box:02d}")
            if (self.upper_box % 5 != 2 or self.lower_box % 5 != 2
                    or self.upper_box - self.lower_box != 5
                    or any(abs(a-b) > 1e-9 for a, b in zip(upper.pose.position[:2], lower.pose.position[:2]))
                    or upper.dimensions_m != lower.dimensions_m
                    or abs(upper.pose.position[2] - lower.pose.position[2]
                           - (upper.dimensions_m[2]+lower.dimensions_m[2])/2) > 0.03):
                raise ValueError("expected vertically adjacent third-column upper/lower boxes")
            if any(item.pose.quaternion_wxyz != Pose().quaternion_wxyz
                   for item in (upper, lower)):
                raise ValueError("task boxes must be axis aligned")
            self.snapshot.object("warehouse_top_door_leaf")

    def to_dict(self):
        from dataclasses import asdict

        output = {"snapshot": self.snapshot.to_dict(), "tasks": dict(self.tasks),
                  "targets": {name: asdict(pose) for name, pose in self.targets}}
        # Preserve the identity and JSON contract of old requests.
        if self.mode != "dual_cycle":
            output.update({name: getattr(self, name) for name in (
                "mode", "upper_box", "lower_box", "support_arm", "transport_arm", "extraction_m", "seed")})
        if self.support_elbow_rise_m != 0.:
            output["support_elbow_rise_m"] = self.support_elbow_rise_m
        if self.cartesian_geometry_tolerance_m != 1e-6:
            output["cartesian_geometry_tolerance_m"] = self.cartesian_geometry_tolerance_m
        return output

    @classmethod
    def from_dict(cls, document):
        return cls(SceneSnapshot.from_dict(document["snapshot"]),
                   tuple(document["tasks"].items()),
                   tuple((name, Pose(**pose)) for name, pose in document["targets"].items()),
                   **{name: document[name] for name in (
                       "mode", "upper_box", "lower_box", "support_arm", "transport_arm", "extraction_m", "seed", "support_elbow_rise_m", "cartesian_geometry_tolerance_m")
                      if name in document})

    @property
    def identity(self):
        return digest(self.to_dict())
