#!/usr/bin/python3
"""Interactive V3.2.2 warehouse pose-to-origin closed-loop Rerun demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import yaml


MODEL_REVISION = "robot_v3.2.2-suction"
MODEL_SCHEMA = "alfa.v322_tool0151_model_snapshot.v1"
EXPECTED_URDF_SHA256 = "537d380a0d8a14a931bd2ca153759879eaeaec8e5f78f9cdf3f5a58c0a122893"
EXPECTED_LINK_COUNT = 21
EXPECTED_VISUAL_MESH_COUNT = 49
EXPECTED_TOOL0_OFFSET_M = 0.151

# Existing warehouse and station contract. Input x/y are relative to the
# nominal pose whose chassis front is 0.90 m from CONTACT_X_M.
WAREHOUSE_OPENING_X_M = -1.18
WAREHOUSE_REAR_X_M = 1.20
WAREHOUSE_HALF_WIDTH_M = 1.19
WAREHOUSE_HEIGHT_M = 2.35
CONTACT_X_M = 0.75
CHASSIS_FRONT_X_M = 0.500000002779484
NOMINAL_BASE_X_M = CONTACT_X_M - CHASSIS_FRONT_X_M - 0.90
SAFETY_CLEARANCE_M = 0.05
INPUT_BOUNDARY_TOLERANCE_M = 1e-6

CONTROL_DT_S = 0.04
MAX_LINEAR_SPEED_MPS = 0.25
MAX_ANGULAR_SPEED_RAD_S = 0.40
MAX_LINEAR_ACCEL_MPS2 = 0.20
MAX_ANGULAR_ACCEL_RAD_S2 = 0.30
LINEAR_FEEDBACK_GAIN = 1.8
CROSS_TRACK_GAIN = 2.4
YAW_FEEDBACK_GAIN = 3.0


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class InputBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    yaw_min_deg: float = -180.0
    yaw_max_deg: float = 180.0


@dataclass(frozen=True)
class TrajectorySample:
    time_s: float
    x: float
    y: float
    yaw: float
    linear_velocity_mps: float
    angular_velocity_rad_s: float
    stage: str
    reference_x: float
    reference_y: float
    reference_yaw: float


@dataclass
class ModelGeometry:
    model_root: Path
    asset_root: Path
    urdf_path: Path
    urdf_text: str
    home_joints: dict[str, float]
    footprint: np.ndarray
    local_min: np.ndarray
    local_max: np.ndarray
    rotation_radius: float
    z_min: float
    z_max: float


@dataclass(frozen=True)
class PlannedReturn:
    start: Pose2D
    drive_heading: float
    signed_distance: float
    drive_direction: str
    initial_rotation: float
    final_rotation: float
    samples: tuple[TrajectorySample, ...]

    @property
    def duration_s(self) -> float:
        return self.samples[-1].time_s

    @property
    def translation_length_m(self) -> float:
        return sum(
            math.hypot(right.x - left.x, right.y - left.y)
            for left, right in zip(self.samples, self.samples[1:])
        )


def find_repo_root() -> Path:
    candidates: list[Path] = []
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        candidates.extend((start, *start.parents))
    for candidate in candidates:
        if (
            candidate
            / "ros2_ws/src/alfa_robot_description/config/"
            "upstream_description.lock.json"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "cannot locate the ALFA repository root; use --model-root"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_xyz(text: str | None, default: Iterable[float]) -> np.ndarray:
    if not text:
        return np.asarray(tuple(default), dtype=float)
    return np.fromstring(text, sep=" ", dtype=float)


def rpy_matrix(text: str | None) -> np.ndarray:
    roll, pitch, yaw = parse_xyz(text, (0.0, 0.0, 0.0))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def origin_transform(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is None:
        return transform
    transform[:3, :3] = rpy_matrix(element.get("rpy"))
    transform[:3, 3] = parse_xyz(element.get("xyz"), (0.0, 0.0, 0.0))
    return transform


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s, d = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.asarray([
        [c + x * x * d, x * y * d - z * s, x * z * d + y * s],
        [y * x * d + z * s, c + y * y * d, y * z * d - x * s],
        [z * x * d - y * s, z * y * d + x * s, c + z * z * d],
    ])


def urdf_fk(root: ET.Element, positions: dict[str, float]) -> dict[str, np.ndarray]:
    links = {element.get("name", "") for element in root.findall("link")}
    joints_by_parent: dict[str, list[ET.Element]] = {}
    child_links: set[str] = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link", "")
        child_name = child.get("link", "")
        joints_by_parent.setdefault(parent_name, []).append(joint)
        child_links.add(child_name)
    roots = sorted(links - child_links)
    if len(roots) != 1:
        raise ValueError(f"expected one URDF root link, got {roots}")
    transforms = {roots[0]: np.eye(4)}
    pending = [roots[0]]
    while pending:
        parent_name = pending.pop()
        for joint in joints_by_parent.get(parent_name, []):
            child_name = joint.find("child").get("link", "")
            motion = np.eye(4)
            value = float(positions.get(joint.get("name", ""), 0.0))
            axis_element = joint.find("axis")
            axis = parse_xyz(
                axis_element.get("xyz") if axis_element is not None else None,
                (1.0, 0.0, 0.0),
            )
            if joint.get("type") in {"revolute", "continuous"}:
                motion[:3, :3] = axis_angle_matrix(axis, value)
            elif joint.get("type") == "prismatic":
                motion[:3, 3] = axis * value
            transforms[child_name] = (
                transforms[parent_name] @ origin_transform(joint.find("origin")) @ motion
            )
            pending.append(child_name)
    if set(transforms) != links:
        raise ValueError("URDF FK did not reach every link")
    return transforms


def binary_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"invalid STL: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * triangle_count:
        raise ValueError(f"expected binary STL: {path}")
    return np.ndarray(
        (triangle_count, 3, 3),
        dtype="<f4",
        buffer=data,
        offset=96,
        strides=(50, 12, 4),
    ).reshape(-1, 3).astype(float)


def cross_2d(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - origin[0]) * (b[1] - origin[1]) -
                 (a[1] - origin[1]) * (b[0] - origin[0]))


def convex_hull(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(points, decimals=10), axis=0)
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]
    if len(ordered) <= 2:
        return ordered
    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross_2d(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross_2d(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def validate_model_manifest(model_root: Path) -> tuple[Path, Path]:
    manifest_path = model_root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != MODEL_SCHEMA
        or manifest.get("model_revision") != MODEL_REVISION
        or int(manifest.get("link_count", 0)) != EXPECTED_LINK_COUNT
        or int(manifest.get("visual_mesh_count", 0)) != EXPECTED_VISUAL_MESH_COUNT
        or not math.isclose(
            float(manifest.get("tool0_offset_local_z_m", 0.0)),
            EXPECTED_TOOL0_OFFSET_M,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"invalid certified V3.2.2 model manifest: {manifest_path}")
    for entry in manifest.get("files", []):
        path = model_root / str(entry["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["size"])
            or sha256(path) != str(entry["sha256"])
        ):
            raise ValueError(f"certified model asset hash mismatch: {path}")
    urdf_path = model_root / "alfa-robot-v322-tool0151.urdf"
    if sha256(urdf_path) != EXPECTED_URDF_SHA256:
        raise ValueError(f"certified model URDF hash mismatch: {urdf_path}")
    asset_root = model_root / "alfa_robot_description"
    if not asset_root.is_dir():
        raise ValueError(f"certified model asset root is missing: {asset_root}")
    return urdf_path, asset_root


def validate_repository_model(repo_root: Path) -> tuple[Path, Path]:
    model_root = repo_root / "tools/v3_warehouse_return_20261009/model"
    urdf_path = model_root / "alfa-robot-v322-tool0151.urdf"
    if not urdf_path.is_file() or sha256(urdf_path) != EXPECTED_URDF_SHA256:
        raise ValueError(f"repository model URDF hash mismatch: {urdf_path}")
    asset_root = repo_root / "ros2_ws/src/alfa_robot_description"
    lock_path = asset_root / "config/upstream_description.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock.get("model_revision") != MODEL_REVISION
        or lock.get("upstream_commit") != "165df6fc15dbb754dad1e63b5afa641c843573ac"
    ):
        raise ValueError(f"repository description lock is not V3.2.2: {lock_path}")
    managed = dict(lock.get("managed_destination_files", {}))
    required = {
        name: expected
        for name, expected in managed.items()
        if name == "config/initial_positions.yaml"
        or name.startswith("model_sources/robot_v3_2_2/")
    }
    if len(required) != 70:
        raise ValueError(
            f"repository V3.2.2 asset lock is incomplete: {len(required)}/70 files"
        )
    for relative, expected in required.items():
        path = asset_root / relative
        if not path.is_file() or sha256(path) != str(expected):
            raise ValueError(f"repository model asset hash mismatch: {path}")
    return urdf_path, asset_root


def load_model_geometry(model_root: Path | None = None) -> ModelGeometry:
    if model_root is None:
        repo_root = find_repo_root()
        certificate_root = (
            repo_root
            / "data/ik_benchmark/v3_scoop_5x5/"
            "range_certification_v322_tool0151/model"
        )
        if (certificate_root / "MANIFEST.json").is_file():
            model_root = certificate_root
            urdf_path, asset_root = validate_model_manifest(model_root)
        else:
            model_root = repo_root / "tools/v3_warehouse_return_20261009/model"
            urdf_path, asset_root = validate_repository_model(repo_root)
    else:
        model_root = model_root.resolve()
        urdf_path, asset_root = validate_model_manifest(model_root)
    urdf_text = urdf_path.read_text(encoding="utf-8")
    root = ET.fromstring(urdf_text)
    links = root.findall("link")
    visual_count = sum(len(link.findall("visual")) for link in links)
    if len(links) != EXPECTED_LINK_COUNT or visual_count != EXPECTED_VISUAL_MESH_COUNT:
        raise ValueError("expanded URDF does not match the V3.2.2 shell contract")
    for side in ("left", "right"):
        joint = root.find(f"joint[@name='{side}_tool0_fixed']")
        if joint is None:
            raise ValueError(f"missing {side}_tool0_fixed")
        xyz = parse_xyz(joint.find("origin").get("xyz"), (0.0, 0.0, 0.0))
        if not np.allclose(xyz, [0.0, 0.0, EXPECTED_TOOL0_OFFSET_M], atol=1e-12):
            raise ValueError(f"wrong {side} Tool0 offset: {xyz.tolist()}")
    home_joints = yaml.safe_load(
        (asset_root / "config/initial_positions.yaml").read_text(encoding="utf-8")
    )["initial_positions"]
    transforms = urdf_fk(root, home_joints)
    point_cloud: list[np.ndarray] = []
    for link in links:
        link_name = link.get("name", "")
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError("primitive collision geometry requires explicit support")
            uri = mesh.get("filename", "")
            prefix = "package://alfa_robot_description/"
            if not uri.startswith(prefix):
                raise ValueError(f"unexpected mesh URI: {uri}")
            vertices = binary_stl_vertices(asset_root / uri.removeprefix(prefix))
            scale = parse_xyz(mesh.get("scale"), (1.0, 1.0, 1.0))
            transform = transforms[link_name] @ origin_transform(collision.find("origin"))
            point_cloud.append(
                (vertices * scale) @ transform[:3, :3].T + transform[:3, 3]
            )
    points = np.concatenate(point_cloud)
    footprint = convex_hull(points[:, :2])
    local_min = points.min(axis=0)
    local_max = points.max(axis=0)
    radius = float(np.linalg.norm(points[:, :2], axis=1).max())
    if local_min[2] < -1e-6:
        raise ValueError("V3.2.2 shell penetrates the warehouse floor")
    if local_max[2] + SAFETY_CLEARANCE_M >= WAREHOUSE_HEIGHT_M:
        raise ValueError("V3.2.2 shell does not clear the warehouse ceiling")
    return ModelGeometry(
        model_root=model_root,
        asset_root=asset_root,
        urdf_path=urdf_path,
        urdf_text=urdf_text,
        home_joints={str(k): float(v) for k, v in home_joints.items()},
        footprint=footprint,
        local_min=local_min,
        local_max=local_max,
        rotation_radius=radius,
        z_min=float(local_min[2]),
        z_max=float(local_max[2]),
    )


def safe_input_bounds(geometry: ModelGeometry) -> InputBounds:
    radius_with_clearance = geometry.rotation_radius + SAFETY_CLEARANCE_M
    return InputBounds(
        x_min=WAREHOUSE_OPENING_X_M - NOMINAL_BASE_X_M,
        x_max=CONTACT_X_M - radius_with_clearance - NOMINAL_BASE_X_M,
        y_min=-WAREHOUSE_HALF_WIDTH_M + radius_with_clearance,
        y_max=WAREHOUSE_HALF_WIDTH_M - radius_with_clearance,
    )


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def footprint_at(geometry: ModelGeometry, pose: Pose2D) -> np.ndarray:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    rotation = np.asarray([[c, -s], [s, c]])
    center = np.asarray([NOMINAL_BASE_X_M + pose.x, pose.y])
    return geometry.footprint @ rotation.T + center


def collision_reason(geometry: ModelGeometry, pose: Pose2D) -> str:
    if NOMINAL_BASE_X_M + pose.x < WAREHOUSE_OPENING_X_M - 1e-9:
        return "base_footprint is outside the warehouse opening"
    footprint = footprint_at(geometry, pose)
    if float(footprint[:, 1].min()) < -WAREHOUSE_HALF_WIDTH_M + SAFETY_CLEARANCE_M - 1e-9:
        return "production shell enters the -Y wall safety clearance"
    if float(footprint[:, 1].max()) > WAREHOUSE_HALF_WIDTH_M - SAFETY_CLEARANCE_M + 1e-9:
        return "production shell enters the +Y wall safety clearance"
    if float(footprint[:, 0].max()) > CONTACT_X_M - SAFETY_CLEARANCE_M + 1e-9:
        return "production shell enters the 5x5 contact-wall safety clearance"
    return ""


def validate_input_pose(
    geometry: ModelGeometry, x: float, y: float, yaw_deg: float
) -> Pose2D:
    values = (x, y, yaw_deg)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("x, y and Yaw must be finite")
    bounds = safe_input_bounds(geometry)
    if not (
        bounds.x_min - INPUT_BOUNDARY_TOLERANCE_M
        <= x
        <= bounds.x_max + INPUT_BOUNDARY_TOLERANCE_M
    ):
        raise ValueError(f"x must be in [{bounds.x_min:.6f}, {bounds.x_max:.6f}] m")
    if not (
        bounds.y_min - INPUT_BOUNDARY_TOLERANCE_M
        <= y
        <= bounds.y_max + INPUT_BOUNDARY_TOLERANCE_M
    ):
        raise ValueError(f"y must be in [{bounds.y_min:.6f}, {bounds.y_max:.6f}] m")
    if not bounds.yaw_min_deg <= yaw_deg <= bounds.yaw_max_deg:
        raise ValueError("Yaw must be in [-180, 180] deg")
    pose = Pose2D(
        x=min(bounds.x_max, max(bounds.x_min, x)),
        y=min(bounds.y_max, max(bounds.y_min, y)),
        yaw=wrap_angle(math.radians(yaw_deg)),
    )
    reason = collision_reason(geometry, pose)
    if reason:
        raise ValueError(reason)
    return pose


def quintic(progress: float) -> tuple[float, float]:
    value = min(1.0, max(0.0, progress))
    position = value**3 * (10.0 - 15.0 * value + 6.0 * value**2)
    derivative = 30.0 * value**2 * (1.0 - value) ** 2
    return position, derivative


def motion_duration(distance: float, max_speed: float, max_acceleration: float) -> float:
    distance = abs(distance)
    if distance < 1e-12:
        return 0.0
    # Quintic smoothstep maxima: ds/du=1.875, d2s/du2=10/sqrt(3).
    return max(
        0.80,
        1.875 * distance / max_speed,
        math.sqrt((10.0 / math.sqrt(3.0)) * distance / max_acceleration),
    )


def limited(value: float, maximum: float) -> float:
    return max(-maximum, min(maximum, value))


def rate_limited(value: float, previous: float, max_rate: float, dt: float) -> float:
    return previous + limited(value - previous, max_rate * dt)


class ClosedLoopBuilder:
    def __init__(self, start: Pose2D) -> None:
        self.pose = start
        self.time_s = 0.0
        self.v = 0.0
        self.w = 0.0
        self.samples = [TrajectorySample(
            time_s=0.0,
            x=start.x,
            y=start.y,
            yaw=start.yaw,
            linear_velocity_mps=0.0,
            angular_velocity_rad_s=0.0,
            stage="ready",
            reference_x=start.x,
            reference_y=start.y,
            reference_yaw=start.yaw,
        )]

    def append(self, stage: str, reference: Pose2D, v_raw: float, w_raw: float) -> None:
        self.v = rate_limited(
            limited(v_raw, MAX_LINEAR_SPEED_MPS),
            self.v,
            MAX_LINEAR_ACCEL_MPS2,
            CONTROL_DT_S,
        )
        self.w = rate_limited(
            limited(w_raw, MAX_ANGULAR_SPEED_RAD_S),
            self.w,
            MAX_ANGULAR_ACCEL_RAD_S2,
            CONTROL_DT_S,
        )
        yaw_mid = self.pose.yaw + 0.5 * self.w * CONTROL_DT_S
        self.pose = Pose2D(
            x=self.pose.x + self.v * math.cos(yaw_mid) * CONTROL_DT_S,
            y=self.pose.y + self.v * math.sin(yaw_mid) * CONTROL_DT_S,
            yaw=wrap_angle(self.pose.yaw + self.w * CONTROL_DT_S),
        )
        self.time_s += CONTROL_DT_S
        self.samples.append(TrajectorySample(
            time_s=self.time_s,
            x=self.pose.x,
            y=self.pose.y,
            yaw=self.pose.yaw,
            linear_velocity_mps=self.v,
            angular_velocity_rad_s=self.w,
            stage=stage,
            reference_x=reference.x,
            reference_y=reference.y,
            reference_yaw=reference.yaw,
        ))

    def rotate(self, target_yaw: float, stage: str) -> None:
        start = self.pose
        delta = wrap_angle(target_yaw - start.yaw)
        duration = motion_duration(delta, MAX_ANGULAR_SPEED_RAD_S, MAX_ANGULAR_ACCEL_RAD_S2)
        if duration == 0.0:
            return
        steps = max(1, int(math.ceil(duration / CONTROL_DT_S)))
        duration = steps * CONTROL_DT_S
        for index in range(steps):
            u = (index + 0.5) / steps
            position, derivative = quintic(u)
            reference = Pose2D(start.x, start.y, wrap_angle(start.yaw + delta * position))
            yaw_error = wrap_angle(reference.yaw - self.pose.yaw)
            self.append(
                stage,
                reference,
                0.0,
                delta * derivative / duration + YAW_FEEDBACK_GAIN * yaw_error,
            )
        self.settle(Pose2D(start.x, start.y, target_yaw), stage)

    def translate(self, target: Pose2D, signed_distance: float, stage: str) -> None:
        start = self.pose
        distance = abs(signed_distance)
        if distance < 1e-12:
            return
        direction = np.asarray([target.x - start.x, target.y - start.y]) / distance
        duration = motion_duration(
            distance, MAX_LINEAR_SPEED_MPS, MAX_LINEAR_ACCEL_MPS2
        )
        steps = max(1, int(math.ceil(duration / CONTROL_DT_S)))
        duration = steps * CONTROL_DT_S
        for index in range(steps):
            u = (index + 0.5) / steps
            position, derivative = quintic(u)
            reference = Pose2D(
                x=start.x + direction[0] * distance * position,
                y=start.y + direction[1] * distance * position,
                yaw=target.yaw,
            )
            dx, dy = reference.x - self.pose.x, reference.y - self.pose.y
            c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
            error_x = c * dx + s * dy
            error_y = -s * dx + c * dy
            yaw_error = wrap_angle(reference.yaw - self.pose.yaw)
            v_feedforward = signed_distance * derivative / duration
            direction_sign = 1.0 if signed_distance > 0.0 else -1.0
            self.append(
                stage,
                reference,
                v_feedforward + LINEAR_FEEDBACK_GAIN * error_x,
                YAW_FEEDBACK_GAIN * yaw_error
                + direction_sign * CROSS_TRACK_GAIN * error_y,
            )
        self.settle(target, stage)

    def settle(self, target: Pose2D, stage: str) -> None:
        for _ in range(int(math.ceil(4.0 / CONTROL_DT_S))):
            dx, dy = target.x - self.pose.x, target.y - self.pose.y
            distance = math.hypot(dx, dy)
            yaw_error = wrap_angle(target.yaw - self.pose.yaw)
            if distance < 5e-5 and abs(yaw_error) < math.radians(0.01):
                if abs(self.v) < 0.002 and abs(self.w) < 0.002:
                    self.v = self.w = 0.0
                    return
            c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
            error_x = c * dx + s * dy
            error_y = -s * dx + c * dy
            sign = 1.0 if error_x >= 0.0 else -1.0
            self.append(
                stage,
                target,
                LINEAR_FEEDBACK_GAIN * error_x,
                YAW_FEEDBACK_GAIN * yaw_error + sign * CROSS_TRACK_GAIN * error_y,
            )
        raise RuntimeError(f"closed-loop controller did not settle during {stage}")


def select_straight_motion(start: Pose2D) -> tuple[float, float, str, float, float]:
    distance = math.hypot(start.x, start.y)
    if distance < 1e-12:
        return start.yaw, 0.0, "stationary", 0.0, wrap_angle(-start.yaw)
    forward_heading = math.atan2(-start.y, -start.x)
    candidates = (
        (forward_heading, distance, "forward"),
        (wrap_angle(forward_heading + math.pi), -distance, "reverse"),
    )
    scored = []
    for heading, signed_distance, label in candidates:
        initial = wrap_angle(heading - start.yaw)
        final = wrap_angle(-heading)
        scored.append((abs(initial) + abs(final), label != "forward",
                       heading, signed_distance, label, initial, final))
    _, _, heading, signed_distance, label, initial, final = min(scored)
    return heading, signed_distance, label, initial, final


def plan_return(geometry: ModelGeometry, start: Pose2D) -> PlannedReturn:
    reason = collision_reason(geometry, start)
    if reason:
        raise ValueError(f"unsafe start pose: {reason}")
    heading, signed_distance, direction, initial, final = select_straight_motion(start)
    builder = ClosedLoopBuilder(start)
    builder.rotate(heading, "initial_yaw_alignment")
    builder.translate(Pose2D(0.0, 0.0, heading), signed_distance, "closed_loop_translation")
    builder.rotate(0.0, "final_yaw_zero")
    samples = tuple(builder.samples)
    for index, sample in enumerate(samples):
        pose = Pose2D(sample.x, sample.y, sample.yaw)
        reason = collision_reason(geometry, pose)
        if reason:
            raise RuntimeError(f"trajectory collision at sample {index}: {reason}")
    goal = samples[-1]
    if math.hypot(goal.x, goal.y) > 1e-3 or abs(wrap_angle(goal.yaw)) > math.radians(0.1):
        raise RuntimeError("closed-loop trajectory did not reach the origin tolerance")
    return PlannedReturn(
        start=start,
        drive_heading=heading,
        signed_distance=signed_distance,
        drive_direction=direction,
        initial_rotation=initial,
        final_rotation=final,
        samples=samples,
    )


def base_transform(pose: Pose2D) -> np.ndarray:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    transform = np.eye(4)
    transform[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    transform[:3, 3] = [NOMINAL_BASE_X_M + pose.x, pose.y, 0.0]
    return transform


def path_markdown(geometry: ModelGeometry, plan: PlannedReturn) -> str:
    bounds = safe_input_bounds(geometry)
    return (
        "# V3.2.2 warehouse return\n\n"
        f"- Input `(x, y, Yaw)`: **({plan.start.x:+.3f}, {plan.start.y:+.3f}, "
        f"{math.degrees(plan.start.yaw):+.2f} deg)**\n"
        "- Origin: chassis front clearance `0.90 m`, `y=0`, `Yaw=0 deg`\n"
        f"- Certified input x: `[{bounds.x_min:+.6f}, {bounds.x_max:+.6f}] m`\n"
        f"- Certified input y: `[{bounds.y_min:+.6f}, {bounds.y_max:+.6f}] m`\n"
        "- Certified input Yaw: `[-180, +180] deg`\n"
        f"- Drive: **{plan.drive_direction}**, straight distance "
        f"**{abs(plan.signed_distance):.3f} m**\n"
        f"- Initial / final rotation: `{math.degrees(plan.initial_rotation):+.2f} / "
        f"{math.degrees(plan.final_rotation):+.2f} deg`\n"
        f"- Duration: **{plan.duration_s:.2f} s**\n"
        f"- Shell: `{MODEL_REVISION}`, `{EXPECTED_LINK_COUNT} links / "
        f"{EXPECTED_VISUAL_MESH_COUNT} meshes`\n"
        f"- Rotation envelope: `{geometry.rotation_radius:.6f} m` + "
        f"`{SAFETY_CLEARANCE_M:.2f} m` clearance\n"
        "- Validation: every controller sample checked against the projected "
        "production-shell collision hull"
    )


def record_rerun(
    geometry: ModelGeometry,
    plan: PlannedReturn,
    output: Path,
    ready_only: bool = False,
) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    rerun_package = geometry.asset_root.as_posix().rstrip("/") + "/"
    urdf_text = geometry.urdf_text.replace(
        "package://alfa_robot_description/", rerun_package
    )
    rerun_src = find_repo_root() / "ros2_ws/src/alfa_robot_rerun"
    if str(rerun_src) not in sys.path:
        sys.path.insert(0, str(rerun_src))
    from alfa_robot_rerun.visualize_rerun import (
        UrdfRobot,
        log_robot_static_model,
        log_transform_matrix,
    )

    robot = UrdfRobot(urdf_text)
    if len(robot.links) != EXPECTED_LINK_COUNT:
        raise ValueError("Rerun robot link count differs from the certified model")
    rr.init("alfa_v322_warehouse_return", recording_id=output.stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(output))
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", contents=["/world/**"], name="Warehouse"),
            rrb.Vertical(
                rrb.TextDocumentView(origin="/status", name="Return status"),
                rrb.TimeSeriesView(origin="/control", name="Closed-loop commands"),
                row_shares=[0.62, 0.38],
            ),
            column_shares=[0.72, 0.28],
        ),
        collapse_panels=True,
    ))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    wall_t = 0.05
    warehouse_center_x = 0.5 * (WAREHOUSE_OPENING_X_M + WAREHOUSE_REAR_X_M)
    warehouse_length = WAREHOUSE_REAR_X_M - WAREHOUSE_OPENING_X_M
    rr.log("world/warehouse", rr.Boxes3D(
        centers=[
            [warehouse_center_x, WAREHOUSE_HALF_WIDTH_M + wall_t / 2, WAREHOUSE_HEIGHT_M / 2],
            [warehouse_center_x, -WAREHOUSE_HALF_WIDTH_M - wall_t / 2, WAREHOUSE_HEIGHT_M / 2],
            [warehouse_center_x, 0.0, WAREHOUSE_HEIGHT_M + wall_t / 2],
            [WAREHOUSE_REAR_X_M + wall_t / 2, 0.0, WAREHOUSE_HEIGHT_M / 2],
            [warehouse_center_x, 0.0, -wall_t / 2],
        ],
        half_sizes=[
            [warehouse_length / 2, wall_t / 2, WAREHOUSE_HEIGHT_M / 2],
            [warehouse_length / 2, wall_t / 2, WAREHOUSE_HEIGHT_M / 2],
            [warehouse_length / 2, WAREHOUSE_HALF_WIDTH_M + wall_t, wall_t / 2],
            [wall_t / 2, WAREHOUSE_HALF_WIDTH_M + wall_t, WAREHOUSE_HEIGHT_M / 2],
            [warehouse_length / 2, WAREHOUSE_HALF_WIDTH_M + wall_t, wall_t / 2],
        ],
        colors=[[90, 150, 205, 50]] * 5,
        labels=["+Y wall", "-Y wall", "ceiling", "rear wall", "floor"],
    ), static=True)
    box_centers = []
    for row in range(5):
        for column in range(5):
            box_centers.append([0.90, 0.80 - 0.40 * column, 1.80 - 0.40 * row])
    rr.log("world/contact_wall/boxes", rr.Boxes3D(
        centers=box_centers,
        half_sizes=[[0.15, 0.20, 0.20]],
        colors=[[230, 145, 45, 160]],
        labels=[str(index) for index in range(1, 26)],
        show_labels=False,
    ), static=True)
    bounds = safe_input_bounds(geometry)
    safe_abs_x_min = NOMINAL_BASE_X_M + bounds.x_min
    safe_abs_x_max = NOMINAL_BASE_X_M + bounds.x_max
    rr.log("world/safety/uniform_input_region", rr.Boxes3D(
        centers=[[(safe_abs_x_min + safe_abs_x_max) / 2, 0.0, 0.012]],
        half_sizes=[[(safe_abs_x_max - safe_abs_x_min) / 2, bounds.y_max, 0.008]],
        colors=[[45, 190, 115, 55]],
        labels=["safe x/y input region (all Yaw)"],
    ), static=True)
    rr.log("world/origin", rr.Arrows3D(
        origins=[[NOMINAL_BASE_X_M, 0.0, 0.03], [NOMINAL_BASE_X_M, 0.0, 0.03]],
        vectors=[[0.42, 0.0, 0.0], [0.0, 0.32, 0.0]],
        colors=[[235, 70, 60], [70, 220, 100]],
        labels=["origin +X", "origin +Y"],
    ), static=True)
    shown_samples = plan.samples[:1] if ready_only else plan.samples
    path = [[NOMINAL_BASE_X_M + sample.x, sample.y, 0.035] for sample in plan.samples]
    rr.log("world/plan/shortest_center_path", rr.LineStrips3D(
        strips=[path], colors=[[35, 215, 125, 230]], radii=[0.012],
        labels=["shortest collision-free center path"],
    ), static=True)
    rr.log("status", rr.TextDocument(
        path_markdown(geometry, plan) + ("\n\n**State: READY - waiting for `start`**" if ready_only else ""),
        media_type=rr.MediaType.MARKDOWN,
    ), static=ready_only)
    rr.log("world/robot", rr.Transform3D(translation=[0.0, 0.0, 0.0]), static=True)
    log_robot_static_model(robot, "world/robot", log_meshes=True)
    link_transforms = robot.fk(geometry.home_joints)
    for sample in shown_samples:
        rr.set_time("return_time", duration=sample.time_s)
        pose = Pose2D(sample.x, sample.y, sample.yaw)
        world_from_base = base_transform(pose)
        for link_name, transform in link_transforms.items():
            log_transform_matrix(f"world/robot/{link_name}", world_from_base @ transform)
        footprint = footprint_at(geometry, pose)
        footprint_line = np.vstack([footprint, footprint[0]])
        rr.log("world/robot_collision/production_shell_hull", rr.LineStrips3D(
            strips=[[[float(x), float(y), 0.025] for x, y in footprint_line]],
            colors=[[245, 200, 55, 220]], radii=[0.006],
        ))
        rr.log("control/linear_velocity_mps", rr.Scalars([sample.linear_velocity_mps]))
        rr.log("control/angular_velocity_rad_s", rr.Scalars([sample.angular_velocity_rad_s]))
        rr.log("control/position_error_m", rr.Scalars([math.hypot(sample.x, sample.y)]))
        rr.log("control/yaw_error_rad", rr.Scalars([abs(wrap_angle(sample.yaw))]))
        if not ready_only:
            rr.log("status", rr.TextDocument(
                path_markdown(geometry, plan)
                + f"\n\n**Stage:** `{sample.stage}`\n\n"
                + f"**Pose:** `({sample.x:+.3f}, {sample.y:+.3f}, "
                + f"{math.degrees(sample.yaw):+.2f} deg)`",
                media_type=rr.MediaType.MARKDOWN,
            ))
    rr.disconnect()


def open_rrd(path: Path) -> None:
    subprocess.Popen(
        ["rerun", "--new", str(path.resolve())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def prompt_value(
    label: str,
    lower: float,
    upper: float,
    unit: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> float:
    while True:
        raw = input_fn(f"请输入 {label} [{lower:+.6f}, {upper:+.6f}] {unit}: ").strip()
        try:
            value = float(raw)
        except ValueError:
            output_fn("输入无效：请输入数值。")
            continue
        if lower <= value <= upper:
            return value
        output_fn(f"超出范围：{label} 必须在 [{lower:+.6f}, {upper:+.6f}] {unit}。")


def output_stem(pose: Pose2D) -> str:
    def slug(value: float, scale: float = 1000.0) -> str:
        sign = "p" if value >= 0.0 else "m"
        return f"{sign}{abs(int(round(value * scale))):04d}"
    return f"x{slug(pose.x)}-y{slug(pose.y)}-yaw{slug(math.degrees(pose.yaw), 10.0)}"


def summary_dict(geometry: ModelGeometry, plan: PlannedReturn) -> dict:
    bounds = safe_input_bounds(geometry)
    final = plan.samples[-1]
    max_linear = max(abs(sample.linear_velocity_mps) for sample in plan.samples)
    max_angular = max(abs(sample.angular_velocity_rad_s) for sample in plan.samples)
    return {
        "schema": "alfa.v322_warehouse_return_demo.v1",
        "success": True,
        "model_revision": MODEL_REVISION,
        "model_urdf_sha256": EXPECTED_URDF_SHA256,
        "model_link_count": EXPECTED_LINK_COUNT,
        "model_visual_mesh_count": EXPECTED_VISUAL_MESH_COUNT,
        "origin_contract": {
            "vehicle_front_to_contact_m": 0.90,
            "base_x_map_m": NOMINAL_BASE_X_M,
            "y_m": 0.0,
            "yaw_deg": 0.0,
        },
        "safety_clearance_m": SAFETY_CLEARANCE_M,
        "production_shell_bounds_base": {
            "minimum": geometry.local_min.tolist(),
            "maximum": geometry.local_max.tolist(),
            "rotation_radius_m": geometry.rotation_radius,
        },
        "certified_input_bounds": asdict(bounds),
        "input_pose": {
            "x_m": plan.start.x,
            "y_m": plan.start.y,
            "yaw_deg": math.degrees(plan.start.yaw),
        },
        "plan": {
            "drive_direction": plan.drive_direction,
            "drive_heading_deg": math.degrees(plan.drive_heading),
            "straight_distance_m": abs(plan.signed_distance),
            "initial_rotation_deg": math.degrees(plan.initial_rotation),
            "final_rotation_deg": math.degrees(plan.final_rotation),
            "duration_s": plan.duration_s,
            "sample_count": len(plan.samples),
            "sampled_translation_length_m": plan.translation_length_m,
        },
        "controller": {
            "closed_loop": True,
            "dt_s": CONTROL_DT_S,
            "max_linear_speed_mps": MAX_LINEAR_SPEED_MPS,
            "max_angular_speed_rad_s": MAX_ANGULAR_SPEED_RAD_S,
            "max_linear_acceleration_mps2": MAX_LINEAR_ACCEL_MPS2,
            "max_angular_acceleration_rad_s2": MAX_ANGULAR_ACCEL_RAD_S2,
            "observed_max_linear_speed_mps": max_linear,
            "observed_max_angular_speed_rad_s": max_angular,
        },
        "validation": {
            "all_samples_collision_free": True,
            "final_position_error_m": math.hypot(final.x, final.y),
            "final_yaw_error_deg": abs(math.degrees(wrap_angle(final.yaw))),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, help="relative x in meters")
    parser.add_argument("--y", type=float, help="relative y in meters")
    parser.add_argument("--yaw-deg", type=float, help="relative Yaw in degrees")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument("--start", action="store_true", help="skip the terminal start gate")
    parser.add_argument("--no-open", action="store_true", help="do not open Rerun")
    args = parser.parse_args()

    geometry = load_model_geometry(args.model_root)
    bounds = safe_input_bounds(geometry)
    print(
        "可输入范围（V3.2.2生产外壳，任意Yaw，含5cm安全余量）：\n"
        f"  x = [{bounds.x_min:+.6f}, {bounds.x_max:+.6f}] m\n"
        f"  y = [{bounds.y_min:+.6f}, {bounds.y_max:+.6f}] m\n"
        f"  Yaw = [{bounds.yaw_min_deg:+.0f}, {bounds.yaw_max_deg:+.0f}] deg\n"
        f"  外壳旋转半径 = {geometry.rotation_radius:.6f} m"
    )
    x = args.x if args.x is not None else prompt_value("x", bounds.x_min, bounds.x_max, "m")
    y = args.y if args.y is not None else prompt_value("y", bounds.y_min, bounds.y_max, "m")
    yaw_deg = (
        args.yaw_deg if args.yaw_deg is not None
        else prompt_value("Yaw", bounds.yaw_min_deg, bounds.yaw_max_deg, "deg")
    )
    try:
        pose = validate_input_pose(geometry, x, y, yaw_deg)
        plan = plan_return(geometry, pose)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    output_dir = args.output_dir or (
        find_repo_root() / "data/ik_benchmark/v3_warehouse_return"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(pose)
    ready_rrd = output_dir / f"{stem}-ready.rrd"
    result_rrd = output_dir / f"{stem}-return.rrd"
    summary = output_dir / f"{stem}-summary.json"
    summary.write_text(
        json.dumps(summary_dict(geometry, plan), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    record_rerun(geometry, plan, ready_rrd, ready_only=True)
    if not args.no_open:
        open_rrd(ready_rrd)
    print(f"机器人已放置到输入位姿：{ready_rrd}")
    if args.ready_only:
        print("状态：READY。输入 start 后可启动闭环回原点。")
        return 0
    if not args.start:
        while input("请输入 start 启动: ").strip().lower() != "start":
            print("尚未启动；请输入命令 start。")
    record_rerun(geometry, plan, result_rrd, ready_only=False)
    if not args.no_open:
        open_rrd(result_rrd)
    print(
        f"SUCCESS：{len(plan.samples)}帧全部无碰撞，"
        f"终点位置误差={math.hypot(plan.samples[-1].x, plan.samples[-1].y):.6f}m，"
        f"Yaw误差={abs(math.degrees(wrap_angle(plan.samples[-1].yaw))):.6f}deg。\n"
        f"Rerun: {result_rrd}\nSummary: {summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
