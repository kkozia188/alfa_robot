from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import threading


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def finite_tuple(values, size=None):
    result = tuple(float(value) for value in values)
    if size is not None and len(result) != size:
        raise ValueError(f"expected {size} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("non-finite value")
    return result


@dataclass(frozen=True)
class Pose:
    position: tuple = (0.0, 0.0, 0.0)
    quaternion_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "position", finite_tuple(self.position, 3))
        quaternion = finite_tuple(self.quaternion_wxyz, 4)
        norm = math.sqrt(sum(value * value for value in quaternion))
        if abs(norm - 1.0) > 1e-5:
            raise ValueError("quaternion must be normalized (wxyz)")
        object.__setattr__(self, "quaternion_wxyz", quaternion)


@dataclass(frozen=True)
class RobotState:
    joint_names: tuple
    positions: tuple
    stamp_ns: int
    base_pose: Pose = Pose()

    def __post_init__(self):
        names = tuple(self.joint_names)
        positions = finite_tuple(self.positions, len(names))
        if not names or len(set(names)) != len(names) or not all(names):
            raise ValueError("joint names must be non-empty and unique")
        if not isinstance(self.stamp_ns, int) or self.stamp_ns < 0:
            raise ValueError("invalid state timestamp")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "positions", positions)

    @classmethod
    def from_mapping(cls, positions, stamp_ns, base_pose=Pose()):
        return cls(tuple(positions), tuple(positions.values()), stamp_ns, base_pose)

    def ordered(self, names):
        positions = dict(zip(self.joint_names, self.positions))
        missing = set(names) - positions.keys()
        if missing:
            raise ValueError(f"missing joints: {sorted(missing)}")
        return tuple(positions[name] for name in names)

    def with_positions(self, updates, base_pose=None):
        positions = dict(zip(self.joint_names, self.positions))
        positions.update(updates)
        return RobotState.from_mapping(positions, self.stamp_ns,
                                       self.base_pose if base_pose is None else base_pose)

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    dimensions_m: tuple
    pose: Pose

    def __post_init__(self):
        dimensions = finite_tuple(self.dimensions_m, 3)
        if not self.object_id or any(value <= 0 for value in dimensions):
            raise ValueError("invalid object identity or dimensions")
        object.__setattr__(self, "dimensions_m", dimensions)


@dataclass(frozen=True)
class MeshObject:
    object_id: str
    vertices_m: tuple
    faces: tuple
    pose: Pose = Pose()

    def __post_init__(self):
        vertices = tuple(finite_tuple(vertex, 3) for vertex in self.vertices_m)
        faces = tuple(tuple(face) for face in self.faces)
        if not self.object_id or len(vertices) < 4 or not faces:
            raise ValueError("invalid mesh identity or geometry")
        for face in faces:
            if len(face) != 3 or any(not isinstance(index, int) or not 0 <= index < len(vertices)
                                     for index in face):
                raise ValueError("invalid triangle indices")
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "faces", faces)


@dataclass(frozen=True)
class AttachedObject:
    object_id: str
    dimensions_m: tuple
    parent_link: str
    tool_to_object: Pose
    touch_links: tuple = ()

    def __post_init__(self):
        dimensions = finite_tuple(self.dimensions_m, 3)
        if not self.object_id or not self.parent_link or any(value <= 0 for value in dimensions):
            raise ValueError("invalid attachment")
        object.__setattr__(self, "dimensions_m", dimensions)
        object.__setattr__(self, "touch_links", tuple(self.touch_links))


@dataclass(frozen=True)
class CollisionPolicy:
    exclude_task_objects_before_contact: bool = True
    defer_payload_until_extract_end: bool = True
    ground_z_m: float = 0.0
    max_box_tilt_deg: float = 89.0
    ground_support_link: str = "base_link"

    def __post_init__(self):
        finite_tuple((self.ground_z_m, self.max_box_tilt_deg))
        if not 0 <= self.max_box_tilt_deg <= 180:
            raise ValueError("invalid tilt limit")


@dataclass(frozen=True)
class SceneSnapshot:
    model_id: str
    state: RobotState
    objects: tuple
    attachments: tuple = ()
    frame_id: str = "map"
    revision: int = 0
    policy: CollisionPolicy = CollisionPolicy()
    meshes: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "attachments", tuple(self.attachments))
        object.__setattr__(self, "meshes", tuple(self.meshes))
        identities = [item.object_id for item in self.objects + self.attachments + self.meshes]
        if len(set(identities)) != len(identities):
            raise ValueError("an object cannot be both world and attached, or duplicated")
        if not self.model_id or not self.frame_id or self.revision < 0:
            raise ValueError("invalid scene identity")

    @property
    def geometry_key(self):
        return digest({
            "model_id": self.model_id, "frame_id": self.frame_id,
            "objects": sorted((asdict(item) for item in self.objects), key=lambda item: item["object_id"]),
            "attachments": sorted((asdict(item) for item in self.attachments), key=lambda item: item["object_id"]),
            "meshes": sorted((asdict(item) for item in self.meshes), key=lambda item: item["object_id"]),
            "base_pose": asdict(self.state.base_pose), "policy": asdict(self.policy),
        })

    @property
    def identity(self):
        return digest((self.geometry_key, self.revision, self.state.identity))

    def object(self, object_id):
        return next(item for item in self.objects if item.object_id == object_id)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        state = dict(data["state"])
        state["base_pose"] = Pose(**state.get("base_pose", {}))
        objects = tuple(SceneObject(item["object_id"], item["dimensions_m"], Pose(**item["pose"]))
                        for item in data["objects"])
        attachments = tuple(AttachedObject(
            item["object_id"], item["dimensions_m"], item["parent_link"],
            Pose(**item["tool_to_object"]), item.get("touch_links", ()),
        ) for item in data.get("attachments", ()))
        meshes = tuple(MeshObject(item["object_id"], item["vertices_m"], item["faces"], Pose(**item["pose"]))
                       for item in data.get("meshes", ()))
        return cls(data["model_id"], RobotState(**state), objects, attachments,
                   data.get("frame_id", "map"), data.get("revision", 0),
                   CollisionPolicy(**data.get("policy", {})), meshes)


class SceneStore:
    def __init__(self, snapshot):
        self._snapshot = snapshot
        self._lock = threading.RLock()

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def _commit(self, expected_revision, **changes):
        with self._lock:
            if expected_revision != self._snapshot.revision:
                raise ValueError("scene revision changed")
            candidate = replace(self._snapshot, revision=expected_revision + 1, **changes)
            self._snapshot = candidate
            return candidate

    def update_state(self, state, expected_revision):
        return self._commit(expected_revision, state=state)

    def upsert(self, item, expected_revision):
        with self._lock:
            objects = tuple(existing for existing in self._snapshot.objects
                            if existing.object_id != item.object_id) + (item,)
            return self._commit(expected_revision, objects=objects)

    def attach(self, attachment, expected_revision):
        return self.attach_many((attachment,), expected_revision)

    def attach_many(self, attachments, expected_revision):
        with self._lock:
            attachments = tuple(attachments)
            for attachment in attachments:
                source = self._snapshot.object(attachment.object_id)
                if source.dimensions_m != attachment.dimensions_m:
                    raise ValueError("attachment dimensions differ from world object")
            identities = {attachment.object_id for attachment in attachments}
            objects = tuple(item for item in self._snapshot.objects
                            if item.object_id not in identities)
            return self._commit(expected_revision, objects=objects,
                                attachments=self._snapshot.attachments + attachments)

    def release(self, object_id, expected_revision, world_pose=None):
        with self._lock:
            attachment = next(item for item in self._snapshot.attachments if item.object_id == object_id)
            attachments = tuple(item for item in self._snapshot.attachments if item.object_id != object_id)
            objects = self._snapshot.objects
            if world_pose is not None:
                objects += (SceneObject(object_id, attachment.dimensions_m, world_pose),)
            return self._commit(expected_revision, objects=objects, attachments=attachments)

    def is_current(self, snapshot):
        with self._lock:
            return self._snapshot.identity == snapshot.identity
