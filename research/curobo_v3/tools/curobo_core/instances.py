from dataclasses import dataclass, replace
import math
import threading

from .adapter import matrix_pose, pose_matrix
from .scene import MeshObject, Pose, SceneObject


@dataclass(frozen=True)
class SceneInstance:
    instance_id: str
    pose: Pose
    stamp_ns: int
    parts: tuple
    collision_enabled: bool = True
    frame_id: str = "map"

    def __post_init__(self):
        parts = tuple(self.parts)
        if not self.instance_id or not parts or not isinstance(self.stamp_ns, int) or self.stamp_ns < 0:
            raise ValueError("invalid instance identity or timestamp")
        if not all(isinstance(part, (MeshObject, SceneObject)) for part in parts):
            raise ValueError("unsupported instance geometry")
        if len({part.object_id for part in parts}) != len(parts):
            raise ValueError("duplicate instance part")
        object.__setattr__(self, "parts", parts)

    def world_parts(self):
        if self.frame_id != "map":
            raise ValueError("instance pose must be in map; transform missing")
        return tuple(replace(part, object_id=f"{self.instance_id}/{part.object_id}",
                             pose=matrix_pose(pose_matrix(self.pose) @ pose_matrix(part.pose)))
                     for part in self.parts)


class LiveScene:
    def __init__(self, template):
        self._template = template
        self._state = template.state
        self._instances = {}
        self._revision = template.revision
        self._lock = threading.RLock()

    def update_state(self, state):
        with self._lock:
            self._state = state
            self._revision += 1

    def upsert(self, instance):
        with self._lock:
            self._instances[instance.instance_id] = instance
            self._revision += 1

    def view(self):
        with self._lock:
            return self._state, tuple(self._instances.values()), self._revision

    def capture(self, now_ns, max_age_s, instance_ids=None):
        if not isinstance(now_ns, int) or not math.isfinite(max_age_s) or max_age_s < 0:
            raise ValueError("invalid capture time or age limit")
        with self._lock:
            selected = tuple(self._instances) if instance_ids is None else tuple(instance_ids)
            if len(set(selected)) != len(selected):
                raise ValueError("duplicate selected instance")
            if any(name not in self._instances for name in selected):
                raise ValueError("required instance missing")
            instances = tuple(self._instances[name] for name in selected)
            stamps = [self._state.stamp_ns] + [item.stamp_ns for item in instances if item.collision_enabled]
            if any(now_ns - stamp < 0 or now_ns - stamp > max_age_s * 1e9 for stamp in stamps):
                raise ValueError("selected scene state is stale or timestamp is in the future")
            objects, meshes = list(self._template.objects), list(self._template.meshes)
            for instance in instances:
                if instance.collision_enabled:
                    for part in instance.world_parts():
                        (meshes if isinstance(part, MeshObject) else objects).append(part)
            return replace(self._template, state=self._state, objects=tuple(objects),
                           meshes=tuple(meshes), revision=self._revision)
