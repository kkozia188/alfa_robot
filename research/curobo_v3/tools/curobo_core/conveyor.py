from dataclasses import replace
from pathlib import Path

from .instances import SceneInstance
from .scene import MeshObject, Pose


DEFAULT_CONVEYOR_MESH = Path(__file__).resolve().parents[2] / "models/conveyor/roller_conveyor_collision_solid_v2.stl"


def conveyor_pair(mesh_path=DEFAULT_CONVEYOR_MESH, pose=Pose((-1.9, 0.0, 0.0)), stamp_ns=0,
                  gap_m=0.05, collision_enabled=True):
    import trimesh

    if gap_m < 0:
        raise ValueError("conveyor gap must be non-negative")
    mesh = trimesh.load(mesh_path, force="mesh")
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("conveyor collision mesh must be closed and consistently oriented")
    width = float(mesh.extents[1])
    offset = (width + gap_m) / 2
    geometry = MeshObject("left", mesh.vertices.tolist(), mesh.faces.tolist(), Pose((0.0, offset, 0.0)))
    return SceneInstance("conveyor_pair", pose, stamp_ns,
                         (geometry, replace(geometry, object_id="right", pose=Pose((0.0, -offset, 0.0)))),
                         collision_enabled)
