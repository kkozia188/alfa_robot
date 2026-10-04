#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import struct
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import JointState, Pose


JOINT_NAMES = [
    "updown",
    *[f"left_joint{index}" for index in range(1, 8)],
    *[f"right_joint{index}" for index in range(1, 8)],
]


def limits_from_urdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    joints = {joint.attrib["name"]: joint for joint in ET.parse(path).getroot().findall("joint")}
    lower, upper = [], []
    for name in JOINT_NAMES:
        limit = joints[name].find("limit")
        lower.append(float(limit.attrib["lower"]))
        upper.append(float(limit.attrib["upper"]))
    return np.asarray(lower), np.asarray(upper)


def build_scene() -> Scene:
    front = 1.4
    cuboids = []
    for box in range(25):
        cuboids.append(Cuboid(
            name=f"wall_box_{box}",
            pose=[
                front + 0.15, (box % 5 - 2) * 0.41, 0.20 + (box // 5) * 0.41,
                1.0, 0.0, 0.0, 0.0,
            ],
            dims=[0.30, 0.40, 0.40],
        ))
    wall_back = front + 0.30 + 1e-6
    for name, center, dims in (
        ("left_wall", [wall_back - 2.0, -1.25, 1.2], [4.0, 0.1, 2.4]),
        ("right_wall", [wall_back - 2.0, 1.25, 1.2], [4.0, 0.1, 2.4]),
        ("front_wall", [wall_back + 0.05, 0.0, 1.2], [0.1, 2.6, 2.4]),
        ("ceiling", [wall_back - 2.0, 0.0, 2.45], [4.2, 2.6, 0.1]),
    ):
        cuboids.append(Cuboid(
            name=name,
            pose=[*center, 1.0, 0.0, 0.0, 0.0],
            dims=dims,
        ))
    return Scene(cuboid=cuboids)


def write_binary(path: Path, states: np.ndarray) -> None:
    with path.open("wb") as stream:
        stream.write(struct.pack("Q", len(states)))
        stream.write(np.asarray(states, dtype=np.float64).tobytes(order="C"))


def gpu_check(checker: RobotCollisionChecker, values: torch.Tensor) -> torch.Tensor:
    horizon = values.shape[0]
    q = values.unsqueeze(0)
    checker.setup_batch_tensors(1, horizon)
    state = checker.kinematics.compute_kinematics(
        JointState.from_position(q, joint_names=JOINT_NAMES)
    )
    spheres = state.robot_spheres.view(1, horizon, -1, 4)
    checker.collision_constraint.update_num_spheres(
        spheres.shape[2], batch_size=1, horizon=horizon
    )
    return (
        checker.get_self_collision(spheres).reshape(1, horizon, -1).sum(-1)
        + checker.collision_constraint.forward(state).reshape(1, horizon, -1).sum(-1)
        + checker.get_bound(q).reshape(1, horizon, -1).sum(-1)
    )


def benchmark_gpu(checker, states: torch.Tensor, batch_size: int) -> dict:
    batch = states[:batch_size]
    for _ in range(4):
        gpu_check(checker, batch)
    torch.cuda.synchronize()
    repeats = max(10, int(200000 / batch_size))
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    result = None
    for _ in range(repeats):
        result = gpu_check(checker, batch)
    finished.record()
    torch.cuda.synchronize()
    elapsed_s = started.elapsed_time(finished) / 1000.0
    return {
        "backend": "curobo_gpu_spheres",
        "batch_size": batch_size,
        "repeats": repeats,
        "checks": repeats * batch_size,
        "elapsed_s": elapsed_s,
        "checks_per_s": repeats * batch_size / elapsed_s,
        "collision_fraction": float((result > 0).float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    parser.add_argument("--fcl-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-states", type=int, default=65536)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()

    lower, upper = limits_from_urdf(args.urdf)
    rng = np.random.default_rng(20260929)
    states = rng.uniform(lower, upper, size=(args.num_states, len(JOINT_NAMES)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.output.parent / "states.bin"
    write_binary(state_path, states)

    fcl_results = []
    for threads in (1, 32):
        fcl_results.append(json.loads(subprocess.check_output([
            str(args.fcl_binary), str(args.urdf), str(args.srdf), str(state_path),
            str(args.duration), str(threads),
        ], text=True)))

    robot = yaml.safe_load(args.robot_config.read_text())
    checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=robot,
        scene_model=build_scene(),
        n_cuboids=40,
        n_meshes=0,
        collision_activation_distance=0.0,
    ))
    device_states = torch.tensor(states, device="cuda", dtype=torch.float32)
    gpu_results = []
    for batch_size in (256, 1024, 4096, 16384, 65536):
        try:
            gpu_results.append(benchmark_gpu(checker, device_states, batch_size))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            gpu_results.append({"backend": "curobo_gpu_spheres", "batch_size": batch_size,
                                "error": "out_of_memory"})
            break
    valid_gpu = [row for row in gpu_results if "checks_per_s" in row]
    sphere_count = sum(
        len(rows) for rows in robot["kinematics"]["collision_spheres"].values()
    )
    best_fcl = max(fcl_results, key=lambda row: row["checks_per_s"])
    best_gpu = max(valid_gpu, key=lambda row: row["checks_per_s"])
    report = {
        "description_commit": "bd1e455",
        "joint_names": JOINT_NAMES,
        "states": args.num_states,
        "scene_cuboids": 29,
        "ground_support_collision": "excluded",
        "fcl": fcl_results,
        "best_fcl": best_fcl,
        "curobo_batches": gpu_results,
        "curobo_spheres": sphere_count,
        "best_curobo": best_gpu,
        "speedup_vs_single_thread_fcl": best_gpu["checks_per_s"] / fcl_results[0]["checks_per_s"],
        "speedup_vs_best_fcl": best_gpu["checks_per_s"] / best_fcl["checks_per_s"],
        "gpu": torch.cuda.get_device_name(0),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
