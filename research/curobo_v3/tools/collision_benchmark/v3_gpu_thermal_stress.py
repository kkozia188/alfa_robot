import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import threading
import time

import numpy as np
import torch
import yaml

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from v3_collision_throughput import build_scene, gpu_check, limits_from_urdf


TELEMETRY_FIELDS = [
    "temperature.gpu", "clocks.sm", "clocks.mem", "power.draw", "power.limit",
    "utilization.gpu", "pstate", "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown", "clocks_throttle_reasons.sw_power_cap",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--window", type=float, default=10)
    parser.add_argument("--replays-per-block", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if args.duration <= 0 or args.batch_size <= 0 or args.replays_per_block <= 0 or args.window <= 0:
        parser.error("duration, batch size, replay count and window must be positive")
    torch.set_grad_enabled(False)
    robot = yaml.safe_load(args.robot_config.read_text())
    kin = robot.get("robot_cfg", robot)["kinematics"]
    identity = json.loads(json.dumps(robot))
    for key in ("asset_root_path", "urdf_path"):
        identity.get("robot_cfg", identity)["kinematics"].pop(key, None)
    kin["urdf_path"] = str(args.urdf.resolve())
    kin["asset_root_path"] = str(args.urdf.resolve().parent)
    lower, upper = limits_from_urdf(args.urdf)
    host_states = np.random.default_rng(args.seed).uniform(
        lower, upper, size=(4, args.batch_size, len(lower))
    ).astype(np.float32)
    states = torch.as_tensor(host_states, device="cuda")
    inputs = states[0].clone()
    checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=robot, scene_model=build_scene(), n_cuboids=40, n_meshes=0,
        collision_activation_distance=0.0,
    ))
    setup_started = time.perf_counter()
    for _ in range(5):
        reference = gpu_check(checker, inputs)
    torch.cuda.synchronize()
    reference = reference.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = gpu_check(checker, inputs)
    graph.replay()
    torch.cuda.synchronize()
    reference_error = float((reference - result).abs().max().item())
    if not torch.allclose(reference, result, atol=1e-6, rtol=1e-5) or not torch.equal(reference > 0, result > 0):
        raise RuntimeError("CUDA Graph result differs from eager collision checks")
    for batch_index in range(1, len(states)):
        inputs.copy_(states[batch_index])
        reference = gpu_check(checker, inputs).clone()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        reference_error = max(reference_error, float((reference - result).abs().max().item()))
        if not torch.allclose(reference, result, atol=1e-6, rtol=1e-5) or not torch.equal(reference > 0, result > 0):
            raise RuntimeError("Rotated input CUDA Graph result differs from eager checks")
    telemetry = []
    errors = []
    stop = threading.Event()
    started = time.perf_counter()

    def monitor():
        while not stop.is_set():
            try:
                output = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=" + ",".join(TELEMETRY_FIELDS),
                    "--format=csv,noheader,nounits", "--id=0",
                ], text=True, timeout=5).strip().split(", ")
                telemetry.append(dict(time_s=time.perf_counter() - started,
                                      **dict(zip(TELEMETRY_FIELDS, output))))
            except (subprocess.SubprocessError, OSError) as error:
                errors.append(str(error))
            stop.wait(1)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    windows = []
    total_batches = 0
    window_batches = 0
    window_started = started
    block = 0
    while time.perf_counter() - started < args.duration:
        inputs.copy_(states[block % len(states)])
        for _ in range(args.replays_per_block):
            graph.replay()
        torch.cuda.synchronize()
        total_batches += args.replays_per_block
        window_batches += args.replays_per_block
        block += 1
        now = time.perf_counter()
        if now - window_started >= args.window or now - started >= args.duration:
            duration = now - window_started
            row = dict(start_s=window_started - started, end_s=now - started,
                       elapsed_s=duration, batches=window_batches,
                       states=window_batches * args.batch_size,
                       states_per_s=window_batches * args.batch_size / duration)
            windows.append(row)
            print(json.dumps(row), flush=True)
            window_started, window_batches = now, 0
    elapsed = time.perf_counter() - started
    stop.set()
    monitor_thread.join(timeout=6)
    report = dict(
        hostname=platform.node(), gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        python_version=platform.python_version(), parameters=vars(args) | {
            "robot_config": str(args.robot_config), "urdf": str(args.urdf), "output": str(args.output),
        }, elapsed_s=elapsed, states_checked=total_batches * args.batch_size,
        states_per_s=total_batches * args.batch_size / elapsed,
        model_hash=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        input_hash=hashlib.sha256(host_states.tobytes()).hexdigest(),
        script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sphere_count=sum(len(spheres) for spheres in kin["collision_spheres"].values()),
        scene_cuboids=29, graph_reference_equal=True, graph_max_absolute_error=reference_error,
        warmup_capture_s=started - setup_started, windows=windows, telemetry=telemetry,
        telemetry_errors=errors, collision_fraction=float((result > 0).float().mean().item()),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        workload="cuRobo FK + self/world sphere collision + joint bounds; four deterministic random batches rotated; CUDA Graph; no tree/IK/planning",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "elapsed_s": elapsed,
                      "states_per_s": report["states_per_s"]}), flush=True)


if __name__ == "__main__":
    main()
