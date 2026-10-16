#!/usr/bin/env python3
"""Benchmark official VAMP RRT-Connect serially and in process batches."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
import os
import platform
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


START = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
GOAL = [2.35, 1.0, 0.0, -0.8, 0.0, 2.5, 0.785]
SPHERES = [
    (0.55, 0.0, 0.25),
    (0.35, 0.35, 0.25),
    (0.0, 0.55, 0.25),
    (-0.55, 0.0, 0.25),
    (-0.35, -0.35, 0.25),
    (0.0, -0.55, 0.25),
    (0.35, -0.35, 0.25),
    (0.35, 0.35, 0.8),
    (0.0, 0.55, 0.8),
    (-0.35, 0.35, 0.8),
    (-0.55, 0.0, 0.8),
    (-0.35, -0.35, 0.8),
    (0.0, -0.55, 0.8),
    (0.35, -0.35, 0.8),
]


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio)))
    return ordered[index]


def run_batch(batch: tuple[int, int, int]) -> dict[str, Any]:
    worker, trials, warmup = batch
    import vamp

    robot, planner, settings, _ = vamp.configure_robot_and_planner_with_kwargs(
        "panda", "rrtc"
    )
    environment = vamp.Environment()
    for center in SPHERES:
        environment.add_sphere(vamp.Sphere(center, 0.2))
    if not robot.validate(START, environment) or not robot.validate(GOAL, environment):
        raise RuntimeError("VAMP benchmark endpoints are not collision-free")
    sampler = robot.halton()
    sampler.skip(worker * (trials + warmup + 97))
    for _ in range(warmup):
        result = planner(START, GOAL, environment, settings, sampler)
        if not result.solved:
            raise RuntimeError("VAMP warmup plan failed")

    internal_us: list[float] = []
    started = time.perf_counter()
    solved = 0
    for _ in range(trials):
        result = planner(START, GOAL, environment, settings, sampler)
        solved += bool(result.solved)
        internal_us.append(float(result.nanoseconds) / 1000.0)
    wall_seconds = time.perf_counter() - started
    return {
        "worker": worker,
        "trials": trials,
        "solved": solved,
        "wall_seconds": wall_seconds,
        "throughput_plans_s": trials / wall_seconds,
        "internal_us_median": statistics.median(internal_us),
        "internal_us_p95": percentile(internal_us, 0.95),
        "internal_us_max": max(internal_us),
    }


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v322_vamp_parallel/vamp-panda-throughput.json"),
    )
    args = parser.parse_args()
    if args.trials < 1 or args.workers < 1 or args.warmup < 0:
        parser.error("trials/workers must be positive and warmup non-negative")
    workers = min(args.workers, args.trials)

    serial = run_batch((0, args.trials, args.warmup))
    base, remainder = divmod(args.trials, workers)
    batches = [
        (worker, base + int(worker < remainder), args.warmup)
        for worker in range(workers)
    ]
    parallel_started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        parallel_workers = list(executor.map(run_batch, batches))
    parallel_wall_seconds = time.perf_counter() - parallel_started
    solved = sum(item["solved"] for item in parallel_workers)
    payload = {
        "schema": "alfa.v322_vamp_parallel_benchmark.v1",
        "success": serial["solved"] == args.trials and solved == args.trials,
        "scope": {
            "robot": "official_precompiled_panda",
            "planner": "VAMP RRT-Connect",
            "environment": "official sphere-cage geometry",
            "note": "This validates VAMP/AVX2 throughput, not V3.2.2 collision fidelity.",
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "vamp_planner": importlib.metadata.version("vamp-planner"),
        },
        "serial": serial,
        "parallel": {
            "workers": workers,
            "trials": args.trials,
            "solved": solved,
            "wall_seconds": parallel_wall_seconds,
            "throughput_plans_s": args.trials / parallel_wall_seconds,
            "speedup_vs_serial": serial["wall_seconds"] / parallel_wall_seconds,
            "worker_results": parallel_workers,
        },
    }
    atomic_write(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
