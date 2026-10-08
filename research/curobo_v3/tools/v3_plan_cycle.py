"""Headless planning demo. Produces untimed joint frames, never hardware commands."""

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from curobo_core.contracts import PlanRequest, PlannerAssets

ENTRY_TIME = time.perf_counter()


def gpu_processes():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                      "--format=csv,noheader"], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()
            and int(line.split(",", 1)[0]) != os.getpid()]


def asset_hashes(args):
    files = {getattr(args, name) for name in (
        "robot_config", "mobile_robot_config", "urdf", "named_poses", "box_fit", "target_poses")}
    files.update(args.urdf.parent.rglob("*.stl"))
    files.update(args.urdf.parent.rglob("*.STL"))
    files.update((Path(__file__).parent / "curobo_core").glob("*.py"))
    files.update(Path(__file__).parent / name for name in (
        "v3_batched_loaded_search.py", "v3_plan_cycle.py", "v3_analytic_bridge.cpp"))
    files.add(args.robot_config.parent / "libv3_analytic_bridge.so")
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


def summarize(results, expected):
    times = [result["measured_wall_ms"] for result in results]
    return {"expected_runs": expected, "completed_runs": len(results),
            "successes": sum(bool(result.get("success")) for result in results),
            "passed": len(results) == expected and all(result.get("success") for result in results),
            "median_ms": statistics.median(times) if times else None,
            "worst_ms": max(times) if times else None,
            "retry_counts": [result.get("retry_count", 0) for result in results],
            "failures": [{"seed": result.get("seed"), "repeat": result.get("repeat"),
                          "error": result.get("error")} for result in results if not result.get("success")]}


def main():
    defaults = PlannerAssets()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("robot_config", "mobile_robot_config", "urdf", "named_poses", "box_fit", "target_poses"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=getattr(defaults, name))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--preflight-only", action="store_true",
                        help="CPU asset/contact inspection only; not a planning or acceptance run")
    parser.add_argument("--left-box", type=int, default=24)
    parser.add_argument("--right-box", type=int, default=20)
    parser.add_argument("--mode", choices=("dual_cycle", "sequential_unload"), default="dual_cycle")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--support-elbow-rise-m", type=float,
                        help="explicit sequential support IK elbow-height preference; default unchanged")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--wait-for-gpu-lock", action="store_true",
                        help="queue behind the shared GPU lock; lock waiting is not request planning time")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or any(not 0 <= seed < 2**32 for seed in (args.seeds or [])):
        parser.error("runs must be positive; seeds must be unsigned 32-bit integers")
    supplied = PlanRequest.from_dict(json.loads(args.request.read_text())) if args.request else None
    sequential = (supplied.mode if supplied else args.mode) == "sequential_unload"
    if args.support_elbow_rise_m is not None and not sequential:
        parser.error("--support-elbow-rise-m requires sequential_unload")
    seeds = (args.seeds or [supplied.seed if supplied else 11]) if sequential else [None]
    document = {"results": [], "command": sys.argv, "time_parameterized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        if sequential:
            document["acceptance"] = summarize(document["results"], len(seeds)*args.runs)
            document["acceptance"]["nine_run_protocol"] = seeds == [11, 29, 41] and args.runs == 3
            document["acceptance"]["passed"] &= document["acceptance"]["nine_run_protocol"]
        temporary = args.output.with_suffix(args.output.suffix+".tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.output)

    if args.preflight_only:
        if not sequential:
            parser.error("--preflight-only is for sequential_unload")
        from curobo_core.planner import FullCyclePlanner
        from curobo_core.sequential import contact_sphere_overlaps, mesh_contact_proof, sequential_request, tool_mesh_vertices
        planner = FullCyclePlanner(args)
        request = supplied or sequential_request(planner, seeds[0])
        if args.support_elbow_rise_m is not None:
            request = replace(request, support_elbow_rise_m=args.support_elbow_rise_m)
        overlaps = [hit for side, box_id in request.tasks for hit in contact_sphere_overlaps(
            planner, side, request.snapshot.object(f"wall_box_{box_id:02d}"))]
        import numpy as np
        from curobo_core.adapter import pose_matrix
        from curobo_core.scene import Pose
        from v3_wall_ik_benchmark import canonical_side_suction_quaternion_wxyz
        proofs = {}
        for side, box_id in request.tasks:
            box = request.snapshot.object(f"wall_box_{box_id:02d}")
            contact = Pose((box.pose.position[0]-box.dimensions_m[0]/2, *box.pose.position[1:]),
                           tuple(canonical_side_suction_quaternion_wxyz(side)))
            proofs[side] = mesh_contact_proof(tool_mesh_vertices(planner, side),
                np.linalg.inv(pose_matrix(box.pose)) @ pose_matrix(contact), box.dimensions_m, .001)
        unsupported = not all(proof["valid"] for proof in proofs.values())
        document.update(request=request.to_dict(), asset_sha256=asset_hashes(args),
                        cpu_preflight={"sphere_overlaps_not_whole_box_exemptions": overlaps,
                                       "exact_tool_mesh_proofs": proofs, "native_gpu_verified": False,
                                       "error": "CONTACT_MODEL_UNREPRESENTABLE" if unsupported else None})
        save()
        print(json.dumps(document["cpu_preflight"], ensure_ascii=False), flush=True)
        raise SystemExit(1 if unsupported else 0)

    # Host-global advisory lock plus external workload check. Never kill other users' jobs.
    with open("/tmp/sevenova-curobo-gpu.lock", "a") as lock:
        if sequential:
            try:
                lock_started = time.perf_counter()
                fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_gpu_lock else fcntl.LOCK_NB))
                document["gpu_lock_wait_ms"] = (time.perf_counter()-lock_started)*1000
                other = gpu_processes()
                document["gpu_preflight"] = other
                if other:
                    raise RuntimeError("GPU occupied: " + "; ".join(other))
            except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
                document["admission_error"] = str(error)
                save()
                print(document["admission_error"], file=sys.stderr)
                raise SystemExit(2)
        import torch
        from curobo_core.planner import FullCyclePlanner
        from curobo_core.sequential import sequential_request
        gpu_started = time.perf_counter()
        torch.cuda.synchronize()
        document["gpu_initialization_ms"] = (time.perf_counter()-gpu_started)*1000
        started = time.perf_counter()
        planner = FullCyclePlanner(args)
        torch.cuda.synchronize()
        document["initialization_ms"] = (time.perf_counter()-started)*1000
        document["process_to_ready_ms"] = (time.perf_counter()-ENTRY_TIME)*1000-document.get("gpu_lock_wait_ms", 0.)
        if sequential:
            document["asset_sha256"] = asset_hashes(args)
        request = supplied or (sequential_request(planner, seeds[0]) if sequential else planner.demo_request(
            {"left": args.left_box, "right": args.right_box}))
        if args.support_elbow_rise_m is not None:
            request = replace(request, support_elbow_rise_m=args.support_elbow_rise_m)
        document["request"] = request.to_dict()
        for seed in seeds:
            for repeat in range(1, args.runs+1):
                if sequential:
                    other = gpu_processes()
                    if other:
                        document["admission_error"] = "GPU became occupied: " + "; ".join(other)
                        save()
                        raise SystemExit(2)
                current = replace(request, seed=seed) if sequential else request
                torch.cuda.synchronize()
                started = time.perf_counter()
                try:
                    result = planner.plan_request(current, lambda message: print(message, file=sys.stderr, flush=True))
                except Exception as error:
                    result = planner.last_partial or {"frames": []}
                    result.update(success=False, error={"code": "INTERNAL_ERROR", "message": str(error)},
                                  traceback=traceback.format_exc())
                torch.cuda.synchronize()
                result["measured_wall_ms"] = (time.perf_counter()-started)*1000
                result.update(seed=seed, repeat=repeat, request_id=current.identity,
                              warm_request=bool(document["results"]))
                document["results"].append(result)
                if sequential:
                    path = args.output.parent / (args.output.stem+f"-s{seed}-r{repeat}.json")
                    path.write_text(json.dumps({"request": current.to_dict(), "result": result},
                                               ensure_ascii=False, indent=2, allow_nan=False)+"\n")
                save()
                if not sequential and not result["success"]:
                    break
        save()
        print(json.dumps(document.get("acceptance", [{key: result.get(key) for key in (
            "success", "total_ms", "error", "scene_id")} for result in document["results"]]),
                         ensure_ascii=False), flush=True)
        if not all(result["success"] for result in document["results"]):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
