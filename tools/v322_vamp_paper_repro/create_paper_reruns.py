#!/usr/bin/env python3
"""Create backend-specific path Reruns for the paper benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


WORKTREE = Path(__file__).resolve().parents[2]
RERUN_PACKAGE = WORKTREE / "ros2_ws/src/alfa_robot_rerun"
if str(RERUN_PACKAGE) not in sys.path:
    sys.path.insert(0, str(RERUN_PACKAGE))

from alfa_robot_rerun.visualize_rerun import (  # noqa: E402
    UrdfRobot,
    log_robot_state,
    log_robot_static_model,
)


COLORS = {
    "neutral": [128, 137, 146, 210],
    "ompl": [38, 118, 201, 235],
    "vamp": [233, 138, 34, 235],
    "certified": [22, 132, 91, 235],
    "warning": [211, 143, 31, 235],
    "failure": [199, 61, 61, 235],
    "inactive": [105, 114, 124, 95],
    "wall": [87, 96, 106, 150],
    "ground": [118, 126, 135, 115],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("model_urdf", type=Path)
    parser.add_argument("replay_capture", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_float(value: str) -> float | None:
    return float(value) if value else None


def read_trials(path: Path) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    with path.open() as source:
        for row in csv.DictReader(source):
            grouped[int(row["box_id"])].append(
                {
                    **row,
                    "endpoint_ready": row["endpoint_ready"] == "True",
                    "ompl_exact": row["ompl_exact"] == "True",
                    "vamp_native_solved": row["vamp_native_solved"] == "True",
                    "vamp_fcl_valid": row["vamp_fcl_valid"] == "True",
                    "paired_certified": row["paired_certified"] == "True",
                    "ompl_ms": optional_float(row["ompl_ms"]),
                    "vamp_ms": optional_float(row["vamp_ms"]),
                    "paper_speedup": optional_float(row["paper_speedup"]),
                    "certified_speedup": optional_float(row["certified_speedup"]),
                }
            )
    return grouped


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def box_rows(grouped: dict[int, list[dict]]) -> list[dict]:
    output = []
    for box_id in range(1, 26):
        trials = grouped[box_id]
        ompl_values = [row["ompl_ms"] for row in trials if row["ompl_ms"] is not None]
        vamp_values = [row["vamp_ms"] for row in trials if row["vamp_ms"] is not None]
        speedups = [
            row["paper_speedup"] for row in trials if row["paper_speedup"] is not None
        ]
        certified_speedups = [
            row["certified_speedup"]
            for row in trials if row["certified_speedup"] is not None
        ]
        speedup = median(speedups)
        output.append(
            {
                "box_id": box_id,
                "endpoint_trials": sum(row["endpoint_ready"] for row in trials),
                "ompl_exact": sum(row["ompl_exact"] for row in trials),
                "vamp_native": sum(row["vamp_native_solved"] for row in trials),
                "vamp_fcl": sum(row["vamp_fcl_valid"] for row in trials),
                "paired_certified": sum(row["paired_certified"] for row in trials),
                "ompl_median_ms": median(ompl_values),
                "vamp_median_ms": median(vamp_values),
                "paper_speedup": speedup,
                "certified_speedup": median(certified_speedups),
                "time_reduction_percent": (
                    100.0 * (1.0 - 1.0 / speedup) if speedup and speedup > 0.0 else None
                ),
            }
        )
    return output


def load_joint_states(experiment_dir: Path) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    starts = []
    goals = {}
    for path in sorted((experiment_dir / "raw").glob("*.json")):
        payload = json.loads(path.read_text())
        benchmark = payload.get("paper_benchmark")
        if not benchmark:
            continue
        names = [str(value) for value in benchmark["joint_names"]]
        starts.append(dict(zip(names, map(float, benchmark["start"]))))
        goals[int(payload["target_box_id"])] = dict(
            zip(names, map(float, benchmark["goal"]))
        )
    if not starts:
        raise RuntimeError("no paired benchmark endpoints found")
    start = starts[0]
    if any(candidate != start for candidate in starts):
        raise RuntimeError("benchmark start states are not identical")
    start.update({"head_joint": 0.0, "head_pitch_joint": 0.0})
    for goal in goals.values():
        goal.update({"head_joint": 0.0, "head_pitch_joint": 0.0})
    return start, goals


def load_replay_capture(path: Path) -> dict[int, dict]:
    output = {}
    for source in sorted((path / "raw").glob("*.json")):
        payload = json.loads(source.read_text())
        box_id = int(payload["target_box_id"])
        benchmark = payload.get("paper_benchmark")
        if not benchmark:
            output[box_id] = {
                "halton_skip": None,
                "joint_names": [],
                "ompl": {"states": [], "success": False},
                "vamp": {"states": [], "native": False, "fcl_valid": False},
            }
            continue
        names = [str(value) for value in benchmark["joint_names"]]

        def states(backend: str) -> list[dict[str, float]]:
            return [
                {
                    **dict(zip(names, map(float, values))),
                    "head_joint": 0.0,
                    "head_pitch_joint": 0.0,
                }
                for values in benchmark[backend].get("replay_states", [])
            ]

        output[box_id] = {
            "halton_skip": int(benchmark["halton_skip"]),
            "joint_names": names,
            "ompl": {
                "states": states("ompl"),
                "success": bool(benchmark["ompl"]["success"]),
            },
            "vamp": {
                "states": states("vamp"),
                "native": int(benchmark["vamp"]["audit"].get("sparse_states", 0)) >= 2,
                "fcl_valid": benchmark["vamp"]["audit"].get("fcl_audit_valid") is True,
                "first_invalid_fcl_state": int(
                    benchmark["vamp"]["audit"].get("first_invalid_fcl_state", 0)
                ),
                "invalid_fcl_states": int(
                    benchmark["vamp"]["audit"].get("invalid_fcl_states", 0)
                ),
            },
        }
    return output


def box_center(box_id: int) -> list[float]:
    row, column = divmod(box_id - 1, 5)
    return [0.9, 0.8 - 0.4 * column, 1.8 - 0.4 * row]


def log_static_scene() -> None:
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/ground",
        rr.Boxes3D(centers=[[0.0, 0.0, -0.055]], half_sizes=[[3.0, 3.0, 0.05]],
                   colors=[COLORS["ground"]]),
        static=True,
    )
    centers = [
        [0.01, 1.215, 1.175],
        [0.01, -1.215, 1.175],
        [0.01, 0.0, 2.375],
        [1.225, 0.0, 1.175],
    ]
    half_sizes = [
        [1.19, 0.025, 1.175],
        [1.19, 0.025, 1.175],
        [1.19, 1.24, 0.025],
        [0.025, 1.24, 1.175],
    ]
    rr.log(
        "world/warehouse",
        rr.Boxes3D(
            centers=centers,
            half_sizes=half_sizes,
            colors=[COLORS["wall"]] * len(centers),
            labels=["left wall", "right wall", "ceiling", "rear wall"],
        ),
        static=True,
    )


def backend_color(backend: str, row: dict) -> list[int]:
    if row["endpoint_trials"] == 0:
        return COLORS["neutral"]
    if backend == "ompl":
        if row["ompl_exact"] == 5:
            return COLORS["ompl"]
        if row["ompl_exact"] > 0:
            return COLORS["warning"]
        return COLORS["failure"]
    if row["vamp_fcl"] > 0:
        return COLORS["certified"]
    if row["vamp_native"] > 0:
        return COLORS["warning"]
    return COLORS["failure"]


def log_box_wall(target_box: int, backend: str, rows: list[dict]) -> None:
    centers = [box_center(box_id) for box_id in range(1, 26)]
    colors = []
    labels = []
    for row in rows:
        box_id = row["box_id"]
        if box_id == target_box:
            colors.append(backend_color(backend, row))
            labels.append(f"Box {box_id} TARGET")
        else:
            colors.append(COLORS["inactive"])
            labels.append(str(box_id))
    rr.log(
        "world/box_wall",
        rr.Boxes3D(
            centers=centers,
            half_sizes=[[0.15, 0.20, 0.20]] * 25,
            colors=colors,
            labels=labels,
            show_labels=True,
        ),
    )
    center = box_center(target_box)
    rr.log(
        "world/target",
        rr.Points3D(
            positions=[center],
            colors=[backend_color(backend, rows[target_box - 1])],
            radii=[0.035],
            labels=[f"Box {target_box}"],
            show_labels=True,
        ),
    )


def display_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f} ms"


def display_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def overview_markdown(backend: str, rows: list[dict], summary: dict) -> str:
    backend_name = (
        "Pure RRT-Connect (MoveIt/FCL)"
        if backend == "ompl" else "VAMP RRT-Connect (native SIMD spheres)"
    )
    lines = [
        f"# {backend_name}",
        "",
        "Actual representative online RRT-Connect paths. Every motion frame is "
        "from the selected backend path; no historical path, shortcut, Cartesian "
        "planner, visibility graph, or fallback is used.",
        "",
        f"- Paper-metric paired median speedup: **{summary['paper_speedup_median']:.2f}x**",
        f"- Final-FCL-certified subset median speedup: **{summary['certified_speedup_median']:.2f}x**",
        f"- Native VAMP path FCL pass rate: **{100.0 * summary['vamp_native_fcl_pass_rate']:.2f}%**",
        "- Timing: RRT-Connect entry to first backend-valid path. IK and final FCL audit are excluded.",
        "- Missing time/speedup bars are not zero-time results; inspect Coverage status.",
        "- Coverage codes: 1=NO IK, 2=NO PATH, 3=PARTIAL/FCL REJECT, 4=VALID PATH.",
        "",
        "|Box|Pure RRT median|RRT exact|VAMP median|VAMP native|FCL pass|Speedup|Reduction|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        reduction = row["time_reduction_percent"]
        lines.append(
            f"|{row['box_id']}|{display_ms(row['ompl_median_ms'])}|"
            f"{row['ompl_exact']}/5|{display_ms(row['vamp_median_ms'])}|"
            f"{row['vamp_native']}/5|{row['vamp_fcl']}/5|"
            f"{display_ratio(row['paper_speedup'])}|"
            f"{'-' if reduction is None else f'{reduction:.2f}%'}|"
        )
    return "\n".join(lines)


def current_markdown(backend: str, row: dict, capture: dict, frame_count: int) -> str:
    speedup = row["paper_speedup"]
    reduction = row["time_reduction_percent"]
    if row["endpoint_trials"] == 0:
        result = "No common online-IK endpoint; RRT was not entered"
    elif backend == "ompl":
        result = f"Exact solution {row['ompl_exact']}/5"
    elif row["vamp_fcl"]:
        result = f"Native solution {row['vamp_native']}/5; final FCL pass {row['vamp_fcl']}/5"
    elif row["vamp_native"]:
        result = f"Native solution {row['vamp_native']}/5; final FCL pass 0/5"
    else:
        result = "No native sphere-model solution 0/5"
    return "\n".join(
        [
            f"# Box {row['box_id']} - {'Pure RRT/FCL' if backend == 'ompl' else 'VAMP/SIMD'}",
            "",
            f"- Current backend: **{result}**",
            f"- Pure RRT search median: **{display_ms(row['ompl_median_ms'])}**",
            f"- VAMP search median: **{display_ms(row['vamp_median_ms'])}**",
            f"- Paired median speedup: **{display_ratio(speedup)}**",
            f"- Search-time reduction: **{'-' if reduction is None else f'{reduction:.2f}%'}**",
            f"- Final-FCL-certified speedup: **{display_ratio(row['certified_speedup'])}**",
            f"- Representative Halton skip: **{capture.get('halton_skip', '-')}**",
            f"- Actual replay frames: **{frame_count}**",
            "- Motion source: actual online RRT-Connect output only.",
        ]
    )


def coverage_code(backend: str, row: dict) -> float:
    if row["endpoint_trials"] == 0:
        return 1.0
    if backend == "ompl":
        if row["ompl_exact"] == 0:
            return 2.0
        return 4.0 if row["ompl_exact"] == 5 else 3.0
    if row["vamp_native"] == 0:
        return 2.0
    return 4.0 if row["vamp_fcl"] > 0 else 3.0


def tool_path_points(
    robot: UrdfRobot,
    states: list[dict[str, float]],
    box_id: int,
) -> list[list[float]]:
    _, column = divmod(box_id - 1, 5)
    side = "left" if column < 2 or box_id == 23 else "right"
    link = f"{side}_tool0"
    points = []
    for state in states:
        transform = robot.fk(state).get(link)
        if transform is not None:
            point = transform[:3, 3].copy()
            point[0] -= 0.35
            points.append(point.tolist())
    return points


def create_recording(
    backend: str,
    output: Path,
    model_urdf: Path,
    rows: list[dict],
    summary: dict,
    start: dict[str, float],
    replay: dict[int, dict],
) -> None:
    backend_title = (
        "Pure RRT-Connect + MoveIt/FCL"
        if backend == "ompl" else "VAMP RRT-Connect + native SIMD spheres"
    )
    rr.init(
        f"alfa_v322_paper_{backend}",
        recording_id=f"v322-paper-{backend}-5x5-final",
        spawn=False,
    )
    rr.save(str(output))
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/world",
                    contents=["/world/**"],
                    name=f"V3.2.2 · {backend_title}",
                    eye_controls=rrb.EyeControls3D(look_target=[0.30, 0.0, 1.05]),
                ),
                rrb.Vertical(
                    rrb.TextDocumentView(origin="/current", name="Current box and path"),
                    rrb.BarChartView(origin="/comparison/ompl_ms", name="Pure RRT search median (ms)"),
                    rrb.BarChartView(origin="/comparison/vamp_ms", name="VAMP search median (ms)"),
                    rrb.BarChartView(origin="/comparison/speedup", name="Paired median speedup (x)"),
                    rrb.BarChartView(origin="/comparison/coverage", name="Coverage status (1..4)"),
                    rrb.TextDocumentView(origin="/overview", name="Experiment contract"),
                    row_shares=[0.22, 0.16, 0.16, 0.16, 0.13, 0.17],
                ),
                column_shares=[0.58, 0.42],
            ),
            collapse_panels=True,
        )
    )
    robot = UrdfRobot(model_urdf.read_text())
    log_robot_static_model(robot, "world/robot", log_meshes=True)
    log_static_scene()
    rr.log(
        "world/robot",
        rr.Transform3D(translation=[-0.35, 0.0, 0.0]),
        static=True,
    )
    rr.log(
        "overview",
        rr.TextDocument(
            overview_markdown(backend, rows, summary),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )
    ids = [row["box_id"] for row in rows]
    ompl_values = [row["ompl_median_ms"] or 0.0 for row in rows]
    vamp_values = [row["vamp_median_ms"] or 0.0 for row in rows]
    speedup_values = [row["paper_speedup"] or 0.0 for row in rows]
    coverage_values = [coverage_code(backend, row) for row in rows]
    rr.log(
        "comparison/ompl_ms",
        rr.BarChart(
            ompl_values,
            abscissa=ids,
            widths=[0.72] * 25,
            color=COLORS["ompl"],
        ),
        static=True,
    )
    rr.log(
        "comparison/vamp_ms",
        rr.BarChart(
            vamp_values,
            abscissa=ids,
            widths=[0.72] * 25,
            color=COLORS["vamp"],
        ),
        static=True,
    )
    rr.log(
        "comparison/speedup",
        rr.BarChart(
            speedup_values,
            abscissa=ids,
            widths=[0.72] * 25,
            color=COLORS["certified"],
        ),
        static=True,
    )
    rr.log(
        "comparison/coverage",
        rr.BarChart(
            coverage_values,
            abscissa=ids,
            widths=[0.72] * 25,
            color=COLORS["ompl" if backend == "ompl" else "vamp"],
        ),
        static=True,
    )
    global_frame = 0
    for row in rows:
        box_id = row["box_id"]
        captured = replay.get(box_id, {})
        backend_capture = captured.get(backend, {})
        states = list(backend_capture.get("states", []))
        rr.set_time("frame", sequence=global_frame)
        log_box_wall(box_id, backend, rows)
        path_points = tool_path_points(robot, states, box_id)
        if len(path_points) >= 2:
          rr.log(
              "world/actual_rrt_path",
              rr.LineStrips3D(
                  strips=[path_points],
                  colors=[COLORS["ompl" if backend == "ompl" else "vamp"]],
                  radii=[0.008],
                  labels=[f"Box {box_id} actual RRT path"],
              ),
          )
        else:
          rr.log("world/actual_rrt_path", rr.Clear(recursive=True))
        rr.log(
            "current",
            rr.TextDocument(
                current_markdown(backend, row, captured, len(states)),
                media_type=rr.MediaType.MARKDOWN,
            ),
        )
        playback_states = (
            [states[0]] * 8 + states + [states[-1]] * 8
            if states else [start] * 24
        )
        for state in playback_states:
            rr.set_time("frame", sequence=global_frame)
            log_robot_state(robot, state, "world/robot")
            global_frame += 1
    rr.disconnect()


def main() -> int:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = read_trials(experiment_dir / "trials.csv")
    rows = box_rows(grouped)
    summary = json.loads((experiment_dir / "summary.json").read_text())
    start, goals = load_joint_states(experiment_dir)
    replay = load_replay_capture(args.replay_capture.resolve())
    outputs = {
        "ompl": output_dir / "pure-rrt-connect-fcl-5x5.rrd",
        "vamp": output_dir / "vamp-rrt-connect-simd-5x5.rrd",
    }
    for backend, output in outputs.items():
        create_recording(
            backend,
            output,
            args.model_urdf.resolve(),
            rows,
            summary,
            start,
            replay,
        )
        print(f"{backend}: {output} sha256={sha256(output)}")
    (output_dir / "rerun-manifest.json").write_text(
        json.dumps(
            {
                "schema": "alfa.v322.vamp_paper_rerun.v1",
                "statistical_replay": False,
                "trajectory_frames_available": True,
                "robot_state": "actual_representative_online_rrt_connect_path",
                "no_path_behavior": "hold_common_start_state",
                "recordings": {
                    backend: {"path": str(path), "sha256": sha256(path)}
                    for backend, path in outputs.items()
                },
            },
            indent=2,
        ) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
