#!/usr/bin/env python3
"""Create two English Reruns for the adaptive matched manipulation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

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
    "ompl": [38, 118, 201, 235],
    "vamp": [233, 138, 34, 235],
    "target": [24, 151, 107, 230],
    "inactive": [120, 128, 138, 78],
    "carried": [246, 183, 58, 235],
    "wall": [87, 96, 106, 150],
    "ground": [118, 126, 135, 115],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    centers = [[0.01, 1.215, 1.175], [0.01, -1.215, 1.175],
               [0.01, 0.0, 2.375], [1.225, 0.0, 1.175]]
    half_sizes = [[1.19, 0.025, 1.175], [1.19, 0.025, 1.175],
                  [1.19, 1.24, 0.025], [0.025, 1.24, 1.175]]
    rr.log(
        "world/warehouse",
        rr.Boxes3D(centers=centers, half_sizes=half_sizes,
                   colors=[COLORS["wall"]] * len(centers),
                   labels=["left wall", "right wall", "ceiling", "rear wall"]),
        static=True,
    )


def selected_trials(experiment_dir: Path) -> dict[int, dict[str, Any]]:
    document = json.loads((experiment_dir / "trials.json").read_text(encoding="utf-8"))
    output: dict[int, dict[str, Any]] = {}
    for trial in document["trials"]:
        box_id = int(trial["box_id"])
        if trial["success"] and box_id not in output:
            output[box_id] = trial
    return output


def backend_frames(payload: dict[str, Any], backend: str) -> list[dict[str, Any]]:
    benchmark = payload["paper_benchmark"]
    names = [str(value) for value in benchmark["joint_names"]]
    output: list[dict[str, Any]] = []
    for segment in benchmark["selected_segments"]:
        for values in segment[backend]["replay_states"]:
            state = dict(zip(names, map(float, values)))
            state.update({"head_joint": 0.0, "head_pitch_joint": 0.0})
            output.append({
                "state": state,
                "stage": str(segment["stage"]),
                "attached": bool(segment["attached"]),
            })
    return output


def log_box_wall(target_box: int, attached: bool, completed: set[int]) -> None:
    visible = [box_id for box_id in range(1, 26) if box_id not in completed]
    if attached and target_box in visible:
        visible.remove(target_box)
    centers = [box_center(box_id) for box_id in visible]
    colors = [COLORS["target"] if box_id == target_box else COLORS["inactive"] for box_id in visible]
    rr.log(
        "world/box_wall",
        rr.Boxes3D(
            centers=centers,
            half_sizes=[[0.15, 0.20, 0.20]] * len(visible),
            colors=colors,
            labels=[str(box_id) for box_id in visible],
            show_labels=True,
        ),
    )


def current_text(
    backend: str, box_id: int, trial: dict[str, Any], row: dict[str, Any],
    stage: str, frame_count: int, sphere_world: bool,
) -> str:
    selected = trial.get("selected") or trial
    backend_name = "Pure RRT-Connect / FCL" if backend == "ompl" else (
        "VAMP RRT-Connect / SIMD sphere collision"
        if sphere_world else "VAMP RRT-Connect / exact FCL"
    )
    return "\n".join([
        f"# Box {box_id} - {backend_name}",
        "",
        f"- Stage: **{stage}**",
        f"- Arm / grasp: **{selected['side']} / {selected['mode']}**",
        f"- Base station X: **{float(selected['base_x']):+.2f} m**",
        f"- Winning OMPL search: **{float(row['ompl_winning_median_ms']):.3f} ms**",
        f"- Winning VAMP search: **{float(row['vamp_winning_median_ms']):.3f} ms**",
        f"- Winning-search speedup: **{float(row['winning_median_speedup']):.2f}x**",
        f"- Cumulative OMPL to solution: **{float(row['ompl_median_ms']):.3f} ms**",
        f"- Cumulative VAMP to solution: **{float(row['vamp_median_ms']):.3f} ms**",
        f"- Cumulative speedup: **{float(row['cumulative_median_speedup']):.2f}x**",
        f"- Actual backend frames: **{frame_count}**",
        "- IK endpoints are shared and excluded from search timing.",
        "- No historical path, shortcut, Cartesian planner, or fallback.",
    ])


def overview_text(
    rows: list[dict[str, Any]], summary: dict[str, Any], sphere_world: bool,
) -> str:
    winning = summary["search_metric"]["winning_search"]
    cumulative = summary["search_metric"]["cumulative_to_solution"]
    lines = [
        "# Matched V3.2.2 adaptive manipulation benchmark",
        "",
        "The same online IK start/goal is used by both RRT-Connect backends. IK is excluded.",
        "Every box searches precontact, contact, loaded retreat, and loaded transport stages.",
        "Two base stations, both arms, and front/top suction are available to the shared policy.",
        ("VAMP uses native SIMD sphere collision online; FCL is not in its timed search or acceptance."
         if sphere_world else
         "This full-task recording uses exact FCL in both planners. Native SIMD paper timing is separate."),
        ("The sphere-world planning geometry uses 30 mm box clearance and a lower-chassis approximation."
         if sphere_world else
         "The native paper kernel is reported separately from this exact-FCL task."),
        "",
        f"- Winning-search aggregate speedup: **{winning['aggregate_speedup']:.2f}x**",
        f"- Cumulative-to-solution aggregate speedup: **{cumulative['aggregate_speedup']:.2f}x**",
        f"- Successful trials: **{summary['coverage']['successful_trials']}/{summary['coverage']['total_trials']}**",
        "",
        "|Box|OMPL win ms|VAMP win ms|Win speedup|OMPL cumulative ms|VAMP cumulative ms|Cumulative speedup|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def value(key: str, suffix: str = "") -> str:
            item = row.get(key)
            return "-" if item is None else f"{float(item):.3f}{suffix}"
        lines.append(
            f"|{row['box_id']}|{value('ompl_winning_median_ms')}|"
            f"{value('vamp_winning_median_ms')}|{value('winning_median_speedup', 'x')}|"
            f"{value('ompl_median_ms')}|{value('vamp_median_ms')}|"
            f"{value('cumulative_median_speedup', 'x')}|"
        )
    return "\n".join(lines)


def tool_path(robot: UrdfRobot, frames: list[dict[str, Any]], side: str, base_x: float) -> list[list[float]]:
    points: list[list[float]] = []
    for frame in frames:
        transform = robot.fk(frame["state"])[f"{side}_tool0"]
        point = transform[:3, 3].copy()
        point[0] += base_x
        points.append(point.tolist())
    return points


def log_carried_box(
    robot: UrdfRobot, payload: dict[str, Any], frame: dict[str, Any], base_x: float,
) -> list[float] | None:
    if not frame["attached"]:
        rr.log("world/carried_box", rr.Clear(recursive=True))
        return None
    tool = robot.fk(frame["state"])[str(payload["tool_link"])]
    center_tool = np.asarray(payload["tool_to_box_center"], dtype=float)
    center = (tool @ np.asarray([*center_tool, 1.0]))[:3]
    center[0] += base_x
    rr.log(
        "world/carried_box",
        rr.Boxes3D(centers=[center.tolist()], half_sizes=[[0.15, 0.20, 0.20]],
                   colors=[COLORS["carried"]], labels=["carried target"]),
    )
    return center.tolist()


def create_recording(
    backend: str, output: Path, model_urdf: Path,
    experiment_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any],
    sphere_world: bool,
) -> None:
    title = "Pure RRT-Connect + MoveIt/FCL" if backend == "ompl" else (
        "VAMP RRT-Connect + SIMD sphere collision"
        if sphere_world else "VAMP RRT-Connect + exact MoveIt/FCL"
    )
    prefix = "sphere-world" if sphere_world else "adaptive"
    rr.init(f"alfa_v322_{prefix}_{backend}", recording_id=f"v322-{prefix}-{backend}", spawn=False)
    rr.save(str(output))
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", contents=["/world/**"], name=title,
                              eye_controls=rrb.EyeControls3D(look_target=[0.25, 0.0, 1.05])),
            rrb.Vertical(
                rrb.TextDocumentView(origin="/current", name="Current task"),
                rrb.BarChartView(origin="/comparison/ompl_winning", name="OMPL winning search (ms)"),
                rrb.BarChartView(origin="/comparison/vamp_winning", name="VAMP winning search (ms)"),
                rrb.BarChartView(origin="/comparison/ompl_cumulative", name="OMPL cumulative to solution (ms)"),
                rrb.BarChartView(origin="/comparison/vamp_cumulative", name="VAMP cumulative to solution (ms)"),
                rrb.BarChartView(origin="/comparison/winning_speedup", name="Winning-search speedup (x)"),
                rrb.BarChartView(origin="/comparison/cumulative_speedup", name="Cumulative speedup (x)"),
                rrb.TextDocumentView(origin="/overview", name="Contract and all boxes"),
                row_shares=[0.19, 0.11, 0.11, 0.11, 0.11, 0.10, 0.10, 0.17],
            ),
            column_shares=[0.57, 0.43],
        ), collapse_panels=True,
    ))
    robot = UrdfRobot(model_urdf.read_text(encoding="utf-8"))
    log_robot_static_model(robot, "world/robot", log_meshes=True)
    log_static_scene()
    rr.log("overview", rr.TextDocument(
        overview_text(rows, summary, sphere_world), media_type=rr.MediaType.MARKDOWN
    ), static=True)
    ids = [row["box_id"] for row in rows]
    charts = {
        "ompl_winning": [row.get("ompl_winning_median_ms") or 0.0 for row in rows],
        "vamp_winning": [row.get("vamp_winning_median_ms") or 0.0 for row in rows],
        "ompl_cumulative": [row.get("ompl_median_ms") or 0.0 for row in rows],
        "vamp_cumulative": [row.get("vamp_median_ms") or 0.0 for row in rows],
        "winning_speedup": [row.get("winning_median_speedup") or 0.0 for row in rows],
        "cumulative_speedup": [row.get("cumulative_median_speedup") or 0.0 for row in rows],
    }
    for name, values in charts.items():
        color = COLORS["ompl"] if name.startswith("ompl") else COLORS["vamp"]
        if "speedup" in name:
            color = COLORS["target"]
        rr.log(f"comparison/{name}", rr.BarChart(values, abscissa=ids, widths=[0.72] * 25, color=color), static=True)

    selected = selected_trials(experiment_dir)
    rows_by_box = {int(row["box_id"]): row for row in rows}
    frame_index = 0
    previous_base_x: float | None = None
    previous_home: dict[str, Any] | None = None
    completed_boxes: set[int] = set()
    for box_id in range(1, 26):
        trial = selected.get(box_id)
        if not trial:
            continue
        selected_config = trial.get("selected") or trial
        payload = json.loads(Path(selected_config["result_json"]).read_text(encoding="utf-8"))
        frames = backend_frames(payload, backend)
        if not frames:
            continue
        base_x = float(selected_config["base_x"])
        side = str(selected_config["side"])
        points = tool_path(robot, frames, side, base_x)
        if previous_base_x is not None and previous_home is not None and abs(base_x - previous_base_x) > 1e-9:
            for step in range(1, 31):
                ratio = step / 30.0
                moving_base_x = previous_base_x + (base_x - previous_base_x) * ratio
                rr.set_time("frame", sequence=frame_index)
                rr.log("world/robot", rr.Transform3D(translation=[moving_base_x, 0.0, 0.0]))
                log_robot_state(robot, previous_home["state"], "world/robot")
                log_box_wall(box_id, False, completed_boxes)
                rr.log("world/carried_box", rr.Clear(recursive=True))
                rr.log("current", rr.TextDocument(
                    f"# Base reposition\n\n- From X: **{previous_base_x:+.2f} m**\n"
                    f"- To X: **{base_x:+.2f} m**\n"
                    "- Continuous deterministic chassis motion; excluded from arm RRT timing.",
                    media_type=rr.MediaType.MARKDOWN,
                ))
                frame_index += 1
        rr.set_time("frame", sequence=frame_index)
        rr.log("world/robot", rr.Transform3D(translation=[base_x, 0.0, 0.0]))
        rr.log("world/actual_rrt_path", rr.LineStrips3D(
            strips=[points], colors=[COLORS[backend]], radii=[0.008],
            labels=[f"Box {box_id} five-stage RRT path"],
        ))
        last_carried_center: list[float] | None = None
        delivered_logged = False
        for frame in [frames[0]] * 6 + frames + [frames[-1]] * 6:
            rr.set_time("frame", sequence=frame_index)
            rr.log("world/robot", rr.Transform3D(translation=[base_x, 0.0, 0.0]))
            log_robot_state(robot, frame["state"], "world/robot")
            center = log_carried_box(robot, payload, frame, base_x)
            if center is not None:
                last_carried_center = center
            elif last_carried_center is not None and not delivered_logged:
                completed_boxes.add(box_id)
                rr.log(
                    f"world/delivered_boxes/box_{box_id:02d}",
                    rr.Boxes3D(
                        centers=[last_carried_center], half_sizes=[[0.15, 0.20, 0.20]],
                        colors=[COLORS["carried"]], labels=[f"delivered {box_id}"],
                    ),
                )
                delivered_logged = True
            log_box_wall(box_id, frame["attached"], completed_boxes)
            rr.log("current", rr.TextDocument(
                current_text(
                    backend, box_id, trial, rows_by_box[box_id], frame["stage"],
                    len(frames), sphere_world,
                ),
                media_type=rr.MediaType.MARKDOWN,
            ))
            frame_index += 1
        previous_base_x = base_x
        previous_home = frames[-1]
    rr.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("model_urdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sphere-world", action="store_true")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir.resolve()
    summary = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))
    rows = summary["per_box"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = ({
        "ompl": output_dir / "sphere-world-pure-rrt-connect-fcl-5x5.rrd",
        "vamp": output_dir / "sphere-world-vamp-rrt-connect-simd-5x5.rrd",
    } if args.sphere_world else {
        "ompl": output_dir / "adaptive-pure-rrt-connect-fcl-5x5.rrd",
        "vamp": output_dir / "adaptive-vamp-rrt-connect-fcl-5x5.rrd",
    })
    for backend, output in outputs.items():
        create_recording(
            backend, output, args.model_urdf.resolve(), experiment_dir, rows, summary,
            args.sphere_world,
        )
        print(f"{backend}: {output} sha256={sha256(output)}")
    (output_dir / "rerun-manifest.json").write_text(json.dumps({
        "schema": ("alfa.v322.vamp_sphere_world_rerun.v1" if args.sphere_world else
                   "alfa.v322.vamp_paper_adaptive_rerun.v1"),
        "english_only_visible_text": True,
        "recordings": {key: {"path": str(path), "sha256": sha256(path)} for key, path in outputs.items()},
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
