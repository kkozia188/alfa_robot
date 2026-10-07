#!/usr/bin/python3
"""Generate, cache, verify, and open Rerun for any two-box ID combination."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKSPACE = Path("/home/tim/alfa_robot-v322-box-wall-motion-family-validation")
RESULTS = WORKSPACE / "results"
SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY = RESULTS / "all-pair-matrix-yaw-summary.json"
DEFAULT_ACCEPTANCE = RESULTS / "all-pair-matrix-acceptance.json"
DEFAULT_MATRIX_ROOT = RESULTS / "all-pair-matrix-yaw"
DEFAULT_CACHE = RESULTS / "pair-rerun-cache"
DEFAULT_SCRIPT_DIR = WORKSPACE / "src/alfa_robot_moveit_config/scripts"
DEFAULT_URDF = RESULTS / "v322-production-shell.urdf"
DEFAULT_RERUN = Path("/home/tim/.local/bin/rerun")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def ros_environment() -> dict[str, str]:
    setup_commands = " && ".join([
        "source /opt/ros/humble/setup.bash",
        "source /home/tim/alfa_robot-alfa_v3_dev/install/setup.bash",
        "source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash",
        "env -0",
    ])
    completed = subprocess.run(
        ["/bin/bash", "-lc", setup_commands],
        stdout=subprocess.PIPE,
        check=True,
    )
    environment: dict[str, str] = {}
    for item in completed.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        name, value = item.split(b"=", 1)
        environment[name.decode()] = value.decode()
    environment["PATH"] = (
        "/usr/bin:/bin:/usr/local/bin:/opt/ros/humble/bin:/home/tim/.local/bin"
    )
    moveit_prefix = WORKSPACE / "backend-install/alfa_robot_moveit_config"
    description_prefix = WORKSPACE / "backend-install/alfa_robot_description"
    for name in ("AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
        old = environment.get(name, "")
        environment[name] = ":".join(
            value for value in (str(moveit_prefix), str(description_prefix), old) if value
        )
    old_library = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(moveit_prefix / "lib"), old_library) if value
    )
    return environment


def pair_result(summary: dict[str, Any], pair: list[int]) -> dict[str, Any]:
    matches = [item for item in summary["results"] if item["pair"] == pair]
    if len(matches) != 1:
        raise RuntimeError(f"matrix summary has no unique pair {pair}")
    return matches[0]


def selected_paths(matrix_root: Path, result: dict[str, Any]) -> tuple[Path, Path]:
    selected = result["selected"]
    left, right = result["pair"]
    candidate = matrix_root / f"pair-{left:02d}-{right:02d}" / (
        f"candidate-{int(selected['candidate_rank']):03d}"
    )
    retry = selected.get("bridge_rescue_retry")
    if retry:
        return (
            candidate / f"replay-rescue-{int(retry)}.json",
            candidate / f"validation-rescue-{int(retry)}.json",
        )
    return candidate / "replay.json", candidate / "validation.json"


def report_markdown(
    result: dict[str, Any], acceptance: dict[str, Any], validation: dict[str, Any] | None
) -> str:
    left, right = result["pair"]
    if result.get("success"):
        selected = result["selected"]
        operation = (validation or {}).get("operations", [{}])[0]
        pose = selected.get("base_pose_map") or [
            selected.get("base_x_m", 0.0),
            selected.get("base_y_m", 0.0),
            selected.get("base_yaw_rad", 0.0),
        ]
        return "\n".join([
            f"# Pair {left} + {right}",
            "",
            "## Certified Full Cycle",
            "",
            f"- Candidate rank: **{selected['candidate_rank']}**",
            f"- Left / right box: **{selected['left_box_id']} / {selected['right_box_id']}**",
            f"- Grasp mode: `{selected['grasp_mode']}`",
            f"- Shared lift: `{float(selected['common_updown_m']):.2f} m`",
            f"- Retreat: `{float(selected['retreat_distance_m']):.2f} m`",
            f"- Base pose: `[{float(pose[0]):.3f}, {float(pose[1]):.3f}, {float(pose[2]):.3f}]`",
            f"- Checked frames: **{int((validation or {}).get('checked_frames', 0))}**",
            f"- Edge samples: **{int((validation or {}).get('checked_edge_samples', 0))}**",
            f"- Operation result: **{'PASS' if operation.get('success') else 'UNKNOWN'}**",
            f"- One-sided attachment frames: **{selected.get('one_sided_attachment_frames', 0)}**",
        ]) + "\n"
    failed = next(
        (item for item in acceptance["failed_pairs"] if item["pair"] == [left, right]),
        None,
    )
    if failed is None:
        raise RuntimeError(f"acceptance report has no failure detail for {[left, right]}")
    counts = "\n".join(
        f"- `{reason}`: {count}" for reason, count in failed["failure_counts"].items()
    )
    return "\n".join([
        f"# Pair {left} + {right}",
        "",
        "## Unsupported By Current Synchronized Family",
        "",
        f"- Rows: **{failed['rows_from_top'][0]} / {failed['rows_from_top'][1]}**",
        f"- Column: **{failed['columns_from_left'][0]}**",
        f"- Candidate attempts: **{failed['candidate_attempts']}**",
        f"- Dominant failure: **{failed['dominant_failure']}**",
        "",
        failed["reason"],
        "",
        "### Failure Counts",
        "",
        counts,
        "",
        "This recording is a failure diagnostic, not a successful motion replay.",
    ]) + "\n"


def record_failure_diagnostic(
    pair: list[int], markdown: str, output: Path
) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    centers = {}
    for box_id in range(1, 26):
        row = (box_id - 1) // 5
        column = (box_id - 1) % 5
        centers[box_id] = [0.9, (2 - column) * 0.4, (4.5 - row) * 0.4]
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init(f"v322_pair_{pair[0]:02d}_{pair[1]:02d}_failure", spawn=False)
    rr.save(output)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world", contents=["/world/**"], name="Unsupported pair geometry"
            ),
            rrb.TextDocumentView(origin="/report", name="Failure diagnosis"),
            column_shares=[0.62, 0.38],
        ),
        collapse_panels=True,
    ))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    remaining = [box_id for box_id in range(1, 26) if box_id not in pair]
    rr.log(
        "world/box_wall/remaining",
        rr.Boxes3D(
            centers=[centers[box_id] for box_id in remaining],
            half_sizes=[[0.15, 0.20, 0.20]],
            colors=[[155, 162, 158, 100]],
        ),
        static=True,
    )
    rr.log(
        "world/box_wall/unsupported_targets",
        rr.Boxes3D(
            centers=[centers[box_id] for box_id in pair],
            half_sizes=[[0.15, 0.20, 0.20]],
            colors=[[210, 75, 75, 235], [210, 75, 75, 235]],
            labels=[f"target {pair[0]}", f"target {pair[1]}"],
            show_labels=True,
        ),
        static=True,
    )
    rr.log(
        "report",
        rr.TextDocument(markdown, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )
    rr.disconnect()


def ensure_urdf(path: Path, environment: dict[str, str]) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    description_prefix = WORKSPACE / "backend-install/alfa_robot_description"
    xacro_file = (
        description_prefix
        / "share/alfa_robot_description/urdf/alfa_robot.urdf.xacro"
    )
    if not xacro_file.is_file():
        raise RuntimeError(f"description xacro is missing: {xacro_file}")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["/opt/ros/humble/bin/xacro", "-o", str(path), str(xacro_file)],
        env=environment,
        check=True,
    )


def verify_and_open(output: Path, rerun: Path, no_open: bool) -> None:
    subprocess.run([str(rerun), "rrd", "verify", str(output)], check=True)
    if not no_open:
        subprocess.run(
            [str(rerun), "--new", "--detach-process", str(output)], check=True
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("box_a", type=int)
    parser.add_argument("box_b", type=int)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--script-dir", type=Path, default=DEFAULT_SCRIPT_DIR)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--rerun", type=Path, default=DEFAULT_RERUN)
    parser.add_argument("--playback-speed", type=float, default=3.0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.box_a <= 25 or not 1 <= args.box_b <= 25:
        parser.error("box IDs must be in [1, 25]")
    if args.box_a == args.box_b:
        parser.error("box IDs must be distinct")
    pair = sorted([args.box_a, args.box_b])
    summary = read_json(args.summary)
    acceptance = read_json(args.acceptance)
    result = pair_result(summary, pair)
    environment = ros_environment()
    pair_name = f"pair-{pair[0]:02d}-{pair[1]:02d}"
    pair_cache = args.cache_dir / pair_name
    pair_cache.mkdir(parents=True, exist_ok=True)
    output = pair_cache / (
        f"{pair_name}-full.rrd" if result.get("success")
        else f"{pair_name}-failure-diagnostic.rrd"
    )
    markdown_path = pair_cache / f"{pair_name}-report.md"

    if result.get("success"):
        replay_path, validation_path = selected_paths(args.matrix_root, result)
        if not replay_path.is_file() or not validation_path.is_file():
            raise RuntimeError(
                f"selected evidence is missing: {replay_path} / {validation_path}"
            )
        validation = read_json(validation_path)
        markdown = report_markdown(result, acceptance, validation)
        markdown_path.write_text(markdown, encoding="utf-8")
        if args.rebuild or not output.is_file():
            ensure_urdf(args.urdf, environment)
            command = [
                "/usr/bin/python3",
                str(SOURCE_DIR / "record_fresh_continuous_rerun.py"),
                "--replay", str(replay_path),
                "--validation", str(validation_path),
                "--report-markdown", str(markdown_path),
                "--script-dir", str(args.script_dir),
                "--urdf", str(args.urdf),
                "--output", str(output),
                "--playback-speed", str(args.playback_speed),
            ]
            subprocess.run(command, env=environment, check=True)
        print(
            f"SUCCESS pair={pair[0]}+{pair[1]} candidate="
            f"{result['selected']['candidate_rank']} recording={output}",
            flush=True,
        )
    else:
        markdown = report_markdown(result, acceptance, None)
        markdown_path.write_text(markdown, encoding="utf-8")
        if args.rebuild or not output.is_file():
            record_failure_diagnostic(pair, markdown, output)
        print(
            f"UNSUPPORTED pair={pair[0]}+{pair[1]} diagnostic={output}",
            flush=True,
        )
    verify_and_open(output, args.rerun, args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
