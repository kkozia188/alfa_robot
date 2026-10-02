#!/usr/bin/python3
"""Plan, validate, and replay a parameterized two-standoff V3 5x5 task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v3_5x5_grasp_sequence_rerun as sequence
import v3_scoop_5x5_dual_sequence as dual
import v3_scoop_5x5_grasp_sequence_rerun as scoop


UPPER_RANGE_M = (0.80, 0.90)
LOWER_RANGE_M = (0.55, 0.65)
VEHICLE_FRONT_X_IN_BASE_M = 0.500000002779484
TASK_ORDER = [
    1, 2, 5, 4, 3,
    6, 7, 10, 9, 8,
    11, 12, 15, 14, 13,
    16, 17, 20, 19, 18,
    21, 22, 25, 24, 23,
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def distance_slug(value: float) -> str:
    micrometers = int(round(value * 1_000_000.0))
    if micrometers % 10_000 == 0:
        return f"{micrometers // 10_000:03d}"
    return f"{micrometers:06d}um"


def base_x_for_front_clearance(front_clearance_m: float) -> float:
    return sequence.CONTACT_X - VEHICLE_FRONT_X_IN_BASE_M - front_clearance_m


def is_centimeter_grid_value(value: float) -> bool:
    return math.isclose(value * 100.0, round(value * 100.0), abs_tol=1e-9)


def round_to_certified_centimeter(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def certified_pair_entry(
    workspace: Path,
    upper_front_clearance_m: float,
    lower_front_clearance_m: float,
) -> dict[str, Any]:
    manifest_path = (
        workspace
        / "data/ik_benchmark/v3_scoop_5x5/range_certification"
        / "clearance-grid-certificate.json"
    )
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema")
        != "alfa.v3_scoop_5x5_clearance_grid_certificate.v1"
        or not manifest.get("success")
        or int(manifest.get("certified_pair_count", 0)) != 121
        or not math.isclose(
            float(manifest.get("input_resolution_m", 0.0)), 0.01, abs_tol=1e-12
        )
    ):
        raise ValueError(f"invalid clearance-grid certificate: {manifest_path}")
    matches = [
        entry for entry in manifest.get("pairs", [])
        if math.isclose(
            float(entry["upper_front_clearance_m"]),
            upper_front_clearance_m,
            abs_tol=1e-9,
        ) and math.isclose(
            float(entry["lower_front_clearance_m"]),
            lower_front_clearance_m,
            abs_tol=1e-9,
        )
    ]
    if len(matches) != 1 or not matches[0].get("success"):
        raise ValueError("requested clearance pair is not certified")
    entry = matches[0]
    for name in ("cache", "replay", "validation"):
        path = Path(str(entry[name]))
        if not path.is_file() or sha256(path) != str(entry[f"{name}_sha256"]):
            raise ValueError(f"certified {name} hash mismatch: {path}")
    validation = read_json(Path(str(entry["validation"])))
    if not validation.get("success"):
        raise ValueError("certified validation report is not successful")
    return entry


def prompt_front_clearance(
    value: float | None,
    label: str,
    bounds_m: tuple[float, float],
    default_m: float,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
) -> float:
    if value is not None:
        return value
    read = input_fn or input
    lower, upper = bounds_m
    prompt = (
        f"请输入{label}车头前接触面至箱体前接触面的 X 净距 "
        f"[{lower:.2f}-{upper:.2f} m，直接回车={default_m:.2f} m]: "
    )
    while True:
        try:
            raw = read(prompt).strip()
        except EOFError as error:
            raise ValueError(
                f"未读取到{label} X 净距；请在终端交互输入，或使用命令行参数"
            ) from error
        if not raw:
            return default_m
        try:
            selected = float(raw)
        except ValueError:
            output_fn(f"输入无效：请输入 {lower:.2f}-{upper:.2f} 之间的米制数值。")
            continue
        if lower <= selected <= upper:
            return selected
        output_fn(f"超出范围：{label} X 净距必须在 {lower:.2f}-{upper:.2f} m。")


def shortest_angle_delta(left: float, right: float) -> float:
    return math.atan2(math.sin(right - left), math.cos(right - left))


def operation_from_task(
    task: dict[str, Any],
    removed: set[int],
    left_attachment: dict[str, Any],
    right_attachment: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    box_id = int(task["box_id"])
    frames = []
    for source in [*task["transition_frames"], *task["payload"]["frames"]]:
        attached = bool(source.get("box_attached", False))
        frames.append({
            "stage": str(source["stage"]),
            "joints": [float(value) for value in source["joints"]],
            "left_attached": attached and task["side"] == "left",
            "right_attached": attached and task["side"] == "right",
        })
    active_attachment = dual.attachment(task)
    if task["side"] == "left":
        left_attachment = active_attachment
    else:
        right_attachment = active_attachment
    operation = {
        "label": f"Box {box_id}",
        "kind": "single_sequential",
        "row": int(task["row"]),
        "left_box_id": box_id if task["side"] == "left" else 0,
        "right_box_id": box_id if task["side"] == "right" else 0,
        "mode": str(task["mode"]),
        "base_pose_map": dual.task_base_pose(task),
        "removed_before": sorted(removed),
        "left_attachment": left_attachment,
        "right_attachment": right_attachment,
        "box_planning": [dual.task_planning_summary(task)],
        "obstacles": dual.obstacles(removed, {box_id}),
        "allow_ground_model_base": True,
        "transition_strategy": str(
            task.get("transition_motion", {}).get(
                "planning_strategies", ["validated_transition_waypoints"]
            )[0]
        ),
        "frames": frames,
    }
    return operation, left_attachment, right_attachment


def base_transition_operation(
    joints: list[float],
    preparation_frames: list[dict[str, Any]],
    from_pose: list[float],
    to_pose: list[float],
    upper_front_clearance_m: float,
    lower_front_clearance_m: float,
    removed: set[int],
    left_attachment: dict[str, Any],
    right_attachment: dict[str, Any],
    step_m: float,
    speed_m_s: float,
    label: str,
    after_row: int,
    coupled_frames: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    distance = math.hypot(to_pose[0] - from_pose[0], to_pose[1] - from_pose[1])
    yaw_distance = abs(shortest_angle_delta(from_pose[2], to_pose[2]))
    steps = max(
        1,
        int(math.ceil(distance / step_m)),
        int(math.ceil(yaw_distance / math.radians(1.0))),
    )
    frames = [{
        "stage": "prepare_base_reposition",
        "joints": [float(value) for value in frame["joints"]],
        "left_attached": False,
        "right_attached": False,
        "base_pose_map": list(from_pose),
    } for frame in preparation_frames]
    if coupled_frames:
        frames.extend(copy_frame for copy_frame in coupled_frames)
    else:
        for index in range(steps + 1):
            ratio = index / steps
            frames.append({
                "stage": "base_reposition",
                "joints": list(joints),
                "left_attached": False,
                "right_attached": False,
                "base_pose_map": [
                    from_pose[0] + (to_pose[0] - from_pose[0]) * ratio,
                    from_pose[1] + (to_pose[1] - from_pose[1]) * ratio,
                    from_pose[2] + shortest_angle_delta(from_pose[2], to_pose[2]) * ratio,
                ],
            })
    return {
        "label": label,
        "kind": "base_transition",
        "row": after_row,
        "after_row": after_row,
        "left_box_id": 0,
        "right_box_id": 0,
        "mode": "base_reposition",
        "base_pose_map": list(from_pose),
        "base_pose_goal_map": list(to_pose),
        "from_front_clearance_m": upper_front_clearance_m,
        "to_front_clearance_m": lower_front_clearance_m,
        "base_speed_m_s": speed_m_s,
        "transition_strategy": "validated_planar_linear",
        "updown_policy": "stationary",
        "removed_before": sorted(removed),
        "left_attachment": left_attachment,
        "right_attachment": right_attachment,
        "box_planning": [],
        "obstacles": dual.obstacles(removed, set()),
        "allow_ground_model_base": True,
        "frames": frames,
    }


def validate_cache_profile(
    cache: dict[str, Any],
    upper_front_clearance_m: float,
    lower_front_clearance_m: float,
) -> None:
    tasks = list(cache.get("tasks", []))
    if [int(task["box_id"]) for task in tasks] != TASK_ORDER:
        raise ValueError("plan cache does not contain the required 25-box order")
    for task in tasks:
        expected = (
            upper_front_clearance_m if int(task["row"]) <= 3
            else lower_front_clearance_m
        )
        pose = dual.task_base_pose(task)
        actual = sequence.CONTACT_X - (pose[0] + VEHICLE_FRONT_X_IN_BASE_M)
        if not math.isclose(actual, expected, abs_tol=1e-6):
            raise ValueError(
                f"box {task['box_id']} cache front clearance {actual:.3f}m "
                f"does not match requested {expected:.3f}m"
            )


def build_replay(
    cache_path: Path,
    output_path: Path,
    upper_front_clearance_m: float,
    lower_front_clearance_m: float,
    base_step_m: float,
    base_speed_m_s: float,
) -> dict[str, Any]:
    cache = read_json(cache_path)
    if cache.get("schema") != scoop.PLAN_CACHE_SCHEMA:
        raise ValueError("input is not a scoop plan cache")
    validate_cache_profile(
        cache, upper_front_clearance_m, lower_front_clearance_m
    )
    tasks = list(cache["tasks"])
    names = [str(value) for value in tasks[0]["payload"]["joint_names"]]
    removed: set[int] = set()
    operations = []
    left_attachment = dual.attachment(next(task for task in tasks if task["side"] == "left"))
    right_attachment = dual.attachment(next(task for task in tasks if task["side"] == "right"))
    for index, task in enumerate(tasks):
        operation, left_attachment, right_attachment = operation_from_task(
            task, removed, left_attachment, right_attachment
        )
        operations.append(operation)
        removed.add(int(task["box_id"]))
        if index + 1 < len(tasks):
            next_task = tasks[index + 1]
            from_pose = dual.task_base_pose(task)
            to_pose = dual.task_base_pose(next_task)
            if max(abs(left - right) for left, right in zip(from_pose, to_pose)) > 1e-9:
                reposition = next_task.get("base_reposition", {})
                if not reposition.get("required") or (
                    reposition.get("stow_required") and not reposition.get("stow_frames")
                ):
                    raise ValueError(
                        f"box {next_task['box_id']} is missing validated pre-base stow frames"
                    )
                from_clearance = sequence.CONTACT_X - (
                    from_pose[0] + VEHICLE_FRONT_X_IN_BASE_M
                )
                to_clearance = sequence.CONTACT_X - (
                    to_pose[0] + VEHICLE_FRONT_X_IN_BASE_M
                )
                operations.append(base_transition_operation(
                    [
                        float(value) for value in reposition["transport_joints"]
                    ],
                    list(reposition.get("stow_frames", [])),
                    from_pose,
                    to_pose,
                    from_clearance,
                    to_clearance,
                    removed,
                    left_attachment,
                    right_attachment,
                    base_step_m,
                    base_speed_m_s,
                    f"Chassis reposition after box {int(task['box_id'])}",
                    int(task["row"]),
                    list(reposition.get("coupled_frames", [])),
                ))
    if removed != set(range(1, 26)):
        raise RuntimeError("replay did not consume all 25 boxes")
    replay = {
        "schema": dual.SCHEMA,
        "source_plan_cache": str(cache_path.resolve()),
        "source_plan_cache_schema": cache["schema"],
        "vehicle_front_clearance_profile": {
            "vehicle_front_x_in_base_m": VEHICLE_FRONT_X_IN_BASE_M,
            "upper_rows_1_to_3_m": upper_front_clearance_m,
            "lower_rows_4_to_5_m": lower_front_clearance_m,
            "base_transition_after_row": 3,
            "base_transition_step_m": base_step_m,
            "base_transition_speed_m_s": base_speed_m_s,
        },
        "joint_names": names,
        "group_order": [[box_id] for box_id in TASK_ORDER],
        "planning_time_definition": {
            "selected_task_core_ms": "validated deterministic task planning",
            "selected_transition_core_ms": "validated deterministic arm transition planning",
            "base_transition": "validated planar-root interpolation; no planning time",
        },
        "box_planning": [
            dual.task_planning_summary(task)
            for task in sorted(tasks, key=lambda item: int(item["box_id"]))
        ],
        "operations": operations,
    }
    write_json(output_path, replay)
    return replay


def run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def cache_is_complete_for_request(
    path: Path, upper_front_clearance_m: float, lower_front_clearance_m: float
) -> bool:
    if not path.is_file():
        return False
    try:
        cache = read_json(path)
        validate_cache_profile(
            cache, upper_front_clearance_m, lower_front_clearance_m
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return len(cache.get("tasks", [])) == 25


def accept_validation_attempt(
    attempt_path: Path, validation_path: Path
) -> dict[str, Any]:
    if not attempt_path.is_file():
        raise RuntimeError(
            "the current MoveIt/FCL validator did not produce a result; "
            "an older validation report will not be reused"
        )
    validation = read_json(attempt_path)
    if not validation.get("success"):
        raise RuntimeError(f"moving-base validation failed: {attempt_path}")
    attempt_path.replace(validation_path)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upper-front-clearance-m",
        type=float,
        help="rows 1-3 vehicle-front X clearance in meters; prompt when omitted",
    )
    parser.add_argument(
        "--lower-front-clearance-m",
        type=float,
        help="rows 4-5 vehicle-front X clearance in meters; prompt when omitted",
    )
    parser.add_argument("--plan-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--planning-timeout", type=float, default=35.0)
    parser.add_argument("--rrt-retries", type=int, default=4)
    parser.add_argument("--base-step-m", type=float, default=0.01)
    parser.add_argument("--base-speed-m-s", type=float, default=0.10)
    parser.add_argument("--force-replan", action="store_true")
    parser.add_argument(
        "--certified-only", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "round in-range inputs to the nearest prevalidated centimeter "
            "using decimal half-up rounding"
        ),
    )
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    try:
        args.upper_front_clearance_m = prompt_front_clearance(
            args.upper_front_clearance_m,
            "前三排",
            UPPER_RANGE_M,
            0.85,
        )
        args.lower_front_clearance_m = prompt_front_clearance(
            args.lower_front_clearance_m,
            "后两排",
            LOWER_RANGE_M,
            0.60,
        )
    except ValueError as error:
        parser.error(str(error))
    if not UPPER_RANGE_M[0] <= args.upper_front_clearance_m <= UPPER_RANGE_M[1]:
        parser.error("upper vehicle-front clearance must be in [0.80, 0.90] m")
    if not LOWER_RANGE_M[0] <= args.lower_front_clearance_m <= LOWER_RANGE_M[1]:
        parser.error("lower vehicle-front clearance must be in [0.55, 0.65] m")
    if args.base_step_m <= 0.0 or args.base_step_m > 0.01:
        parser.error("base-step-m must be in (0, 0.01] m")
    if args.base_speed_m_s <= 0.0:
        parser.error("base-speed-m-s must be positive")

    requested_upper_front_clearance_m = args.upper_front_clearance_m
    requested_lower_front_clearance_m = args.lower_front_clearance_m
    if args.certified_only:
        args.upper_front_clearance_m = round_to_certified_centimeter(
            requested_upper_front_clearance_m
        )
        args.lower_front_clearance_m = round_to_certified_centimeter(
            requested_lower_front_clearance_m
        )
        print(
            "\n认证精度换算（十进制四舍五入到最近 1 cm）："
            f"前三排 输入 {requested_upper_front_clearance_m:.6f} m -> "
            f"实际执行 {args.upper_front_clearance_m:.2f} m；"
            f"后两排 输入 {requested_lower_front_clearance_m:.6f} m -> "
            f"实际执行 {args.lower_front_clearance_m:.2f} m。",
            flush=True,
        )

    upper_base_x = base_x_for_front_clearance(args.upper_front_clearance_m)
    lower_base_x = base_x_for_front_clearance(args.lower_front_clearance_m)
    print(
        "\n已接收车头前接触面 X 净距："
        f"前三排 {args.upper_front_clearance_m:.3f} m (base_x={upper_base_x:.3f} m)，"
        f"后两排 {args.lower_front_clearance_m:.3f} m (base_x={lower_base_x:.3f} m)。",
        flush=True,
    )
    print(
        "用户输入只控制 X 净距；Y 由规划器在边缘箱不可达时自动补偿，并进入同一套碰撞验证。",
        flush=True,
    )
    workspace = sequence.workspace_root()
    certified_entry = None
    if args.certified_only and not args.force_replan and args.plan_cache is None:
        try:
            certified_entry = certified_pair_entry(
                workspace,
                args.upper_front_clearance_m,
                args.lower_front_clearance_m,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
    if certified_entry:
        print(
            "认证网格：121/121组合已通过；当前缓存、回放和FCL报告SHA-256校验通过。\n",
            flush=True,
        )
    else:
        print(
            "实验模式：当前组合将现场重规划和全量验证，成功后才打开 Rerun。\n",
            flush=True,
        )
    run_name = (
        f"front-u{distance_slug(args.upper_front_clearance_m)}-"
        f"l{distance_slug(args.lower_front_clearance_m)}"
    )
    certified_output_dir = (
        Path(str(certified_entry["cache"])).resolve().parent
        if certified_entry else None
    )
    output_dir = (args.output_dir or certified_output_dir or (
        workspace / "data/ik_benchmark/v3_scoop_5x5/two_standoff_runs" / run_name
    )).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cold_rrd = output_dir / f"v3-scoop-{run_name}-cold.rrd"
    cold_cache = scoop.plan_cache_path(cold_rrd)
    runtime_cache = output_dir / f"v3-scoop-{run_name}-runtime-plan-cache.json"
    replay_path = output_dir / f"v3-scoop-{run_name}-moving-base-replay.json"
    validation_path = output_dir / f"v3-scoop-{run_name}-moving-base-validation.json"
    validation_attempt_path = output_dir / (
        f"v3-scoop-{run_name}-moving-base-validation-attempt.json"
    )
    final_rrd = output_dir / f"v3-scoop-{run_name}-moving-base.rrd"
    summary = output_dir / f"v3-scoop-{run_name}-moving-base-summary.json"
    metrics = output_dir / f"v3-scoop-{run_name}-moving-base-metrics.csv"
    planning_csv = output_dir / f"v3-scoop-{run_name}-box-planning.csv"

    supplied_cache = args.plan_cache.resolve() if args.plan_cache else None
    if certified_entry:
        runtime_cache = Path(str(certified_entry["cache"])).resolve()
        replay_path = Path(str(certified_entry["replay"])).resolve()
        validation_path = Path(str(certified_entry["validation"])).resolve()
    elif supplied_cache:
        runtime_cache = supplied_cache
    elif args.force_replan or not cache_is_complete_for_request(
        runtime_cache,
        args.upper_front_clearance_m,
        args.lower_front_clearance_m,
    ):
        if args.force_replan or not cache_is_complete_for_request(
            cold_cache,
            args.upper_front_clearance_m,
            args.lower_front_clearance_m,
        ):
            run([
                "/usr/bin/python3",
                str(SCRIPT_DIR / "v3_scoop_5x5_grasp_sequence_rerun.py"),
                "--save", str(cold_rrd),
                "--upper-front-clearance-m", f"{args.upper_front_clearance_m:.6f}",
                "--lower-front-clearance-m", f"{args.lower_front_clearance_m:.6f}",
                "--planning-timeout", f"{args.planning_timeout:.3f}",
                "--rrt-retries", str(args.rrt_retries),
                "--no-resume" if args.force_replan else "--resume",
                "--no-spawn",
            ])
        run([
            "/usr/bin/python3",
            str(SCRIPT_DIR / "v3_scoop_5x5_hotstart_cache.py"),
            "--source", str(cold_cache),
            "--output", str(runtime_cache),
            "--timeout", f"{args.planning_timeout:.3f}",
        ])

    if not certified_entry:
        build_replay(
            runtime_cache,
            replay_path,
            args.upper_front_clearance_m,
            args.lower_front_clearance_m,
            args.base_step_m,
            args.base_speed_m_s,
        )
        validation_attempt_path.unlink(missing_ok=True)
        run([
            "ros2", "launch", "alfa_robot_moveit_config",
            "v3_dual_arm_5x5_replay_validator.launch.py",
            f"replay_json_path:={replay_path}",
            f"result_json_path:={validation_attempt_path}",
        ])
        accept_validation_attempt(validation_attempt_path, validation_path)
    run([
        "/usr/bin/python3",
        str(SCRIPT_DIR / "v3_scoop_5x5_dual_rerun.py"),
        "--input", str(replay_path),
        "--save", str(final_rrd),
        "--summary", str(summary),
        "--metrics-csv", str(metrics),
        "--box-planning-csv", str(planning_csv),
        "--validation-report", str(validation_path),
        "--no-spawn",
    ])
    run(["rerun", "rrd", "verify", str(final_rrd)])
    print(
        f"RESULT success=25/25 upper_front={args.upper_front_clearance_m:.2f}m "
        f"lower_front={args.lower_front_clearance_m:.2f}m "
        f"base_motion=validated rrd={final_rrd}",
        flush=True,
    )
    if args.spawn:
        subprocess.Popen(
            ["rerun", "--new", str(final_rrd)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
