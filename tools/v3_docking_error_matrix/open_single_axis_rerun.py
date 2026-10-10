#!/usr/bin/env python3
"""Prompt for physical X, Y, and Yaw, then open a certified Rerun replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable

from matrix_common import case_slug, clearance_slug


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_WORKSPACE = Path("/home/tim/alfa_robot-alfa_v3_dev")
YAW_WORKSPACE = Path("/home/tim/alfa_robot-motion274-docking-error")
STRONG_UPPER_RANGE_M = (0.80, 0.90)
STRONG_LOWER_RANGE_M = (0.55, 0.65)
MAX_UPPER_RANGE_M = (0.74, 1.02)
MAX_LOWER_RANGE_M = (0.36, 0.79)
MAX_Y_RANGE_M = (-0.120, 0.0875)
MAX_YAW_RANGE_DEG = (-5.0, 5.0)
DEFAULT_UPPER_M = 0.85
DEFAULT_LOWER_M = 0.60
Y_RESULTS_ROOT = ROOT / "data/ik_benchmark/v3_docking_error_matrix"
X_UPPER_RESULTS_ROOT = Y_RESULTS_ROOT / "x-clearance-upper"
X_LOWER_RESULTS_ROOT = Y_RESULTS_ROOT / "x-clearance-lower"
ON_DEMAND_RESULTS_ROOT = Y_RESULTS_ROOT / "on-demand-full"
YAW_CASES_ROOT = YAW_WORKSPACE / "data/ik_benchmark/v3_yaw_robustness/cases"
YAW_RERUN_ROOT = YAW_WORKSPACE / "data/ik_benchmark/v3_yaw_robustness/release/rerun"
CASE_RUNNER = ROOT / "tools/v3_docking_error_matrix/run_pose_case.py"
X_DEMO = (
    PRIMARY_WORKSPACE
    / "ros2_ws/src/alfa_robot_moveit_config/scripts/v3_scoop_5x5_conveyor_demo.py"
)
RERUN_RECORDER = (
    PRIMARY_WORKSPACE
    / "ros2_ws/src/alfa_robot_moveit_config/scripts/v3_scoop_5x5_dual_rerun.py"
)
X_CERTIFICATE = (
    PRIMARY_WORKSPACE
    / "data/ik_benchmark/v3_scoop_5x5/range_certification_v322_tool0151"
    / "clearance-grid-certificate.json"
)
BASELINE_CACHE = (
    PRIMARY_WORKSPACE
    / "data/ik_benchmark/v3_scoop_5x5/releases"
    / "2026-10-01-v322-tool0151-mobile-base-conveyor/v322-plan-cache.json"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json_if_changed(path: Path, value: dict[str, Any]) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_centimeter(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )


def round_to_step(value: float, step: str) -> float:
    quantum = Decimal(step)
    units = (Decimal(str(value)) / quantum).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return float(units * quantum)


def round_y(value: float) -> float:
    # Preserve the measured +87.5 mm boundary; all other on-demand IDs use 1 mm.
    if 0.08725 <= value < 0.08775:
        return 0.0875
    return round_to_step(value, "0.001")


def round_yaw(value: float) -> float:
    return round_to_step(value, "0.1")


def prompt_float(
    value: float | None,
    prompt: str,
    default: float,
    input_fn: Callable[[str], str] | None = None,
) -> float:
    if value is not None:
        if not math.isfinite(value):
            raise ValueError("输入必须是有限数字")
        return value
    read = input_fn or input
    while True:
        raw = read(f"{prompt} [直接回车={default:g}]: ").strip()
        if not raw:
            return default
        try:
            numeric = float(raw)
        except ValueError:
            print("输入无效：请输入有限数字。")
            continue
        if math.isfinite(numeric):
            return numeric
        print("输入无效：请输入有限数字。")


def selected_axis(upper: float, lower: float, y_m: float, yaw_deg: float) -> str:
    x_changed = (
        not math.isclose(upper, DEFAULT_UPPER_M, abs_tol=1e-12)
        or not math.isclose(lower, DEFAULT_LOWER_M, abs_tol=1e-12)
    )
    changed = [x_changed, abs(y_m) > 1e-12, abs(yaw_deg) > 1e-12]
    if sum(changed) > 1:
        raise ValueError("只允许X、Y、Yaw中的一个变量偏离名义值")
    if abs(y_m) > 1e-12:
        return "y"
    if abs(yaw_deg) > 1e-12:
        return "yaw"
    return "x"


def in_range(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] - 1e-12 <= value <= bounds[1] + 1e-12


def x_case_mode(upper: float, lower: float) -> str:
    if not in_range(upper, MAX_UPPER_RANGE_M):
        raise ValueError(
            f"前三排X净距必须在[{MAX_UPPER_RANGE_M[0]:.2f}, "
            f"{MAX_UPPER_RANGE_M[1]:.2f}]m"
        )
    if not in_range(lower, MAX_LOWER_RANGE_M):
        raise ValueError(
            f"后两排X净距必须在[{MAX_LOWER_RANGE_M[0]:.2f}, "
            f"{MAX_LOWER_RANGE_M[1]:.2f}]m"
        )
    if in_range(upper, STRONG_UPPER_RANGE_M) and in_range(
        lower, STRONG_LOWER_RANGE_M
    ):
        return "strong_grid"
    if math.isclose(lower, DEFAULT_LOWER_M, abs_tol=1e-12):
        return "upper_extension"
    if math.isclose(upper, DEFAULT_UPPER_M, abs_tol=1e-12):
        return "lower_extension"
    raise ValueError(
        "超出121组合强认证区间时，只能扩展一组X净距："
        "前三排扩展要求后两排=0.60m；后两排扩展要求前三排=0.85m"
    )


def successful_x_extension_cases() -> dict[tuple[float, float], Path]:
    cases: dict[tuple[float, float], Path] = {}
    for root in (X_UPPER_RESULTS_ROOT, X_LOWER_RESULTS_ROOT):
        for result_path in root.glob("*/case-result.json"):
            result = read_json(result_path)
            clearance = result.get("front_clearance", {})
            validation = result.get("validation", {})
            if (
                result.get("status") not in {"passed", "degraded"}
                or int(result.get("completed_boxes", 0)) != 25
                or not validation.get("success")
            ):
                continue
            key = (
                round(float(clearance["upper_rows_1_to_3_m"]), 2),
                round(float(clearance["lower_rows_4_to_5_m"]), 2),
            )
            cases[key] = result_path
    return cases


def exact_x_extension_case(
    upper: float,
    lower: float,
    cases: dict[tuple[float, float], Path],
) -> Path:
    key = (round(upper, 2), round(lower, 2))
    if key not in cases:
        raise ValueError(
            f"X={upper:.2f}/{lower:.2f}m缺少完整25箱/FCL扩展证书"
        )
    return cases[key]


def successful_y_cases(root: Path = Y_RESULTS_ROOT) -> dict[float, Path]:
    candidates: dict[float, list[Path]] = {}
    for result_path in root.glob("**/case-result.json"):
        result = read_json(result_path)
        error = result.get("error")
        validation = result.get("validation", {})
        if not isinstance(error, dict):
            continue
        if (
            abs(float(error.get("dx_m", 0.0))) > 1e-12
            or abs(float(error.get("yaw_deg", 0.0))) > 1e-12
            or result.get("status") not in {"passed", "degraded"}
            or int(result.get("completed_boxes", 0)) != 25
            or not validation.get("success")
        ):
            continue
        replay = result_path.parent / "v322-pose-conveyor-replay.json"
        validation_path = result_path.parent / "v322-pose-conveyor-validation.json"
        if replay.is_file() and validation_path.is_file():
            value = round(float(error.get("dy_m", 0.0)), 6)
            candidates.setdefault(value, []).append(result_path)
    return {
        value: sorted(paths, key=lambda path: ("/full/" not in str(path), str(path)))[0]
        for value, paths in candidates.items()
    }


def exact_y_case(value: float, cases: dict[float, Path]) -> tuple[float, Path]:
    matches = [item for item in cases if math.isclose(item, value, abs_tol=1e-9)]
    if len(matches) != 1:
        available = ", ".join(f"{item:+g}" for item in sorted(cases))
        raise ValueError(
            f"Y={value:+g} m没有完整Rerun证书；可直接打开的Y值为: {available}"
        )
    selected = matches[0]
    return selected, cases[selected]


def certified_integer_yaw(value: float) -> int | None:
    rounded = round(value)
    if abs(value - rounded) <= 1e-9 and -5 <= rounded <= 5:
        return int(rounded)
    return None


def yaw_case(yaw_deg: int) -> tuple[Path, Path]:
    slug = f"yaw-{'p' if yaw_deg >= 0 else 'm'}{abs(yaw_deg):02d}"
    case_dir = YAW_CASES_ROOT / slug
    summary = read_json(case_dir / "summary.json")
    validation = read_json(case_dir / "v322-yaw-conveyor-validation.json")
    rrd = YAW_RERUN_ROOT / f"{slug}-v322-production-shell.rrd"
    if (
        int(summary.get("completed_boxes", 0)) != 25
        or int(summary.get("cycle_count", 0)) != 15
        or not validation.get("success")
        or not rrd.is_file()
    ):
        raise ValueError(f"Yaw {yaw_deg:+d}度缺少完整25箱/FCL/Rerun证书")
    return case_dir, rrd


def model_snapshot() -> tuple[Path, Path, str]:
    certificate = read_json(X_CERTIFICATE)
    snapshot = certificate["model_snapshot"]
    manifest = X_CERTIFICATE.parent / str(snapshot["manifest"])
    if sha256(manifest) != str(snapshot["manifest_sha256"]):
        raise ValueError("V3.2.2模型清单SHA-256不匹配")
    model = read_json(manifest)
    urdf = manifest.parent / str(snapshot["urdf"])
    asset_root = manifest.parent / str(snapshot["asset_root"])
    if (
        model.get("model_revision") != "robot_v3.2.2-suction"
        or int(model.get("link_count", 0)) != 21
        or int(model.get("visual_mesh_count", 0)) != 49
        or not urdf.is_file()
        or not asset_root.is_dir()
        or sha256(urdf) != str(snapshot["urdf_sha256"])
    ):
        raise ValueError("V3.2.2生产外壳模型证书无效")
    for entry in model.get("files", []):
        path = manifest.parent / str(entry["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["size"])
            or sha256(path) != str(entry["sha256"])
        ):
            raise ValueError(f"模型资产SHA-256不匹配: {path}")
    return urdf, asset_root, str(snapshot["urdf_sha256"])


def runtime_environment() -> dict[str, str]:
    setup = """
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset LD_LIBRARY_PATH PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
source /home/tim/alfa_robot-motion261-v322/install-full/setup.bash
source /home/tim/alfa_robot-motion274-docking-error/install-motion274-yaw/setup.bash
env -0
"""
    completed = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", setup],
        check=True,
        stdout=subprocess.PIPE,
    )
    environment = dict(os.environ)
    for item in completed.stdout.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            environment[key.decode()] = value.decode()
    return environment


def y_slug(value: float) -> str:
    sign = "p" if value >= 0.0 else "m"
    return f"y-{sign}{int(round(abs(value) * 10000)):04d}d1mm"


def yaw_slug_precise(value: float) -> str:
    sign = "p" if value >= 0.0 else "m"
    return f"yaw-{sign}{int(round(abs(value) * 10)):03d}d10"


def recording_is_current(
    rrd: Path,
    summary_path: Path,
    replay: Path,
    validation: Path,
    urdf_sha: str,
) -> bool:
    if not rrd.is_file() or not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
        model = summary.get("rerun_model", {})
        return (
            summary.get("validation_success") is True
            and Path(str(summary["input"])).resolve() == replay.resolve()
            and Path(str(summary["validation_report"])).resolve()
            == validation.resolve()
            and model.get("model_revision") == "robot_v3.2.2-suction"
            and model.get("urdf_sha256") == urdf_sha
            and rrd.stat().st_mtime_ns
            >= max(replay.stat().st_mtime_ns, validation.stat().st_mtime_ns)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def open_rrd(rrd: Path, environment: dict[str, str], spawn: bool) -> None:
    subprocess.run(["rerun", "rrd", "verify", str(rrd)], check=True, env=environment)
    if spawn:
        subprocess.Popen(
            ["rerun", "--new", str(rrd)],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def open_x(upper: float, lower: float, environment: dict[str, str], spawn: bool) -> None:
    command = [
        "/usr/bin/python3", str(X_DEMO),
        "--upper-front-clearance-m", f"{upper:.2f}",
        "--lower-front-clearance-m", f"{lower:.2f}",
    ]
    if not spawn:
        command.append("--no-spawn")
    subprocess.run(command, check=True, env=environment)


def open_pose_result(
    prefix: str,
    output_name: str,
    result_path: Path,
    environment: dict[str, str],
    spawn: bool,
) -> Path:
    result = read_json(result_path)
    case_dir = result_path.parent
    replay = case_dir / "v322-pose-conveyor-replay.json"
    validation = case_dir / "v322-pose-conveyor-validation.json"
    urdf, asset_root, urdf_sha = model_snapshot()
    output = Y_RESULTS_ROOT / "rerun" / output_name
    output.mkdir(parents=True, exist_ok=True)
    replay_for_rerun = output / f"{prefix}-replay-with-model.json"
    replay_payload = read_json(replay)
    replay_payload.update({
        "model_revision": str(result["model_revision"]),
        "tool0_offset_local_z_m": float(result["tool0_offset_local_z_m"]),
        "upstream_base_commit": str(result["upstream_base_commit"]),
    })
    write_json_if_changed(replay_for_rerun, replay_payload)
    rrd = output / f"{prefix}-v322-production-shell.rrd"
    summary = output / f"{prefix}-summary.json"
    metrics = output / f"{prefix}-metrics.csv"
    planning = output / f"{prefix}-box-planning.csv"
    if not recording_is_current(
        rrd, summary, replay_for_rerun, validation, urdf_sha
    ):
        subprocess.run([
            "/usr/bin/python3", str(RERUN_RECORDER),
            "--input", str(replay_for_rerun),
            "--save", str(rrd),
            "--summary", str(summary),
            "--metrics-csv", str(metrics),
            "--box-planning-csv", str(planning),
            "--validation-report", str(validation),
            "--model-urdf", str(urdf),
            "--model-asset-root", str(asset_root),
            "--no-spawn",
        ], check=True, env=environment)
    if result.get("status") not in {"passed", "degraded"}:
        raise ValueError("结果不是完整成功状态")
    open_rrd(rrd, environment, spawn)
    return rrd


def ensure_on_demand_case(
    *,
    y_m: float,
    yaw_deg: float,
    environment: dict[str, str],
    retry_failed: bool,
) -> Path:
    slug = case_slug(0.0, y_m, yaw_deg)
    result_path = ON_DEMAND_RESULTS_ROOT / slug / "case-result.json"
    if result_path.is_file():
        result = read_json(result_path)
        validation = result.get("validation", {})
        if (
            result.get("status") in {"passed", "degraded"}
            and int(result.get("completed_boxes", 0)) == 25
            and validation.get("success")
        ):
            return result_path
        if not retry_failed:
            raise ValueError(
                "该输入已有完整失败记录；未打开Rerun: "
                f"stage={result.get('failure_stage', '')} "
                f"reason={result.get('failure_reason', '')}。"
                "如需重新随机规划，请增加--retry-failed"
            )
    print(
        "该点没有缓存，开始完整规划25箱并运行MoveIt/FCL验证；"
        "只有验证通过才会打开Rerun。",
        flush=True,
    )
    command = [
        "/usr/bin/python3", str(CASE_RUNNER),
        "--base-dx-m", "0",
        "--base-dy-m", f"{y_m:.6f}",
        "--yaw-deg", f"{yaw_deg:.6f}",
        "--baseline-cache", str(BASELINE_CACHE),
        "--output-root", str(ON_DEMAND_RESULTS_ROOT),
    ]
    if not retry_failed:
        command.append("--resume")
    completed = subprocess.run(command, check=False, env=environment)
    if not result_path.is_file():
        raise ValueError(
            f"规划器未生成结果，退出码={completed.returncode}: {slug}"
        )
    result = read_json(result_path)
    if (
        completed.returncode != 0
        or result.get("status") not in {"passed", "degraded"}
        or int(result.get("completed_boxes", 0)) != 25
        or not result.get("validation", {}).get("success")
    ):
        raise ValueError(
            "该输入未通过完整任务验证: "
            f"status={result.get('status')} "
            f"stage={result.get('failure_stage', '')} "
            f"reason={result.get('failure_reason', '')}"
        )
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper-x-m", type=float)
    parser.add_argument("--lower-x-m", type=float)
    parser.add_argument("--y-m", type=float)
    parser.add_argument("--yaw-deg", type=float)
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="重新规划已有失败记录；默认直接返回已知失败原因",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    y_cases = successful_y_cases()
    print(
        "单变量合同：X使用两组车头净距；Y或Yaw非零时X必须为0.85/0.60m。\n"
        "X最大独立范围：前三排0.74-1.02m，后两排0.36-0.79m；"
        "超出121组合强认证区间时只能扩展其中一组。",
        flush=True,
    )
    upper_requested = prompt_float(
        args.upper_x_m, "请输入前三排X净距(m)", DEFAULT_UPPER_M
    )
    lower_requested = prompt_float(
        args.lower_x_m, "请输入后两排X净距(m)", DEFAULT_LOWER_M
    )
    print(
        f"Y最大输入范围: {MAX_Y_RANGE_M[0]:+.4f}..{MAX_Y_RANGE_M[1]:+.4f}m；"
        "已有缓存值: "
        + ", ".join(f"{value:+g}" for value in sorted(y_cases)),
        flush=True,
    )
    y_requested = prompt_float(args.y_m, "请输入Y偏差(m)", 0.0)
    print(
        f"Yaw最大输入范围: {MAX_YAW_RANGE_DEG[0]:+g}.."
        f"{MAX_YAW_RANGE_DEG[1]:+g}deg（0.1deg精度）",
        flush=True,
    )
    yaw_requested = prompt_float(args.yaw_deg, "请输入Yaw偏差(deg)", 0.0)
    upper = round_centimeter(upper_requested)
    lower = round_centimeter(lower_requested)
    y_m = round_y(y_requested)
    yaw_value = round_yaw(yaw_requested)
    print(
        "输入精度换算："
        f"X={upper_requested:g}/{lower_requested:g} -> {upper:.2f}/{lower:.2f}m；"
        f"Y={y_requested:g} -> {y_m:+g}m；"
        f"Yaw={yaw_requested:g} -> {yaw_value:+g}deg。",
        flush=True,
    )
    result_path: Path | None = None
    yaw_deg: int | None = None
    try:
        axis = selected_axis(upper, lower, y_m, yaw_value)
        x_mode = x_case_mode(upper, lower)
        if not in_range(y_m, MAX_Y_RANGE_M):
            raise ValueError(
                f"Y必须在[{MAX_Y_RANGE_M[0]:+.4f}, "
                f"{MAX_Y_RANGE_M[1]:+.4f}]m"
            )
        if not in_range(yaw_value, MAX_YAW_RANGE_DEG):
            raise ValueError(
                f"Yaw必须在[{MAX_YAW_RANGE_DEG[0]:+g}, "
                f"{MAX_YAW_RANGE_DEG[1]:+g}]deg"
            )
        if axis == "y":
            result_path = next((
                path for value, path in y_cases.items()
                if math.isclose(value, y_m, abs_tol=1e-9)
            ), None)
        elif axis == "yaw":
            yaw_deg = certified_integer_yaw(yaw_value)
            if yaw_deg is not None:
                _, rrd = yaw_case(yaw_deg)
        elif x_mode != "strong_grid":
            result_path = exact_x_extension_case(
                upper, lower, successful_x_extension_cases()
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"选择 axis={axis} X={upper:.2f}/{lower:.2f}m "
        f"Y={y_m:+g}m Yaw={yaw_value:+g}deg",
        flush=True,
    )
    if axis == "x" and x_mode != "strong_grid":
        print(
            "提示：当前是独立X功能扩展点；完整任务已通过，但不属于121组合的"
            "全部任务核心低于3秒强保证。",
            flush=True,
        )
        if upper <= 0.76:
            print("姿态警告：前三排0.74-0.76m曾出现约6.093deg携箱倾角。", flush=True)
        if lower in {0.51, 0.52}:
            print("性能警告：后两排0.51/0.52m曾有任务核心超过3秒。", flush=True)
    if args.dry_run:
        cache_state = "已有完整缓存" if (
            axis == "x" or result_path is not None or axis == "yaw" and yaw_deg is not None
        ) else "运行时将先完整规划和FCL验证"
        print(f"DRY_RUN 输入范围通过；{cache_state}", flush=True)
        return 0
    try:
        environment = runtime_environment()
        if axis == "x":
            if x_mode == "strong_grid":
                open_x(upper, lower, environment, args.spawn)
            else:
                rrd = open_pose_result(
                    clearance_slug(upper, lower), "single-axis-x",
                    result_path, environment, args.spawn,
                )
                action = "已打开" if args.spawn else "校验通过（未打开）"
                print(
                    f"Rerun{action}: X={upper:.2f}/{lower:.2f}m rrd={rrd}",
                    flush=True,
                )
        elif axis == "y":
            if result_path is None:
                result_path = ensure_on_demand_case(
                    y_m=y_m, yaw_deg=0.0, environment=environment,
                    retry_failed=args.retry_failed,
                )
            rrd = open_pose_result(
                y_slug(y_m), "single-axis-y", result_path, environment, args.spawn
            )
            action = "已打开" if args.spawn else "校验通过（未打开）"
            print(f"Rerun{action}: Y={y_m:+g}m rrd={rrd}", flush=True)
        else:
            if yaw_deg is None:
                result_path = ensure_on_demand_case(
                    y_m=0.0, yaw_deg=yaw_value, environment=environment,
                    retry_failed=args.retry_failed,
                )
                rrd = open_pose_result(
                    yaw_slug_precise(yaw_value), "single-axis-yaw",
                    result_path, environment, args.spawn,
                )
            else:
                open_rrd(rrd, environment, args.spawn)
            action = "已打开" if args.spawn else "校验通过（未打开）"
            print(f"Rerun{action}: Yaw={yaw_value:+g}deg rrd={rrd}", flush=True)
    except (
        OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError
    ) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
