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


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_WORKSPACE = Path("/home/tim/alfa_robot-alfa_v3_dev")
YAW_WORKSPACE = Path("/home/tim/alfa_robot-motion274-docking-error")
UPPER_RANGE_M = (0.80, 0.90)
LOWER_RANGE_M = (0.55, 0.65)
DEFAULT_UPPER_M = 0.85
DEFAULT_LOWER_M = 0.60
Y_RESULTS_ROOT = ROOT / "data/ik_benchmark/v3_docking_error_matrix"
YAW_CASES_ROOT = YAW_WORKSPACE / "data/ik_benchmark/v3_yaw_robustness/cases"
YAW_RERUN_ROOT = YAW_WORKSPACE / "data/ik_benchmark/v3_yaw_robustness/release/rerun"
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


def prompt_float(
    value: float | None,
    prompt: str,
    default: float,
    input_fn: Callable[[str], str] | None = None,
) -> float:
    if value is not None:
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


def parse_yaw(value: float) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-9 or not -5 <= rounded <= 5:
        raise ValueError("Yaw必须是-5到+5之间的整数度")
    return int(rounded)


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


def y_recording_is_current(
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


def open_y(
    y_m: float,
    result_path: Path,
    environment: dict[str, str],
    spawn: bool,
) -> Path:
    result = read_json(result_path)
    case_dir = result_path.parent
    replay = case_dir / "v322-pose-conveyor-replay.json"
    validation = case_dir / "v322-pose-conveyor-validation.json"
    urdf, asset_root, urdf_sha = model_snapshot()
    output = Y_RESULTS_ROOT / "rerun" / "single-axis-y"
    output.mkdir(parents=True, exist_ok=True)
    prefix = y_slug(y_m)
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
    if not y_recording_is_current(
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
        raise ValueError("Y结果不是完整成功状态")
    open_rrd(rrd, environment, spawn)
    return rrd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper-x-m", type=float)
    parser.add_argument("--lower-x-m", type=float)
    parser.add_argument("--y-m", type=float)
    parser.add_argument("--yaw-deg", type=float)
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    y_cases = successful_y_cases()
    print(
        "单变量合同：X使用两组车头净距；Y或Yaw非零时X必须为0.85/0.60m。",
        flush=True,
    )
    upper_requested = prompt_float(
        args.upper_x_m, "请输入前三排X净距(m)", DEFAULT_UPPER_M
    )
    lower_requested = prompt_float(
        args.lower_x_m, "请输入后两排X净距(m)", DEFAULT_LOWER_M
    )
    print(
        "已有完整Y回放值: "
        + ", ".join(f"{value:+g}" for value in sorted(y_cases)),
        flush=True,
    )
    y_m = prompt_float(args.y_m, "请输入Y偏差(m)", 0.0)
    yaw_value = prompt_float(args.yaw_deg, "请输入Yaw偏差(deg)", 0.0)
    upper = round_centimeter(upper_requested)
    lower = round_centimeter(lower_requested)
    if not UPPER_RANGE_M[0] <= upper_requested <= UPPER_RANGE_M[1]:
        raise SystemExit("前三排X净距必须在[0.80, 0.90]m强认证范围内")
    if not LOWER_RANGE_M[0] <= lower_requested <= LOWER_RANGE_M[1]:
        raise SystemExit("后两排X净距必须在[0.55, 0.65]m强认证范围内")
    try:
        axis = selected_axis(upper, lower, y_m, yaw_value)
        if axis == "y":
            y_m, result_path = exact_y_case(y_m, y_cases)
        elif axis == "yaw":
            yaw_deg = parse_yaw(yaw_value)
            _, rrd = yaw_case(yaw_deg)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"选择 axis={axis} X={upper:.2f}/{lower:.2f}m "
        f"Y={y_m:+g}m Yaw={yaw_value:+g}deg",
        flush=True,
    )
    if args.dry_run:
        print("DRY_RUN 输入合同与完整证书检查通过", flush=True)
        return 0
    environment = runtime_environment()
    if axis == "x":
        open_x(upper, lower, environment, args.spawn)
    elif axis == "y":
        rrd = open_y(y_m, result_path, environment, args.spawn)
        action = "已打开" if args.spawn else "校验通过（未打开）"
        print(f"Rerun{action}: Y={y_m:+g}m rrd={rrd}", flush=True)
    else:
        open_rrd(rrd, environment, args.spawn)
        action = "已打开" if args.spawn else "校验通过（未打开）"
        print(f"Rerun{action}: Yaw={yaw_deg:+d}deg rrd={rrd}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
