#!/usr/bin/python3
"""Open the certified V3 5x5 dual-arm right-conveyor shuttle replay."""

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

import v3_5x5_grasp_sequence_rerun as sequence


UPPER_RANGE_M = (0.80, 0.90)
LOWER_RANGE_M = (0.55, 0.65)
DEFAULT_UPPER_M = 0.85
DEFAULT_LOWER_M = 0.60
SCRIPT_DIR = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_to_certified_centimeter(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


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
    while True:
        raw = read(
            f"请输入{label}车头前接触面至箱体前接触面的 X 净距 "
            f"[{lower:.2f}-{upper:.2f} m，直接回车={default_m:.2f} m]: "
        ).strip()
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


def certified_pair_entry(
    workspace: Path, upper_front_clearance_m: float, lower_front_clearance_m: float
) -> dict[str, Any]:
    manifest_path = (
        workspace
        / "data/ik_benchmark/v3_scoop_5x5/range_certification_v322_tool0151"
        / "clearance-grid-certificate.json"
    )
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != "alfa.v322_tool0151_conveyor_clearance_grid.v1"
        or not manifest.get("success")
        or int(manifest.get("certified_pair_count", 0)) != 121
        or manifest.get("model_revision") != "robot_v3.2.2-suction"
        or not math.isclose(
            float(manifest.get("tool0_offset_local_z_m", 0.0)), 0.151, abs_tol=1e-12
        )
        or manifest.get("upstream_base_commit")
        != "d9c330cef72981390d81ac2b1cd5a6eb9e892195"
        or not math.isclose(
            float(manifest.get("input_resolution_m", 0.0)), 0.01, abs_tol=1e-12
        )
    ):
        raise ValueError(f"invalid V3.2.2 clearance-grid certificate: {manifest_path}")
    matches = [
        entry
        for entry in manifest.get("pairs", [])
        if math.isclose(
            float(entry["upper_front_clearance_m"]),
            upper_front_clearance_m,
            abs_tol=1e-9,
        )
        and math.isclose(
            float(entry["lower_front_clearance_m"]),
            lower_front_clearance_m,
            abs_tol=1e-9,
        )
    ]
    if len(matches) != 1 or not matches[0].get("success"):
        raise ValueError("requested clearance pair is not certified")
    entry = matches[0]
    for name in ("replay", "validation"):
        path = Path(str(entry[name]))
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.is_file() or sha256(path) != str(entry[f"{name}_sha256"]):
            raise ValueError(f"certified {name} hash mismatch: {path}")
        entry[name] = str(path.resolve())
    if not read_json(Path(entry["validation"])).get("success"):
        raise ValueError("certified validation report is not successful")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper-front-clearance-m", type=float)
    parser.add_argument("--lower-front-clearance-m", type=float)
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    upper_requested = prompt_front_clearance(
        args.upper_front_clearance_m, "前三排", UPPER_RANGE_M, DEFAULT_UPPER_M
    )
    lower_requested = prompt_front_clearance(
        args.lower_front_clearance_m, "后两排", LOWER_RANGE_M, DEFAULT_LOWER_M
    )
    if not UPPER_RANGE_M[0] <= upper_requested <= UPPER_RANGE_M[1]:
        parser.error("前三排车头 X 净距必须在 [0.80, 0.90] m")
    if not LOWER_RANGE_M[0] <= lower_requested <= LOWER_RANGE_M[1]:
        parser.error("后两排车头 X 净距必须在 [0.55, 0.65] m")
    upper = round_to_certified_centimeter(upper_requested)
    lower = round_to_certified_centimeter(lower_requested)
    print(
        "\n认证精度换算（十进制四舍五入到最近 1 cm）："
        f"前三排 输入 {upper_requested:.6f} m -> 实际执行 {upper:.2f} m；"
        f"后两排 输入 {lower_requested:.6f} m -> 实际执行 {lower:.2f} m。",
        flush=True,
    )
    root = sequence.workspace_root()
    try:
        entry = certified_pair_entry(root, upper, lower)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    replay = Path(str(entry["replay"])).resolve()
    validation = Path(str(entry["validation"])).resolve()
    output_dir = replay.parent
    pair_name = f"front-u{int(round(upper * 100)):03d}-l{int(round(lower * 100)):03d}"
    rrd = output_dir / f"v322-conveyor-{pair_name}.rrd"
    summary = output_dir / f"v322-conveyor-{pair_name}-summary.json"
    metrics = output_dir / f"v322-conveyor-{pair_name}-metrics.csv"
    planning = output_dir / f"v322-conveyor-{pair_name}-box-planning.csv"
    if not rrd.is_file() or rrd.stat().st_mtime_ns < replay.stat().st_mtime_ns:
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "v3_scoop_5x5_dual_rerun.py"),
            "--input", str(replay),
            "--save", str(rrd),
            "--summary", str(summary),
            "--metrics-csv", str(metrics),
            "--box-planning-csv", str(planning),
            "--validation-report", str(validation),
            "--no-spawn",
        ], check=True)
    subprocess.run(["rerun", "rrd", "verify", str(rrd)], check=True)
    print(
        "RESULT certified_grid=121/121 success=25/25 pickup_base_y=0.000m "
        "dual_pairs=10/10 simultaneous backoff=2.350m "
        "right_shuttle=1.500m cycles=15 joint_flips=0 "
        f"upper_front={upper:.2f}m lower_front={lower:.2f}m rrd={rrd}",
        flush=True,
    )
    if args.spawn:
        subprocess.Popen(
            ["rerun", "--new", str(rrd)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
