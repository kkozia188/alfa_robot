#!/usr/bin/env python3
"""Prompt for a certified base-yaw error and open its complete Rerun replay."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CASES = REPO / "data/ik_benchmark/v3_yaw_robustness/cases"
DEFAULT_RERUN = REPO / "data/ik_benchmark/v3_yaw_robustness/release/rerun"


def yaw_slug(yaw_deg: int) -> str:
    return f"yaw-{'p' if yaw_deg >= 0 else 'm'}{abs(yaw_deg):02d}"


def parse_certified_yaw(value: str) -> int:
    try:
        numeric = float(value.strip())
    except ValueError as error:
        raise ValueError("请输入数字，例如 -5、0 或 3") from error
    if not math.isfinite(numeric):
        raise ValueError("角度必须是有限数字")
    rounded = round(numeric)
    if abs(numeric - rounded) > 1e-9:
        raise ValueError("当前证书按 1 度取样，请输入 -5 到 +5 的整数")
    yaw = int(rounded)
    if not -5 <= yaw <= 5:
        raise ValueError("角度超出认证范围，请输入 -5 到 +5")
    return yaw


def certified_case(cases_root: Path, yaw_deg: int) -> Path:
    case_dir = cases_root / yaw_slug(yaw_deg)
    summary_path = case_dir / "summary.json"
    validation_path = case_dir / "v322-yaw-conveyor-validation.json"
    if not summary_path.is_file() or not validation_path.is_file():
        raise RuntimeError(f"缺少 Yaw {yaw_deg:+d} 度的本地证书")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        summary.get("completed_boxes") != 25
        or summary.get("cycle_count") != 15
        or not validation.get("success")
    ):
        raise RuntimeError(f"Yaw {yaw_deg:+d} 度的证书未通过完整验收")
    return case_dir


def ensure_rerun(case_dir: Path, rerun_root: Path) -> Path:
    rrd = rerun_root / f"{case_dir.name}-v322-production-shell.rrd"
    if rrd.is_file():
        return rrd
    raise RuntimeError(
        f"缺少 {rrd.name}。请先按 tools/v3_yaw_robustness/README.md 生成生产外壳 Rerun。"
    )


def prompt_yaw() -> int:
    while True:
        value = input("请输入底盘 Yaw 角度偏差（-5 到 +5，整数度）：").strip()
        try:
            return parse_certified_yaw(value)
        except ValueError as error:
            print(f"输入无效：{error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaw-deg", help="跳过交互提示，直接打开指定整数角度")
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--rerun-root", type=Path, default=DEFAULT_RERUN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    yaw = parse_certified_yaw(args.yaw_deg) if args.yaw_deg is not None else prompt_yaw()
    case_dir = certified_case(args.cases_root.resolve(), yaw)
    rrd = ensure_rerun(case_dir, args.rerun_root.resolve())
    print(f"已选择 Yaw {yaw:+d} 度，正在校验 Rerun：{rrd}")
    subprocess.run(["rerun", "rrd", "verify", str(rrd)], check=True)
    subprocess.run(["rerun", "--new", "--detach-process", str(rrd)], check=True)
    print(f"Rerun 已打开：Yaw {yaw:+d} 度，完整 25 箱 / 15 循环")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
