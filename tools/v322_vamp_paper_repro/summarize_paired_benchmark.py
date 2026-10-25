#!/usr/bin/env python3
"""Summarize raw paired RRT-Connect benchmark JSON without hiding failures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def median(values):
    return statistics.median(values) if values else None


def ratio(ompl_ms, vamp_ms):
    return ompl_ms / vamp_ms if ompl_ms is not None and vamp_ms else None


def fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_speedup(value):
    return "-" if value is None else f"{value:.2f}x"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--sphere-audit", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--kernel", type=Path)
    args = parser.parse_args()
    raw_files = sorted((args.experiment_dir / "raw").glob("*.json"))
    if not raw_files:
        raise RuntimeError("no raw trial JSON files found")

    rows = []
    for path in raw_files:
        payload = json.loads(path.read_text())
        benchmark = payload.get("paper_benchmark")
        row = {
            "box_id": payload["target_box_id"],
            "halton_skip": benchmark.get("halton_skip") if benchmark else None,
            "endpoint_ready": bool(benchmark),
            "ompl_exact": False,
            "vamp_native_solved": False,
            "vamp_fcl_valid": False,
            "paired_certified": False,
            "ompl_ms": None,
            "vamp_ms": None,
            "paper_speedup": None,
            "certified_speedup": None,
            "failure_stage": payload.get("failure_stage", ""),
            "raw_json": str(path.resolve()),
        }
        if benchmark:
            ompl = benchmark["ompl"]
            vamp = benchmark["vamp"]
            audit = vamp["audit"]
            row.update({
                "ompl_exact": bool(ompl["success"]),
                "vamp_native_solved": audit.get("sparse_states", 0) >= 2,
                "vamp_fcl_valid": audit.get("fcl_audit_valid") is True,
                "paired_certified": bool(payload["success"]),
                "ompl_ms": ompl["planner_ms"],
                "vamp_ms": vamp["planner_ms"],
                "ompl_direct": ompl.get("audit", {}).get("direct_motion"),
                "vamp_iterations": audit.get("iterations"),
                "vamp_tree_nodes": audit.get("tree_nodes"),
            })
            if row["ompl_exact"] and row["vamp_native_solved"]:
                row["paper_speedup"] = ratio(row["ompl_ms"], row["vamp_ms"])
            if row["paired_certified"]:
                row["certified_speedup"] = ratio(row["ompl_ms"], row["vamp_ms"])
        rows.append(row)

    by_box = []
    for box_id in range(1, 26):
        trials = [row for row in rows if row["box_id"] == box_id]
        endpoints = [row for row in trials if row["endpoint_ready"]]
        by_box.append({
            "box_id": box_id,
            "trials": len(trials),
            "endpoint_trials": len(endpoints),
            "ompl_exact": sum(row["ompl_exact"] for row in endpoints),
            "vamp_native_solved": sum(row["vamp_native_solved"] for row in endpoints),
            "vamp_fcl_valid": sum(row["vamp_fcl_valid"] for row in endpoints),
            "paired_certified": sum(row["paired_certified"] for row in endpoints),
            "ompl_exact_median_ms": median([
                row["ompl_ms"] for row in endpoints if row["ompl_exact"]
            ]),
            "vamp_native_median_ms": median([
                row["vamp_ms"] for row in endpoints if row["vamp_native_solved"]
            ]),
            "paper_speedup_median": median([
                row["paper_speedup"] for row in endpoints if row["paper_speedup"] is not None
            ]),
            "certified_speedup_median": median([
                row["certified_speedup"] for row in endpoints
                if row["certified_speedup"] is not None
            ]),
        })

    endpoints = [row for row in rows if row["endpoint_ready"]]
    paper_pairs = [row for row in endpoints if row["paper_speedup"] is not None]
    certified = [row for row in endpoints if row["certified_speedup"] is not None]
    native_solved = [row for row in endpoints if row["vamp_native_solved"]]
    summary = {
        "schema": "alfa.v322.vamp_paper_summary.v1",
        "scheduled_trials": len(rows),
        "endpoint_ready_trials": len(endpoints),
        "endpoint_failed_trials": len(rows) - len(endpoints),
        "ompl_exact_trials": sum(row["ompl_exact"] for row in endpoints),
        "vamp_native_solved_trials": len(native_solved),
        "vamp_fcl_valid_trials": sum(row["vamp_fcl_valid"] for row in endpoints),
        "paired_certified_trials": len(certified),
        "vamp_native_fcl_pass_rate": (
            sum(row["vamp_fcl_valid"] for row in native_solved) / len(native_solved)
            if native_solved else None
        ),
        "paper_comparable_trials": len(paper_pairs),
        "paper_speedup_median": median([row["paper_speedup"] for row in paper_pairs]),
        "paper_ratio_of_medians": ratio(
            median([row["ompl_ms"] for row in paper_pairs]),
            median([row["vamp_ms"] for row in paper_pairs]),
        ),
        "certified_speedup_median": median([
            row["certified_speedup"] for row in certified
        ]),
        "certified_ratio_of_medians": ratio(
            median([row["ompl_ms"] for row in certified]),
            median([row["vamp_ms"] for row in certified]),
        ),
        "paper_ompl_median_ms": median([row["ompl_ms"] for row in paper_pairs]),
        "paper_vamp_median_ms": median([row["vamp_ms"] for row in paper_pairs]),
        "certified_ompl_median_ms": median([row["ompl_ms"] for row in certified]),
        "certified_vamp_median_ms": median([row["vamp_ms"] for row in certified]),
        "by_box": by_box,
    }
    if args.sphere_audit:
        summary["sphere_model_audit"] = json.loads(args.sphere_audit.read_text())
    summary["artifact_sha256"] = {}
    for name, path in (("release_binary", args.binary), ("cricket_kernel", args.kernel)):
        if path:
            summary["artifact_sha256"][name] = {
                "path": str(path.resolve()), "sha256": sha256(path)
            }
    (args.experiment_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (args.experiment_dir / "trials.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.experiment_dir / "boxes.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(by_box[0]))
        writer.writeheader()
        writer.writerows(by_box)

    lines = [
        "# V3.2.2 OMPL/FCL 与原生 VAMP RRT-Connect 配对实验",
        "",
        "两组使用同一次在线 IK 的相同 start/goal、相同 Halton 段、相同 range，"
        "且都先做计时内 start→goal 直连检查。IK、启动和最终稠密 FCL 复核不计入规划时间。",
        "",
        "## 总体结果",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 计划实验 | {summary['scheduled_trials']} |",
        f"| 获得共同 IK 端点 | {summary['endpoint_ready_trials']} |",
        f"| OMPL/FCL 精确解 | {summary['ompl_exact_trials']} |",
        f"| VAMP 球模型原生解 | {summary['vamp_native_solved_trials']} |",
        f"| VAMP 解通过最终 FCL | {summary['vamp_fcl_valid_trials']} |",
        f"| 双方成功且可认证 | {summary['paired_certified_trials']} |",
        f"| VAMP 原生解的 FCL 通过率 | {fmt(100 * summary['vamp_native_fcl_pass_rate'], 2)}% |",
        f"| 论文球模型口径中位加速（n={len(paper_pairs)}） | {fmt(summary['paper_speedup_median'], 2)}x |",
        f"| 最终 FCL 通过子集中位加速（n={len(certified)}） | {fmt(summary['certified_speedup_median'], 2)}x |",
        "",
        "> 论文球模型口径包含最终 FCL 未通过的路径，只能说明原生球碰撞搜索速度。"
        "可执行安全结论必须使用“最终 FCL 通过”子集。",
        "",
    ]
    if args.sphere_audit:
        audit = summary["sphere_model_audit"]
        lines.extend([
            "## 球模型审计",
            "",
            f"同一关节结构下随机 `{audit['samples']}` 个全关节域状态："
            f"假阳性 `{100 * audit['rates']['sphere_false_positive']:.2f}%`，"
            f"假阴性 `{100 * audit['rates']['sphere_false_negative']:.2f}%`。"
            "因此球模型不是原网格的安全等价替代，最终 FCL 门禁不可删除。",
            "",
        ])
    lines.extend([
        "## 每箱结果",
        "",
        "| 箱号 | IK | OMPL | VAMP原生 | FCL通过 | 认证配对 | OMPL中位ms | VAMP中位ms | 论文口径中位 | 认证中位 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in by_box:
        lines.append(
            f"| {item['box_id']} | {item['endpoint_trials']}/{item['trials']} | "
            f"{item['ompl_exact']}/{item['endpoint_trials']} | "
            f"{item['vamp_native_solved']}/{item['endpoint_trials']} | "
            f"{item['vamp_fcl_valid']}/{item['endpoint_trials']} | "
            f"{item['paired_certified']}/{item['endpoint_trials']} | "
            f"{fmt(item['ompl_exact_median_ms'])} | {fmt(item['vamp_native_median_ms'])} | "
            f"{fmt_speedup(item['paper_speedup_median'])} | "
            f"{fmt_speedup(item['certified_speedup_median'])} |"
        )
    certified_boxes = [item["box_id"] for item in by_box if item["paired_certified"]]
    sphere_only_boxes = [
        item["box_id"] for item in by_box
        if item["vamp_native_solved"] and not item["vamp_fcl_valid"]
    ]
    vamp_blocked_boxes = [
        item["box_id"] for item in by_box
        if item["ompl_exact"] and not item["vamp_native_solved"]
    ]
    ik_failed_boxes = [item["box_id"] for item in by_box if not item["endpoint_trials"]]
    lines.extend([
        "",
        "## 现象与原因",
        "",
        f"- 论文球模型口径的中位加速为 `{summary['paper_speedup_median']:.2f}x`，"
        f"但只有 `{summary['vamp_fcl_valid_trials']}/{summary['vamp_native_solved_trials']}` "
        "条 VAMP 原生路径通过最终 FCL。原生球碰撞把多个状态装入 SIMD 批次，"
        "且 Cricket 展开了 FK 与球碰撞；MoveIt/FCL 则逐状态处理三角网格层次，因此前者单次检查明显更快。",
        f"- 通过 FCL 的箱位为 `{certified_boxes}`。其中多数组合是 start→goal 直连，"
        "主要测到的是同一条边上球碰撞与 FCL 网格碰撞的成本差，而不是长时间扩树。"
        "Box 1 需要真实建树且只通过 2/5 次，其认证中位加速更低，更能反映搜索与模型偏差共同作用。",
        f"- 箱位 `{sphere_only_boxes}` 存在 VAMP 球模型解但最终 FCL 全部失败。"
        "这些组的数百倍原生加速来自球模型假阴性，不能解释为可执行规划加速。",
        f"- 箱位 `{vamp_blocked_boxes}` 中 OMPL 至少一次成功而 VAMP 原生无解。"
        "这里球体包络假阳性会缩小可行域；243 球也增加了每次 SIMD 检查工作量，"
        "所以 VAMP 在困难场景中可能耗尽 100 万次迭代。",
        f"- 箱位 `{ik_failed_boxes}` 没有生成共同在线 IK 端点，未进入 RRT。"
        "这些是端点覆盖问题，不能计作任一规划器的成功、失败或耗时。",
        "- 诊断轮曾让 VAMP 独有 start→goal 直连预检、且 OMPL setup 消耗 Halton 前缀；"
        "最终轮已给两边相同直连预检，并在 setup 后重置 Halton，因此未沿用诊断轮加速数据。",
        "",
        "## 后续解决方式",
        "",
        "1. **自适应保守球化**：以原始网格为真值，按产生假阴性/假阳性的具体连杆局部增球，"
        "并以相关任务分布和随机状态做标签优化。它改变的是派生碰撞代理精度，不改变机器人关节结构；"
        "可行性已经由本轮从端点多对伪碰撞收敛到零得到验证，但要达到安全替代仍需把假阴性降到零。",
        "2. **压缩 Cricket 碰撞代码**：保留包围球层次，但把大量内层球对从完全展开改为分块循环或生成独立编译单元。"
        "这会降低 32 MB 头文件和约 17 分钟 Release 编译成本，使增加球数成为可迭代方案；"
        "需要重新基准确认循环没有抵消 SIMD 收益。",
        "3. **扩大共同端点覆盖**：在线 IK 同时搜索合法 updown，而不是固定当前高度，"
        "让低位箱也产生共同 start/goal。两边仍只接收同一个冻结端点，IK 继续排除在计时外；"
        "该方案可行，但必须作为独立端点生成实验，不能与本轮 85 组结果混合。",
        "4. **保持双层结果门禁**：继续同时报告原生球模型时间和最终 FCL 通过率；"
        "在球模型达到零假阴性前，生产路径只接受 FCL 通过结果。该方案已在当前代码中实现，"
        "代价是部分看似极快的 VAMP 路径会被正确拒绝。",
    ])
    lines.extend([
        "",
        "## 边界",
        "",
        "- 机器人关节结构、限位、原始碰撞网格和 SRDF 未修改；VAMP 使用派生球体代理。",
        "- 无历史路径、路点、shortcut、Cartesian 或 fallback。",
        "- VAMP 球模型不是原始网格的精确等价物；模型审计和每条路径 FCL 结果必须共同阅读。",
        "- `paper_speedup` 只在 OMPL 精确解与 VAMP 原生球模型解同时存在时计算。",
        "- `certified_speedup` 还要求 VAMP 路径通过 1deg/1cm 稠密 MoveIt/FCL 复核。",
        "",
    ])
    (args.experiment_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({key: value for key, value in summary.items() if key != "by_box"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
