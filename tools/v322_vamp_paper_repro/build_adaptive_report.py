#!/usr/bin/env python3
"""Build a self-contained Chinese report for the adaptive matched benchmark."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def number(value, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("rerun_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--paper-summary", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    paper_path = args.paper_summary or (
        Path(__file__).resolve().parents[2]
        / "results/v322_vamp_paper_repro/paper-paired-5x5-final/summary.json"
    )
    paper = json.loads(paper_path.read_text(encoding="utf-8"))
    rows = summary["per_box"]
    winning = summary["search_metric"]["winning_search"]
    cumulative = summary["search_metric"]["cumulative_to_solution"]
    table_rows = "\n".join(
        "<tr>" + "".join([
            f"<td>{row['box_id']}</td>",
            f"<td>{row['successes']}/{row['rounds']}</td>",
            f"<td>{html.escape(row['selected_base_x'])}</td>",
            f"<td>{html.escape(row['selected_side'])}</td>",
            f"<td>{html.escape(row['selected_mode'])}</td>",
            f"<td>{number(row['ompl_winning_median_ms'])}</td>",
            f"<td>{number(row['vamp_winning_median_ms'])}</td>",
            f"<td>{number(row['winning_median_speedup'], 2)}x</td>",
            f"<td>{number(row['ompl_median_ms'])}</td>",
            f"<td>{number(row['vamp_median_ms'])}</td>",
            f"<td>{number(row['cumulative_median_speedup'], 2)}x</td>",
            f"<td>{number(row['median_reduction_percent'], 2)}%</td>",
        ]) + "</tr>" for row in rows
    )
    data = json.dumps(rows, ensure_ascii=False)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V3.2.2 RRT-Connect / VAMP 同端点对比</title>
<style>
:root{{--ink:#182028;--muted:#5e6871;--line:#d5d9dc;--paper:#f7f8f8;--ompl:#2676c9;--vamp:#e98a22;--good:#16845b;--warn:#b6443c}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,"Noto Sans CJK SC",sans-serif}}
header{{padding:26px max(24px,5vw) 20px;background:#fff;border-bottom:1px solid var(--line)}} h1{{margin:0 0 8px;font-size:25px;letter-spacing:0}} p{{margin:6px 0;color:var(--muted)}}
main{{max-width:1500px;margin:auto;padding:22px max(20px,4vw) 50px}} .metrics{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}} .metric{{background:#fff;padding:16px}} .metric b{{display:block;font-size:24px;letter-spacing:0}} .metric span{{color:var(--muted)}}
.downloads{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}} a{{color:#075b9e}} .downloads a{{display:inline-flex;align-items:center;padding:9px 12px;background:#fff;border:1px solid var(--line);border-radius:4px;text-decoration:none}}
section{{margin-top:24px}} h2{{font-size:18px;margin:0 0 12px;letter-spacing:0}} .charts{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .chart{{background:#fff;border:1px solid var(--line);padding:14px;min-width:0}} canvas{{display:block;width:100%;height:280px}}
.table-wrap{{overflow:auto;background:#fff;border:1px solid var(--line)}} table{{border-collapse:collapse;width:100%;min-width:1120px}} th,td{{padding:8px 10px;border-bottom:1px solid #e5e7e8;text-align:right;white-space:nowrap}} th{{position:sticky;top:0;background:#eef1f2;color:#3d4851}} th:nth-child(-n+5),td:nth-child(-n+5){{text-align:left}}
.note{{border-left:4px solid var(--good);padding:10px 14px;background:#fff}} @media(max-width:850px){{.metrics,.charts{{grid-template-columns:1fr 1fr}}}} @media(max-width:520px){{.metrics,.charts{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>RRT-Connect 导入 VAMP 后的同端点规划加速</h1><p>V3.2.2 · 25 箱 · 共享 IK · 论文核心 VAMP/SIMD 与工业完整 VAMP/FCL 分层对比</p></header>
<main>
<div class="metrics">
<div class="metric"><b>{number(paper['paper_speedup_median'],2)}x</b><span>论文球模型逐配对中位加速</span></div>
<div class="metric"><b>{number(paper['certified_speedup_median'],2)}x</b><span>FCL 认证子集中位加速</span></div>
<div class="metric"><b>{number(winning['aggregate_speedup'],2)}x</b><span>成功单次累计加速</span></div>
<div class="metric"><b>{number(cumulative['aggregate_speedup'],2)}x</b><span>从开始到首组成功累计加速</span></div>
<div class="metric"><b>{summary['coverage']['successful_trials']}/{summary['coverage']['total_trials']}</b><span>成功轮次</span></div>
<div class="metric"><b>{'是' if summary['coverage']['all_25_boxes_have_success'] else '否'}</b><span>25 箱均有成功结果</span></div>
</div>
<div class="downloads"><a href="{args.rerun_dir.name}/paper-pure-rrt-connect-fcl-5x5.rrd">论文核心 Pure RRT/FCL</a><a href="{args.rerun_dir.name}/paper-vamp-rrt-connect-simd-5x5.rrd">论文核心 VAMP/SIMD</a><a href="{args.rerun_dir.name}/adaptive-pure-rrt-connect-fcl-5x5.rrd">工业完整 Pure RRT/FCL</a><a href="{args.rerun_dir.name}/adaptive-vamp-rrt-connect-fcl-5x5.rrd">工业完整 VAMP RRT/FCL</a><a href="per-box.csv">逐箱 CSV</a><a href="per-stage.csv">逐阶段 CSV</a></div>
<div class="note">本页完整任务对比让 OMPL 与 VAMP 共用精确 MoveIt/FCL，以保证 25 箱动作安全闭环。原生 SIMD 的论文式核心加速与 FCL 通过率在主报告单独展示，不能与本页混为同一指标。</div>
<section><h2>论文式实验的成功率分解</h2>
<p>计划试验 {paper['scheduled_trials']} 次。共同 IK 端点 {paper['endpoint_ready_trials']}/{paper['scheduled_trials']}（{100*paper['endpoint_ready_trials']/paper['scheduled_trials']:.2f}%）；这一步发生在 RRT 之前，失败来自固定抓取配置的 IK 不可用，不属于 VAMP 搜索失败。</p>
<p>在 {paper['endpoint_ready_trials']} 个共同端点上，OMPL 精确解 {paper['ompl_exact_trials']} 个（{100*paper['ompl_exact_trials']/paper['endpoint_ready_trials']:.2f}%），VAMP 球模型原生解 {paper['vamp_native_solved_trials']} 个（{100*paper['vamp_native_solved_trials']/paper['endpoint_ready_trials']:.2f}%）。VAMP 原生解中只有 {paper['vamp_fcl_valid_trials']}/{paper['vamp_native_solved_trials']}（{100*paper['vamp_native_fcl_pass_rate']:.2f}%）通过最终 FCL；相对共同端点为 {100*paper['vamp_fcl_valid_trials']/paper['endpoint_ready_trials']:.2f}%，相对全部计划试验为 {100*paper['vamp_fcl_valid_trials']/paper['scheduled_trials']:.2f}%。</p>
<p>球模型的 12.61% 假阴性会让真实碰撞构型进入 VAMP 路径；路径包含许多状态和边，任一点命中漏检区都会让整条路径在 FCL 门禁失败。19.18% 假阳性则会错误封闭真实自由空间，增加窄通道搜索超时。两种误差分别解释“快速得到但不可认证”和“找不到路径”。</p></section>
<section><h2>为什么原生 SIMD 不能直接完成当前 25 箱闭环</h2>
<p>当前球模型相对生产 MoveIt/FCL 网格的碰撞漏检率为 12.61%。漏检会把真实碰撞构型标成自由构型，因此原生 VAMP RRT-Connect 能很快返回一条“球模型有效”路径，但稠密 FCL 会在事后否决它。当前 Cricket 内核还没有把动态附着箱编码成机器人碰撞球；携箱抽离和搬运时，搜索器看不到箱体体积，系统性误判会进一步增加。</p>
<p>Box 1 诊断中，站位 X=-0.60m 的 VAMP 原生搜索累计 148.201ms、站位 X=-0.35m 累计 951.856ms；两个站位各尝试两个共享 IK 候选、每候选 5 个 seed，仍没有路径通过稠密 FCL。换 seed 只改变随机路径，不能修正自由空间标签或补上附着箱几何。因此本页完整任务改用 VAMP RRT-Connect + 精确 FCL，并把原生 SIMD 的论文指标单独保留。</p>
<p>可行修复是重新生成保守球模型并把假阴性降为 0，再为左/右吸附状态分别生成包含箱体的碰撞内核；随后必须重跑状态标签审计、边标签审计和 25 箱多 seed 回归。只有这些通过后，原生 SIMD 才能替换本页的精确 FCL 后端。</p></section>
<section><h2>安全 SIMD 外包络实验结果</h2>
<p>4 球/网格的保守外包络在 10,000 随机状态上把假阴性降为 0，但假阳性 71.04%，没有一个共同自由状态。按空间焊接顶点、拆连通分量并增加球数后，仍被中央机架/头部与 link1/2 的系统性重叠阻塞。</p>
<p>定位原快速模型的假阴性后发现，1,286 个漏检状态主要来自被排除的 base_link 与双臂 link3–7。将底座改为 768 个聚类 OBB 后，10,000 随机状态假阴性为 0、共同自由 43.07%、假阳性 27.97%；但在 1,000 个实际新搜索路径状态上假阳性仍达 83.10%，仅 16.90% 共同自由，无法完成 25 箱搜索。</p>
<p>因此本轮没有生成“安全纯 SIMD 25 箱 Rerun”：生成它会掩盖搜索失败。下一可行实现是动态臂继续使用 SIMD 球，固定底座和动态载荷使用专用 cuboid/capsule，或只对这些少量 pair 做局部 FCL；完成后仍需达到 25/25 和逐边 FCL 零否决才可发布。</p></section>
<section><h2>25 箱搜索时间</h2><div class="charts"><div class="chart"><canvas id="winning"></canvas></div><div class="chart"><canvas id="cumulative"></canvas></div></div></section>
<section><h2>逐箱完整数据</h2><div class="table-wrap"><table><thead><tr><th>箱号</th><th>成功</th><th>站位 X</th><th>机械臂</th><th>吸附</th><th>OMPL 成功单次 ms</th><th>VAMP 成功单次 ms</th><th>单次加速</th><th>OMPL 累计 ms</th><th>VAMP 累计 ms</th><th>累计加速</th><th>时间变化</th></tr></thead><tbody>{table_rows}</tbody></table></div></section>
</main><script>
const rows={data};
function chart(id,title,okey,vkey){{const c=document.getElementById(id),d=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*d;c.height=h*d;const x=c.getContext('2d');x.scale(d,d);x.font='12px system-ui';x.fillStyle='#182028';x.fillText(title,8,15);const max=Math.max(1,...rows.flatMap(r=>[r[okey]||0,r[vkey]||0]));const top=28,bottom=24,left=40,right=8,plotH=h-top-bottom,bw=(w-left-right)/rows.length;for(let i=0;i<5;i++){{const y=top+plotH*i/4;x.strokeStyle='#e1e4e6';x.beginPath();x.moveTo(left,y);x.lineTo(w-right,y);x.stroke();x.fillStyle='#6b747c';x.fillText((max*(1-i/4)).toFixed(0),2,y+4)}}rows.forEach((r,i)=>{{const a=(r[okey]||0)/max*plotH,b=(r[vkey]||0)/max*plotH,base=left+i*bw;x.fillStyle='#2676c9';x.fillRect(base+bw*.12,top+plotH-a,bw*.32,a);x.fillStyle='#e98a22';x.fillRect(base+bw*.52,top+plotH-b,bw*.32,b);if(i%2===0){{x.fillStyle='#4e5962';x.fillText(String(r.box_id),base+bw*.32,h-6)}}}})}}
chart('winning','Winning search time: OMPL blue / VAMP orange','ompl_winning_median_ms','vamp_winning_median_ms');chart('cumulative','Cumulative time to first complete solution','ompl_median_ms','vamp_median_ms');addEventListener('resize',()=>{{chart('winning','Winning search time: OMPL blue / VAMP orange','ompl_winning_median_ms','vamp_winning_median_ms');chart('cumulative','Cumulative time to first complete solution','ompl_median_ms','vamp_median_ms')}});
</script></body></html>"""
    args.output.resolve().write_text(document, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
