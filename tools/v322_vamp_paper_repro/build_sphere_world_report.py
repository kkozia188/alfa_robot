#!/usr/bin/env python3
"""Build the self-contained report for the completed SIMD sphere-world 5x5 run."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def value(number: float | None, digits: int = 3) -> str:
    return "-" if number is None else f"{float(number):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = summary["per_box"]
    coverage = summary["coverage"]
    metric = summary["search_metric"]["cumulative_to_solution"]
    stage_totals: dict[str, dict[str, float]] = {}
    with (root / "per-stage.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            stage = row["stage"]
            total = stage_totals.setdefault(stage, {"ompl": 0.0, "vamp": 0.0, "count": 0.0})
            total["ompl"] += float(row["ompl_cumulative_ms"])
            total["vamp"] += float(row["vamp_cumulative_ms"])
            total["count"] += 1.0
    stage_labels = {
        "to_precontact": "到预接触",
        "precontact_to_contact": "预接触到接触",
        "contact_to_retreat_loaded": "携箱抽离",
        "retreat_to_transport_loaded": "携箱搬运",
        "released_transport_to_home": "释放后回位",
    }
    stage_table = "".join(
        "<tr>" + "".join([
            f"<td>{stage_labels.get(stage, stage)}</td>",
            f"<td>{int(total['count'])}</td>",
            f"<td>{value(total['ompl'])}</td>",
            f"<td>{value(total['vamp'])}</td>",
            f"<td>{value(total['ompl'] / total['vamp'], 2)}x</td>",
        ]) + "</tr>"
        for stage, total in stage_totals.items()
    )

    table = []
    for row in rows:
        reduction = float(row["median_reduction_percent"])
        table.append("<tr>" + "".join([
            f"<td>{row['box_id']}</td>",
            f"<td>{html.escape(row['selected_side'])}</td>",
            f"<td>{html.escape(row['selected_mode'])}</td>",
            f"<td>{row['successes']}/{row['rounds']}</td>",
            f"<td>{value(row['ompl_median_ms'])}</td>",
            f"<td>{value(row['vamp_median_ms'])}</td>",
            f"<td>{value(row['cumulative_median_speedup'], 2)}x</td>",
            f"<td class='gain'>-{value(reduction, 2)}%</td>",
            f"<td>{value(row['ompl_p95_ms'])}</td>",
            f"<td>{value(row['vamp_p95_ms'])}</td>",
        ]) + "</tr>")

    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VAMP SIMD 球碰撞 5x5 完整任务实验</title>
<style>
:root{{--bg:#f4f6f7;--surface:#fff;--ink:#182127;--muted:#5d6971;--line:#d6dde1;--blue:#2475b9;--orange:#e47c25;--green:#16835b;--red:#b33d38}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,"Noto Sans CJK SC",sans-serif;letter-spacing:0}}
header{{background:var(--surface);border-bottom:1px solid var(--line);padding:24px max(22px,4vw) 18px}} header h1{{font-size:25px;margin:0 0 6px}} header p{{margin:0;color:var(--muted)}}
main{{max-width:1480px;margin:auto;padding:22px max(18px,3vw) 48px}} h2{{font-size:18px;margin:0 0 10px}} h3{{font-size:15px;margin:18px 0 7px}} p{{margin:7px 0;color:#354149}}
.metrics{{display:grid;grid-template-columns:repeat(5,minmax(145px,1fr));border:1px solid var(--line);background:var(--line);gap:1px}}
.metric{{background:var(--surface);padding:14px}} .metric b{{display:block;font-size:22px}} .metric span{{color:var(--muted)}}
.links{{display:flex;gap:9px;flex-wrap:wrap;margin:16px 0 0}} .links a{{padding:8px 10px;border:1px solid var(--line);border-radius:4px;background:var(--surface);color:#075b95;text-decoration:none}}
section{{margin-top:26px}} .band{{background:var(--surface);border:1px solid var(--line);padding:16px}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .chart{{min-width:0;background:var(--surface);border:1px solid var(--line);padding:12px}} .chart h3{{margin:0 0 4px}} .chart p{{font-size:12px;margin:0 0 6px}}
svg{{display:block;width:100%;height:360px}} .legend{{display:flex;gap:18px;color:var(--muted);font-size:12px}} .swatch{{display:inline-block;width:12px;height:8px;margin-right:5px}} .ompl{{background:var(--blue)}} .vamp{{background:var(--orange)}}
.table-wrap{{overflow:auto;border:1px solid var(--line);background:var(--surface)}} table{{border-collapse:collapse;width:100%;min-width:980px}} th,td{{padding:8px 10px;border-bottom:1px solid #e7ebed;text-align:right;white-space:nowrap}} th{{position:sticky;top:0;background:#edf1f3;color:#3b474f}} th:nth-child(-n+4),td:nth-child(-n+4){{text-align:left}} .gain{{color:var(--green);font-weight:650}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} ul,ol{{margin:7px 0;padding-left:21px}} li{{margin:6px 0}} .warning{{border-left:4px solid var(--orange)}} .ok{{border-left:4px solid var(--green)}} code{{font-family:ui-monospace,monospace;background:#edf1f3;padding:1px 4px;border-radius:3px}}
@media(max-width:900px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.charts,.split{{grid-template-columns:1fr}}}} @media(max-width:520px){{.metrics{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>VAMP SIMD 球碰撞 5×5 完整任务实验</h1><p>Pure RRT-Connect + MoveIt/FCL 对比 VAMP RRT-Connect + 原生 SIMD 球碰撞 · 共享 IK · Release · 三轮</p></header>
<main>
<div class="metrics">
<div class="metric"><b>{coverage['successful_trials']}/{coverage['total_trials']}</b><span>完整任务成功</span></div>
<div class="metric"><b>{value(metric['aggregate_speedup'],2)}x</b><span>总搜索时间加速</span></div>
<div class="metric"><b>{value(metric['paired_speedup_median'],2)}x</b><span>逐试验加速中位数</span></div>
<div class="metric"><b>{value(metric['ompl_total_ms'],1)} ms</b><span>OMPL/FCL 总搜索</span></div>
<div class="metric"><b>{value(metric['vamp_total_ms'],1)} ms</b><span>VAMP/SIMD 总搜索</span></div>
</div>
<div class="links"><a href="../rerun/sphere-world-pure-rrt-connect-fcl-5x5.rrd">Pure RRT Rerun</a><a href="../rerun/sphere-world-vamp-rrt-connect-simd-5x5.rrd">VAMP SIMD Rerun</a><a href="../per-box.csv">逐箱 CSV</a><a href="../per-stage.csv">逐阶段 CSV</a><a href="../per-round.csv">逐轮 CSV</a><a href="../summary.json">Summary JSON</a><a href="../trials.json">Trial manifest</a></div>

<section><div class="band ok"><h2>实验结论</h2>
<p>25 个箱子、3 轮、75 次完整任务全部成功。每次都从共享 IK 端点重新建立 RRT-Connect 树，依次搜索预接触、接触、携箱抽离、携箱搬运、释放后回位五段；没有历史路径、shortcut、Cartesian 或 fallback。</p>
<p>75 次累计搜索时间从 OMPL/FCL 的 <strong>{value(metric['ompl_total_ms'],3)} ms</strong> 降到 VAMP/SIMD 的 <strong>{value(metric['vamp_total_ms'],3)} ms</strong>，总量加速 <strong>{value(metric['aggregate_speedup'],2)}×</strong>。逐试验加速中位数为 <strong>{value(metric['paired_speedup_median'],2)}×</strong>。两者差异较大，是因为 Box 2、4、21、22、24 等复杂箱在 VAMP 总时间中占比很高，拉低总量加速。</p></div></section>

<section><h2>直观时间对比</h2><div class="charts">
<div class="chart"><h3>逐箱三轮中位搜索时间</h3><p>对数纵轴；每个值是五个 RRT 段之和，IK 不计入。</p><div class="legend"><span><i class="swatch ompl"></i>OMPL/FCL</span><span><i class="swatch vamp"></i>VAMP/SIMD</span></div><svg id="time-chart" role="img" aria-label="逐箱规划时间对比"></svg></div>
<div class="chart"><h3>按箱序累计搜索时间</h3><p>将 25 个箱子的三轮中位数顺序累加。</p><div class="legend"><span><i class="swatch ompl"></i>OMPL/FCL</span><span><i class="swatch vamp"></i>VAMP/SIMD</span></div><svg id="cumulative-chart" role="img" aria-label="累计规划时间对比"></svg></div>
</div></section>

<section><h2>逐阶段汇总</h2><div class="table-wrap"><table><thead><tr><th>阶段</th><th>成功试验</th><th>OMPL/FCL 总计 ms</th><th>VAMP/SIMD 总计 ms</th><th>加速</th></tr></thead><tbody>{stage_table}</tbody></table></div><p>携箱搬运阶段的总量加速最低，说明剩余成本主要来自受载荷和环境共同约束的扩树次数，而不是 SIMD 单次碰撞检查本身。</p></section>

<section><h2>逐箱完整数据</h2><p>75 个任务的五个阶段均在第一次搜索成功，所以“成功单次搜索之和”与“累计到整组成功”相同；逐阶段原值和尝试次数保存在逐阶段 CSV。</p><div class="table-wrap"><table><thead><tr><th>箱号</th><th>机械臂</th><th>吸附</th><th>成功</th><th>RRT/FCL ms</th><th>VAMP/SIMD ms</th><th>加速倍数</th><th>相对 RRT 时间</th><th>RRT P95 ms</th><th>VAMP P95 ms</th></tr></thead><tbody>{''.join(table)}</tbody></table></div></section>

<section><div class="split">
<div class="band"><h2>为什么 VAMP 快</h2><ul>
<li>两边都是在线 RRT-Connect；差异主要在碰撞有效性检查，不是 RRT 与非 RRT。</li>
<li>VAMP 将 15 维 FK、243 个机器人球、环境长方体和 48 个附着箱球编译进 AVX2 内核，批量检查多个状态。</li>
<li>OMPL 每次状态/边检查需要构造 MoveIt RobotState，并走通用 FCL 网格碰撞路径；固定框架和虚调用形成约几十毫秒的成本底线。</li>
<li>普通箱路径接近直接连接时，VAMP 的五段总时间约 0.37–2.4 ms；FCL 仍多为 45–160 ms，因此常见百倍级差异。</li>
</ul><h3>为什么有些箱加速较低</h3><ul>
<li>Box 2、4、21 的走廊命中率低，需要更多树扩展，VAMP 时间升到约 45–58 ms。</li>
<li>Box 22、24 是底层顶吸，附着箱与地面、底盘及倾角约束共同缩窄自由空间，VAMP 仍需约 40–41 ms。</li>
<li>复杂箱占据 VAMP 总时间的大部分，所以总量加速 22.60× 低于逐试验中位数 112.58×。</li>
</ul></div>
<div class="band warning"><h2>严谨边界与潜在问题</h2><ul>
<li>本轮 VAMP 在线搜索和接受完全使用球世界，没有最终 FCL 门禁；因此数值确实反映原生 SIMD 搜索，但安全性只相对于当前球世界成立。</li>
<li>为让紧密堆叠箱可抽离，双方共同采用 30 mm 的规划碰撞内缩。它不是原尺寸实体的保守外包络。</li>
<li>底盘目前只用一个 0–0.30 m 的下部盒近似；上部复杂凹空尚未做具备 ACM 语义的球/胶囊模型。</li>
<li>为消除一个经共享 FCL 端点确认的假阳性，左 link7 的单球半径从 81.789 mm 调整到 67 mm。仍需重新做全状态与连续边漏检审计。</li>
<li>相同 seed 与采样分布已统一，但 OMPL 与 VAMP 内部批处理和树扩展顺序不同，不能声称逐次消耗完全相同的样本索引。</li>
</ul></div></div></section>

<section><div class="band"><h2>未来解决方式</h2><ol>
<li><strong>高分辨率附着箱球化：</strong>先以更密小球或球+胶囊覆盖箱体表面，降低当前 3×4×4 外包球的角部外凸；改变附件几何精度，使所需内缩从 30 mm 降到工业许可间隙。可行性高，但球数增加会降低 VAMP 倍速，需要测量 384、800、3072 球的拐点。</li>
<li><strong>底盘分区建模：</strong>将下部实体、侧梁和立柱分别球化，并把固定连接/允许接触关系编入生成内核；改变单一实心包围盒填满凹空的问题。Cricket 已支持生成自碰撞对，工程可行，但需要新的 URDF/SRDF 生成流水线。</li>
<li><strong>反例驱动细化：</strong>对 FCL 碰撞但球模型放行的状态增加局部球或 margin，对 FCL 自由但球模型阻塞的区域缩分具体球；改变全局统一 margin 带来的漏检和假阳性冲突。现有端点与逐构件诊断已经能自动定位反例。</li>
<li><strong>边级安全审计：</strong>不仅随机采样状态，还审计本次 125 段路径的所有离散点和相邻边；改变“端点有效即认为整条边安全”的不足。达到原尺寸几何下状态和边假阴性均为 0 后，才可评估删除生产 FCL。</li>
<li><strong>性能复验：</strong>固定 Release、CPU 亲和性和 5–10 轮重复，报告中位数、P95、成功率与迭代数；改变当前三轮对长尾估计有限的问题，使结果可与论文指标稳定对照。</li>
</ol></div></section>
</main>
<script>
const rows={data};
const C={{ompl:'#2475b9',vamp:'#e47c25',grid:'#d6dde1',ink:'#182127',muted:'#5d6971'}};
function el(n,a={{}}){{const e=document.createElementNS('http://www.w3.org/2000/svg',n);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));return e}}
function text(svg,x,y,s,anchor='middle'){{const t=el('text',{{x,y,'text-anchor':anchor,fill:C.muted,'font-size':11}});t.textContent=s;svg.appendChild(t)}}
function drawTimes(){{const svg=document.getElementById('time-chart'),W=Math.max(620,svg.clientWidth),H=360,m={{l:60,r:12,t:14,b:42}},iw=W-m.l-m.r,ih=H-m.t-m.b;svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);svg.replaceChildren();const lo=.1,hi=2000,y=v=>m.t+ih*(Math.log10(hi)-Math.log10(Math.max(lo,v)))/(Math.log10(hi)-Math.log10(lo)),x=i=>m.l+(i+.5)*iw/25;[.1,1,10,100,1000].forEach(v=>{{const yy=y(v);svg.appendChild(el('line',{{x1:m.l,x2:m.l+iw,y1:yy,y2:yy,stroke:C.grid}}));text(svg,m.l-8,yy+4,v+' ms','end')}});rows.forEach((r,i)=>{{const xx=x(i),bw=Math.max(3,iw/25*.28);[['ompl_median_ms',C.ompl,-bw],['vamp_median_ms',C.vamp,1]].forEach(([k,c,dx])=>{{const yy=y(r[k]);svg.appendChild(el('rect',{{x:xx+dx,y:yy,width:bw,height:m.t+ih-yy,fill:c}}))}});if(i%2===0)text(svg,xx,H-17,String(r.box_id))}});text(svg,m.l+iw/2,H-2,'Box');}}
function drawCumulative(){{const svg=document.getElementById('cumulative-chart'),W=Math.max(620,svg.clientWidth),H=360,m={{l:64,r:16,t:14,b:42}},iw=W-m.l-m.r,ih=H-m.t-m.b;svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);svg.replaceChildren();let ao=0,av=0;const pts=rows.map((r,i)=>({{x:i+1,o:ao+=r.ompl_median_ms,v:av+=r.vamp_median_ms}})),hi=Math.max(...pts.map(p=>p.o)),x=n=>m.l+(n-1)*iw/24,y=v=>m.t+ih-v/hi*ih;for(let i=0;i<=4;i++){{const val=hi*i/4,yy=y(val);svg.appendChild(el('line',{{x1:m.l,x2:m.l+iw,y1:yy,y2:yy,stroke:C.grid}}));text(svg,m.l-8,yy+4,val.toFixed(0)+' ms','end')}};[['o',C.ompl],['v',C.vamp]].forEach(([k,c])=>{{const d=pts.map((p,i)=>(i?'L':'M')+x(p.x)+' '+y(p[k])).join(' ');svg.appendChild(el('path',{{d,fill:'none',stroke:c,'stroke-width':2.5}}));pts.forEach(p=>svg.appendChild(el('circle',{{cx:x(p.x),cy:y(p[k]),r:2.4,fill:c}})))}});[1,5,10,15,20,25].forEach(n=>text(svg,x(n),H-17,String(n)));text(svg,m.l+iw/2,H-2,'Completed boxes');}}
function redraw(){{drawTimes();drawCumulative()}} redraw();new ResizeObserver(redraw).observe(document.querySelector('.charts'));
</script></body></html>"""
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
