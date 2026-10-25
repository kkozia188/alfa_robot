#!/usr/bin/env python3
"""Build a self-contained browser report for the paired paper-metric experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("rerun_dir", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def optional_float(value: str) -> float | None:
    return float(value) if value else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def build_rows(trials_path: Path) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    with trials_path.open() as source:
        for source_row in csv.DictReader(source):
            row = {
                **source_row,
                "endpoint_ready": source_row["endpoint_ready"] == "True",
                "ompl_exact": source_row["ompl_exact"] == "True",
                "vamp_native_solved": source_row["vamp_native_solved"] == "True",
                "vamp_fcl_valid": source_row["vamp_fcl_valid"] == "True",
                "paired_certified": source_row["paired_certified"] == "True",
                "ompl_ms": optional_float(source_row["ompl_ms"]),
                "vamp_ms": optional_float(source_row["vamp_ms"]),
                "paper_speedup": optional_float(source_row["paper_speedup"]),
                "certified_speedup": optional_float(source_row["certified_speedup"]),
            }
            grouped[int(source_row["box_id"])].append(row)
    rows = []
    for box_id in range(1, 26):
        group = grouped[box_id]
        ompl_values = [row["ompl_ms"] for row in group if row["ompl_ms"] is not None]
        vamp_values = [row["vamp_ms"] for row in group if row["vamp_ms"] is not None]
        speedups = [
            row["paper_speedup"] for row in group if row["paper_speedup"] is not None
        ]
        certified = [
            row["certified_speedup"]
            for row in group if row["certified_speedup"] is not None
        ]
        speedup = median(speedups)
        endpoint_trials = sum(row["endpoint_ready"] for row in group)
        vamp_native = sum(row["vamp_native_solved"] for row in group)
        vamp_fcl = sum(row["vamp_fcl_valid"] for row in group)
        if endpoint_trials == 0:
            status = "无共同IK端点"
            status_class = "neutral"
        elif vamp_fcl > 0:
            status = "VAMP通过最终FCL"
            status_class = "certified"
        elif vamp_native > 0:
            status = "VAMP球路径未通过FCL"
            status_class = "warning"
        else:
            status = "VAMP原生无解"
            status_class = "failure"
        rows.append(
            {
                "box": box_id,
                "endpoint": endpoint_trials,
                "omplExact": sum(row["ompl_exact"] for row in group),
                "vampNative": vamp_native,
                "vampFcl": vamp_fcl,
                "paired": sum(row["paired_certified"] for row in group),
                "omplMs": median(ompl_values),
                "vampMs": median(vamp_values),
                "speedup": speedup,
                "certifiedSpeedup": median(certified),
                "reduction": 100.0 * (1.0 - 1.0 / speedup) if speedup else None,
                "status": status,
                "statusClass": status_class,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    experiment = args.experiment_dir.resolve()
    rerun_dir = args.rerun_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads((experiment / "summary.json").read_text())
    reruns = json.loads((rerun_dir / "rerun-manifest.json").read_text())
    rows = build_rows(experiment / "trials.csv")
    data_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    summary_json = json.dumps(
        {
            "scheduled": summary["scheduled_trials"],
            "endpointReady": summary["endpoint_ready_trials"],
            "omplExact": summary["ompl_exact_trials"],
            "vampNative": summary["vamp_native_solved_trials"],
            "vampFcl": summary["vamp_fcl_valid_trials"],
            "paperComparable": summary["paper_comparable_trials"],
            "paperSpeedup": summary["paper_speedup_median"],
            "paperRatioOfMedians": summary["paper_ratio_of_medians"],
            "paperOmplMs": summary["paper_ompl_median_ms"],
            "paperVampMs": summary["paper_vamp_median_ms"],
            "certifiedSpeedup": summary["certified_speedup_median"],
            "certifiedRatioOfMedians": summary["certified_ratio_of_medians"],
            "certifiedOmplMs": summary["certified_ompl_median_ms"],
            "certifiedVampMs": summary["certified_vamp_median_ms"],
            "fclPassRate": summary["vamp_native_fcl_pass_rate"],
            "falsePositive": summary["sphere_model_audit"]["rates"]["sphere_false_positive"],
            "falseNegative": summary["sphere_model_audit"]["rates"]["sphere_false_negative"],
        },
        separators=(",", ":"),
    )
    ompl_rrd = Path(reruns["recordings"]["ompl"]["path"]).name
    vamp_rrd = Path(reruns["recordings"]["vamp"]["path"]).name
    ompl_sha = reruns["recordings"]["ompl"]["sha256"]
    vamp_sha = reruns["recordings"]["vamp"]["sha256"]
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V3.2.2 纯RRT 与 VAMP 避障搜索对比</title>
<style>
:root{color-scheme:light dark;--bg:light-dark(#f7f9fb,#111519);--surface:light-dark(#fff,#171c21);--text:light-dark(#17202a,#eef3f7);--muted:light-dark(#5f6d7a,#a9b4bd);--line:light-dark(#d7dee5,#35404a);--blue:#2878c8;--orange:#e58a25;--green:#18845c;--red:#c74848;--amber:#c58a25;--gray:#7c8792}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;letter-spacing:0}main{max-width:1440px;margin:0 auto;padding:24px 28px 60px}h1{font-size:26px;margin:0 0 6px;font-weight:650}h2{font-size:19px;margin:32px 0 14px}h3{font-size:15px;margin:0 0 8px}.lede{margin:0;color:var(--muted);font-size:14px}.meta{display:flex;flex-wrap:wrap;gap:8px 20px;margin:18px 0 22px;padding:12px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);font-size:13px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.metric{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px}.metric .name{font-size:12px;color:var(--muted)}.metric .value{font-size:24px;font-weight:650;margin-top:5px;font-variant-numeric:tabular-nums}.metric .note{font-size:12px;color:var(--muted);margin-top:4px}.downloads{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:18px}.download{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--blue);border-radius:6px;padding:14px}.download.vamp{border-left-color:var(--orange)}.download a{color:var(--text);font-weight:650;text-decoration:none}.download a:hover{text-decoration:underline}.hash{margin-top:8px;color:var(--muted);font:11px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}.chart-tools{display:flex;flex-wrap:wrap;gap:14px;margin:4px 0 10px;font-size:13px}.chart-tools label{display:flex;align-items:center;gap:6px}.chart{position:relative;background:var(--surface);border:1px solid var(--line);border-radius:6px;min-height:440px}.chart svg{display:block;width:100%;height:auto}.tooltip{position:absolute;display:none;pointer-events:none;z-index:2;background:var(--surface);border:1px solid var(--line);padding:8px 10px;font-size:12px;line-height:1.5;max-width:260px}.detail{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:center;margin:12px 0 0}.detail select{width:100%;padding:8px;border:1px solid var(--line);border-radius:4px;background:var(--surface);color:var(--text)}.detail-output{font-size:13px;color:var(--muted)}.detail-output strong{color:var(--text)}.table-wrap{overflow-x:auto;border-top:1px solid var(--line)}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px 8px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}th{position:sticky;top:0;background:var(--bg);color:var(--muted);font-weight:600}th:first-child,td:first-child,th:last-child,td:last-child{text-align:left}.status{font-weight:600}.status.certified{color:var(--green)}.status.warning{color:var(--amber)}.status.failure{color:var(--red)}.status.neutral{color:var(--gray)}.analysis-grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.analysis-grid section{border-top:2px solid var(--line);padding-top:12px}.analysis-grid ul,.analysis-grid ol{margin:8px 0 0;padding-left:21px}.analysis-grid li{margin:7px 0;line-height:1.55;font-size:14px}.callout{border-left:4px solid var(--amber);padding:10px 14px;background:var(--surface);font-size:14px;line-height:1.55}.foot{margin-top:30px;padding-top:14px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
@media(max-width:900px){main{padding:18px 14px 42px}.metrics{grid-template-columns:1fr 1fr}.downloads,.analysis-grid{grid-template-columns:1fr}.detail{grid-template-columns:1fr}.chart{min-height:340px}}
@media(max-width:480px){.metrics{grid-template-columns:1fr}h1{font-size:22px}.metric .value{font-size:22px}}
</style>
</head>
<body><main>
<h1>纯 RRT-Connect（FCL） vs VAMP RRT-Connect（SIMD 球碰撞）</h1>
<p class="lede">在 ALFA V3.2.2 与同一 5×5 仓内箱墙上复现论文核心搜索指标：进入 RRT-Connect 到首次后端碰撞模型有效路径。IK、启动、路径简化与最终 FCL 复核不计入搜索时间。</p>
<div class="meta"><span>Release 二进制</span><span>15维：updown + 双臂14轴</span><span>相同 start/goal</span><span>相同 Halton 段</span><span>无历史路径 / shortcut / Cartesian / fallback</span></div>
<div class="metrics">
  <div class="metric"><div class="name">论文口径配对中位加速</div><div class="value" id="paper-speedup"></div><div class="note" id="paper-count"></div></div>
  <div class="metric"><div class="name">论文口径搜索中位时间</div><div class="value" id="paper-time"></div><div class="note" id="paper-time-note"></div></div>
  <div class="metric"><div class="name">最终FCL认证子集加速</div><div class="value" id="cert-speedup"></div><div class="note" id="cert-time"></div></div>
  <div class="metric"><div class="name">VAMP原生路径FCL通过率</div><div class="value" id="fcl-rate"></div><div class="note" id="fcl-count"></div></div>
</div>

<div class="downloads">
 <div class="download"><h3>纯 RRT-Connect / MoveIt FCL</h3><a href="rerun/pure-rrt-connect-fcl-5x5.rrd">下载或打开真实路径 Rerun</a><div class="hash">SHA-256: __OMPL_SHA__</div></div>
 <div class="download vamp"><h3>VAMP RRT-Connect / 原生 SIMD 球碰撞</h3><a href="rerun/vamp-rrt-connect-simd-5x5.rrd">下载或打开真实路径 Rerun</a><div class="hash">SHA-256: __VAMP_SHA__</div></div>
</div>

<h2>逐箱避障搜索时间</h2>
<div class="chart-tools"><label><input id="show-ompl" type="checkbox" checked>纯RRT/FCL</label><label><input id="show-vamp" type="checkbox" checked>VAMP/SIMD</label><span>Y轴为对数刻度，单位 ms；中位时间包含超时与最终FCL拒绝的试验。</span></div>
<div id="time-chart" class="chart" aria-label="25个箱子的纯RRT和VAMP搜索时间对比"><div class="tooltip"></div></div>
<div class="detail"><label for="box-select">查看箱号</label><select id="box-select"></select><div id="box-detail" class="detail-output" aria-live="polite"></div></div>

<h2>逐箱加速倍数</h2>
<p class="lede">主指标为同一箱、同一 Halton 段内的 <code>纯RRT搜索时间 ÷ VAMP搜索时间</code>，再对可比试验取中位数。绿色表示至少一条 VAMP 路径通过最终 FCL；橙色仅代表论文球模型搜索速度。</p>
<div id="speedup-chart" class="chart" aria-label="逐箱纯RRT相对VAMP的配对中位加速倍数"><div class="tooltip"></div></div>

<h2>逐箱数据表</h2>
<div class="table-wrap"><table><thead><tr><th>箱号</th><th>纯RRT中位 ms</th><th>RRT精确解</th><th>VAMP中位 ms</th><th>VAMP原生解</th><th>最终FCL通过</th><th>配对中位加速</th><th>时间减少</th><th>结论</th></tr></thead><tbody id="rows"></tbody></table></div>

<h2>数据分析</h2>
<div class="analysis-grid">
 <section><h3>为什么旧纯RRT能完成25箱</h3><ul><li>旧实验是完整分段抓取流程：每箱在线选择抓取侧、正吸/顶吸、阶段IK和多个RRT段；当前论文口径是统一home到单一预接触goal。</li><li>旧纯RRT的任务规划中位每箱231.304秒、最大773.165秒，25箱没有一个低于5秒；25/25表示长预算与重试后可完成，不表示2秒内都能完成。</li><li>旧结果还使用两个底盘站位并组合多轮/重试产物；当前实验固定同一场景、每箱5个Halton段并保留全部失败。</li></ul></section>
 <section><h3>为什么 VAMP 快</h3><ul><li>Cricket 把 FK 与球体碰撞对编译成专用代码，并用 SIMD 同时验证多状态。</li><li>MoveIt/FCL 逐状态遍历三角网格层次；在直连与宽可行域任务上，碰撞检查成本成为主要差异。</li><li>论文口径 55 个同试验可比样本的配对中位加速为 189.78×。</li></ul></section>
 <section><h3>为什么有时 VAMP 慢或无解</h3><ul><li>球体包络假阳性会缩小可行域：箱3、8中纯RRT成功而VAMP原生无解。</li><li>243球提高精度，也提高每次 SIMD 验证工作量；困难场景可能耗尽100万次迭代。</li><li>箱2、5、7等虽然VAMP得到球模型路径，但未通过最终网格FCL。</li></ul></section>
 <section><h3>为什么没有复现论文最高倍数</h3><ul><li>论文使用成熟机器人专用球模型与MotionBenchMaker任务；我们的15维双臂+升降、243球和29个环境长方体更重。</li><li>我们的论文口径189.78×其实与论文Fetch/Baxter量级接近；显著较低的是最终FCL认证口径36.92×。</li><li>完整抓取流程中在线IK可占约91%，VAMP不加速IK，因此搜索内核的大倍数不会等比例转化成端到端收益。</li></ul></section>
 <section><h3>结论</h3><ul><li>在论文定义的球模型搜索指标上，VAMP明显快于纯RRT/FCL。</li><li>当前球模型不能替代原始网格安全模型：70条VAMP原生路径只有22条通过最终FCL。</li><li>189.78×是原生球碰撞搜索加速，不是当前机器人安全可执行规划的整体加速。</li><li>最终FCL认证子集的配对中位加速为36.92×，但该子集主要由直连任务组成。</li></ul></section>
 <section><h3>潜在问题</h3><ul><li>10,000状态审计：球模型假阴性12.61%，假阳性19.18%。</li><li>箱16、18、20–25没有共同在线IK端点，未进入RRT，不能计作规划器结果。</li><li>本实验复现的是论文指标，不是论文原机器人与原场景；不能直接与论文1000×数字等同。</li><li>时间统计来自最终125组；动作回放是在相同合同下按固定规则重新捕获的每箱代表试验。Rerun只播放实际搜索器返回路径，失败任务保持start并显示状态。</li></ul></section>
</div>

<h2>未来展望</h2>
<div class="callout"><ol><li>以原始FCL网格为真值，对产生假阴性/假阳性的具体连杆做自适应保守球化，目标是任务分布和全域审计均零假阴性。</li><li>把FCL反例回填到球化与搜索流程，形成“VAMP候选 → FCL拒绝 → 局部模型加密”的闭环。</li><li>分别报告直连检查和真实扩树任务；避免大量直连样本掩盖复杂搜索性能。</li><li>独立扩展在线IK端点覆盖，但始终冻结同一start/goal再比较两种RRT后端。</li><li>继续保留最终1°/1cm MoveIt/FCL门禁，直到球模型得到零假阴性证书。</li></ol></div>
<div class="foot">两份真实路径 Rerun 均已通过 <code>rerun rrd verify</code>。时间柱使用最终125组统计，动作使用相同实验合同下的固定代表试验。</div>
</main>
<script>
const rows=__ROWS__;const summary=__SUMMARY__;
const fmt=v=>v==null?'—':v.toLocaleString('zh-CN',{minimumFractionDigits:3,maximumFractionDigits:3});
const pct=v=>v==null?'—':v.toFixed(2)+'%';const ratio=v=>v==null?'—':v.toFixed(2)+'×';
document.getElementById('paper-speedup').textContent=ratio(summary.paperSpeedup);
document.getElementById('paper-count').textContent='时间减少 '+(100*(1-1/summary.paperSpeedup)).toFixed(2)+'% · 可比 '+summary.paperComparable+' 组';
document.getElementById('paper-time').textContent=fmt(summary.paperOmplMs)+' / '+fmt(summary.paperVampMs)+' ms';
document.getElementById('paper-time-note').textContent='纯RRT / VAMP · 中位数之比 '+ratio(summary.paperRatioOfMedians)+'（非主指标）';
document.getElementById('cert-speedup').textContent=ratio(summary.certifiedSpeedup);
document.getElementById('cert-time').textContent='时间减少 '+(100*(1-1/summary.certifiedSpeedup)).toFixed(2)+'% · '+fmt(summary.certifiedOmplMs)+' / '+fmt(summary.certifiedVampMs)+' ms';
document.getElementById('fcl-rate').textContent=(100*summary.fclPassRate).toFixed(2)+'%';
document.getElementById('fcl-count').textContent=summary.vampFcl+' / '+summary.vampNative+' 条原生路径';
const tbody=document.getElementById('rows');
rows.forEach(r=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${r.box}</td><td>${fmt(r.omplMs)}</td><td>${r.omplExact}/5</td><td>${fmt(r.vampMs)}</td><td>${r.vampNative}/5</td><td>${r.vampFcl}/5</td><td>${ratio(r.speedup)}</td><td>${pct(r.reduction)}</td><td><span class="status ${r.statusClass}">${r.status}</span></td>`;tbody.appendChild(tr)});
const select=document.getElementById('box-select');rows.forEach(r=>{const o=document.createElement('option');o.value=r.box;o.textContent='Box '+r.box;select.appendChild(o)});select.value='1';
function detail(){const r=rows[Number(select.value)-1];document.getElementById('box-detail').innerHTML=`<strong>纯RRT ${fmt(r.omplMs)} ms (${r.omplExact}/5)</strong>　vs　<strong>VAMP ${fmt(r.vampMs)} ms (${r.vampNative}/5)</strong>　配对加速 ${ratio(r.speedup)}　时间减少 ${pct(r.reduction)}　最终FCL ${r.vampFcl}/5`}
select.addEventListener('change',detail);detail();
const chart=document.getElementById('time-chart');const tip=chart.querySelector('.tooltip');
function draw(){chart.querySelectorAll('svg').forEach(e=>e.remove());const W=Math.max(680,chart.clientWidth),H=440,m={l:72,r:22,t:24,b:48},iw=W-m.l-m.r,ih=H-m.t-m.b;const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);chart.appendChild(svg);const mk=(n,a={})=>{const e=document.createElementNS(ns,n);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));return e};const max=5000,min=.1,y=v=>m.t+ih*(Math.log10(max)-Math.log10(Math.max(min,v)))/(Math.log10(max)-Math.log10(min)),x=b=>m.l+(b-.5)*iw/25;
[.1,1,10,100,1000].forEach(v=>{const yy=y(v),line=mk('line',{x1:m.l,x2:m.l+iw,y1:yy,y2:yy,stroke:'var(--line)','stroke-width':1});svg.appendChild(line);const t=mk('text',{x:m.l-12,y:yy+5,'text-anchor':'end',fill:'var(--muted)','font-size':12});t.textContent=v;svg.appendChild(t)});const frame=mk('rect',{x:m.l,y:m.t,width:iw,height:ih,fill:'none',stroke:'var(--line)'});svg.appendChild(frame);rows.forEach(r=>{const xx=x(r.box);const tick=mk('text',{x:xx,y:m.t+ih+26,'text-anchor':'middle',fill:'var(--muted)','font-size':11});tick.textContent=r.box;svg.appendChild(tick);[['omplMs','show-ompl','var(--blue)',-5,'circle'],['vampMs','show-vamp','var(--orange)',5,'rect']].forEach(([key,id,color,dx,shape])=>{if(!document.getElementById(id).checked||r[key]==null)return;let mark;if(shape==='circle'){mark=mk('circle',{cx:xx+dx,cy:y(r[key]),r:5,fill:color})}else{mark=mk('rect',{x:xx+dx-5,y:y(r[key])-5,width:10,height:10,fill:color,transform:`rotate(45 ${xx+dx} ${y(r[key])})`})}mark.style.cursor='pointer';mark.addEventListener('pointermove',ev=>{tip.style.display='block';tip.innerHTML=`<strong>Box ${r.box}</strong><br>纯RRT ${fmt(r.omplMs)} ms · ${r.omplExact}/5<br>VAMP ${fmt(r.vampMs)} ms · ${r.vampNative}/5<br>加速 ${ratio(r.speedup)} · 最终FCL ${r.vampFcl}/5`;const b=chart.getBoundingClientRect();tip.style.left=Math.min(b.width-270,Math.max(6,ev.clientX-b.left+10))+'px';tip.style.top=Math.max(6,ev.clientY-b.top-75)+'px'});mark.addEventListener('pointerleave',()=>tip.style.display='none');svg.appendChild(mark)})});const xt=mk('text',{x:m.l+iw/2,y:H-9,'text-anchor':'middle',fill:'var(--text)','font-size':13});xt.textContent='箱号';svg.appendChild(xt);const yt=mk('text',{x:16,y:m.t+ih/2,'text-anchor':'middle',fill:'var(--text)','font-size':13,transform:`rotate(-90 16 ${m.t+ih/2})`});yt.textContent='避障搜索时间 (ms, 对数刻度)';svg.appendChild(yt)}
document.getElementById('show-ompl').addEventListener('change',draw);document.getElementById('show-vamp').addEventListener('change',draw);new ResizeObserver(draw).observe(chart);draw();
const speedChart=document.getElementById('speedup-chart'),speedTip=speedChart.querySelector('.tooltip');
function drawSpeedup(){speedChart.querySelectorAll('svg').forEach(e=>e.remove());const W=Math.max(680,speedChart.clientWidth),H=440,m={l:72,r:22,t:24,b:48},iw=W-m.l-m.r,ih=H-m.t-m.b,ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);speedChart.appendChild(svg);const mk=(n,a={})=>{const e=document.createElementNS(ns,n);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));return e};const max=1000,min=1,y=v=>m.t+ih*(Math.log10(max)-Math.log10(Math.max(min,v)))/(Math.log10(max)-Math.log10(min)),x=b=>m.l+(b-.5)*iw/25,bw=Math.max(9,iw/25*.58);[1,10,100,1000].forEach(v=>{const yy=y(v);svg.appendChild(mk('line',{x1:m.l,x2:m.l+iw,y1:yy,y2:yy,stroke:'var(--line)','stroke-width':v===1?2:1}));const t=mk('text',{x:m.l-12,y:yy+5,'text-anchor':'end',fill:'var(--muted)','font-size':12});t.textContent=v+'×';svg.appendChild(t)});svg.appendChild(mk('rect',{x:m.l,y:m.t,width:iw,height:ih,fill:'none',stroke:'var(--line)'}));rows.forEach(r=>{const xx=x(r.box),tick=mk('text',{x:xx,y:m.t+ih+26,'text-anchor':'middle',fill:'var(--muted)','font-size':11});tick.textContent=r.box;svg.appendChild(tick);if(r.speedup==null)return;const yy=y(r.speedup),base=y(1),bar=mk('rect',{x:xx-bw/2,y:yy,width:bw,height:Math.max(1,base-yy),fill:r.vampFcl>0?'var(--green)':'var(--orange)'});bar.style.cursor='pointer';bar.addEventListener('pointermove',ev=>{speedTip.style.display='block';speedTip.innerHTML=`<strong>Box ${r.box}: ${ratio(r.speedup)}</strong><br>时间减少 ${pct(r.reduction)}<br>纯RRT ${fmt(r.omplMs)} ms<br>VAMP ${fmt(r.vampMs)} ms<br>最终FCL ${r.vampFcl}/5`;const b=speedChart.getBoundingClientRect();speedTip.style.left=Math.min(b.width-270,Math.max(6,ev.clientX-b.left+10))+'px';speedTip.style.top=Math.max(6,ev.clientY-b.top-75)+'px'});bar.addEventListener('pointerleave',()=>speedTip.style.display='none');svg.appendChild(bar);const label=mk('text',{x:xx,y:Math.max(14,yy-5),'text-anchor':'middle',fill:'var(--text)','font-size':10});label.textContent=r.speedup>=100?r.speedup.toFixed(0)+'×':r.speedup.toFixed(1)+'×';svg.appendChild(label)});const xt=mk('text',{x:m.l+iw/2,y:H-9,'text-anchor':'middle',fill:'var(--text)','font-size':13});xt.textContent='箱号';svg.appendChild(xt);const yt=mk('text',{x:16,y:m.t+ih/2,'text-anchor':'middle',fill:'var(--text)','font-size':13,transform:`rotate(-90 16 ${m.t+ih/2})`});yt.textContent='配对中位加速倍数 (对数刻度)';svg.appendChild(yt)}
new ResizeObserver(drawSpeedup).observe(speedChart);drawSpeedup();
</script></body></html>'''
    html = (
        template.replace("__ROWS__", data_json)
        .replace("__SUMMARY__", summary_json)
        .replace("__OMPL_SHA__", ompl_sha)
        .replace("__VAMP_SHA__", vamp_sha)
    )
    output.write_text(html)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
