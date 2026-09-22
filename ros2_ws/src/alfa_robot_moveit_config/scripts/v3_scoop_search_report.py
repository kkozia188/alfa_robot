#!/usr/bin/env python3
"""Build an offline search and TCP-path report from a scoop sequence summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def report_data(summary: dict) -> dict:
    rows = []
    for box in summary["boxes"]:
        search = box.get("planning_search", {})
        task = search.get("task", {})
        transition = search.get("transition", {})
        selected = search.get("selected_task", {})
        analysis = box.get("trajectory_analysis", {})
        transition_analysis = box.get("transition_trajectory_analysis", {})
        segments = box.get("segment_search", [])
        transition_segments = [
            segment
            for leg in box.get("transition_motion", {}).get("search_legs", [])
            for segment in leg.get("segments", [])
        ]
        rows.append({
            "id": box["box_id"], "side": box["side"], "mode": box["mode"],
            "selected_s": round(float(selected.get("process_wall_ms", 0.0)) / 1000.0, 3),
            "planner_s": round(float(selected.get("planner_wall_ms", 0.0)) / 1000.0, 3),
            "total_s": round((float(task.get("process_wall_ms", 0.0)) +
                float(transition.get("process_wall_ms", 0.0))) / 1000.0, 3),
            "task_attempts": task.get("attempts", 0),
            "transition_attempts": transition.get("attempts", 0),
            "repair_attempts": task.get("shortcut_repair_rrt_attempts", 0) +
                transition.get("shortcut_repair_rrt_attempts", 0),
            "repair_successes": task.get("shortcut_repair_rrt_successes", 0) +
                transition.get("shortcut_repair_rrt_successes", 0),
            "joint_fallbacks": task.get("joint_rrt_fallbacks", 0) +
                transition.get("joint_rrt_fallbacks", 0),
            "tcp_m": round(float(analysis.get("tcp_path_length_m", 0.0)), 3),
            "tcp_orientation_deg": round(float(analysis.get("tcp_orientation_travel_deg", 0.0)), 1),
            "shortcut_position_deviation_m": round(float(analysis.get(
                "max_shortcut_tcp_position_deviation_m", 0.0)), 4),
            "shortcut_orientation_deviation_deg": round(float(analysis.get(
                "max_shortcut_tcp_orientation_deviation_deg", 0.0)), 2),
            "travel_deg": round(float(box.get("motion_selection", {}).get(
                "total_joint_travel_deg", 0.0)), 1),
            "reversals": analysis.get("total_joint_reversals", 0),
            "joint_flip_events": analysis.get("joint_flip_events", 0),
            "axis_reversals": analysis.get("joint_reversals_by_axis", {}),
            "axis_ranges": analysis.get("joint_range_deg_by_axis", {}),
            "tilt_deg": round(float(box.get("carried_box_orientation", {}).get(
                "max_tilt_deg", 0.0)), 2),
            "stage_paths": {
                **{
                    f"transition/{stage}": points
                    for stage, points in transition_analysis.get(
                        "tcp_stage_positions", {}).items()
                },
                **{
                    f"task/{stage}": points
                    for stage, points in analysis.get("tcp_stage_positions", {}).items()
                },
            },
            "segments": [*transition_segments, *segments],
            "transition_strategies": box.get("transition_motion", {}).get(
                "planning_strategies", []),
        })
    return {
        "boxes": rows,
        "completed_boxes": summary["completed_boxes"],
        "validated_transitions": summary["validated_transitions"],
        "playback_s": round(float(summary.get("playback_timing", {}).get(
            "recording_duration_s", 0.0)), 2),
        "total_search_s": round(sum(row["total_s"] for row in rows), 2),
        "max_tilt_deg": max((row["tilt_deg"] for row in rows), default=0.0),
        "total_joint_travel_deg": round(float(summary.get(
            "total_sequence_joint_travel_deg", 0.0)), 1),
    }


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>铲式抓箱 · 搜索与轨迹</title>
<style>
:root{color-scheme:light;--ink:#16232a;--muted:#627078;--line:#dbe3e4;--paper:#f7f9f8;--green:#087970;--orange:#c6692d;--red:#b4433c}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;letter-spacing:0}
header{border-bottom:1px solid var(--line);padding:18px max(22px,calc((100% - 1440px)/2)) 16px;background:var(--paper)}
h1{font-size:21px;line-height:1.3;margin:0 0 14px;font-weight:650;letter-spacing:0}
.totals{display:flex;flex-wrap:wrap;gap:10px 24px;align-items:baseline}.totals span{white-space:nowrap;color:var(--muted)}.totals strong{font-size:18px;color:var(--ink);font-variant-numeric:tabular-nums}
main{max-width:1440px;margin:auto;padding:20px 22px 32px;display:grid;grid-template-columns:minmax(420px,1.05fr) minmax(360px,.95fr);gap:26px}
h2{font-size:15px;margin:0 0 10px;letter-spacing:0}.caption{color:var(--muted);font-size:12px;margin:0 0 9px}
.table-wrap{overflow:auto;border-top:1px solid var(--line)}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;white-space:nowrap}th,td{text-align:right;padding:8px 7px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{position:sticky;top:0;background:#fff;color:var(--muted);font-weight:550;font-size:12px}tbody tr{cursor:pointer}tbody tr:hover,tbody tr:focus{background:#eff5f3;outline:none}tbody tr[aria-selected=true]{background:#e1f1ed;box-shadow:inset 3px 0 var(--green)}
.bar-cell{width:116px}.track{height:7px;background:#e6ebea;width:106px;position:relative}.fill{height:100%;background:var(--green)}.fill.slow{background:var(--orange)}.target{position:absolute;left:var(--target);top:-2px;height:11px;border-left:1px dashed #8a9695}
.status{font-weight:650;color:var(--green)}.status.slow{color:var(--orange)}
.detail{border-top:1px solid var(--line);padding-top:14px;min-width:0}.detail-head{display:flex;justify-content:space-between;gap:12px;align-items:baseline}.detail h2{font-size:17px}.detail small{color:var(--muted)}
.metric-line{display:flex;gap:10px 20px;flex-wrap:wrap;margin:4px 0 16px;font-variant-numeric:tabular-nums}.metric-line span{color:var(--muted)}.metric-line strong{color:var(--ink)}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:12px}.plot{min-width:0}.plot canvas{width:100%;height:238px;display:block;border:1px solid var(--line);background:#fafcfc}.plot figcaption{color:var(--muted);font-size:12px;padding-top:4px}figure{margin:0}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin:12px 0 15px;font-size:12px;color:var(--muted)}.legend i{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:5px}
.stages{border-top:1px solid var(--line);padding-top:13px}.stages table td,.stages table th{padding:6px 7px}.stages table td:nth-child(2),.stages table th:nth-child(2){text-align:left}.axes{color:var(--muted);font-size:12px;margin-top:13px;line-height:1.7}
@media(max-width:960px){main{display:block}.detail{margin-top:28px}.table-wrap{max-height:420px}}@media(max-width:520px){main{padding:15px 12px}.plots{grid-template-columns:1fr}.plot canvas{height:220px}header{padding:15px 12px}.table-wrap{margin-right:-12px}.bar-cell{display:none}}
</style>
</head>
<body>
<header><h1>铲式抓箱 · 搜索与轨迹</h1><div class="totals" id="totals"></div></header>
<main><section><h2>逐箱搜索</h2><p class="caption">核心规划目标 2–5s；进程耗时另含约0.9s启动开销。总搜索包含候选试算、失败与箱间过渡。竖线标记5s。</p><div class="table-wrap"><table><thead><tr><th>箱</th><th>臂</th><th>核心 / s</th><th class="bar-cell"></th><th>进程 / s</th><th>总搜索 / s</th><th>尝试</th><th>修补</th><th>换向</th><th>倾角</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section class="detail"><div class="detail-head"><h2 id="selected-title"></h2><small id="selected-method"></small></div><div class="metric-line" id="selected-metrics"></div><div class="plots"><figure class="plot"><canvas id="xy" aria-label="TCP 的 XY 路径"></canvas><figcaption>TCP 俯视 · X / Y (m)</figcaption></figure><figure class="plot"><canvas id="xz" aria-label="TCP 的 XZ 路径"></canvas><figcaption>TCP 侧视 · X / Z (m)</figcaption></figure></div><div class="legend" id="legend"></div><div class="stages"><h2>分段搜索</h2><table><thead><tr><th>阶段</th><th>策略</th><th>耗时 / s</th><th>修补关节</th></tr></thead><tbody id="stages"></tbody></table></div><div class="axes" id="axes"></div></section></main>
<script id="dataset" type="application/json">__DATA__</script>
<script>
(()=>{
const root=JSON.parse(document.getElementById('dataset').textContent);
const colors={'task/cartesian_approach':'#0a9475','task/cartesian_retreat':'#d28a36','task/rrt_to_place':'#c5574d','task/tcp_to_precontact':'#358baa','task/precontact':'#358baa','task/updown_to_place':'#788996','task/updown_at_place':'#788996','task/release_at_place':'#46555e','transition/between_boxes':'#775aa7'};
const value=(n,d=1)=>Number(n||0).toFixed(d);
document.getElementById('totals').innerHTML=`<span>抓取 / 过渡 <strong>${root.completed_boxes}/${root.validated_transitions}</strong></span><span>搜索累计 <strong>${value(root.total_search_s,1)}s</strong></span><span>Rerun 时间轴 <strong>${value(root.playback_s,1)}s</strong></span><span>七轴总行程 <strong>${value(root.total_joint_travel_deg,0)}°</strong></span><span>最大携箱倾角 <strong>${value(root.max_tilt_deg,1)}°</strong></span>`;
const maxSelected=Math.max(5,...root.boxes.map(x=>x.planner_s));
let selected=0;
const rows=document.getElementById('rows');
root.boxes.forEach((box,index)=>{
 const tr=document.createElement('tr');tr.tabIndex=0;tr.setAttribute('aria-selected',index===0?'true':'false');
 const slow=box.planner_s>5;const width=Math.max(1,Math.min(100,100*box.planner_s/maxSelected));
 tr.innerHTML=`<td>${box.id}</td><td>${box.side==='left'?'左':'右'}</td><td class="status ${slow?'slow':''}">${value(box.planner_s,2)}</td><td class="bar-cell"><div class="track" style="--target:${100*5/maxSelected}%"><div class="fill ${slow?'slow':''}" style="width:${width}%"></div><i class="target"></i></div></td><td>${value(box.selected_s,2)}</td><td>${value(box.total_s,2)}</td><td>${box.task_attempts}+${box.transition_attempts}</td><td>${box.repair_successes}/${box.repair_attempts}</td><td>${box.reversals}</td><td>${value(box.tilt_deg,1)}°</td>`;
 tr.addEventListener('click',()=>select(index));tr.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(index)}});rows.appendChild(tr);
});
function plot(canvas,box,yIndex){
 const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));
 const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const w=rect.width,h=rect.height,pad=23;
 const entries=Object.entries(box.stage_paths).filter(([,points])=>points.length>1);
 const points=entries.flatMap(([,values])=>values);if(!points.length)return;
 let xmin=Math.min(...points.map(p=>p[0])),xmax=Math.max(...points.map(p=>p[0]));
 let ymin=Math.min(...points.map(p=>p[yIndex])),ymax=Math.max(...points.map(p=>p[yIndex]));
 const span=Math.max(xmax-xmin,ymax-ymin,0.02)*1.12;let xc=(xmin+xmax)/2,yc=(ymin+ymax)/2;
 const scale=Math.min((w-2*pad)/span,(h-2*pad)/span);
 const coord=p=>[w/2+(p[0]-xc)*scale,h/2-(p[yIndex]-yc)*scale];
 ctx.strokeStyle='#dbe3e4';ctx.lineWidth=1;ctx.strokeRect(pad-.5,pad-.5,w-2*pad+1,h-2*pad+1);
 for(const [name,values] of entries){ctx.strokeStyle=colors[name]||'#647c86';ctx.lineWidth=2;ctx.beginPath();values.forEach((p,i)=>{const[x,y]=coord(p);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke()}
 const [sx,sy]=coord(points[0]),[ex,ey]=coord(points[points.length-1]);
 for(const [x,y,color] of [[sx,sy,'#087970'],[ex,ey,'#16232a']]){ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill()}
 ctx.fillStyle='#627078';ctx.font='12px system-ui';ctx.fillText(value(xmin,2),pad,h-6);ctx.textAlign='right';ctx.fillText(value(xmax,2),w-pad,h-6);ctx.textAlign='left';
}
function select(index){selected=index;const b=root.boxes[index];
 [...rows.children].forEach((r,i)=>r.setAttribute('aria-selected',i===index?'true':'false'));
 document.getElementById('selected-title').textContent=`${b.id} 号箱 · ${b.side==='left'?'左臂':'右臂'} · ${b.mode==='front'?'正面':'顶吸'}`;
 document.getElementById('selected-method').textContent=b.transition_strategies.join(' / ');
 document.getElementById('selected-metrics').innerHTML=`<span>核心 / 进程 <strong>${value(b.planner_s,2)}s / ${value(b.selected_s,2)}s</strong></span><span>TCP 行程 <strong>${value(b.tcp_m,2)}m</strong></span><span>相对 shortcut 偏离 <strong>${value(b.shortcut_position_deviation_m,3)}m / ${value(b.shortcut_orientation_deviation_deg,1)}°</strong></span><span>关节行程 <strong>${value(b.travel_deg,0)}°</strong></span><span>换向 <strong>${b.reversals}</strong></span><span>大步翻转 <strong>${b.joint_flip_events}</strong></span><span>携箱倾角 <strong>${value(b.tilt_deg,1)}°</strong></span>`;
 const names=[...new Set(Object.keys(b.stage_paths))].filter(n=>b.stage_paths[n].length>1);
 document.getElementById('legend').innerHTML=names.map(n=>`<span><i style="background:${colors[n]||'#647c86'}"></i>${n}</span>`).join('');
 const stages=document.getElementById('stages');stages.replaceChildren();
 b.segments.forEach(s=>{const tr=document.createElement('tr');const parts=[s.stage||'transition',s.strategy||'failed',value(s.total_ms/1000,3),(s.repaired_joints||[]).join(', ')||'—'];parts.forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.appendChild(td)});stages.appendChild(tr)});
 document.getElementById('axes').textContent='各关节换向 / 活动范围：'+Object.entries(b.axis_reversals).map(([name,count])=>`${name.split('_').pop().toUpperCase()} ${count}次 / ${value(b.axis_ranges[name],0)}°`).join(' · ');
 plot(document.getElementById('xy'),b,1);plot(document.getElementById('xz'),b,2);
}
new ResizeObserver(()=>select(selected)).observe(document.querySelector('.plots'));
select(0);
})();
</script>
</body>
</html>
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    output = args.output or args.summary.with_name(
        args.summary.stem.replace("-summary", "") + "-search-report.html")
    data = json.dumps(report_data(summary), ensure_ascii=False, separators=(",", ":"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(HTML.replace("__DATA__", data.replace("</", "<\\/")), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
