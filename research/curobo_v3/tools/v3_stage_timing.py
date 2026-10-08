"""Chinese labels and truthful timing presentation for sequential-unload replay."""

STAGES = {
    "initialize": ("初始化与初始状态检查", "cuRobo 碰撞/限位检查 + URDF FK"),
    "right_contact_model": ("上箱接触几何评估", "原工具网格半空间/接触足迹检查"),
    "right_precontact": ("右臂到上箱预接触点", "cuRobo IK + GPU RRTConnect"),
    "right_contact": ("右臂接触上箱", "解析 IK 笛卡尔续解 + 通道检查"),
    "right_attach": ("右臂吸附上箱", "实际 FK 附件变换 + 携箱检查"),
    "left_contact_model": ("下箱接触几何评估", "箱面内抓取候选 + 网格足迹检查"),
    "left_extract_target": ("下箱抽离终点求解", "cuRobo IK（左臂 7 维）"),
    "left_precontact": ("左臂到下箱预接触点", "GPU RRTConnect（左臂 7 维）"),
    "left_contact": ("左臂接触下箱", "解析 IK 笛卡尔续解（含逆向接触构造）"),
    "left_attach": ("左臂吸附下箱", "实际 FK 附件变换 + 携箱检查"),
    "left_extract": ("下箱直线抽离 36 厘米", "逆向解析 IK 构造 + 正向轨迹验收"),
    "left_place": ("下箱搬运至左后卸载区", "cuRobo IK + GPU RRTConnect（7 维）"),
    "left_release": ("下箱释放", "实际 FK 位姿核验 + 移除附件"),
    "left_retreat": ("左臂空载撤离", "解析 IK 笛卡尔续解"),
    "left_park": ("左臂安全停靠", "命名姿态/升降扫掠筛选 + GPU RRTConnect"),
    "right_lower": ("上箱竖直下降至下层", "升降轴几何插值 + 全路径校验（无 RRT）"),
    "right_extract": ("上箱直线抽离 36 厘米", "cuRobo IK + 解析 IK 笛卡尔续解（8 维）"),
    "right_place": ("上箱搬运至右后卸载区", "cuRobo IK + GPU RRTConnect（8 维）"),
    "right_release": ("上箱释放与完成", "实际 FK 位姿核验 + 移除附件"),
}


def stage_label(phase):
    return STAGES.get(phase, (phase, "未记录"))[0]


def stage_method(phase, result=None):
    if result and result.get("cartesian_strategy", "").startswith("f044_fixed_swivel"):
        methods = {
            "left_precontact": "cuRobo IK + GPU RRTConnect（左臂 7 维优先）",
            "left_contact": "正向解析 IK（固定冗余角、1 cm 采样）",
            "left_extract": "终点引导选解析分支 + 正向固定冗余角 IK（1 cm；无 RRT）",
            "right_lower": "固定冗余角解析 IK + 升降插值 + 路径校验（无 RRT）",
            "right_extract": "固定角基线；必要时8维终点 IK + 1 cm局部冗余续解（每失败段最多14候选；无 RRT）",
        }
        if phase in methods:
            return methods[phase]
    return STAGES.get(phase, (phase, "未记录"))[1]


def timing_markdown(result):
    """Never infer stage time from frames or apportion an aggregate to stages."""
    totals = result.get("stage_timing_ms", {})
    breakdown = result.get("stage_breakdown_ms", {})
    known = set(result.get("phases", ())) | set(totals)
    known.update(a["stage"] for a in result.get("attempts", ()) if a["stage"] in STAGES)
    phases = [p for p in STAGES if p in known] + sorted(known-set(STAGES))
    wall = result.get("measured_wall_ms", result.get("total_ms"))
    header = f"**本次规划请求：{wall/1000:.3f} 秒**" if wall is not None else "**本次规划请求：未记录**"
    lines = [header, "", "以下是计算墙钟，不是播放/执行时间；包含失败候选、预检和同阶段重复访问。", ""]
    if totals:
        lines += [f"阶段合计：**{sum(totals.values())/1000:.3f} 秒**。GPU 在阶段边界同步；进程初始化和GPU锁等待不计入阶段。", "",
                  "**耗时最高的阶段**"]
        lines += [f"- {stage_label(p)}：**{duration/1000:.3f} 秒**" for p, duration in sorted(totals.items(), key=lambda item: -item[1])[:3]]
        lines.append("")
        if wall is not None:
            overhead = wall-sum(totals.values())
            if overhead >= 0.:
                lines.append(f"阶段之外的请求开销（差额）：**{overhead/1000:.3f} 秒**。")
        lines.append("")
    else:
        lines += ["**旧结果未记录逐阶段总耗时：下表不作估算。** 只显示原始 RRT 记录和全局分类耗时。", ""]
    lines += ["| 中文阶段 | 耗时（秒） | 求解方式 |", "| --- | ---: | --- |"]
    for phase in phases:
        duration = f"{totals[phase]/1000:.3f}" if phase in totals else "未记录"
        lines.append(f"| {stage_label(phase)} | {duration} | {stage_method(phase, result)} |")
    if totals:
        lines += ["", "### 分阶段计算细项（秒）", "IK含初始化/筛选；笛卡尔含解析构造/验证；验收含原生检查与独立URDF FK。", "",
                  "| 阶段 | IK | RRTConnect | 笛卡尔 | 验收/FK | 其他 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for phase in phases:
            if phase not in totals:
                continue
            details = breakdown.get(phase, {})
            parts = [details.get(key, 0.) for key in ("ik", "search", "cartesian", "audit")]
            other = totals[phase]-sum(parts)
            values = [f"{value/1000:.3f}" for value in parts]
            values.append(f"{other/1000:.3f}" if other >= -1e-3 else "计时重叠")
            lines.append(f"| {stage_label(phase)} | " + " | ".join(values) + " |")
        lines += ["", "### 显式碰撞检查（嵌套子项，不可再相加）", "不含 cuRobo IK 优化器内部未单独暴露的碰撞内核时间。", "",
                  "| 阶段 | 检查耗时（秒） |", "| --- | ---: |"]
        lines += [f"| {stage_label(p)} | {breakdown.get(p, {}).get('collision', 0.)/1000:.3f} |" for p in phases if p in totals]
    else:
        lines += ["", "### 原始记录中可确认的 RRTConnect 耗时", "这是阶段的一部分，不是阶段总耗时。", "",
                  "| 阶段 | RRTConnect耗时（秒） |", "| --- | ---: |"]
        for phase in phases:
            searches = [a["stats"]["search_ms"] for a in result.get("attempts", ())
                        if a["stage"] == phase and a.get("operation") == "rrtconnect" and "search_ms" in a.get("stats", {})]
            if searches:
                lines.append(f"| {stage_label(phase)} | {sum(searches)/1000:.3f} |")
        names = {"ik": "cuRobo IK", "search": "GPU RRTConnect", "cartesian": "解析笛卡尔构造", "collision": "显式碰撞检查"}
        lines += ["", "### 旧结果全局分类（含嵌套，不能相加）"]
        lines += [f"- {names.get(k, k)}：{v/1000:.3f} 秒" for k, v in result.get("timing_ms", {}).items()]
    ik_attempts = [a for a in result.get("attempts", ()) if "initialization_ms" in a]
    if ik_attempts:
        lines += ["", "### IK 实测细项（秒，已包含在阶段总耗时中）", "",
                  "| 阶段/尝试 | 初始化或缓存查找 | 实际求解 | 筛选 | 缓存命中 |",
                  "| --- | ---: | ---: | ---: | --- |"]
        for index, attempt in enumerate(ik_attempts, 1):
            values = [f"{attempt[k]/1000:.3f}" if k in attempt else "未记录"
                      for k in ("initialization_ms", "solve_ms", "filter_ms")]
            hit = "是" if attempt.get("cache_hit") else "否"
            lines.append(f"| {stage_label(attempt['stage'])}/{index} | " + " | ".join(values) + f" | {hit} |")
    return "\n".join(lines)
