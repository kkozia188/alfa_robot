# 2026-10-03 Linear初步适配节点收口

按用户授权完成：

- MOTION-238～243：六项全部Done，保留历史探索与失败结论。
- MOTION-240补5条：MorphIt参数与覆盖缺陷、人工核心/面边角球、10内部球、20内部球+12边+8角定版、附着坐标修复与最终验收。
- MOTION-241补5条：串行IK/官方TrajOpt、批量分层图与首边诊断、代理模型IK/FK回归、32候选固定Updown/ψ解析抽离、阶段验收。
- MOTION-238补4条：机器人拟合与爆炸图、Tool0与mobile模型同源修复、代理分支同步与目标冻结、接入验收。
- MOTION-239补3条：离线/在线候选、Pose纠正、基础IK验收；保持原Done。
- MOTION-243补5条：GPU/实时性、官方无参考Graph/TrajOpt边界、因子化搜索实现边界、GPU搜索路线归属、阶段研究收口。
- MOTION-242补6条：参考轨迹优化纠正、官方路线比较、完整6D目标/yaw碰撞微调、完整周期、计时与对象复用、Demo收口。
- MOTION-275补1条并解除MOTION-237父关系：独立具体GPU算法研究，保留同Project/Milestone和In Progress，更新描述承接已收口基础成果。
- MOTION-237补1条并重写标题/描述：三代机适配与基本功能Demo验证；公司节点为初步适配，不再要求11/60组算法成熟度。设In Review，开发完成待张圣涵验证。

本轮共30条新评论。Done表示本期适配/基础实验研究收口；不宣称球近似消除边角漏检、官方长距离算法全部成功、最新模型批量全部通过、生产接口/实机交付或真正端到端算法已完成。未完成具体研究归独立MOTION-275或未来按单项创建issue。

MCP状态：当前会话未加载Linear工具；本机独立MCP启动实际OAuth刷新失败`invalid_grant: Grant not found`。已告知用户并通过已登录浏览器完成授权更新。

链接：
- https://linear.app/sevenova/issue/MOTION-237
- https://linear.app/sevenova/issue/MOTION-275

UI核对：237子任务6/6 Done，237 In Review；275无父任务、In Progress。截图：/tmp/linear-237-closure.png、/tmp/linear-275-independent.png。
