# 左臂初始至接触录制回放

feature/ipc-left-prefix-control，用户已确认只回放左7轴，不控制升降/右臂、不承诺TCP到原箱面。基于contact-roll-alignment实验，未PR合并或实机运动验收。

## 已完成

- generated/left_prefix_control_20261018/prefixes.json保存六组左臂前缀。截断在attach之前最后contact_approach帧，完全排除吸附、抽离、运输、放箱、释放；回Home只逆序同一前缀。
- 工控机/home/ar/alfa_left_prefix_control_20261018存放独立标准库脚本和数据；alfa-left-prefix-control.service运行在127.0.0.1:18122，ROS_DOMAIN_ID=77。没有更改RT控制源码、没有自动使能。
- 本机alfa-left-prefix-tunnel-8155.service将localhost:8155转发到上述后端。前端可选择六组任务、查看真实反馈/目标角度、最高速度、预计时长、获取/释放18121控制权及受控停止。
- 最新控制台反馈左[65,-75,20,-90,0,40,90]°，J3/J6与自制模型同号。旧反号映射已经取消。
- 后端通过18121既有认证建立独立会话、SSE心跳及lease互斥，令牌只在工控机读取、不会发给前端或记录内容。实际软限位从18121读取，每次发送再检查；不跳过起点/新鲜反馈/停稳/active/ENABLED/Action服务门禁。
- 停止调用当前Action目标的取消接口，等待控制器取消结果，停稳后才允许沿原前缀逆序回Home；不是硬件急停。页面失去心跳或控制权丢失时请求取消，未结束前不释放lease。

## 2026-10-19接口已对齐

已读取实际ROS图：/left_arm_jtc/joint_trajectory没有订阅者，left_arm_jtc仅订阅/parameter_events。18121真正执行接口为/left_arm_jtc/follow_joint_trajectory Action。

用户确认沿用既定FollowJointTrajectory Action，已将后端切换到/left_arm_jtc/follow_joint_trajectory，并接入接受/拒绝、执行结果、错误码和取消。已部署工控机服务并更新8155前端，未修改RT控制器。旧--topic参数移除，使用--action（默认即上述接口）。

已确认真实Action服务可见、反馈/控制器可读；本轮未使能、未接管其他终端、未发送实机动作，实机motion_publications=0。若18121持有控制权，应先在那里释放，再到8155获取。若机器人不在记录Home，不会自动规划回Home或硬连线，须先用原控制台调整到左[65,-75,20,-90,0,40,90]°，再释放控制权。

## 验证

六项标准库单元测试通过（导出截断、仅7轴/映射、返回方向、速度/加速度/jerk、拒绝后段、状态门禁）。
实际ROS2收发在隔离Domain231完成，脚本integration_smoke.py拒绝在其他Domain运行。最新ActionServer测试包含成功、逆序回Home、取消、拒绝及执行失败；已接受5个左7轴目标，最终Home误差0，退出码0。
证据/tmp/left_prefix_action_smoke_20261019.log；此前topic测试仅为历史模拟，不是实机执行验收。

时间参数化保留每段原几何，逐记录点停稳，默认最高5°/s、10°/s²、50°/s³；L18/R22约218.6秒。禁止为加速静默简化路径。

## 重启

工控机服务已设置完整ROS环境，令牌文件会自动定位当前18121进程：

```bash
ssh ar@192.168.100.40 'systemctl --user restart alfa-left-prefix-control.service'
systemctl --user restart alfa-left-prefix-tunnel-8155.service
```

Action、反馈、控制器、使能、停稳和控制权均满足后才可发送。原18121使能和现场停止流程保持独立。
