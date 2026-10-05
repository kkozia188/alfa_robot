# 官方cuRobo main适配

2026-10-05固定官方main提交`78fd485fa82d9b9a063fb4985e371814587e666a`。
本机第三方源码目录仍是`/mnt/mydisk/ALFA/curobo_v2_ws/src/curobo`，运行环境仍是
`/mnt/mydisk/ALFA/.venvs/curobo-v2-py310`。不新增仓库副本、不改系统Python。

## 变化与验证范围

官方已修复旧版`wp.torch.device_from_torch`调用；Warp1.17使用顶层接口。
官方连接器已包含首碰截断修复，本项目不再叠加旧算法补丁；旧补丁保留用于历史版本。
新版`LinearConnector.set_dependencies`需要共享索引缓冲区。旧测试只设置action_dim，
会在真正测试算法前失败；当前测试按完整初始化合同设置依赖，并覆盖首碰、无碰撞及起点碰撞。

工具中21组直接cuRobo导入均可解析；CPU核心22项通过，连接器3项GPU回归通过，
BIT搜索3项通过；六种规划器首组完整周期全部成功。核心GPU专项覆盖缓存复用/更新、
过期结果拒绝、无GUI调用；龙头车Mesh远离/重叠/关闭检查通过。
这些是接口和首组回归，不是历史全部实验或11/60组的重新验收。
历史JSON维持原样；此次证据摘要见`generated/main_compatibility/validation.json`。

## 测试命令

```bash
cd /mnt/mydisk/ALFA/alfa_robot_v3_curobo/research/curobo_v3
/mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python -m unittest discover \
  -s tests \
  -p 'test_core_*.py' \
  -v
/mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python -m unittest discover \
  -s tests \
  -p 'test_gpu_curobo_compat.py' \
  -v
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /mnt/mydisk/ALFA/.venvs/curobo-v2-py310/bin/python tools/v3_cycle_compare.py \
  --planners informed_rrt rrt rrtconnect prm informed_connect bitstar \
  --output /tmp/curobo_main_six_planners.json
```

大批量GPU验证应避免同时运行多个规划演示。源码哈希是版本标识；浅克隆加本地测试
修改会让包版本字符串显示dev/dirty，它不意味着仍在运行旧提交。

`alfa_robot_v3_curobo`为当前正式开发工作树；`alfa_robot_curobo_compare`为同仓历史对比
及后续压测工作树，含本地路径重定位改动和后续提交，本次未删除或覆盖。
