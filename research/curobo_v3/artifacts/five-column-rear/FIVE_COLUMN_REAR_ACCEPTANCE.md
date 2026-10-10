# 五列顶部两层车后卸载验收（2026-10-10）

- 结果：**6/6，通过**。
- 每个箱体必须搬到当前车辆默认 `unloading` 后方目标，记录实际位姿后释放；抽离即消失不计通过。
- 范围：五列顶部两层，共10箱；底部15箱保留。
- 顺序：列2 → 列1 → 列3 → 列4 → 列0。
- yaw=0、base_x=0、base_y∈[-0.5,0.5]m，桥宽2.1m。

| seed | repeat | 帧数 | 后方释放 | 最大放置误差(mm) | 最大角误差(°) | RRT重试 | 审计 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 11 | 1 | 14658 | 10 | 0.0223 | 0.006124 | 0 | 通过 |
| 11 | 2 | 14658 | 10 | 0.0223 | 0.006124 | 0 | 通过 |
| 29 | 1 | 14820 | 10 | 0.0007 | 0.000011 | 0 | 通过 |
| 29 | 2 | 14820 | 10 | 0.0007 | 0.000011 | 0 | 通过 |
| 41 | 1 | 14844 | 10 | 0.0007 | 0.000015 | 0 | 通过 |
| 41 | 2 | 14895 | 10 | 0.0007 | 0.000015 | 1 | 通过 |

## 语义

- 中间三列：右臂支撑上箱，左臂抽离并搬到左后释放；上箱下降、抽离并搬到右后释放。
- 外列0：右臂依次处理上/下箱，两箱都搬到右后释放。
- 外列4：左臂镜像处理，两箱都搬到左后释放。
- 列间仅在无附件时收拢双臂并横移底盘。

## 验收边界

- 每次20个生命周期事件：10次吸附、10次后方释放。
- 每次独立比较实际箱体世界位姿和目标位姿，≤1mm/0.5°。
- 底盘投影始终在桥面，base_x/yaw恒0。
- 仍为几何规划与附件生命周期验证，不是动力学或实机。

## Viser只读回放

```bash
cd /home/astesia/Sevenova/.v3-curobo-five-column-rear-20261010/research/curobo_v3 && \
gzip -dc artifacts/five-column-rear/five-column-rear-s11-r1.json.gz > /tmp/five-column-rear-s11.json && \
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/home/astesia/Sevenova/golden_curobo_adapter/mentor-repro-20261006/curobo-main:tools \
/home/astesia/Sevenova/curobo_v2_ros/bin/python -u tools/v3_five_column_demo.py \
  --result /tmp/five-column-rear-s11.json \
  --port 8095
```

页面：`http://127.0.0.1:8095`。箱体仅在后方 `*_release` 事件发生时消失。
