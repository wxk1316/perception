# 校园小车原始点云感知与追踪

日期：2026-11-07。版本0.2.0；ROS2 包名：`campus_perception`。

独立订阅原始 `/rslidar_points`，完成5m范围点云处理、CenterPoint全部5类检测、可开关的最近两帧融合、卡尔曼/匈牙利跨帧追踪，发布检测、追踪和 RViz。检测/追踪保留 car、truck、bus、bicycle、pedestrian；`/rda/obstacles` 为原规划保留兼容映射。定位 `/localization` 只读；本仓库不包含定位、规划或底盘控制节点，也不会替换 `rda_dwa_real`。

适配目标为 AGX Orin 64GB、Ubuntu22.04、ROS2 Humble、Python3.10、CUDA12.6、TensorRT10.3；32线 Helios LiDAR、10Hz；行人速度上限0.6m/s、外部车辆1.0m/s、自行车独立默认1.0m/s。

- [编译、安装与启动](docs/INSTALL_zh.md)
- [话题、参数与规划接入](docs/USAGE_zh.md)
- [两帧融合、5m与预处理评估](docs/FUSION_AND_PREPROCESSING_zh.md)
- [验证记录与边界](verification/README.md)
- [感知追踪主模块](campus_perception/campus_perception/perception_tracking_node.py)
- [独立参数](campus_perception/config/perception_tracking.yaml)
- [RViz 配置](campus_perception/rviz/perception_tracking.rviz)
- [打包下载](dist/campus_perception_20261107.zip)
- [预训练模型与第三方条款](campus_perception/models/THIRD_PARTY_NOTICES.md)

## 快速使用

先 source ROS2 与原车机工作区，让已有 `rda_dwa_real_msgs` 可见；在独立感知工作区中构建：

```bash
colcon build --symlink-install --packages-select campus_perception
source install/setup.bash
```

在 Orin 可写目录生成 TensorRT engine 后启动（详细路径和命令见安装文档）：

```bash
ros2 launch campus_perception campus_perception.launch.py with_rviz:=true
```

同一 `/rda/obstacles` 只能有一个有效感知发布者，启动新模块前停用原 `lidar_track_node`。模型缺 engine 时几何支路继续运行，状态中明确报告模型不可用。

代码声明 MIT；预训练 base/tiny 权重保留官方 Apache-2.0 及训练数据条款。权重来源为 [AutowareFoundation/lidar_centerpoint](https://huggingface.co/AutowareFoundation/lidar_centerpoint)，固定 revision `494c8171def40bd36cc2feb323e0a5acbfab132b`。本机离线验证不能代替 Orin 实际吞吐与校园准确率验证。

## 两帧开关

模型默认使用当前帧与最近一帧，几何占据始终基于当前帧。所有障碍物输出默认只保留最近5m；按包围框最近位置判断，而不是仅按中心。

```bash
ros2 launch campus_perception campus_perception.launch.py two_frame_fusion:=false with_rviz:=true
ros2 launch campus_perception campus_perception.launch.py two_frame_fusion:=true with_rviz:=true
```

两条命令分别执行一次实验，不要同时运行两个感知发布者。不传开关时遵循YAML中的 two_frame_fusion_enabled。可用 `/rda/perception/fused_points_body` 查看当前body坐标中的XYZ与time_lag；`/rda/perception/points_map` 是当前几何非地面点。
