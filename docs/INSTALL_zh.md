# 编译、安装与启动

日期：2026-11-07。仓库：<https://github.com/wxk1316/perception>。

## 1. 工作区与依赖

本仓库新增独立 ROS2 Python 包 `campus_perception`，不覆盖车机原来的 `rda_dwa_real`，不包含定位或底层控制算法。模块通过现有 `rda_dwa_real_msgs/ObstacleArray` 与规划通信，因此使用原车机工作区作为 underlay。按 [ROS2 Humble 工作区说明](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html) source 原环境，再构建新的 overlay。

目标环境：Ubuntu22.04、ROS2 Humble、Python3.10、AGX Orin64GB、CUDA12.6、TensorRT10.3。几何/KF 支路使用 NumPy、SciPy；配置与模型使用 PyYAML；ROS 消息、RViz 及 launch 来自 ROS2。模型通过现有 TensorRT/CUDA runtime 执行，不需要 torch、spconv、pycuda 或完整 Autoware。

在车机终端安装缺失的通用依赖：

```bash
sudo apt-get update
sudo apt-get install -y python3-colcon-common-extensions python3-numpy python3-scipy python3-yaml \
  ros-humble-rviz2 ros-humble-rclpy ros-humble-ament-index-python \
  ros-humble-launch-ros ros-humble-sensor-msgs ros-humble-nav-msgs \
  ros-humble-geometry-msgs ros-humble-std-msgs ros-humble-visualization-msgs
```

沿用 MiiVii/Jetson 已有 CUDA/TensorRT 安装，不在此流程替换厂商驱动或升级 JetPack。

## 2. 确认原消息包可见

下面 `CAR_WS` 为示例路径，改为车机原工作区的实际根目录：

```bash
CAR_WS=/home/nvidia/car_ros
source /opt/ros/humble/setup.bash
source "$CAR_WS/install/setup.bash"
ros2 pkg prefix rda_dwa_real_msgs
ros2 interface show rda_dwa_real_msgs/msg/Obstacle
python3 -c "import numpy, scipy, yaml, rclpy; print('基础依赖可用')"
```

若找不到消息包，在原工作区构建原有消息源码后重新 source：

```bash
cd "$CAR_WS"
colcon build --packages-select rda_dwa_real_msgs
source install/setup.bash
```

使用原消息定义。不要在 overlay 另装一份与原车机不同的同名消息包。本模块同时支持基础字段版与含 `has_shape_heading/shape_heading` 的扩展版：扩展版主动发布 `has_shape_heading=true, shape_heading=0`，明确 length/width 是 map AABB，heading 是运动方向。

## 3. 获取源码并编译

独立工作区路径可自行调整；以下为首次安装：

```bash
PERCEPTION_WS="$HOME/campus_perception_ws"
mkdir -p "$PERCEPTION_WS/src"
git clone https://github.com/wxk1316/perception.git "$PERCEPTION_WS/src/perception"
cd "$PERCEPTION_WS"
colcon build --symlink-install --packages-select campus_perception
source install/setup.bash
ros2 pkg prefix campus_perception
```

没有 C++、CUDA 自定义算子或 spconv 需要编译；colcon 安装 Python 模块、launch、配置和权重。`--symlink-install` 便于调参和编辑，但所有修改仍在新的感知目录中。

若使用仓库 `dist/` ZIP，先把 ZIP 下载到本机并验证同目录 SHA256，在新工作区中解压替代 git clone：

```bash
sha256sum -c campus_perception_20261107.zip.sha256
mkdir -p "$PERCEPTION_WS/src/perception"
unzip campus_perception_20261107.zip -d "$PERCEPTION_WS/src/perception"
```

GitHub 自带 Download ZIP 也可使用；解压后让目录位于独立工作区 src 内即可。完整工程的定位、驱动、地图和规划保持原版本。

## 4. Orin 上生成 TensorRT engine

先确认系统 Python 可以使用厂商 TensorRT：

```bash
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

将完整模型目录复制到用户可写位置，生成本机 engine：

```bash
SHARE_DIR="$(ros2 pkg prefix campus_perception)/share/campus_perception"
MODEL_DIR="$HOME/.local/share/campus_perception/centerpoint/base"
mkdir -p "$MODEL_DIR"
cp -a "$SHARE_DIR/models/centerpoint/base/." "$MODEL_DIR/"
ros2 run campus_perception campus_model --model-directory "$MODEL_DIR" --build-only
```

base/tiny 已带公开 ONNX。TensorRT engine 缓存绑定 GPU 架构、TensorRT 版本、模型 hash 和输入 profile，不能将其他机器 engine 当成本机通用文件。encoder `[N,32,9]` 为动态 N；head 使用 `[1,32,480,480]` canonical profile。详情见 [TensorRT10.3 官方指南](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1030/developer-guide/index.html)。节点不在点云回调里编译 engine。

需要 tiny 时，从 `models/centerpoint/tiny` 复制到另一目录并单独构建，然后在启动配置中指向该目录。模型文件损坏可从固定公开 revision 恢复：

```bash
python3 "$PERCEPTION_WS/src/perception/tools/prepare_campus_model.py" \
  --variant base --destination "$MODEL_DIR"
```

## 5. 配置原始点云和安装外参

创建用户配置副本：

```bash
CONFIG_FILE="$HOME/.config/campus_perception/perception_tracking.yaml"
mkdir -p "$(dirname "$CONFIG_FILE")"
cp "$SHARE_DIR/config/perception_tracking.yaml" "$CONFIG_FILE"
```

编辑该文件，顶层节点名保持 `perception_tracking_node`。核对 `points_topic`、`lidar_frame_ids`、`odom_topic`、LiDAR→定位 body 外参、body 离地高度和逐点时间约定。

默认原始 `/rslidar_points`、sensor 输入、未声称上游去畸变；frame 允许 rslidar/lidar/laser。`sensor_z=0.28m`、`body_height=0.17m` 是原配置初值，并非本轮独立物理测量。`point_time_mode=auto` 只接受与 header 同时间域的绝对 timestamp 秒；相对 time/t 必须配置单位及首/末点约定。详见使用文档。

## 6. 启动与 RViz

在原方式运行雷达与 `/localization` 后启动感知，选择已经生成 engine 的目录：

```bash
ros2 launch campus_perception campus_perception.launch.py \
  config:="$CONFIG_FILE" with_rviz:=true
```

启动前在用户配置中把 `centerpoint_model_directory` 设为 `MODEL_DIR` 的实际绝对路径，例如 `/home/nvidia/.local/share/campus_perception/centerpoint/base`。YAML 中不自动展开 `$HOME` 或 `~`。该 launch 只启动感知和可选 RViz。

不生成 engine、只验证几何/KF 话题时可运行：

```bash
ros2 run campus_perception perception_tracking_node --ros-args \
  --params-file "$CONFIG_FILE" -p centerpoint_enabled:=false
```

新模块与原 `lidar_track_node` 只运行一个。规划继续订阅 `/rda/obstacles`，无需修改规划算法。原一键 launch 若固定启动旧感知，应在自己的启动组合中停用旧感知 Node，保持其他节点原样；本仓库不替你改写原启动文件。

## 7. 日常更新与回放

更新源码后在独立工作区重新编译：

```bash
cd "$PERCEPTION_WS/src/perception"
git pull --ff-only
cd "$PERCEPTION_WS"
colcon build --symlink-install --packages-select campus_perception
source install/setup.bash
```

新终端 source 顺序为 ROS2 → 原车机 underlay → 感知 overlay。回放时感知/RViz 使用 `use_sim_time:=true`，rosbag 发布 `/clock`，定位源采用相同回放时钟：

```bash
ros2 launch campus_perception campus_perception.launch.py \
  config:="$CONFIG_FILE" with_rviz:=true use_sim_time:=true
```

## 8. 常见问题

| 现象 | 检查与处理 |
|---|---|
| 找不到 campus_perception | 独立工作区编译成功后 source install/setup.bash |
| 找不到 rda_dwa_real_msgs | 先 source 原工作区；必要时只构建原有消息包 |
| 模型 unavailable / engine missing | 在 Orin 可写目录执行 build-only，并设置模型目录绝对路径 |
| 点云有输入但没有输出 | 检查 frame 白名单、源 stamp/时钟、定位 map frame、位姿是否覆盖源时间及超时 |
| deskew=missing 或其他退化状态 | 核对实际逐点时间字段、单位和扫描覆盖；不把该状态当作已补偿 |
| RViz 无点云 | Fixed Frame=map；PointCloud2 QoS=Best Effort；检查 points_map 发布 |
| 障碍物跳动或双份 | 用 ros2 topic info --verbose 检查是否仍有旧感知发布者 |
| localization jump latched | 感知拒绝不连续位姿，确认地图/定位后重启感知；不会改动定位算法 |

本机没有 Orin/ROS DDS/RViz 图形运行环境，源码与离线测试通过不表示已完成真实车机 build 或10Hz验收。实车验证指标见使用文档。
