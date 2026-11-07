# 使用、参数与规划接入

## 1. 完整处理流程

`/rslidar_points` 原始点云 → 字段与源时间校验 → 可信逐点时间/位姿去畸变 → LiDAR→body 外参 → 自车剔除、全周 ROI → 局部地面分割 → 几何聚类与异步 CenterPoint → 两条源时间 CV-KF/匈牙利支路 → 占据覆盖融合 → 追踪、规划、RViz。

主模块为 `campus_perception/campus_perception/perception_tracking_node.py`；数值和模型辅助文件在同包。`obstacle_fusion.py` 在本独立包中仅保留感知几何，不包含原规划 TrackManager，也不导入规划包。感知读取 `/localization`，不发布或修改定位。

CenterPoint base/tiny 公开预训练权重随仓库提供；全部直接训练类别 car、truck、bus、bicycle、pedestrian 分别输出。仅在 /rda/obstacles 兼容原规划时将前三类映射成 vehicle。TRAILER/MOTORBIKE 在重映射配置中出现，不是本权重直接训练的类别。不要求网络细分类树干、墙、花坛、路沿等静态物体，它们通过几何保留点进入占据。模型不可用或漏检时，几何继续工作；不根据类别丢掉未知障碍。

## 2. 话题

| 话题 | 类型 | 约定 |
|---|---|---|
| `/rslidar_points` | sensor_msgs/msg/PointCloud2 | 默认原始 sensor 点云；XYZ 必需 |
| `/localization` | nav_msgs/msg/Odometry | 现有 map 中 body 位姿，只读 |
| `/rda/perception/detections` | rda_dwa_real_msgs/msg/ObstacleArray | 本帧源点云聚类 AABB，id=-1、speed=-1；不含跨帧身份 |
| `/rda/perception/model_detections` | 同上 | 接受的异步模型源帧检测，保留原模型源时间，不冒充本帧检测 |
| `/rda/perception/tracks` | 同上 | 融合后的稳定轨迹/占据与可信运动或 unknown 状态 |
| `/rda/obstacles` | 同上 | 原规划兼容消息：car/truck/bus→vehicle；速度不可信→unknown；ID/几何/源时间保留 |
| `/rda/perception/fused_points_body` | sensor_msgs/msg/PointCloud2 | 当前body坐标中的模型输入XYZ与time_lag，含地面、可含一帧历史 |
| `/rda/perception/points_map` | sensor_msgs/msg/PointCloud2 | 非地面保留点 XYZ，真实物理 map 坐标 |
| `/rda/perception/detection_markers` | visualization_msgs/msg/MarkerArray | 当前检测蓝色轮廓，det 编号仅为帧内索引 |
| `/rda/perception/tracking_markers` | 同上 | 稳定 ID、类别、速度与追踪足迹 |
| `/rda/obstacles_markers` | 同上 | 原 RViz 追踪话题兼容别名 |
| `/rda/perception_status` | std_msgs/msg/String | JSON 状态、源龄、去畸变、点数、检测数、轨迹数、model/solver 与运动可信性 |

障碍消息和 points_map 使用 map；fused_points_body 使用当前body坐标，需要已有map→body TF在RViz显示。length/width 是 map AABB 尺寸；heading 是速度方向，不能拿它旋转碰撞框。检测/追踪保留五类语义，未知速度以speed=-1表示，语义不会因此丢失；兼容规划消息仍使用type=unknown、speed=0的既有约定。扩展消息存在 shape_heading 时设置为0并声明有效；基础消息版由原规划的 AABB 约定消费，务必核对消费端几何契约。

模型实体 ID 从1000000001开始，几何父对象/瓦片使用独立区间。一个对象首次从仅几何升级到模型实体时可能切换一次公开 ID；已有模型轨迹不会仅因跨过瓦片格子换 ID。暂时遮挡保持约0.6s，长遮挡或完全相似对象交换位置不承诺永不换 ID。

## 3. 主要参数

| 参数 | 初始值 | 含义 |
|---|---:|---|
| points_topic / odom_topic | /rslidar_points / /localization | 原始点云和只读定位 |
| input_frame_type / input_deskewed | sensor / false | 独立原始 sensor 路线 |
| lidar_frame_ids | rslidar, lidar, laser | 必须核对实际 header.frame_id |
| sensor_z / body_height | 0.28m / 0.17m | LiDAR→body 垂直外参 / body 离地高度初值 |
| sensor_x/y/roll/pitch/yaw | 0 | 感知独立外参，按标定更新 |
| max_obstacle_range | 5m | 车体参考点到占据AABB最近位置；最终所有障碍输出裁剪 |
| model_input_range | 5m | 模型点云上下文；可增至8m，而规划输出仍为5m |
| two_frame_fusion_enabled | true | 模型当前+上一帧；几何始终用当前源 |
| fusion_max_age | 0.15s | 仅最近相邻10Hz源帧；较长缺口退为单帧 |
| ground_distance / ground_cell_size | 0.025m / 2m | 局部地面距离 / 网格尺度 |
| ground_max_fit_points | 256 | 拟合代表点上限；分类仍作用于全部保留点 |
| track_gate / mahalanobis_gate | 0.8m / 9.21 | 绝对关联门 / 二维创新平方门 |
| track_lost_ttl_seconds | 0.6s | 有限遮挡保持 |
| motion_observation_window | 0.5s | 低速源时间回归窗口 |
| pedestrian_max_speed | 0.6m/s | 场景运动假设 |
| vehicle_max_speed / bicycle_max_speed | 1.0m/s / 1.0m/s | 独立的两类速度假设 |
| dyn_speed_thr / static_speed_thr | 0.12 / 0.06m/s | 运动/静止滞回阈值 |
| dyn_confirm_seconds / static_confirm_seconds | 0.3s / 1.2s | 按源时间确认 |
| centerpoint_enabled / backend | true / tensorrt | 模型开关和推理后端 |
| centerpoint_score_threshold | 0.35 | 模型初始分数门 |
| centerpoint_max_age | 0.25s | 接受模型源帧年龄上限 |
| centerpoint_reference_height | 1.8m | 虚拟模型参考高度，物理 map 坐标保持不变 |
| cloud_timeout / loc_timeout | 0.5s / 0.5s | 输入新鲜度约束 |

sensor_z 与 body_height 来自原项目配置初值，不是本轮独立测量。不要把离地高度0.45m再次直接当作 LiDAR→body 外参。感知不自动读取或修改 FAST-LIO 在线外参；安装标定需同步给感知配置。

模型保留地面上下文；几何支路去地面。ground_enabled=false用于几何消融时保留路面，不隐式启用旧15cm高度截断。静态地面使用本项目稳健局部平面实现，不宣称是 Patchwork++ 移植。默认先分地面，不先用旧15cm最低高度截掉低路沿。不确定地面点保留占据，能否检出低物体仍取决于真实雷达返回。

## 4. 去畸变和源时间

`point_time_mode=auto` 只自动接受可校验的绝对 timestamp 秒，要求点云、header、定位同一时间域。相对 time/t 需显式选择 relative_seconds 或 relative_nanoseconds，并按驱动扫描首/末点约定设置 point_time_header_is_end。没有可信时间时不从方位角伪造时间。

逐点时间与完整位姿历史可靠且覆盖扫描时支持 SE(3) 补偿；仅平面历史可支持 SE(2)。这使用现有定位历史插值，不声称等同于高频 IMU 去畸变。无时间、历史不足或跨时域时点保留未去畸变，并报告 missing、relative_time_requires_configuration、time_outside_source_clock 或 pose_history_does_not_cover_scan。

无逐点时间不阻止基础几何输出，但转弯形变仍需实测；必须看状态确认实际是否补偿。硬件说明没有给出真实字段/header结果，所以默认配置不代表已确认驱动时间约定。

重复源、坏解码、过期/未来源不刷新输出。有效空帧产生空检测，并按源时间保持/删除已有轨迹。位姿尚未覆盖源时刻时最多保留一份最新待处理点云。定位跳变只在感知侧拒绝并报告，不修改定位算法。

每个模型源帧仅进入模型 KF 一次。融合缓存每个接受的原始云都更新，与推理是否跳帧独立；缓存只保存未融合的上一源帧，不累积第三帧。模型本身不再隐式叠加上一推理帧。位姿变换为 T_current_world @ T_world_previous，历史点转换到当前body后再次按当前范围及车体过滤；time_lag=当前源时间-历史源时间。

只补偿自车运动，不消除行人/车辆自身位移。历史点可辅助模型稀疏输入，但不作为几何当前占据/KF新观测，避免拖影。空当前云不单靠历史生成新的模型输入。模型输出保留当前源时间，其输入内历史点有独立年龄。异步结果保持其原始时间；只读预测到当前云源时刻时不增加量测次数。源龄和投递延迟增加占据包络，过旧时撤销速度可信状态。只删除完全被模型包络覆盖的几何框，模型框外的几何碎片继续保留。复杂静态结构以瓦片保留凹形空间。

## 5. RViz 与状态检查

配置文件 `campus_perception/rviz/perception_tracking.rviz` 固定 frame=map。蓝色轮廓为本帧检测，追踪红色为运动、绿色为静止、黄色为unknown；标签含ID和速度，箭头长度按1秒速度位移。绘制是二维占据示意，不是测得的完整三维高度。Marker 在输入超时后自动消失。

```bash
ros2 topic hz /rslidar_points
ros2 topic hz /rda/perception/tracks
ros2 topic echo /rda/perception_status --once
ros2 topic info /rda/obstacles --verbose
```

检查 input_topic/input_frame/type、point_time_status、deskew、model.status、model_error、source_age_seconds、model_result_age_seconds、processing_seconds、skipped_busy和measurement_age。ready表示模型可用，unavailable表示仍需处理模型环境或engine；几何可继续。

## 6. 下层规划接入

原规划继续读取 `/rda/obstacles`，消息类型不变。新包名独立，不把原 `rda_dwa_real` 的 Python 模块覆盖到 overlay 中。首先确认旧感知已停用，不能让两个发布者向同一规划话题交替投递不同ID。

本仓库只交付感知。可以在自己的 launch 组合中移除旧感知 Node、加入新感知 Node，并保持原定位/规划/控制配置：

```python
Node(package='campus_perception', executable='perception_tracking_node',
     name='perception_tracking_node', parameters=[perception_config], output='screen')
```

该 snippet 不启动其他节点，perception_config 指向用户独立参数文件。

## 7. 校园验收

校内需要观测行人、车辆、自行车和静态物体，分别记录漏检、假占据、ID切换和轨迹源龄。重点回放后方/侧方低路沿、停放车起步、两人交叉、短时遮挡、转弯以及模型暂时不可用。

在定位/规划并行运行时测原始源到输出的p95/p99年龄与持续积压，而不是只看平均网络推理时间。10Hz与校园准确率尚未在Orin上实测。本机独立包离线测试以及权重/CPU探针仅验证代码与输入输出契约。


评估代码和具体消融步骤见[两帧与预处理评估](FUSION_AND_PREPROCESSING_zh.md)，本版模型类别、5m与融合回归记录见verification/。
