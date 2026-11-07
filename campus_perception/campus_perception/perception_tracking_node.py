"""Independent raw-LiDAR perception and tracking ROS module.

/rslidar_points + read-only /localization -> decode/deskew -> ego/ROI/ground
filter -> current detections + asynchronous CenterPoint -> map CV-KF/Hungarian
tracks -> RViz and the existing /rda/obstacles planner contract.
No localization, planner, controller or sensor driver is started or modified.
Helper modules implement the reusable numerical algorithms. Every observation
retains its source stamp; malformed clouds publish nothing, valid empty clouds
publish empty detections and age existing tracks.
"""
from __future__ import annotations

import math
import struct
import time
import json
from collections import deque
from pathlib import Path
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry

try:
    from rda_dwa_real_msgs.msg import ObstacleArray, Obstacle
except Exception:
    ObstacleArray = Obstacle = None

from .obstacle_fusion import (Clusterer, WorldObstacleRepresenter,
                              RepresentationLimitError)
from .frame_utils import quat_to_yaw, body_pose_to_base_link, SourcePoseBuffer
from .kf_tracker import GlobalKFTracker
from .campus_cloud import (decode_cloud,deskew_cloud,sensor_matrix,transform_points,
                           LocalGroundFilter,semantic_clusters)
from .campus_cloud import pose_matrix,interpolate_pose_matrix,model_frame_from_body
from .campus_model_worker import ModelWorker
from .campus_fusion import model_detections,fuse_coverage,expand_publication_delay,MODEL_ID_OFFSET
from .semantics import MODEL_CLASSES, planner_obstacles
from .temporal_cloud import TwoFrameCloudFusion, nearby_obstacles, nearby_boxes


class PerceptionTrackingNode(Node):
    """Campus geometry + pretrained detection regions + source-time KF IDs."""

    def __init__(self, node_name='perception_tracking_node'):
        super().__init__(node_name)

        # ---------------- 参数 ----------------
        self.declare_parameter('points_topic', '/rslidar_points')
        self.declare_parameter('odom_topic', '/localization')
        self.declare_parameter('obstacles_topic', '/rda/obstacles')
        output_defaults = {
            'detections_topic': '/rda/perception/detections',
            'model_detections_topic': '/rda/perception/model_detections',
            'tracks_topic': '/rda/perception/tracks',
            'detection_markers_topic': '/rda/perception/detection_markers',
            'tracking_markers_topic': '/rda/perception/tracking_markers',
            'points_map_topic': '/rda/perception/points_map',
            'fused_points_topic': '/rda/perception/fused_points_body',
        }
        for name, default in output_defaults.items():
            self.declare_parameter(name, default)
        self._output_topics = {name: str(self.get_parameter(name).value)
                               for name in output_defaults}
        # 兼容原配置的离地高度；实际转换分用 sensor_z 和 body_height。
        self.declare_parameter('lidar_height', 0.45)
        # === 对齐原项目 cloud_seg 流程（base_link 系）===
        # ① 第一级直通滤波范围（base_link 系）
        self.declare_parameter('pass_x_min', -12.0)
        self.declare_parameter('pass_x_max', 12.0)
        self.declare_parameter('pass_y_min', -12.0)
        self.declare_parameter('pass_y_max', 12.0)
        self.declare_parameter('pass_z_min', 0.15)
        self.declare_parameter('pass_z_max', 1.95)
        # ③ 车体多边形剔除（对齐原项目：agv_length=1.2, agv_width=0.8, outside=0.05）
        self.declare_parameter('vehicle_length', 1.2)
        self.declare_parameter('vehicle_width', 0.8)
        self.declare_parameter('vehicle_outside', 0.0)
        # 旧前向参数仅保留配置兼容；主流程始终处理全周 ROI。
        self.declare_parameter('front_x_min', 0.6)
        self.declare_parameter('front_x_max', 3.0)
        # body→base_link x 偏移（与 bridge_node 一致；/localization 给的是 body 位姿）
        self.declare_parameter('body_to_base_link_x', 0.0)
        # 聚类
        self.declare_parameter('cluster_gate', 0.6)
        self.declare_parameter('cluster_min_points', 2)
        self.declare_parameter('cluster_max_radius', 30.0)
        # 卡尔曼 + 关联
        self.declare_parameter('track_dt', 0.1)       # 控制周期（雷达 10Hz 时 0.1）
        self.declare_parameter('motion_guard_enable', True)
        self.declare_parameter('measurement_position_slack', 0.15)
        self.declare_parameter('velocity_innovation_limit', 0.8)
        self.declare_parameter('velocity_reconfirm_time', 0.2)
        self.declare_parameter('track_gate', 0.8)     # 匈牙利关联门限 m
        self.declare_parameter('track_lost_ttl', 10)  # 连续 miss 帧数 → 删除（10 帧≈1s）
        self.declare_parameter('kf_q', 0.3)           # 过程噪声
        self.declare_parameter('kf_r', 0.12)          # 观测噪声（聚类质心）
        self.declare_parameter('obstacle_max_speed', 1.0)
        # Campus motion state is independent of pedestrian/vehicle/bicycle semantics.
        # Initial campus hysteresis; calibrate against stationary track noise.
        self.declare_parameter('dyn_speed_thr', 0.12)       # KF 速度>此值 → "在动"
        self.declare_parameter('dyn_hold_frames', 5)       # 连续 N 帧在动 → 动态(行人)
        self.declare_parameter('static_speed_thr', 0.06)   # KF 速度<此值 → "静止"
        self.declare_parameter('static_hold_frames', 10)   # 连续 N 帧静止 → 静态
        self.declare_parameter('static_small_thr', 0.6)    # 静态细分：长边≤此值 → block，否则 static_large
        # 定位超时（s）：超过则不刷新障碍（bridge 侧 localization_timeout 会急停）
        self.declare_parameter('loc_timeout', 0.5)
        self.declare_parameter('max_obstacle_range', 5.0)  # full surround includes mode wait/recovery horizon

        self.declare_parameter('use_z_filter', True)
        self.declare_parameter('rslidar_z_min', -0.3)
        self.declare_parameter('rslidar_z_max', 1.5)
        self.declare_parameter('vehicle_half_length', 0.6)
        self.declare_parameter('vehicle_half_width', 0.4)
        self.declare_parameter('vehicle_center_x', 0.0)
        self.declare_parameter('sensor_x', 0.0)
        self.declare_parameter('sensor_y', 0.0)
        self.declare_parameter('sensor_yaw', 0.0)
        self.declare_parameter('cloud_timeout', 0.5)
        self.declare_parameter('pose_sync_tolerance', 0.15)
        self.declare_parameter('track_lost_ttl_seconds', 0.6)
        self.declare_parameter('localization_max_speed', 0.4)
        self.declare_parameter('localization_max_yaw_rate', 0.8)
        self.declare_parameter('localization_jump_position_margin', 0.15)
        self.declare_parameter('localization_jump_yaw_margin_deg', 5.0)
        self.declare_parameter('tile_size', 0.5)
        self.declare_parameter('tile_extent_threshold', 2.0)
        self.declare_parameter('tile_occupancy_threshold', 0.5)
        self.declare_parameter('max_scene_tiles', 512)
        campus_defaults = {
            'enhanced_association':True,'mahalanobis_gate':9.21,'association_unmatched_cost':.45,
            'dyn_confirm_seconds':.3,'static_confirm_seconds':1.2,
            'motion_observation_window':.5,
            'pedestrian_max_speed':.6,'vehicle_max_speed':1.0,'bicycle_max_speed':1.0,
            'ground_enabled':True,'ground_cell_size':2.0,'ground_min_points':12,'ground_max_fit_points':256,
            'ground_distance':.025,'ground_max_slope_deg':15.0,'ground_anchor_tolerance':.03,
            'sensor_roll':0.0,'sensor_pitch':0.0,
            'point_time_field':'','point_time_mode':'auto','point_time_header_is_end':True,
            'scan_period':.1,'centerpoint_enabled':True,'centerpoint_backend':'tensorrt',
            'lidar_frame_ids':['rslidar','lidar','laser'],'input_frame_type':'sensor','input_deskewed':False,
            'body_height':.17,'sensor_z':.28,'centerpoint_reference_height':1.8,'pose_exact_tolerance':.005,
            'centerpoint_model_directory':'','centerpoint_score_threshold':.35,'centerpoint_max_age':.25,
            'two_frame_fusion_enabled':True,'fusion_max_age':.15,'model_input_range':5.0,
        }
        for name,value in campus_defaults.items():
            self.declare_parameter(name,value)
        self._campus = {name:self.get_parameter(name).value for name in campus_defaults}
        if self._campus['input_frame_type'] not in ('body','sensor'):
            raise ValueError('input_frame_type must be body or sensor')
        for name in ('body_height','sensor_z','sensor_roll','sensor_pitch','centerpoint_reference_height','pose_exact_tolerance','scan_period'):
            if not math.isfinite(float(self._campus[name])):
                raise ValueError('Nonfinite perception parameter: '+name)
        if not 0<=self._campus['body_height']<=2.0 or any(self._campus[name]<=0 for name in
            ('centerpoint_reference_height','pose_exact_tolerance','scan_period')):
            raise ValueError('Invalid perception height or source timing')
        self._cloud_fusion = TwoFrameCloudFusion(float(self._campus['fusion_max_age']))
        self._fusion_generation = 0
        self._fusion_enabled = bool(self._campus['two_frame_fusion_enabled'])
        if not math.isfinite(float(self._campus['model_input_range'])) or self._campus['model_input_range'] <= 0:
            raise ValueError('Invalid model input range')
        self._pending_cloud = None
        self._points_topic = str(self.get_parameter('points_topic').value)
        self._odom_topic = str(self.get_parameter('odom_topic').value)
        self._obstacles_topic = str(self.get_parameter('obstacles_topic').value)
        # 雷达高度
        self._lidar_h = float(self.get_parameter('lidar_height').value)
        # 第一级直通滤波（base_link 系）
        self._pass_xmin = float(self.get_parameter('pass_x_min').value)
        self._pass_xmax = float(self.get_parameter('pass_x_max').value)
        self._pass_ymin = float(self.get_parameter('pass_y_min').value)
        self._pass_ymax = float(self.get_parameter('pass_y_max').value)
        self._pass_zmin = float(self.get_parameter('pass_z_min').value)
        self._pass_zmax = float(self.get_parameter('pass_z_max').value)
        # 车体多边形参数
        self._veh_len = float(self.get_parameter('vehicle_length').value)
        self._veh_wid = float(self.get_parameter('vehicle_width').value)
        self._veh_out = float(self.get_parameter('vehicle_outside').value)
        self._veh_hl = float(self.get_parameter("vehicle_half_length").value) + self._veh_out
        self._veh_hw = float(self.get_parameter("vehicle_half_width").value) + self._veh_out
        # front 区域
        self._front_xmin = float(self.get_parameter('front_x_min').value)
        self._front_xmax = float(self.get_parameter('front_x_max').value)
        self._body_to_base_x = float(self.get_parameter('body_to_base_link_x').value)
        self._loc_timeout = float(self.get_parameter('loc_timeout').value)
        self._max_range = float(self.get_parameter('max_obstacle_range').value)

        self._lidar_filter_cfg = {
            'lidar_height':self._lidar_h,
            'z_min':float(self.get_parameter('rslidar_z_min').value),
            'z_max':float(self.get_parameter('rslidar_z_max').value),
            'use_z_filter':bool(self.get_parameter('use_z_filter').value),
            'x_min':self._pass_xmin,'x_max':self._pass_xmax,
            'y_min':self._pass_ymin,'y_max':self._pass_ymax,
            'base_z_min':self._pass_zmin,'base_z_max':self._pass_zmax,
            'half_length':self._veh_hl,'half_width':self._veh_hw,'max_range':self._max_range,
            'vehicle_center_x':float(self.get_parameter('vehicle_center_x').value),
            'sensor_x':float(self.get_parameter('sensor_x').value),
            'sensor_y':float(self.get_parameter('sensor_y').value),
            'sensor_yaw':float(self.get_parameter('sensor_yaw').value)}

        self._clusterer = Clusterer(
            gate=float(self.get_parameter('cluster_gate').value),
            min_points=int(self.get_parameter('cluster_min_points').value),
            max_radius=float(self.get_parameter('cluster_max_radius').value),
        )
        self._tracker = GlobalKFTracker(
            motion_guard_enable=bool(self.get_parameter('motion_guard_enable').value),
            measurement_position_slack=float(self.get_parameter('measurement_position_slack').value),
            velocity_innovation_limit=float(self.get_parameter('velocity_innovation_limit').value),
            velocity_reconfirm_time=float(self.get_parameter('velocity_reconfirm_time').value),
            dt=float(self.get_parameter('track_dt').value),
            lost_ttl_seconds=float(self.get_parameter("track_lost_ttl_seconds").value),
            gate=float(self.get_parameter('track_gate').value),
            lost_ttl=int(self.get_parameter('track_lost_ttl').value),
            q=float(self.get_parameter('kf_q').value),
            r=float(self.get_parameter('kf_r').value),
            max_speed=float(self.get_parameter('obstacle_max_speed').value),
            dyn_speed_thr=float(self.get_parameter('dyn_speed_thr').value),
            dyn_hold_frames=int(self.get_parameter('dyn_hold_frames').value),
            static_speed_thr=float(self.get_parameter('static_speed_thr').value),
            static_hold_frames=int(self.get_parameter('static_hold_frames').value),
            static_small_thr=float(self.get_parameter('static_small_thr').value),
            enhanced_association=bool(self._campus['enhanced_association']),
            mahalanobis_gate=float(self._campus['mahalanobis_gate']),
            unmatched_cost=float(self._campus['association_unmatched_cost']),
            dyn_confirm_seconds=float(self._campus['dyn_confirm_seconds']),
            static_confirm_seconds=float(self._campus['static_confirm_seconds']),
            class_speed_limits={kind:float(self._campus['pedestrian_max_speed' if kind=='pedestrian' else 'bicycle_max_speed' if kind=='bicycle' else 'vehicle_max_speed']) for kind in (*MODEL_CLASSES,'vehicle')},
            preserve_semantics=True,
            motion_observation_window=float(self._campus['motion_observation_window']),
            velocity_max_age=float(self._campus['centerpoint_max_age']),
        )
        self._representer = WorldObstacleRepresenter(self._tracker,
            tile_size=float(self.get_parameter('tile_size').value),
            extent_threshold=float(self.get_parameter('tile_extent_threshold').value),
            occupancy_threshold=float(self.get_parameter('tile_occupancy_threshold').value),
            max_tiles=int(self.get_parameter('max_scene_tiles').value),
            half_length=self._veh_len/2,half_width=self._veh_wid/2)
        self._model_tracker = GlobalKFTracker(
            dt=self._tracker.dt,gate=self._tracker.gate,lost_ttl_seconds=self._tracker.lost_ttl_seconds,
            q=self._tracker.q,r=self._tracker.r,max_speed=self._tracker.max_speed,
            motion_guard_enable=self._tracker.motion_guard_enable,
            enhanced_association=self._tracker.enhanced_association,
            dyn_speed_thr=self._tracker.dyn_speed_thr,static_speed_thr=self._tracker.static_speed_thr,
            dyn_confirm_seconds=self._tracker.dyn_confirm_seconds,
            static_confirm_seconds=self._tracker.static_confirm_seconds,
            measurement_position_slack=self._tracker.measurement_position_slack,
            velocity_innovation_limit=self._tracker.velocity_innovation_limit,
            velocity_reconfirm_time=self._tracker.velocity_reconfirm_time,
            class_speed_limits=self._tracker.class_speed_limits,preserve_semantics=True,
            motion_observation_window=self._tracker.motion_observation_window,
            velocity_max_age=self._tracker.velocity_max_age,
            mahalanobis_gate=self._tracker.mahalanobis_gate,unmatched_cost=self._tracker.unmatched_cost)
        self._consumed_model_stamp = None

        # ---------------- 状态 ----------------
        self._loc = None            # (x, y, yaw) 最近定位
        self._full_poses = deque(maxlen=200)
        self._loc_t = 0.0  # last accepted source stamp
        self._pose_buffer = SourcePoseBuffer(self._loc_timeout,
            float(self.get_parameter('pose_sync_tolerance').value),
            max_speed=float(self.get_parameter('localization_max_speed').value),
            max_yaw_rate=float(self.get_parameter('localization_max_yaw_rate').value),
            position_margin=float(self.get_parameter('localization_jump_position_margin').value),
            yaw_margin=math.radians(float(self.get_parameter('localization_jump_yaw_margin_deg').value)))
        self._last_cloud_stamp = None
        self._loc_fault_logged = False
        self._representation_fault_logged = False
        self._cloud_timeout = float(self.get_parameter('cloud_timeout').value)
        self._extrinsics = {'x':self._lidar_filter_cfg['sensor_x'],'y':self._lidar_filter_cfg['sensor_y'],
            'height':float(self._campus['sensor_z']),'yaw':self._lidar_filter_cfg['sensor_yaw'],
            'roll':float(self._campus['sensor_roll']),'pitch':float(self._campus['sensor_pitch'])}
        self._ground = LocalGroundFilter(
            cell_size=self._campus['ground_cell_size'],min_points=self._campus['ground_min_points'],
            distance=self._campus['ground_distance'],max_slope_deg=self._campus['ground_max_slope_deg'],
            anchor_tolerance=self._campus['ground_anchor_tolerance'],max_fit_points=self._campus['ground_max_fit_points'])
        model_directory = self._campus['centerpoint_model_directory'] or str(
            Path(get_package_share_directory('campus_perception'))/'models'/'centerpoint'/'base')
        self._model = ModelWorker(model_directory,backend=self._campus['centerpoint_backend'],
            score_threshold=self._campus['centerpoint_score_threshold'],max_range=self._max_range,
            enabled=self._campus['centerpoint_enabled'])
        self._model_error_logged = ''

        # ---------------- 订阅 / 发布 ----------------
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST)
        self._sub_cloud = self.create_subscription(
            PointCloud2, self._points_topic, self._cb_cloud, qos)
        self._sub_odom = self.create_subscription(
            Odometry, self._odom_topic, self._cb_odom, qos)
        if ObstacleArray is not None:
            self._pub_obs = self.create_publisher(
                ObstacleArray, self._obstacles_topic, 10)
            self._pub_detections = self.create_publisher(
                ObstacleArray, self._output_topics['detections_topic'], 10)
            self._pub_model_detections = self.create_publisher(
                ObstacleArray, self._output_topics['model_detections_topic'], 10)
            if self._output_topics['tracks_topic'] == self._obstacles_topic:
                raise ValueError('tracks_topic must differ from the legacy planner topic')
            # The legacy planner and full-class inspection messages are separate.
            self._pub_tracks = (self._pub_obs if self._output_topics['tracks_topic'] == self._obstacles_topic
                else self.create_publisher(ObstacleArray, self._output_topics['tracks_topic'], 10))
        else:
            self._pub_obs = None
            self.get_logger().error(
                'rda_dwa_real_msgs 未找到（请先 source 原车机工作区的 install/setup.bash），'
                '/rda/obstacles 发布被禁用')
        self._pub_markers = self.create_publisher(
            MarkerArray, '/rda/obstacles_markers', 10)
        self._pub_detection_markers = self.create_publisher(
            MarkerArray, self._output_topics['detection_markers_topic'], 10)
        self._pub_tracking_markers = self.create_publisher(
            MarkerArray, self._output_topics['tracking_markers_topic'], 10)
        self._pub_points_map = self.create_publisher(
            PointCloud2, self._output_topics['points_map_topic'], qos)
        self._pub_fused_points = self.create_publisher(PointCloud2,self._output_topics['fused_points_topic'],qos)
        self._pub_perception_diag = self.create_publisher(String,'/rda/perception_status',10)

        self.get_logger().info(
            f'{node_name} 启动: points={self._points_topic} '
            f'odom={self._odom_topic} → {self._obstacles_topic} '
            f'cluster_gate={self._clusterer.gate} track_gate={self._tracker.gate}')

    # ==================================================================
    # 回调
    # ==================================================================
    def _cb_odom(self, msg: Odometry):
        p = msg.pose.pose
        quaternion = (p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w)
        if not all(math.isfinite(float(q)) for q in (*quaternion,p.position.z)) or abs(math.hypot(*quaternion)-1.0) > 0.025:
            return
        x_body = p.position.x
        y_body = p.position.y
        yaw_body = quat_to_yaw(p.orientation.x, p.orientation.y,
                               p.orientation.z, p.orientation.w)
        # User confirmed /localization and lidar are already at the geometric
        # center. Default planar offset is zero; retain the parameter for explicit
        # future calibration, without applying the obsolete -0.15m offset.
        x_base, y_base, yaw_base = body_pose_to_base_link(
            x_body, y_body, yaw_body, self._body_to_base_x)
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        now = self.get_clock().now().nanoseconds*1e-9
        pose = (x_base,y_base,yaw_base)
        if msg.header.frame_id.lstrip('/') != 'map':
            return
        if self._pose_buffer.add(stamp,pose,now):
            self._loc,self._loc_t = pose,stamp
            self._full_poses.append((stamp,pose_matrix((x_base,y_base,p.position.z),quaternion)))
            if self._pending_cloud is not None:
                pending = self._pending_cloud
                self._pending_cloud = None
                self._cb_cloud(pending)
        elif self._pose_buffer.fault and not self._loc_fault_logged:
            self.get_logger().error(f'Localization jump latched: {self._pose_buffer.last_rejected}; '
                'stop and restart after confirming global pose')
            self._loc_fault_logged = True

    def _cb_cloud(self, msg: PointCloud2):
        started = time.perf_counter()
        now = self.get_clock().now().nanoseconds*1e-9
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if (self._pub_obs is None or not self._pose_buffer.fresh(now) or stamp <= 0
                or now-stamp > self._cloud_timeout or stamp-now > 0.15):
            return
        if self._last_cloud_stamp is not None and stamp <= self._last_cloud_stamp:
            return
        if self._cloud_fusion.previous is not None and stamp <= self._cloud_fusion.previous[1]:
            return
        if msg.header.frame_id.lstrip('/') not in self._campus['lidar_frame_ids']:
            return
        pose_times = [item[0] for item in self._pose_buffer.poses]
        # Raw scan headers may be first/last point times. Interpolate a bracketed
        # source pose, or retain only the latest cloud until its pose arrives.
        bracketed = pose_times[0]<=stamp<=pose_times[-1]
        exact = min(abs(value-stamp) for value in pose_times)<=self._campus['pose_exact_tolerance']
        if not (bracketed or exact):
            self._pending_cloud = msg
            return
        # An exact measured endpoint remains usable after a gap. The buffer's
        # interpolation gap guard applies only when a pose must be interpolated.
        pose = (min(self._pose_buffer.poses, key=lambda item: abs(item[0]-stamp))[1]
                if exact else self._pose_buffer.at(stamp))
        if pose is None:
            return
        try:
            cloud = decode_cloud(msg,time_field=self._campus['point_time_field'],
                time_mode='disabled' if self._campus['input_deskewed'] else self._campus['point_time_mode'],header_is_end=self._campus['point_time_header_is_end'],
                scan_period=self._campus['scan_period'])
            if (cloud.point_times is not None and len(cloud.point_times) and
                cloud.point_times.max()>pose_times[-1]+self._campus['pose_exact_tolerance']):
                self._pending_cloud = msg
                return
            full_pose = interpolate_pose_matrix(list(self._full_poses),stamp) if self._full_poses else None
            world_base = full_pose if full_pose is not None else sensor_matrix(pose,height=0.)
            mounting = self._extrinsics if self._campus['input_frame_type']=='sensor' else {'height':0.}
            if self._campus['input_deskewed']:
                input_points,deskew_status = cloud.xyz,'upstream_compensated'
            else:
                input_points,deskew_status = deskew_cloud(cloud,stamp,list(self._pose_buffer.poses),mounting,list(self._full_poses))
            body_points = transform_points(input_points,sensor_matrix((0,0,0),**mounting))
            base_points = body_points+np.array([0.,0.,self._campus['body_height']])
        except (ValueError,TypeError,IndexError,KeyError,BufferError,struct.error) as exc:
            self.get_logger().error(f'Cloud decode rejected: {exc}')
            return
        self_mask = ((np.abs(base_points[:,0]-self._lidar_filter_cfg['vehicle_center_x'])<=self._veh_hl)&
                     (np.abs(base_points[:,1])<=self._veh_hw))
        radius = np.linalg.norm(base_points[:,:2],axis=1)
        crop = ((~self_mask)&(radius<=self._max_range)&(base_points[:,2]>=-1.)&(base_points[:,2]<=self._pass_zmax)&
                (base_points[:,0]>=self._pass_xmin)&(base_points[:,0]<=self._pass_xmax)&
                (base_points[:,1]>=self._pass_ymin)&(base_points[:,1]<=self._pass_ymax))
        if self._campus['ground_enabled']:
            selected = np.flatnonzero(crop)
            selected = selected[self._ground.nonground_mask(base_points[selected])]
            obstacle_body = body_points[selected]
        else:
            selected = crop
            # A ground-off comparison keeps road points; no hidden 15cm gate.
            obstacle_body = body_points[selected]
        rx,ry,_ = pose
        # World-fixed voxel and tile grids are essential: rotating the ego frame
        # must not alter static geometry centers or create apparent wall speed.
        world_xyz = transform_points(obstacle_body,world_base)
        world_points = world_xyz[:,:2]
        enabled = bool(self.get_parameter('two_frame_fusion_enabled').value)
        if enabled != self._fusion_enabled:
            self._fusion_enabled = enabled
            self._fusion_generation += 1
            self._model_tracker.reset()
            self._consumed_model_stamp = None
            # Discard semantic evidence from the other A/B condition, while
            # preserving genuine current geometry motion and occupancy IDs.
            for track in self._tracker.tracks.values():
                track.semantic_class,track.semantic_score = 'unknown',0.
                track._pending_class,track._pending_class_hits = '',0
                track.max_speed = self._tracker.max_speed
        boxes = self._model.current_boxes(stamp,float(self._campus['centerpoint_max_age']),generation=self._fusion_generation)
        boxes = nearby_boxes(boxes,(rx,ry),self._max_range)
        model_stamp = self._model.stats.get('source_stamp')
        new_model_source = (model_stamp is not None and 0<=stamp-model_stamp<=self._campus['centerpoint_max_age']
            and self._model.stats.get('fusion_generation',0)==self._fusion_generation
            and (self._consumed_model_stamp is None or model_stamp>self._consumed_model_stamp))
        if new_model_source:
            self._model_tracker.step(model_detections(boxes),stamp=model_stamp,now=model_stamp)
            self._consumed_model_stamp = model_stamp
        if self._model.error and self._model.error != self._model_error_logged:
            self.get_logger().error('CenterPoint unavailable; geometric occupancy remains active: '+self._model.error)
            self._model_error_logged = self._model.error
        world_clusters = semantic_clusters(world_points,boxes,self._clusterer,stamp,
            max_age=self._campus['centerpoint_max_age'],min_points=self._clusterer.min_points,
            speed_limits=self._tracker.class_speed_limits,origin=(rx,ry))
        # The model retains road points and sees the latest two accepted raw
        # frames. Geometric occupancy uses current returns to avoid actor ghosts.
        model_source = body_points[~self_mask]
        fused_body = self._cloud_fusion.fuse(model_source,stamp,world_base,
            enabled=enabled,radius=float(self._campus['model_input_range']))
        fusion_self = ((np.abs(fused_body[:,0]-self._lidar_filter_cfg['vehicle_center_x'])<=self._veh_hl)&
                       (np.abs(fused_body[:,1])<=self._veh_hw))
        fused_body = fused_body[~fusion_self]
        self._cloud_fusion.stats['current_points'] = int(np.count_nonzero(fused_body[:,3]==0))
        self._cloud_fusion.stats['history_points'] = int(np.count_nonzero(fused_body[:,3]>0))
        model_xyz,model_world = model_frame_from_body(fused_body[:,:3],world_base,
            body_height=self._campus['body_height'],reference_height=self._campus['centerpoint_reference_height'])
        model_xyzt = np.column_stack((model_xyz,fused_body[:,3]))
        self._model.submit(model_xyzt,stamp,model_world,metadata={
            'fusion_generation':self._fusion_generation,'input_fusion_enabled':enabled,
            'input_current_points':self._cloud_fusion.stats['current_points'],
            'input_history_points':self._cloud_fusion.stats['history_points'],
            'input_previous_source_stamp':self._cloud_fusion.stats['previous_source_stamp']})
        try:
            tracks = self._representer.step(world_clusters,stamp,reference_origin=(rx,ry),vehicle_pose=pose)
        except RepresentationLimitError as exc:
            if not self._representation_fault_logged:
                self.get_logger().error(f'Perception representation overload: {exc}')
                self._representation_fault_logged = True
            return  # Explicit fail-closed input timeout, never fill a U cavity.
        self._representation_fault_logged = False
        model_entities = self._model_tracker.snapshot(stamp) if self._consumed_model_stamp is not None else []
        tracks = fuse_coverage(tracks,model_entities)
        if len(tracks)>self._representer.max_tiles:
            self.get_logger().error(f'Combined perception overload: {len(tracks)} obstacle parts')
            return
        publication_delay = self.get_clock().now().nanoseconds*1e-9-stamp
        if publication_delay>self._cloud_timeout or publication_delay<-.15:
            return
        measurement_ages = {out_id:stamp-self._tracker.tracks[pid].last_seen
            for (pid,_),out_id in self._representer._ids.items() if pid in self._tracker.tracks}
        measurement_ages.update({MODEL_ID_OFFSET+tid:stamp-tr.last_seen for tid,tr in self._model_tracker.tracks.items()})
        tracks = expand_publication_delay(tracks,publication_delay,measurement_ages,
            self._tracker.class_speed_limits,self._tracker.velocity_max_age)
        tracks = nearby_obstacles(tracks,(rx,ry),self._max_range)
        self._last_cloud_stamp = stamp
        # Current raw detections have no cross-frame identity or measured velocity.
        detections = nearby_obstacles(self._geometry_detections(world_clusters),(rx,ry),self._max_range)
        self._pub_detections.publish(self._obstacle_array(detections, msg.header.stamp))
        self._pub_detection_markers.publish(self._markers(detections, msg.header.stamp, detected=True))
        self._pub_points_map.publish(self._point_cloud(world_xyz, msg.header.stamp))
        # This cloud is explicitly in the current body frame and contains age.
        # RViz can use the existing map->body TF; points_map needs no new TF.
        self._pub_fused_points.publish(self._fused_cloud(fused_body,msg.header.stamp))
        if new_model_source:
            # Neural output is asynchronous. Never relabel it with this cloud's time.
            model_time = type(msg.header.stamp)()
            model_ns = round(float(model_stamp)*1e9)
            model_time.sec, model_time.nanosec = divmod(model_ns, 1_000_000_000)
            neural_detections = [(-1,x,y,-1.,0.,kind,L,W)
                                 for x,y,kind,L,W,_ in model_detections(boxes)]
            self._pub_model_detections.publish(self._obstacle_array(neural_detections, model_time))
        self._publish_obstacles(tracks,msg.header.stamp)
        status = String()
        status.data = json.dumps({'source_stamp':stamp,
            'source_age_seconds':self.get_clock().now().nanoseconds*1e-9-stamp,
            'source_age_on_arrival_seconds':now-stamp,'input_topic':self._points_topic,
            'processing_seconds':time.perf_counter()-started,'deskew':deskew_status,
            'point_time_status':cloud.time_status,'input_frame':msg.header.frame_id,
            'input_frame_type':self._campus['input_frame_type'],
            'raw_point_count':len(cloud.xyz),'nonground_point_count':len(world_xyz),
            'detection_count':len(detections),'track_count':len(tracks),
            'model_classes':list(MODEL_CLASSES),'max_obstacle_range':self._max_range,
            'model_input_range':self._campus['model_input_range'],'fusion':dict(self._cloud_fusion.stats),
            'geometry_source':'current_frame_only',
            'ground':self._ground.stats,'model':dict(self._model.stats),'model_error':self._model.error,
            'model_result_age_seconds':stamp-self._model.stats['source_stamp'] if 'source_stamp' in self._model.stats else None,
            'model_regions_used':len(boxes),'association':self._tracker.stats,
            'model_association':self._model_tracker.stats,
            'model_parents':[{'id':MODEL_ID_OFFSET+tr.id,'semantic_class':tr.semantic_class,
                'velocity_trusted':tr.velocity_trusted and stamp-tr.last_seen+max(publication_delay,0.)<=self._tracker.velocity_max_age,'lifecycle':tr.lifecycle,
                'measurement_age':stamp-tr.last_seen+max(publication_delay,0.)} for tr in self._model_tracker.tracks.values()],
            'parents':[{'id':tr.id,'semantic_class':tr.semantic_class,'score':tr.semantic_score,
                'motion_state':'static' if tr.static_confirmed else 'moving' if tr.moving else 'unconfirmed',
                'velocity_trusted':tr.velocity_trusted and stamp-tr.last_seen+max(publication_delay,0.)<=self._tracker.velocity_max_age,'lifecycle':tr.lifecycle,
                'measurement_age':stamp-tr.last_seen+max(publication_delay,0.)} for tr in self._tracker.tracks.values()]},allow_nan=False)
        self._pub_perception_diag.publish(status)

    def destroy_node(self):
        self._model.close()
        return super().destroy_node()

    @staticmethod
    def _geometry_detections(clusters):
        """Map AABBs from only this source cloud, before KF smoothing/tiling."""
        result = []
        for cluster in clusters:
            xs, ys = zip(*cluster)
            lo_x, hi_x, lo_y, hi_y = (getattr(cluster, 'raw_bounds', None)
                                     or (min(xs), max(xs), min(ys), max(ys)))
            margin = float(getattr(cluster, 'voxel_margin', 0.))
            result.append((-1, (lo_x+hi_x)/2, (lo_y+hi_y)/2, -1., 0.,
                           getattr(cluster, 'semantic_class', 'unknown'),
                           max(hi_x-lo_x+2*margin, .05), max(hi_y-lo_y+2*margin, .05)))
        return result

    @staticmethod
    def _obstacle_array(items, source_stamp):
        msg = ObstacleArray()
        msg.header.stamp, msg.header.frame_id = source_stamp, 'map'
        for tid, x, y, speed, heading, kind, length, width in items:
            ob = Obstacle()
            ob.id, ob.type = int(tid), str(kind)
            ob.x, ob.y = float(x), float(y)
            ob.length, ob.width = float(length), float(width)
            # heading is velocity direction; dimensions always describe map AABBs.
            ob.speed, ob.heading = float(speed), float(heading)
            # Some existing car interfaces separate footprint yaw from motion yaw.
            # Our dimensions are map AABBs, so the footprint yaw is always zero.
            if hasattr(ob, 'has_shape_heading') and hasattr(ob, 'shape_heading'):
                ob.has_shape_heading, ob.shape_heading = True, 0.
            msg.obstacles.append(ob)
        return msg

    @staticmethod
    def _point_cloud(world_xyz, source_stamp):
        """Publish filtered physical XYZ in map, without requiring a sensor TF."""
        points = np.asarray(world_xyz, dtype='<f4').reshape(-1, 3)
        msg = PointCloud2()
        msg.header.stamp, msg.header.frame_id = source_stamp, 'map'
        msg.height, msg.width = 1, len(points)
        msg.fields = [PointField(name=name, offset=4*i, datatype=PointField.FLOAT32, count=1)
                      for i, name in enumerate(('x', 'y', 'z'))]
        msg.is_bigendian, msg.is_dense = False, True
        msg.point_step, msg.row_step = 12, 12*len(points)
        msg.data = points.tobytes(order='C')
        return msg

    @staticmethod
    def _fused_cloud(xyzt, source_stamp):
        points = np.asarray(xyzt,dtype='<f4').reshape(-1,4)
        msg = PointCloud2()
        msg.header.stamp,msg.header.frame_id = source_stamp,'body'
        msg.height,msg.width = 1,len(points)
        msg.fields = [PointField(name=name,offset=4*i,datatype=PointField.FLOAT32,count=1)
                      for i,name in enumerate(('x','y','z','time_lag'))]
        msg.is_bigendian,msg.is_dense = False,True
        msg.point_step,msg.row_step,msg.data = 16,16*len(points),points.tobytes()
        return msg

    def _markers(self, items, source_stamp, detected=False):
        """Detection outlines, tracked footprints, labels and trusted velocity arrows.

        Markers expire after the input timeout, including if input stops. Rendered
        z/height is a schematic 2D footprint, not a measured object height.
        """
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id, clear.header.stamp = 'map', source_stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        prefix = 'detections' if detected else 'tracks'

        def marker(tid, namespace, marker_type):
            m = Marker()
            m.header.frame_id, m.header.stamp = 'map', source_stamp
            m.ns, m.id, m.type, m.action = namespace, int(tid), marker_type, Marker.ADD
            m.pose.orientation.w = 1.
            duration_ns = round(max(self._cloud_timeout, .1)*1e9)
            m.lifetime.sec, m.lifetime.nanosec = divmod(duration_ns, 1_000_000_000)
            return m

        for index, (tid, x, y, speed, heading, kind, length, width) in enumerate(items):
            mid = index if detected else tid
            colour = ((.15, .7, 1.) if detected else
                      (1., .75, .1) if kind == 'unknown' or speed < 0 else
                      (1., .2, .2) if speed > .06 else (.2, 1., .2))
            box = marker(mid, prefix+'_boxes', Marker.LINE_STRIP if detected else Marker.CUBE)
            if detected:
                box.scale.x = .025
                corners = [(x-length/2, y-width/2), (x+length/2, y-width/2),
                           (x+length/2, y+width/2), (x-length/2, y+width/2),
                           (x-length/2, y-width/2)]
                box.points = [Point(x=float(px), y=float(py), z=.12) for px, py in corners]
            else:
                box.pose.position.x, box.pose.position.y = float(x), float(y)
                box.pose.position.z = .05
                # A velocity heading must never rotate an axis-aligned collision box.
                box.scale.x, box.scale.y, box.scale.z = float(length), float(width), .1
            box.color.r, box.color.g, box.color.b = colour
            box.color.a = .9 if detected else .45
            markers.markers.append(box)

            label = marker(mid, prefix+'_labels', Marker.TEXT_VIEW_FACING)
            label.pose.position.x, label.pose.position.y = float(x), float(y)
            label.pose.position.z = .5 if detected else .3
            label.scale.z = .18 if detected else .25
            label.color.r, label.color.g, label.color.b, label.color.a = 1., 1., 1., 1.
            label.text = (f'det:{index} {kind}' if detected else
                          f'id:{tid} {kind}  v:'+('unknown' if speed<0 else f'{speed:.2f}m/s'))
            markers.markers.append(label)
            if not detected and kind != 'unknown' and speed > .06:
                arrow = marker(mid, prefix+'_velocity', Marker.ARROW)
                arrow.points = [Point(x=float(x), y=float(y), z=.2),
                                Point(x=float(x+speed*math.cos(heading)),
                                      y=float(y+speed*math.sin(heading)), z=.2)]
                arrow.scale.x, arrow.scale.y, arrow.scale.z = .04, .08, .08
                arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1., .3, .2, 1.
                markers.markers.append(arrow)
        return markers

    def _publish_obstacles(self, tracks, source_stamp):
        """Identical tracked messages feed inspection and the existing planner."""
        msg = self._obstacle_array(tracks, source_stamp)
        self._pub_obs.publish(self._obstacle_array(planner_obstacles(tracks),source_stamp))
        if self._pub_tracks is not self._pub_obs:
            self._pub_tracks.publish(msg)
        markers = self._markers(tracks, source_stamp)
        self._pub_tracking_markers.publish(markers)
        self._pub_markers.publish(markers)  # old RViz topic remains compatible


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
