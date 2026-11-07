"""
frame_utils.py — 实车适配层坐标系工具（纯 Python，无 ROS 依赖，可单测）。

坐标系约定
==========
算法内部坐标系（= EnhancedDWA 消费的 world 系）:
  - x 前、y 左、θ 逆时针为正（与仿真 Scene world 系一致）
  - robot.x/y/theta 为全局位姿；obstacle.x/y 为全局位置；
  - goal 为全局坐标；输出 (v_cmd, ω_cmd) 为车体纵向线速度 + 绕车体
    几何中心的角速度。

外部真实话题可能的坐标系:
  - 'map'    : 全局静态地图系（AMCL/定位话题常见）
  - 'odom'   : 里程计系（/odom 常见）
  - 车体系    : 'base_link' / 'base_footprint' / 'laser_frame' / 'lidar' 等
    （雷达/点云/自定义障碍物检测常见，消息 header.frame_id 指明）
本模块提供:
  - quaternion → yaw（ROS 四元数 x y z w 顺序）
  - yaw → quaternion
  - body_to_world_2d : 车体系 → 算法系（用定位位姿 (x,y,θ) 做 2D 刚体变换）
  - world_to_body_2d : 逆变换（目标点取车身系等场景）
  - is_body_frame    : 判断 frame 名是否属于车体系（参数化）
  传感器外参（sensor→base_link 的固定平移/旋转）以参数提供：
  sensor_x/sensor_y/sensor_yaw（雷达装在车体上相对几何中心的位姿）。
"""
from __future__ import annotations

import math
from collections import deque


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """ROS geometry_msgs/Quaternion → 绕 Z 轴偏航角（rad, [-π, π)）。"""
    # 归一化防止坏四元数
    n = math.hypot(x, y, z, w)
    if n < 1e-12:
        return 0.0
    x, y, z, w = x / n, y / n, z / n, w / n
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    # 归一化到 [-π, π)
    if yaw > math.pi:
        yaw -= 2.0 * math.pi
    elif yaw < -math.pi:
        yaw += 2.0 * math.pi
    return yaw


def yaw_to_quat(yaw: float):
    """偏航角 → (x, y, z, w) 四元数（绕 Z）。"""
    y = yaw * 0.5
    return (0.0, 0.0, math.sin(y), math.cos(y))


def wrap_angle(a: float) -> float:
    """归一化到 [-π, π)。"""
    return (a + math.pi) - 2.0 * math.pi * math.floor((a + math.pi) / (2.0 * math.pi)) - math.pi


def is_body_frame(frame_id: str, body_frames: list[str]) -> bool:
    """frame_id 是否属于车体系/传感器系（需要按定位位姿转换）。

    匹配规则：与 body_frames 任一项相等，或 frame_id 以该前缀开头
    （如 'laser_frame_2' → 'laser' 命中）。
    """
    f = (frame_id or '').strip().lower()
    if not f:
        return False
    for bf in body_frames:
        b = bf.strip().lower()
        if not b:
            continue
        if f == b or f.startswith(b + '_') or f.startswith(b + '/'):
            return True
    return False


def body_to_world_2d(x_b: float, y_b: float, heading_b: float,
                     rx: float, ry: float, rtheta: float):
    """车体系 (x_b, y_b) → 算法系（map/odom）。

    定位位姿 (rx, ry, rtheta) 为 "算法系 → 车体系" 的逆变换参数：
        p_world = R(rtheta) · p_body + t_world
    heading_world = wrap(heading_body + rtheta)
    返回 (x_w, y_w, heading_world)。
    """
    c, s = math.cos(rtheta), math.sin(rtheta)
    xw = rx + x_b * c - y_b * s
    yw = ry + x_b * s + y_b * c
    hw = wrap_angle(heading_b + rtheta)
    return xw, yw, hw


def world_to_body_2d(x_w: float, y_w: float, rx: float, ry: float, rtheta: float):
    """算法系 → 车体系（目标点取车身系、控制回显等）。

    p_body = R(rtheta)^T · (p_world - t_world)。返回 (x_b, y_b)。
    """
    c, s = math.cos(rtheta), math.sin(rtheta)
    dx, dy = x_w - rx, y_w - ry
    return dx * c + dy * s, -dx * s + dy * c


class SensorExtrinsics:
    """传感器（雷达）相对车体几何中心的外参：平移 (x, y) + 偏航 yaw（rad）。

    - 若激光 frame 与 base_link 重合（常见）,全 0 即可。
    - 若激光装在车头前方 0.3m、左偏 0.05m、水平朝向与车头一致:
      SensorExtrinsics(0.3, 0.05, 0.0)
    正方向约定：x 前正、y 左正、yaw 逆时针为正（与算法系一致）。
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)

    def sensor_to_body(self, x_s: float, y_s: float, heading_s: float):
        """传感器系 → 车体系（先转 yaw，再平移）。"""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        xb = self.x + x_s * c - y_s * s
        yb = self.y + x_s * s + y_s * c
        hb = wrap_angle(heading_s + self.yaw)
        return xb, yb, hb

    def body_to_sensor(self, x_b: float, y_b: float, heading_b: float):
        """车体系 → 传感器系（逆变换）。"""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = x_b - self.x, y_b - self.y
        xs = dx * c + dy * s
        ys = -dx * s + dy * c
        hs = wrap_angle(heading_b - self.yaw)
        return xs, ys, hs

    def to_map(self, x_s: float, y_s: float, heading_s: float,
               rx: float, ry: float, rtheta: float):
        """传感器系 → 算法系（map/odom）：sensor→body→world 两级级联。

        返回 (x_w, y_w, heading_world)。
        """
        xb, yb, hb = self.sensor_to_body(x_s, y_s, heading_s)
        return body_to_world_2d(xb, yb, hb, rx, ry, rtheta)


def body_pose_to_base_link(x_body: float, y_body: float, yaw_body: float,
                           body_to_base_x: float):
    """Apply only an explicitly configured planar localization-reference offset.

    User confirmed current /localization, base_link and lidar x/y origins are
    already the vehicle geometric center, so the deployed offset is 0.0 and this
    function is an identity. A nonzero value is reserved for an explicitly
    calibrated future frame, with the corresponding static TF changed together:
        x_base = x_body + body_to_base_x * cos(yaw)
        y_base = y_body + body_to_base_x * sin(yaw)
    Yaw is unchanged.
    """
    x_base = x_body + body_to_base_x * math.cos(yaw_body)
    y_base = y_body + body_to_base_x * math.sin(yaw_body)
    return x_base, y_base, yaw_body


class SourcePoseBuffer:
    """Keep source-stamped poses; duplicate delivery does not renew freshness."""

    def __init__(self, max_age=0.5, sync_tolerance=0.15, history=200,
                 max_speed=0.4, max_yaw_rate=0.8, position_margin=0.15,
                 yaw_margin=math.radians(5.0)):
        self.max_age = float(max_age)
        self.sync_tolerance = float(sync_tolerance)
        self.poses = deque(maxlen=history)
        self.max_speed, self.max_yaw_rate = float(max_speed), float(max_yaw_rate)
        self.position_margin, self.yaw_margin = float(position_margin), float(yaw_margin)
        self.fault = ''
        self.last_rejected = None

    def add(self, stamp, pose, now):
        if self.fault:
            return False
        stamp, now = float(stamp), float(now)
        if not all(math.isfinite(v) for v in (stamp, now, *pose)):
            return False
        if stamp <= 0 or now-stamp > self.max_age or stamp-now > self.sync_tolerance:
            return False
        if self.poses and stamp <= self.poses[-1][0]:
            return False
        if self.poses:
            old_stamp, old_pose = self.poses[-1]
            dt = stamp-old_stamp
            bounded_dt = min(dt,self.max_age)
            displacement = math.hypot(pose[0]-old_pose[0],pose[1]-old_pose[1])
            yaw_delta = abs(wrap_angle(pose[2]-old_pose[2]))
            if (displacement > self.max_speed*bounded_dt+self.position_margin
                    or yaw_delta > self.max_yaw_rate*bounded_dt+self.yaw_margin):
                # A map reset cannot become fresh continuous feedback. Hold
                # invalid until an explicit system reinitialization/restart.
                self.fault = 'localization_jump'
                self.last_rejected = {'stamp':stamp,'source_dt':dt,
                    'displacement':displacement,'yaw_delta':yaw_delta,
                    'position_limit':self.max_speed*bounded_dt+self.position_margin,
                    'yaw_limit':self.max_yaw_rate*bounded_dt+self.yaw_margin}
                return False
        self.poses.append((stamp, tuple(pose)))
        return True

    def fresh(self, now):
        return bool(not self.fault and self.poses and 0 <= float(now)-self.poses[-1][0] <= self.max_age)

    def at(self, stamp):
        """Interpolate position and shortest yaw at cloud acquisition time."""
        if not self.poses:
            return None
        for i in range(1, len(self.poses)):
            ta, pa = self.poses[i-1]
            tb, pb = self.poses[i]
            if ta <= stamp <= tb:
                if tb-ta > 2*self.sync_tolerance:
                    return None
                u = (stamp-ta)/(tb-ta)
                return (pa[0]+u*(pb[0]-pa[0]), pa[1]+u*(pb[1]-pa[1]),
                        wrap_angle(pa[2]+u*wrap_angle(pb[2]-pa[2])))
        t, pose = min(self.poses, key=lambda item: abs(item[0]-stamp))
        return pose if abs(t-stamp) <= self.sync_tolerance else None


def oriented_box_aabb(length, width, yaw):
    """Map-aligned dimensions enclosing an oriented box at the same center."""
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return c*length+s*width, s*length+c*width


def filter_lidar_points(points, *, lidar_height=0.45, z_min=-0.3, z_max=1.5,
                        use_z_filter=True, x_min=-10.0, x_max=10.0,
                        y_min=-5.0, y_max=5.0, base_z_min=0.15, base_z_max=1.95,
                        half_length=0.6, half_width=0.4, max_range=5.0,
                        vehicle_center_x=0.0,
                        sensor_x=0.0, sensor_y=0.0, sensor_yaw=0.0):
    """Full-surround lidar crop with the installed +height TF convention."""
    result = []
    c, s = math.cos(sensor_yaw), math.sin(sensor_yaw)
    for point in points:
        # sensor_msgs_py may yield numpy.void structured records (no slicing).
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        if not all(math.isfinite(v) for v in (x,y,z)):
            continue
        if use_z_filter and not z_min <= z <= z_max:
            continue
        # base_link->rslidar is +height: a lidar point maps by adding height.
        z_base = z+lidar_height
        x,y = sensor_x+c*x-s*y, sensor_y+s*x+c*y
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue
        if use_z_filter and not base_z_min <= z_base <= base_z_max:
            continue
        if abs(x-vehicle_center_x) <= half_length and abs(y) <= half_width:
            continue
        if math.hypot(x,y) <= max_range:
            result.append((x,y))
    return result
