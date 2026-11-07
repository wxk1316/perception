"""Instantiate active ROS adapters with an in-memory ROS test double.

This checks constructor/configuration contracts only. It does not start ROS,
send DDS messages, open UDP sockets, or prove ROS middleware compatibility.
Run with Python + numpy + PyYAML; source packages are added from this workspace.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'campus_perception')]
CONFIG = yaml.safe_load((ROOT/'campus_perception/config/perception_tracking.yaml').read_text())


def module(name):
    result = types.ModuleType(name)
    sys.modules[name] = result
    return result


class Message:
    def __init__(self, **kwargs):
        self.header = types.SimpleNamespace(
            stamp=types.SimpleNamespace(sec=1000, nanosec=0), frame_id='map')
        self.data = None
        self.poses = []
        self.obstacles = []
        self.markers = []
        self.points = []
        self.scale = types.SimpleNamespace(x=0., y=0., z=0.)
        self.color = types.SimpleNamespace(r=0., g=0., b=0., a=0.)
        self.lifetime = types.SimpleNamespace(sec=0, nanosec=0)
        self.pose = types.SimpleNamespace(
            position=types.SimpleNamespace(x=0., y=0., z=0.),
            orientation=types.SimpleNamespace(x=0., y=0., z=0., w=1.))
        self.twist = types.SimpleNamespace(
            linear=types.SimpleNamespace(x=0., y=0., z=0.),
            angular=types.SimpleNamespace(x=0., y=0., z=0.))
        self.__dict__.update(kwargs)


class Clock:
    def now(self):
        return types.SimpleNamespace(
            nanoseconds=1000_000_000_000,
            to_msg=lambda: types.SimpleNamespace(sec=1000, nanosec=0))


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class Node:
    def __init__(self, name):
        self.name = name
        self.overrides = CONFIG.get(name, {}).get('ros__parameters', {})
        self.declared = {}
        self.read = set()
        self.subscriptions = []
        self.publishers = {}

    def declare_parameter(self, name, default=None):
        value = self.overrides.get(name, default)
        if default is not None and type(value) is not type(default):
            raise TypeError(f'{self.name}.{name}: YAML {type(value).__name__}, '
                            f'declaration {type(default).__name__}')
        self.declared[name] = value
        return types.SimpleNamespace(value=value)

    def get_parameter(self, name):
        self.read.add(name)
        return types.SimpleNamespace(value=self.declared[name])

    def create_subscription(self, msg_type, topic, callback, qos, **kwargs):
        self.subscriptions.append(topic)
        return types.SimpleNamespace(callback=callback)

    def create_publisher(self, msg_type, topic, qos, **kwargs):
        result = Publisher()
        self.publishers[topic] = result
        return result

    def create_timer(self, *args, **kwargs):
        return None

    def get_clock(self):
        return Clock()

    def get_logger(self):
        return types.SimpleNamespace(**{key: lambda *args, **kwargs: None
                                       for key in ('info', 'warning', 'warn', 'error', 'debug')})


def install_doubles():
    ament = module('ament_index_python')
    ament.packages = module('ament_index_python.packages')
    ament.packages.get_package_share_directory = lambda package:str(ROOT/package)
    ros = module('rclpy')
    ros.init = ros.shutdown = ros.spin = lambda *args, **kwargs: None
    module('rclpy.node').Node = Node
    qos = module('rclpy.qos')
    qos.QoSProfile = lambda *args, **kwargs: types.SimpleNamespace(**kwargs)
    for name in ('DurabilityPolicy', 'ReliabilityPolicy', 'HistoryPolicy'):
        setattr(qos, name, types.SimpleNamespace(
            RELIABLE=1, BEST_EFFORT=2, VOLATILE=1, TRANSIENT_LOCAL=2, KEEP_LAST=1))
    qos.qos_profile_sensor_data = types.SimpleNamespace()
    callback = module('rclpy.callback_groups')
    callback.MutuallyExclusiveCallbackGroup = callback.ReentrantCallbackGroup = object
    module('rclpy.executors').MultiThreadedExecutor = object
    groups = {
        'std_msgs': ('Float32MultiArray', 'String', 'Float32', 'Int32', 'Bool', 'Header'),
        'sensor_msgs': ('LaserScan', 'NavSatFix', 'PointCloud2', 'PointField'),
        'geometry_msgs': ('Point', 'Twist', 'TwistStamped', 'PoseStamped', 'PoseWithCovarianceStamped'),
        'nav_msgs': ('Odometry', 'Path'),
        'control_msgs': ('CarState', 'CarCtr'),
        'rda_dwa_real_msgs': ('ObstacleArray', 'Obstacle'),
        'visualization_msgs': ('Marker', 'MarkerArray'),
    }
    for package, names in groups.items():
        parent = module(package)
        child = module(f'{package}.msg')
        parent.msg = child
        for name in names:
            cls = type(name, (Message,), {'ADD': 0, 'DELETEALL': 3, 'CUBE': 1,
                       'TEXT_VIEW_FACING': 9, 'ARROW': 0, 'LINE_STRIP': 4, 'FLOAT32': 7})
            setattr(child, name, cls)
    sensor = module('sensor_msgs_py')
    sensor.point_cloud2 = module('sensor_msgs_py.point_cloud2')
    sensor.point_cloud2.read_points = lambda *args, **kwargs: []

