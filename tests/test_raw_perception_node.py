"""Exercise source-time ROS contracts with test doubles, never DDS/vehicle IO."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def adapters():
    previous = dict(sys.modules)
    spec = importlib.util.spec_from_file_location('raw_adapter_double', ROOT/'tests/ros_doubles.py')
    double = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(double)
    double.install_doubles()
    for name in ('perception_tracking_node',):
        double.CONFIG[name]['ros__parameters']['centerpoint_enabled'] = False
    from campus_perception.perception_tracking_node import PerceptionTrackingNode
    yield NS(Node=PerceptionTrackingNode, double=double)
    for key in list(sys.modules):
        if key not in previous:
            sys.modules.pop(key, None)
        elif sys.modules[key] is not previous[key]:
            sys.modules[key] = previous[key]


def stamp(t):
    sec, nanosec = divmod(round(t*1e9), 1_000_000_000)
    return NS(sec=sec, nanosec=nanosec)


def now(node, t):
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=round(t*1e9)))


def odom(t, x=0., y=0., yaw=0., z=0.):
    return NS(header=NS(stamp=stamp(t), frame_id='map'), child_frame_id='body',
              pose=NS(pose=NS(position=NS(x=x,y=y,z=z),
                              orientation=NS(x=0.,y=0.,z=math.sin(yaw/2),w=math.cos(yaw/2)))))


def cloud(t=1000., points=((-1.,.5,-.4),(-1.,.55,-.4)), frame='rslidar', times=None):
    fields = [NS(name=n, offset=4*i, datatype=7, count=1) for i,n in enumerate(('x','y','z'))]
    if times is None:
        step = 12
        data = b''.join(struct.pack('<fff', *p) for p in points)
    else:
        step = 20
        fields.append(NS(name='timestamp',offset=12,datatype=8,count=1))
        data = b''.join(struct.pack('<fffd', *p, t_point) for p,t_point in zip(points,times))
    return NS(header=NS(stamp=stamp(t),frame_id=frame),fields=fields,point_step=step,
              row_step=step*len(points),width=len(points),height=1,is_bigendian=False,data=data)


def node_at(adapters, t=1000., x=0., yaw=0.):
    node = adapters.Node()
    now(node,t)
    node._cb_odom(odom(t,x=x,yaw=yaw))
    return node


def test_independent_package_uses_raw_input_without_original_nodes(adapters):
    standalone = adapters.Node()
    assert standalone.subscriptions == ['/rslidar_points','/localization']
    assert standalone._campus['input_frame_type'] == 'sensor'
    assert standalone._campus['input_deskewed'] is False
    assert set(standalone.overrides) <= set(standalone.declared)
    assert set(standalone.overrides) <= standalone.read
    assert '/localization' not in standalone.publishers
    assert not any(t in standalone.publishers for t in ('/Odometry','/cmd_vel','/car_cmd'))
    config = yaml.safe_load((ROOT/'campus_perception/config/perception_tracking.yaml').read_text())
    assert set(config) == {'perception_tracking_node'}


def test_raw_rear_low_obstacle_detection_tracks_and_map_cloud(adapters):
    node = node_at(adapters, x=4., yaw=math.pi/2)
    node._cb_cloud(cloud())
    detected = node._pub_detections.messages[-1]
    tracked = node._pub_tracks.messages[-1]
    assert detected.header.frame_id == tracked.header.frame_id == 'map'
    assert detected.obstacles[0].id == -1 and detected.obstacles[0].speed == -1.
    assert tracked.obstacles[0].id > 0
    assert node._pub_obs.messages[-1].obstacles[0].id == tracked.obstacles[0].id
    points = node._pub_points_map.messages[-1]
    xyz = np.frombuffer(points.data,dtype='<f4').reshape(-1,3)
    assert points.width == 2 and points.row_step == 24 and points.point_step == 12
    assert all(f.datatype == 7 for f in points.fields)
    np.testing.assert_allclose(xyz,[[3.5,-1.,-.12],[3.45,-1.,-.12]],atol=1e-6)
    assert detected.header.stamp.sec == points.header.stamp.sec == 1000
    status = json.loads(node._pub_perception_diag.messages[-1].data)
    assert status['deskew'] == 'missing' and status['nonground_point_count'] == 2


def test_empty_source_has_empty_detections_but_holds_ids_then_expires(adapters):
    node = node_at(adapters)
    node._cb_cloud(cloud())
    tid = node._pub_obs.messages[-1].obstacles[0].id
    for dt in (.1,.7):
        now(node,1000+dt)
        node._cb_odom(odom(1000+dt))
        node._cb_cloud(cloud(1000+dt,points=()))
        assert node._pub_detections.messages[-1].obstacles == []
        assert node._pub_points_map.messages[-1].data == b''
        if dt < .6:
            assert node._pub_tracks.messages[-1].obstacles[0].id == tid
        else:
            assert node._pub_tracks.messages[-1].obstacles == []


@pytest.mark.parametrize('failure',['duplicate','frame','truncated','stale','future','no_pose'])
def test_invalid_cloud_does_not_refresh_any_output(adapters,failure):
    node = node_at(adapters)
    node._cb_cloud(cloud())
    counts = {topic:len(pub.messages) for topic,pub in node.publishers.items()}
    msg = cloud(1000.1)
    now(node,1000.1)
    node._cb_odom(odom(1000.1))
    if failure == 'duplicate': msg = cloud()
    elif failure == 'frame': msg.header.frame_id = 'body'
    elif failure == 'truncated': msg.data = b'bad'
    elif failure == 'stale': now(node,1001.)
    elif failure == 'future': msg.header.stamp = stamp(1001.)
    elif failure == 'no_pose': msg.header.stamp = stamp(1000.2)
    node._cb_cloud(msg)
    assert counts == {topic:len(pub.messages) for topic,pub in node.publishers.items()}


def test_raw_cloud_waits_for_pose_then_runs_complete_pipeline(adapters):
    node = node_at(adapters)
    now(node,1000.1)
    pending = cloud(1000.1)
    node._cb_cloud(pending)
    assert node._pending_cloud is pending and not node._pub_obs.messages
    node._cb_odom(odom(1000.1))
    assert node._pending_cloud is None
    assert len(node._pub_detections.messages) == len(node._pub_tracks.messages) == 1


def test_first_point_stamp_waits_for_end_pose_and_deskews_using_real_times(adapters):
    node = node_at(adapters)
    msg = cloud(times=(1000.,1000.1))
    node._cb_cloud(msg)
    assert node._pending_cloud is msg and not node._pub_obs.messages
    now(node,1000.1)
    node._cb_odom(odom(1000.1,x=.04))
    points = node._pub_points_map.messages[-1]
    xyz = np.frombuffer(points.data,dtype='<f4').reshape(-1,3)
    np.testing.assert_allclose(xyz[:,0],[-1.,-.96],atol=1e-6)
    assert points.header.stamp.sec == 1000 and points.header.stamp.nanosec == 0
    assert json.loads(node._pub_perception_diag.messages[-1].data)['deskew'] == 'pose_3d_compensated'


def test_delayed_neural_detection_keeps_source_stamp_and_consumes_once(adapters):
    node = node_at(adapters)
    points = ((2.,1.,-.2),(2.,1.02,-.2))
    node._cb_cloud(cloud(points=points))
    for dt in (.1,.2):
        now(node,1000+dt)
        node._cb_odom(odom(1000+dt))
        node._model.stats['source_stamp'] = 1000.
        node._model.boxes = [dict(x=2.,y=1.01,yaw=0.,length=.5,width=.5,score=.9,
                                 stamp=1000.,**{'class':'pedestrian'})]
        node._cb_cloud(cloud(1000+dt,points=points))
        assert node._pub_obs.messages[-1].obstacles[0].id == 1_000_000_001
    assert node._model_tracker.tracks[1].hits == 1
    assert len(node._pub_model_detections.messages) == 1
    detected = node._pub_model_detections.messages[0]
    assert detected.header.stamp.sec == 1000 and detected.header.stamp.nanosec == 0
    assert detected.obstacles[0].type == 'pedestrian' and detected.obstacles[0].speed == -1.
    assert node._pub_tracks.messages[-1].header.stamp.nanosec == 200_000_000


def test_rviz_separates_detection_ids_aabb_and_velocity_and_expires(adapters):
    node = adapters.Node()
    tracks = [(41,2.,1.,.4,math.pi/2,'pedestrian',.5,.3)]
    markers = node._markers(tracks,stamp(1000.))
    box, label, arrow = markers.markers[1:]
    assert box.id == label.id == arrow.id == 41
    assert box.pose.orientation.z == 0 and box.pose.orientation.w == 1
    assert (box.scale.x,box.scale.y) == (.5,.3)
    assert arrow.points[1].y == pytest.approx(1.4)
    assert 'id:41' in label.text
    assert all(m.lifetime.nanosec == 500_000_000 for m in markers.markers[1:])
    detections = [(-1,*tracks[0][1:]),(-1,*tracks[0][1:])]
    dm = node._markers(detections,stamp(1000.),detected=True)
    assert [m.id for m in dm.markers if getattr(m,'ns','').endswith('_boxes')] == [0,1]
    assert len(dm.markers[1].points) == 5
    assert len(node._markers([],stamp(1000.)).markers) == 1


def test_extended_car_message_keeps_aabb_yaw_separate_from_velocity(adapters):
    from campus_perception import perception_tracking_node as module
    original = module.Obstacle
    class ExtendedObstacle(original):
        def __init__(self):
            super().__init__()
            self.has_shape_heading = False
            self.shape_heading = -1.
    module.Obstacle = ExtendedObstacle
    try:
        msg = adapters.Node._obstacle_array([(1,2.,1.,.4,math.pi/2,'vehicle',2.,1.)],stamp(1000.))
        ob = msg.obstacles[0]
        assert ob.has_shape_heading is True and ob.shape_heading == 0.
        assert ob.heading == math.pi/2 and (ob.length,ob.width) == (2.,1.)
    finally:
        module.Obstacle = original


def test_rviz_topics_qos_and_launch_match_standalone_module(adapters):
    node = adapters.Node()
    rviz = yaml.safe_load((ROOT/'campus_perception/rviz/perception_tracking.rviz').read_text())
    assert rviz['Visualization Manager']['Global Options']['Fixed Frame'] == 'map'
    topics = {d['Topic']['Value']:d['Topic'] for d in rviz['Visualization Manager']['Displays'] if 'Topic' in d}
    assert set(topics) <= set(node.publishers)
    assert topics['/rda/perception/points_map']['Reliability Policy'] == 'Best Effort'
    launch = (ROOT/'campus_perception/launch/campus_perception.launch.py').read_text()
    assert "executable='perception_tracking_node'" in launch
    assert "executable='rviz2'" in launch
    assert "executable='bridge_node'" not in launch


def test_default_five_meter_roi_and_full_labels_in_status(adapters):
    node=node_at(adapters)
    node._cb_cloud(cloud(points=((4.,1.,-.4),(4.,1.02,-.4),(6.,1.,-.4),(6.,1.02,-.4))))
    assert len(node._pub_detections.messages[-1].obstacles)==1
    xyz=np.frombuffer(node._pub_points_map.messages[-1].data,dtype='<f4').reshape(-1,3)
    assert len(xyz)==2 and np.all(np.linalg.norm(xyz[:,:2],axis=1)<=5.)
    status=json.loads(node._pub_perception_diag.messages[-1].data)
    assert status['max_obstacle_range']==5.
    assert set(status['model_classes'])=={'car','truck','bus','bicycle','pedestrian'}


def test_raw_frame_fusion_is_independent_of_inference_and_geometry_is_current(adapters):
    node=node_at(adapters)
    for i in range(3):
        t=1000+i*.1;now(node,t);node._cb_odom(odom(t))
        node._cb_cloud(cloud(t,points=((2.+i*.06,1.,-.4),(2.+i*.06,1.02,-.4))))
    fused=node._pub_fused_points.messages[-1]
    xyzt=np.frombuffer(fused.data,dtype='<f4').reshape(-1,4)
    assert fused.header.frame_id=='body' and len(xyzt)==4
    np.testing.assert_allclose(xyzt[:,0],[2.12,2.12,2.06,2.06],atol=1e-6)
    np.testing.assert_allclose(xyzt[:,3],[0.,0.,.1,.1],atol=1e-6)
    assert node._pub_points_map.messages[-1].width==2
    assert node._pub_detections.messages[-1].obstacles[0].x==pytest.approx(2.12)


def test_runtime_fusion_off_drops_old_generation_and_has_zero_point_ages(adapters):
    node=node_at(adapters);node._cb_cloud(cloud())
    node._tracker.tracks[1].semantic_class='truck'
    node.declared['two_frame_fusion_enabled']=False
    node._model.stats['source_stamp']=1000.
    node._model.boxes=[dict(x=2.,y=1.,yaw=0.,length=1.,width=1.,score=.9,stamp=1000.,**{'class':'bus'})]
    now(node,1000.1);node._cb_odom(odom(1000.1));node._cb_cloud(cloud(1000.1))
    xyzt=np.frombuffer(node._pub_fused_points.messages[-1].data,dtype='<f4').reshape(-1,4)
    assert len(xyzt)==2 and not np.any(xyzt[:,3])
    assert not node._model_tracker.tracks and node._fusion_generation==1
    assert node._tracker.tracks[1].semantic_class=='unknown'


def test_full_model_detection_and_planner_vehicle_mapping(adapters):
    node=node_at(adapters)
    for i in range(4):
        t=1000+i*.1;now(node,t);node._cb_odom(odom(t))
        node._model.stats['source_stamp']=t
        node._model.boxes=[dict(x=2.,y=1.,yaw=0.,length=2.,width=1.,score=.9,stamp=t,**{'class':'truck'})]
        node._cb_cloud(cloud(t,points=((2.,1.,-.4),(2.,1.02,-.4))))
    assert node._pub_model_detections.messages[-1].obstacles[0].type=='truck'
    assert node._pub_tracks.messages[-1].obstacles[0].type=='truck'
    assert node._pub_obs.messages[-1].obstacles[0].type=='vehicle'


@pytest.mark.parametrize('value,override',[('config',None),('true',True),('false',False)])
def test_launch_preserves_yaml_fusion_switch_unless_explicitly_overridden(adapters,monkeypatch,value,override):
    import types,importlib.util,sys
    class Config:
        def __init__(self,name):self.name=name
        def perform(self,context):return context[self.name]
    defs={
        'launch':{'LaunchDescription':lambda x:x},
        'launch.actions':{'DeclareLaunchArgument':lambda *a,**k:None,'OpaqueFunction':lambda **k:None},
        'launch.conditions':{'IfCondition':lambda x:x},
        'launch.substitutions':{'LaunchConfiguration':Config},
        'launch_ros':{},'launch_ros.actions':{'Node':lambda **k:NS(**k)},
        'launch_ros.parameter_descriptions':{'ParameterValue':lambda x,**k:x}}
    for name,attrs in defs.items():
        m=types.ModuleType(name);m.__dict__.update(attrs);monkeypatch.setitem(sys.modules,name,m)
    path=ROOT/'campus_perception/launch/campus_perception.launch.py'
    spec=importlib.util.spec_from_file_location('fusion_launch',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    nodes=module._make_nodes({'two_frame_fusion':value})
    changes=[p['two_frame_fusion_enabled'] for p in nodes[0].parameters if isinstance(p,dict) and 'two_frame_fusion_enabled' in p]
    assert changes==([] if override is None else [override])
