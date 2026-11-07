"""ROS-independent cloud decoding, planar deskew and local road-ground fitting.

Unknown ground patches remain occupied. Point times are never synthesized from
azimuth; auto mode only accepts absolute seconds in the cloud's clock domain.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class DecodedCloud:
    xyz: np.ndarray
    point_times: np.ndarray | None = None
    time_status: str = 'missing'


def decode_cloud(msg, time_field='', time_mode='auto', header_is_end=True,
                 scan_period=.1):
    """Decode padded/organized and big-endian PointCloud2 without ROS imports."""
    if time_mode not in ('auto','absolute','relative_seconds','relative_nanoseconds','disabled'):
        raise ValueError('Unsupported point_time_mode')
    fields = {field.name: field for field in msg.fields}
    scalar_types = {1:'i1',2:'u1',3:'i2',4:'u2',5:'i4',6:'u4',7:'f4',8:'f8'}
    endian = '>' if msg.is_bigendian else '<'
    names,formats,offsets = [],[],[]
    selected_time = time_field or next((name for name in ('timestamp','time','t') if name in fields),'')
    wanted = ['x','y','z']+([selected_time] if selected_time in fields and time_mode != 'disabled' else [])
    if msg.point_step <= 0 or msg.row_step < msg.width*msg.point_step:
        raise ValueError('Invalid PointCloud2 strides')
    required = (msg.height-1)*msg.row_step+msg.width*msg.point_step if msg.height else 0
    if len(msg.data) < required:
        raise ValueError('Truncated PointCloud2 data')
    for name in wanted:
        field = fields.get(name)
        if field is None or field.datatype not in scalar_types or field.count != 1:
            raise ValueError('Missing or nonscalar cloud field: '+name)
        dtype = np.dtype(endian+scalar_types[field.datatype])
        if field.offset < 0 or field.offset+dtype.itemsize > msg.point_step:
            raise ValueError('Cloud field outside point stride')
        names.append(name);formats.append(dtype);offsets.append(field.offset)
    dtype = np.dtype({'names':names,'formats':formats,'offsets':offsets,'itemsize':msg.point_step})
    if msg.width*msg.height == 0:
        return DecodedCloud(np.empty((0,3),dtype=np.float32))
    records = np.ndarray((msg.height,msg.width), dtype=dtype, buffer=msg.data,
                         strides=(msg.row_step,msg.point_step))
    xyz = np.column_stack([records[name].reshape(-1) for name in ('x','y','z')]).astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if not len(xyz):
        raise ValueError('No finite returns in a nonempty lidar cloud')
    if selected_time not in names:
        return DecodedCloud(xyz,None,'disabled' if time_mode == 'disabled' else 'missing')
    values = records[selected_time].reshape(-1)[finite].astype(np.float64)
    stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    if time_mode in ('relative_seconds','relative_nanoseconds'):
        if time_mode == 'relative_nanoseconds':
            values *= 1e-9
        if np.any(values < 0) or np.any(values > scan_period+1e-5):
            return DecodedCloud(xyz,None,'invalid_relative_time')
        values += stamp-scan_period if header_is_end else stamp
    elif time_mode == 'auto' and selected_time != 'timestamp':
        return DecodedCloud(xyz,None,'relative_time_requires_configuration')
    if (not np.isfinite(values).all() or
        np.any(np.abs(values-stamp) > scan_period+1e-4)):
        return DecodedCloud(xyz,None,'time_outside_source_clock')
    return DecodedCloud(xyz,values,'validated')


def sensor_matrix(pose, x=0., y=0., height=.45, roll=0., pitch=0., yaw=0.):
    cr,sr,cp,sp,cy,sy = math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
    sensor_rotation = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])@np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])@np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    if np.asarray(pose).shape == (4,4):
        mounting = np.eye(4)
        mounting[:3,:3],mounting[:3,3] = sensor_rotation,[x,y,height]
        return np.asarray(pose)@mounting
    ego_c,ego_s = math.cos(pose[2]),math.sin(pose[2])
    ego_rotation = np.array([[ego_c,-ego_s,0],[ego_s,ego_c,0],[0,0,1]])
    matrix = np.eye(4)
    matrix[:3,:3] = ego_rotation@sensor_rotation
    matrix[:3,3] = np.array([pose[0],pose[1],0])+ego_rotation@np.array([x,y,height])
    return matrix


def pose_matrix(position,quaternion):
    """ROS xyzw quaternion and position -> full source pose matrix."""
    from scipy.spatial.transform import Rotation
    matrix = np.eye(4)
    matrix[:3,:3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3,3] = position
    return matrix


def interpolate_pose_matrix(poses,stamp,tolerance=.15):
    from scipy.spatial.transform import Rotation,Slerp
    if not poses:
        return None
    times = np.array([item[0] for item in poses])
    matrices = np.array([item[1] for item in poses])
    if stamp<times[0] or stamp>times[-1] or len(times)==1:
        nearest = int(np.argmin(np.abs(times-stamp)))
        return matrices[nearest] if abs(times[nearest]-stamp)<=tolerance else None
    result = np.eye(4)
    result[:3,:3] = Slerp(times,Rotation.from_matrix(matrices[:,:3,:3]))([stamp]).as_matrix()[0]
    result[:3,3] = [np.interp(stamp,times,matrices[:,axis,3]) for axis in range(3)]
    return result


def transform_points(xyz, matrix):
    return np.asarray(xyz)@matrix[:3,:3].T+matrix[:3,3]


def model_frame_from_body(points,world_body,body_height=.17,reference_height=1.8):
    """Normalize road-relative z without changing the physical world points."""
    points = np.asarray(points,dtype=np.float32).copy()
    points[:,2] += float(body_height)-float(reference_height)
    origin = np.eye(4);origin[2,3] = float(reference_height)-float(body_height)
    return points,np.asarray(world_body)@origin


def deskew_cloud(cloud, stamp, poses, extrinsics,full_poses=None):
    """Compensate available SE(3)/SE(2) source poses; retain the sensor frame."""
    if cloud.point_times is None or not len(cloud.xyz):
        return cloud.xyz,cloud.time_status
    if full_poses and len(full_poses)>=2:
        from scipy.spatial.transform import Rotation,Slerp
        times = np.array([item[0] for item in full_poses])
        matrices = np.array([item[1] for item in full_poses])
        if (stamp<times[0] or stamp>times[-1] or
            cloud.point_times.min()<times[0] or cloud.point_times.max()>times[-1]):
            return cloud.xyz,'pose_history_does_not_cover_scan'
        base_points = transform_points(cloud.xyz,sensor_matrix((0,0,0),**extrinsics))
        rotations = Slerp(times,Rotation.from_matrix(matrices[:,:3,:3]))(cloud.point_times)
        translations = np.column_stack([np.interp(cloud.point_times,times,matrices[:,axis,3]) for axis in range(3)])
        world = rotations.apply(base_points)+translations
        reference = interpolate_pose_matrix(full_poses,stamp,tolerance=0.)
        return transform_points(world,np.linalg.inv(sensor_matrix(reference,**extrinsics))).astype(np.float32),'pose_3d_compensated'
    times = np.array([item[0] for item in poses],dtype=float)
    states = np.array([item[1] for item in poses],dtype=float)
    if (len(times)<2 or stamp<times[0] or stamp>times[-1] or
        cloud.point_times.min()<times[0] or cloud.point_times.max()>times[-1]):
        return cloud.xyz,'pose_history_does_not_cover_scan'
    yaw = np.unwrap(states[:,2])
    reference = tuple(np.interp(stamp,times,values) for values in (states[:,0],states[:,1],yaw))
    sensor_to_base = sensor_matrix((0,0,0),**extrinsics)
    points = transform_points(cloud.xyz,sensor_to_base)
    px,py,theta = [np.interp(cloud.point_times,times,values) for values in (states[:,0],states[:,1],yaw)]
    c,s = np.cos(theta),np.sin(theta)
    world_x,world_y = px+c*points[:,0]-s*points[:,1],py+s*points[:,0]+c*points[:,1]
    inv_reference_sensor = np.linalg.inv(sensor_matrix(reference,**extrinsics))
    world = np.column_stack((world_x,world_y,points[:,2]))
    return transform_points(world,inv_reference_sensor).astype(np.float32),'planar_compensated'


class LocalGroundFilter:
    """Robust neighbourhood planes anchored to the calibrated road height.

    This is a small independent CPU implementation, not a Patchwork++ port.
    Sparse, elevated or inconsistent fits cannot remove points as ground.
    """
    def __init__(self,cell_size=2.0,min_points=12,distance=.025,
                 max_slope_deg=15.,anchor_tolerance=.05,max_fit_points=256):
        self.cell_size,self.min_points = float(cell_size),int(min_points)
        self.distance = float(distance)
        self.max_slope = math.tan(math.radians(max_slope_deg))
        self.anchor_tolerance = float(anchor_tolerance)
        self.max_fit_points = int(max_fit_points)
        if (not all(math.isfinite(value) for value in (self.cell_size,self.distance,self.max_slope,self.anchor_tolerance))
            or self.cell_size<=0 or self.min_points<3 or self.distance<=0 or self.anchor_tolerance<=0
            or not 0<max_slope_deg<45 or self.max_fit_points<self.min_points):
            raise ValueError('Invalid local ground parameters')
        self.stats = {}

    def nonground_mask(self,xyz):
        xyz = np.asarray(xyz,dtype=float)
        occupied = np.ones(len(xyz),dtype=bool)
        if not len(xyz):
            self.stats = {'ground':0,'nonground':0,'fitted_patches':0}
            return occupied
        cells = np.floor(xyz[:,:2]/self.cell_size).astype(np.int64)
        keys,inverse = np.unique(cells,axis=0,return_inverse=True)
        order = np.argsort(inverse,kind='stable')
        starts = np.searchsorted(inverse[order],np.arange(len(keys)+1))
        groups = {tuple(key):order[starts[i]:starts[i+1]] for i,key in enumerate(keys)}
        fitted = 0
        for key,indices in groups.items():
            neighbours = [groups[(key[0]+dx,key[1]+dy)] for dx in (-1,0,1) for dy in (-1,0,1)
                          if (key[0]+dx,key[1]+dy) in groups]
            nearby = xyz[np.concatenate(neighbours)]
            candidates = nearby[np.abs(nearby[:,2])<=.25]
            if len(candidates)<self.min_points:
                continue
            if len(candidates)>self.max_fit_points:
                candidates = candidates[np.linspace(0,len(candidates)-1,self.max_fit_points,dtype=int)]
            seed = candidates[candidates[:,2]<=np.quantile(candidates[:,2],.4)+self.distance]
            if len(seed)<self.min_points:
                continue
            coefficients = None
            for _ in range(3):
                design = np.column_stack((seed[:,:2],np.ones(len(seed))))
                coefficients,_,rank,_ = np.linalg.lstsq(design,seed[:,2],rcond=None)
                if rank<3:
                    coefficients = None;break
                residual = candidates[:,2]-np.column_stack((candidates[:,:2],np.ones(len(candidates))))@coefficients
                seed = candidates[np.abs(residual)<=self.distance]
                if len(seed)<self.min_points:
                    coefficients = None;break
            if coefficients is None:
                continue
            if np.linalg.norm(coefficients[:2])>self.max_slope or abs(coefficients[2])>self.anchor_tolerance:
                continue
            residual = xyz[indices,2]-np.column_stack((xyz[indices,:2],np.ones(len(indices))))@coefficients
            occupied[indices] = np.abs(residual)>self.distance
            fitted += 1
        self.stats = {'ground':int(np.count_nonzero(~occupied)),
                      'nonground':int(np.count_nonzero(occupied)),'fitted_patches':fitted}
        return occupied


def semantic_clusters(world_points, boxes, clusterer, stamp, max_age=.25,
                      min_points=2, speed_limits=None,origin=(0.,0.)):
    """Use recent model regions to partition current geometry, preserving it all.

    Positions always come from this source cloud. A prior model result supplies
    only class/partition evidence, never a second or artificially fresh KF update.
    Unclaimed points retain their geometric clusters.
    """
    from .obstacle_fusion import VoxelCluster
    points = np.asarray(world_points,dtype=float).reshape(-1,2)
    owned = np.zeros(len(points),dtype=bool)
    result = []
    limits = speed_limits or {'pedestrian':.6,'vehicle':1.,'bicycle':1.}
    boxes = [box for box in boxes if 0<=stamp-box['stamp']<=max_age and box['class'] in limits]
    ownership,best_cost = np.full(len(points),-1,dtype=int),np.full(len(points),np.inf)
    for index,box in enumerate(boxes):
        age = stamp-box['stamp']
        c,s = math.cos(box['yaw']),math.sin(box['yaw'])
        delta = points-np.array([box['x'],box['y']])
        local_x,local_y = c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]
        margin = .10+limits[box['class']]*age
        half_x,half_y = box['length']/2+margin,box['width']/2+margin
        cost = (local_x/half_x)**2+(local_y/half_y)**2+.1*(1-box['score'])
        selected = (np.abs(local_x)<=half_x)&(np.abs(local_y)<=half_y)&(cost<best_cost)
        ownership[selected],best_cost[selected] = index,cost[selected]
    for index,box in enumerate(boxes):
        selected = ownership==index
        if np.count_nonzero(selected)<min_points:
            continue
        part = points[selected]
        bounds = (float(part[:,0].min()),float(part[:,0].max()),float(part[:,1].min()),float(part[:,1].max()))
        cluster = VoxelCluster([tuple(point) for point in part],0.,bounds)
        cluster.semantic_class,cluster.semantic_score = box['class'],float(box['score'])
        result.append(cluster);owned |= selected
    result.extend(clusterer.cluster([tuple(point) for point in points[~owned]],origin=origin))
    return result
