"""Full-surround grid clustering and measurement-driven obstacle fusion.

The real lidar adapter supplies source-stamped, KF-estimated motion. Tracks
refresh shape and velocity only on new measurements; occluded tracks survive a
bounded time and snapshots extrapolate without modifying measurement anchors.
Legacy messages lacking separate shape yaw use a conservative square envelope.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional
from .semantics import ACTOR_CLASSES



# ---------------------------------------------------------------------------
# 类型推断 / 尺寸基准
# ---------------------------------------------------------------------------
TYPE_BASE_SIZE = {
    'pedestrian': (0.15, 0.15),
    'bicycle': (0.8, 0.5),
    'vehicle': (1.4, 0.8),
    'static_large': (1.5, 1.5),
    'block': (0.4, 0.4),
}

# 类型判定阈值（按长边）：<=0.35 → pedestrian；<=0.95 → bicycle；
# <=2.0 → vehicle；否则 static_large。
_TYPE_MAX_LONG = {
    'pedestrian': 0.35,
    'bicycle': 0.95,
    'vehicle': 2.0,
}


def infer_type(L: float, W: float) -> str:
    """由实测包围盒推断四类现实障碍类型。"""
    a, b = max(L, W), min(L, W)
    if a <= _TYPE_MAX_LONG['pedestrian']:
        return 'pedestrian'
    if a <= _TYPE_MAX_LONG['bicycle']:
        return 'bicycle'
    if a <= _TYPE_MAX_LONG['vehicle'] and b <= 1.1:
        return 'vehicle'
    return 'static_large'


def resolve_size(obs_type: str, L_meas: Optional[float], W_meas: Optional[float]):
    """实测尺寸可用则保留，否则回退类型基准尺寸（保守）。

    返回 (L, W)。对动态类型（行人/自行车/车辆）实测过小/不可信时
    至少给到基准尺寸下限，防止半径过小造成漏碰。
    """
    base = TYPE_BASE_SIZE.get(obs_type, (1.0, 1.0))
    if L_meas is None or W_meas is None or L_meas <= 0.02 or W_meas <= 0.02:
        return base
    L = max(L_meas, base[0] * 0.5)
    W = max(W_meas, base[1] * 0.5)
    return L, W


# ---------------------------------------------------------------------------
# 点集聚类 → 障碍物矩形
# ---------------------------------------------------------------------------
class VoxelCluster(list):
    """Voxel centroids plus a conservative bound on discarded point offsets."""

    def __init__(self, points, margin, raw_bounds=None):
        super().__init__(points)
        self.voxel_margin = float(margin)
        self.raw_bounds = raw_bounds


class Clusterer:
    """简单 2D 距离聚类（单遍近邻合并），用于激光/点云障碍物提取。

    参数:
      gate: 同一障碍物内两点最大间距（默认 0.6m，行人/车辆尺度适用）
      min_points: 簇最少点数（默认 2，过滤单点噪声）
      max_radius: 单簇质心距离原点最大范围（默认 30m，滤远点）
    """

    def __init__(self, gate: float = 0.6, min_points: int = 2, max_radius: float = 30.0):
        self.gate = float(gate)
        self.min_points = int(min_points)
        self.max_radius = float(max_radius)

    def cluster(self, points: list[tuple[float, float]], origin=(0.0,0.0)):
        """Spatial-grid connected components; order-independent Euclidean clusters.

        Voxel centroids bound the cost on dense full-surround lidar scans. Retain
        each voxel's original point count so min_points keeps its old meaning.
        """
        voxel = max(self.gate / 8.0, 0.02)
        accum = {}
        for x, y in points:
            x,y = round(float(x),9),round(float(y),9)
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if math.hypot(x-origin[0], y-origin[1]) > self.max_radius:
                continue
            key = (math.floor(x / voxel), math.floor(y / voxel))
            old = accum.get(key)
            if old is None:
                accum[key] = (x,y,1,x,x,y,y)
            else:
                ax,ay,count,lo_x,hi_x,lo_y,hi_y = old
                accum[key] = (ax+x,ay+y,count+1,x if x<lo_x else lo_x,
                              x if x>hi_x else hi_x,y if y<lo_y else lo_y,y if y>hi_y else hi_y)
        values = [value for _,value in sorted(accum.items())]
        samples = [(value[0]/value[2],value[1]/value[2],value[2]) for value in values]
        bounds = [value[3:] for value in values]
        grid = defaultdict(set)
        cell = max(self.gate, 1e-6)
        for i, (x, y, _) in enumerate(samples):
            grid[(math.floor(x / cell), math.floor(y / cell))].add(i)
        seen, result = set(), []
        # Account for voxel averaging in the connectivity radius.
        gate2 = (self.gate + math.sqrt(2.0) * voxel) ** 2
        for start in range(len(samples)):
            if start in seen:
                continue
            seen.add(start)
            x,y,_ = samples[start]
            grid[(math.floor(x/cell),math.floor(y/cell))].discard(start)
            todo, indices = [start], []
            while todo:
                i = todo.pop()
                indices.append(i)
                x, y, _ = samples[i]
                gx, gy = math.floor(x / cell), math.floor(y / cell)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        bucket = grid.get((gx+dx,gy+dy))
                        if not bucket:
                            continue
                        # Consume discovered candidates so dense cells do not
                        # repeatedly scan already visited points at every node.
                        for j in tuple(bucket):
                            jx, jy, _ = samples[j]
                            if (jx - x) ** 2 + (jy - y) ** 2 <= gate2:
                                seen.add(j)
                                bucket.remove(j)
                                todo.append(j)
            if sum(samples[i][2] for i in indices) >= self.min_points:
                # Avoid expanding every voxel into four synthetic corner
                # records. Bounding-box projections add the equivalent (or
                # larger) margin while preserving all raw points' coverage.
                result.append(VoxelCluster([(samples[i][0],samples[i][1])
                                             for i in indices],voxel,
                    (min(bounds[i][0] for i in indices),max(bounds[i][1] for i in indices),
                     min(bounds[i][2] for i in indices),max(bounds[i][3] for i in indices))))
        return result


def cluster_bounding_box(cluster: list[tuple[float, float]]):
    """簇 → 轴对齐包围盒 (cx, cy, L, W, heading)。

    简单实现：沿主轴（协方差最大特征向量）取旋转矩形；主轴无法可靠
    估计时退化为轴对齐包围盒。heading 为矩形长边方向（rad）。
    返回 (cx, cy, L, W, heading)；静态默认 heading=0 由调用方决定。
    """
    xs = [p[0] for p in cluster]
    ys = [p[1] for p in cluster]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    n = len(cluster)
    # 主轴（PCA）
    mxx = sum((p[0] - cx) ** 2 for p in cluster) / n
    myy = sum((p[1] - cy) ** 2 for p in cluster) / n
    mxy = sum((p[0] - cx) * (p[1] - cy) for p in cluster) / n
    if abs(mxy) < 1e-9 or math.hypot(mxx, myy) < 1e-12:
        theta = 0.0
    else:
        theta = 0.5 * math.atan2(2.0 * mxy, mxx - myy)
    c, s = math.cos(theta), math.sin(theta)
    proj_u = [p[0] * c + p[1] * s for p in cluster]
    proj_v = [-p[0] * s + p[1] * c for p in cluster]
    u_min, u_max = min(proj_u), max(proj_u)
    v_min, v_max = min(proj_v), max(proj_v)
    # Projection of the retained +/-voxel square: exactly the envelope formerly
    # represented by four corners, without adding avoidable shape inflation.
    margin = (abs(c)+abs(s))*float(getattr(cluster,'voxel_margin',0.0))
    u_min,u_max = u_min-margin,u_max+margin
    v_min,v_max = v_min-margin,v_max+margin
    L = max(u_max - u_min, 0.05)
    W = max(v_max - v_min, 0.05)
    uc, vc = (u_min + u_max) / 2.0, (v_min + v_max) / 2.0
    return uc * c - vc * s, uc * s + vc * c, L, W, theta


def split_world_cluster(cluster, tile_size=0.5, extent_threshold=2.0,
                        occupancy_threshold=0.5,protected_point=None,protected_pose=None,
                        half_length=.6,half_width=.4):
    """Represent one map cluster by an AABB or fixed-world tile AABBs.

    Returns [(tile_key_or_None,cx,cy,L,W)]. Sparse concave or long thin clusters
    split; small people and filled dense objects retain their whole footprint.
    The voxel offset margin is included in every box, preserving raw coverage.
    """
    if not cluster:
        return []
    if tile_size <= 0 or extent_threshold <= 0 or not 0 <= occupancy_threshold <= 1:
        raise ValueError('Invalid world tile parameters')
    margin = float(getattr(cluster,'voxel_margin',0.0))
    points = [(round(float(x),9),round(float(y),9)) for x,y in cluster]
    if not all(math.isfinite(v) for point in points for v in point):
        raise ValueError('Nonfinite world cluster')
    def box(points):
        xs,ys = zip(*points)
        lo_x,hi_x,min_y,max_y = min(xs)-margin,max(xs)+margin,min(ys)-margin,max(ys)+margin
        return ((lo_x+hi_x)/2,(min_y+max_y)/2,
                max(hi_x-lo_x,.05),max(max_y-min_y,.05))
    cx,cy,L,W = box(points)
    raw_bounds = getattr(cluster,'raw_bounds',None)
    if raw_bounds is not None:
        lo_x,hi_x,lo_y,hi_y = raw_bounds
        cx,cy = (lo_x+hi_x)/2,(lo_y+hi_y)/2
        L,W = max(hi_x-lo_x+2*margin,.05),max(hi_y-lo_y+2*margin,.05)
    bins = defaultdict(list)
    for x,y in points:
        bins[(math.floor(x/tile_size),math.floor(y/tile_size))].append((x,y))
    ix,iy = zip(*bins)
    fill = len(bins)/((max(ix)-min(ix)+1)*(max(iy)-min(iy)+1))
    hollow = False
    if protected_pose is not None or protected_point is not None:
        px,py,yaw = protected_pose if protected_pose is not None else (*protected_point,0.0)
        contained = abs(px-cx)<=L/2 and abs(py-cy)<=W/2
        c,s = math.cos(yaw),math.sin(yaw)
        # Confirm that retained voxel cells do not occupy the actual rectangle.
        # This is a representation trigger, with no planner safety margin and
        # no enclosing circle that would hide a narrow but clear U cavity.
        projection_margin = margin*(abs(c)+abs(s))
        bound_squared = (math.hypot(half_length,half_width)+math.sqrt(2)*margin)**2
        nearest = math.inf
        for x,y in points:
            dx,dy = x-px,y-py
            if dx*dx+dy*dy > bound_squared:
                continue  # prefilter only; final occupation uses the rectangle
            gap = math.hypot(max(abs(c*dx+s*dy)-half_length-projection_margin,0.0),
                             max(abs(-s*dx+c*dy)-half_width-projection_margin,0.0))
            nearest = min(nearest,gap)
            if nearest <= 1e-9:
                break
        hollow = contained and nearest>1e-9
    if max(L,W) < extent_threshold or (not hollow and fill >= occupancy_threshold and min(L,W) > tile_size*1.2):
        return [(None,cx,cy,L,W)]
    return [(key,*box(bins[key])) for key in sorted(bins)]


class RepresentationLimitError(RuntimeError):
    """Explicit fail-closed overflow; never merge tiles across a free cavity."""


class WorldObstacleRepresenter:
    """Track whole-object motion while representing concave geometry by tiles.

    Fixed map voxels/tiles and stable anchors prevent ego rotation from creating
    fake tile velocity. Dynamic and unconfirmed tiles inherit parent motion,
    rather than treating an occupied fixed grid cell as a stationary object.
    """
    def __init__(self,tracker=None,tile_size=.5,extent_threshold=2.0,
                 occupancy_threshold=.5,max_tiles=512,half_length=.6,half_width=.4):
        if tracker is None:
            from .kf_tracker import GlobalKFTracker
            tracker = GlobalKFTracker(dt=.1,lost_ttl_seconds=.5)
        self.tracker = tracker
        self.tile_size,self.extent_threshold = float(tile_size),float(extent_threshold)
        self.occupancy_threshold,self.max_tiles = float(occupancy_threshold),int(max_tiles)
        self.half_length,self.half_width = float(half_length),float(half_width)
        if self.tile_size <= 0 or self.extent_threshold <= 0 or not 0 <= self.occupancy_threshold <= 1 or self.max_tiles < 1:
            raise ValueError('Invalid world representation parameters')
        self._parts,self._anchors,self._ids = {},{},{}
        self._next_id,self._last_stamp,self._last_output = 1,None,[]
        self.stats = {}

    def step(self,world_clusters,stamp,reference_origin=None,vehicle_pose=None):
        stamp = float(stamp)
        if self._last_stamp is not None and stamp <= self._last_stamp:
            return list(self._last_output)
        groups = [split_world_cluster(cluster,self.tile_size,self.extent_threshold,
                                      self.occupancy_threshold,reference_origin,vehicle_pose,
                                      self.half_length,self.half_width) for cluster in world_clusters]
        # Parent centers are the complete map AABB center, never a tile center.
        detections = []
        for cluster in world_clusters:
            margin = float(getattr(cluster,'voxel_margin',0))
            xs,ys = zip(*cluster)
            lo_x,hi_x,lo_y,hi_y = getattr(cluster,'raw_bounds',None) or (min(xs),max(xs),min(ys),max(ys))
            L,W = max(hi_x-lo_x+2*margin,.05),max(hi_y-lo_y+2*margin,.05)
            detections.append(((lo_x+hi_x)/2,(lo_y+hi_y)/2,
                               getattr(cluster,'semantic_class','unknown') if self.tracker.preserve_semantics else infer_type(L,W),
                               L,W,float(getattr(cluster,'semantic_score',0.0))))
        parents = self.tracker.step(detections,stamp=stamp,now=stamp)
        # Semantic actors are whole entities. Map tile keys would give a moving
        # car new public IDs each time it crossed a grid boundary.
        if self.tracker.preserve_semantics:
            for di,det in enumerate(detections):
                pid = self.tracker.last_detection_ids.get(di)
                if pid is not None and self.tracker.tracks[pid].semantic_class in ACTOR_CLASSES:
                    groups[di] = [(None,det[0],det[1],det[3],det[4])]
        live = set(self.tracker.tracks)
        for pid in list(self._parts):
            if pid not in live:
                del self._parts[pid]
                for key in [k for k in self._anchors if k[0]==pid]:
                    self._anchors.pop(key,None)
                    self._ids.pop(key,None)
        parent_states = {item[0]:item for item in parents}
        for di,parts in enumerate(groups):
            pid = self.tracker.last_detection_ids.get(di)
            if pid is None:
                continue
            current = {}
            for key,cx,cy,L,W in parts:
                if key is None:
                    state = parent_states[pid]
                    current[key] = (state[1],state[2],state[6],state[7],stamp)
                else:
                    anchor_key = (pid,key)
                    ax,ay,aL,aW = self._anchors.get(anchor_key,(cx,cy,L,W))
                    aL,aW = max(aL,2*abs(cx-ax)+L),max(aW,2*abs(cy-ay)+W)
                    self._anchors[anchor_key] = (ax,ay,aL,aW)
                    current[key] = (ax,ay,aL,aW,stamp)
            self._parts[pid] = current
        count = sum(len(parts) for parts in self._parts.values())
        self.stats = {'parent_count':len(parents),'output_count':count,
                      'tiled_parents':sum(any(k is not None for k in p) for p in self._parts.values())}
        if count > self.max_tiles:
            raise RepresentationLimitError(f'{count} obstacle parts exceeds {self.max_tiles}; preserving free-space geometry requires stopping')
        output = []
        for pid,state in sorted(parent_states.items()):
            parent = self.tracker.tracks[pid]
            speed,heading,kind = state[3],state[4],state[5]
            for key,(cx,cy,L,W,seen) in sorted(self._parts.get(pid,{}).items(),key=lambda item:str(item[0])):
                tiled = key is not None
                if tiled:
                    stale = (self.tracker.velocity_max_age is not None and stamp-parent.last_seen>self.tracker.velocity_max_age)
                    kind = ('unknown' if stale else 'block' if parent.static_confirmed else
                            'pedestrian' if parent.velocity_trusted else 'unknown')
                    if self.tracker.preserve_semantics:
                        kind = (parent.semantic_class if parent.semantic_class!='unknown'
                                else 'unknown' if stale else 'block' if parent.static_confirmed else 'unknown')
                identity = (pid,key)
                if identity not in self._ids:
                    self._ids[identity],self._next_id = self._next_id,self._next_id+1
                age = max(stamp-seen,0.0)
                motion_speed = max(speed,0.)
                unknown_rate = parent.max_speed+motion_speed if kind=='unknown' or parent.miss or speed<0 else 0.0
                growth = (0.4+2*unknown_rate)*age
                output.append((self._ids[identity],cx+motion_speed*math.cos(heading)*age,
                               cy+motion_speed*math.sin(heading)*age,speed,heading,
                               kind,max(L+growth,.05),max(W+growth,.05)))
        self._last_stamp,self._last_output = stamp,list(output)
        return output


# ---------------------------------------------------------------------------
