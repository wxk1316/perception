"""Exactly two raw source frames, aligned to the current body reference.

The cache stores an unfused source cloud and its source-time SE(3) transform.
It advances on clouds, independently of inference throughput. Age remains the
fourth feature; ego alignment does not compensate an actor's own motion.
"""
import math
import numpy as np
from .campus_cloud import transform_points


class TwoFrameCloudFusion:
    def __init__(self, max_age=.15):
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError('Invalid fusion maximum age')
        self.max_age = float(max_age)
        self.previous = None
        self.last_enabled = None
        self.stats = {}

    def fuse(self, xyz, stamp, world_body, enabled=True, radius=5.):
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        matrix = np.asarray(world_body, dtype=float)
        if (not math.isfinite(stamp) or stamp <= 0 or matrix.shape != (4, 4)
            or not np.isfinite(matrix).all() or not np.isfinite(xyz).all()
            or not math.isfinite(radius) or radius <= 0):
            raise ValueError('Invalid source cloud or frame transform')
        if self.previous is not None and stamp <= self.previous[1]:
            raise ValueError('Fusion requires strictly increasing source timestamps')
        if self.last_enabled is not None and bool(enabled) != self.last_enabled:
            self.previous = None  # no history from the previous A/B condition
        self.last_enabled = bool(enabled)
        current = xyz[np.linalg.norm(xyz[:, :2], axis=1) <= radius]
        samples = [np.column_stack((current, np.zeros(len(current), dtype=np.float32)))]
        mode, age, previous_stamp, count = 'disabled' if not enabled else 'first_frame', None, None, 0
        if enabled and self.previous is not None and len(current):
            prev_xyz, previous_stamp, prev_matrix = self.previous
            age = float(stamp-previous_stamp)
            if 0 < age <= self.max_age+1e-9:
                history = transform_points(prev_xyz, np.linalg.inv(matrix) @ prev_matrix)
                history = history[np.linalg.norm(history[:, :2], axis=1) <= radius]
                samples.append(np.column_stack((history, np.full(len(history), age, dtype=np.float32))))
                count, mode = len(history), 'two_frames'
            else:
                mode = 'gap_single_frame'
        if enabled and not len(current):
            mode = 'empty_current'  # never create a new detection from history alone
        # Store the original source cloud, never the already merged samples.
        self.previous = (xyz.copy(), float(stamp), matrix.copy())
        self.stats = dict(enabled=bool(enabled), mode=mode, current_points=len(current),
                          history_points=count, previous_source_stamp=previous_stamp,
                          history_age_seconds=age, source_stamp=float(stamp))
        return np.concatenate(samples).astype(np.float32)


def rectangle_distance(x, y, length, width, origin=(0., 0.), yaw=0.):
    """Distance to a footprint, rather than its center (large objects matter)."""
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = origin[0]-x, origin[1]-y
    lx, ly = c*dx+s*dy, -s*dx+c*dy
    return math.hypot(max(abs(lx)-length/2, 0.), max(abs(ly)-width/2, 0.))


def nearby_obstacles(items, origin, radius):
    return [item for item in items if rectangle_distance(item[1], item[2], item[6], item[7], origin) <= radius+1e-9]


def nearby_boxes(boxes, origin, radius):
    return [box for box in boxes if rectangle_distance(box['x'], box['y'], box['length'], box['width'], origin, box['yaw']) <= radius+1e-9]
