"""Source-time CV tracking with observed-motion consistency guards.

Motion bounds are configurable operating assumptions, not vehicle calibration or
statistical confidence bounds. An incompatible detection stays represented as a
new/updated unknown obstacle; it never disappears because a model gate failed.
"""
from __future__ import annotations
import math
from collections import deque
import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    linear_sum_assignment = None
    _HAS_SCIPY = False


class KFTrack:
    def __init__(self, tid, x, y, dt, q=0.3, r=0.15, obs_type='unknown',
                 length=1.0, width=1.0, max_speed=1.5, dyn_speed_thr=0.3,
                 dyn_hold_frames=5, static_speed_thr=0.15, static_hold_frames=10,
                 motion_guard_enable=True, measurement_position_slack=0.15,
                 velocity_innovation_limit=0.8, velocity_reconfirm_time=0.2,
                 dyn_confirm_seconds=None, static_confirm_seconds=None,
                 motion_observation_window=None):
        self.id, self.dt, self.q = int(tid), float(dt), float(q)
        self.max_speed = float(max_speed)
        self.motion_guard_enable = bool(motion_guard_enable)
        self.measurement_position_slack = float(measurement_position_slack)
        self.velocity_innovation_limit = float(velocity_innovation_limit)
        self.velocity_reconfirm_time = float(velocity_reconfirm_time)
        self.motion_observation_window = float(motion_observation_window or velocity_reconfirm_time)
        self._regression_velocity = motion_observation_window is not None
        self.velocity_trusted = False
        self.motion_status = 'first_observation'
        self._reconfirm_elapsed = 0.0
        self._last_raw_velocity = None
        self._measurement_elapsed = 0.0
        self._motion_window = deque([(0.0,float(x),float(y))])
        self.dyn_speed_thr, self.static_speed_thr = dyn_speed_thr, static_speed_thr
        self._mv_required, self._st_required = dyn_hold_frames*self.dt, static_hold_frames*self.dt
        if dyn_confirm_seconds is not None:
            self._mv_required = float(dyn_confirm_seconds)
        if static_confirm_seconds is not None:
            self._st_required = float(static_confirm_seconds)
        self._mv_duration = self._st_duration = 0.0
        self.x = np.array([x,y,0.0,0.0], dtype=float)
        self.P = np.diag([1.0,1.0,2.0,2.0])
        self.H = np.array([[1,0,0,0],[0,1,0,0]],dtype=float)
        self.R = np.eye(2)*float(r)**2
        self.obs_type, self.length, self.width = obs_type, float(length), float(width)
        self.observed_length, self.observed_width = float(length), float(width)
        self.semantic_class = obs_type if obs_type in ('pedestrian', 'vehicle', 'bicycle') else 'unknown'
        self.semantic_score = 0.0
        self._pending_class,self._pending_class_hits = '',0
        self.hits, self.lifecycle = 1, 'tentative'
        self.association_ambiguous = False
        self.moving, self.static_confirmed = False, False
        self.last_seen, self.miss = 0.0, 0
        self._last_measurement = (float(x),float(y))
        self._measurement_count = 1
        self._set_dt(self.dt)

    def _set_dt(self, dt):
        self.dt = max(float(dt),1e-6)
        d, d2 = self.dt, self.dt**2
        self.F = np.array([[1,0,d,0],[0,1,0,d],[0,0,1,0],[0,0,0,1]],dtype=float)
        self.Q = self.q*np.array([[d2*d2/4,0,d2*d/2,0],[0,d2*d2/4,0,d2*d/2],
                                 [d2*d/2,0,d2,0],[0,d2*d/2,0,d2]])

    def predict(self, dt=None):
        if dt is not None:
            self._set_dt(dt)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, zx, zy, measurement_dt=None):
        z = np.array([zx,zy],dtype=float)
        elapsed = max(float(measurement_dt or self.dt),1e-6)
        raw_velocity = (z-np.asarray(self._last_measurement))/elapsed
        self._measurement_elapsed += elapsed
        self._motion_window.append((self._measurement_elapsed,float(zx),float(zy)))
        # A source-time window avoids interpreting high-rate centroid noise as
        # repeated extreme acceleration. Keep the sample bracketing its start.
        cutoff = self._measurement_elapsed-self.motion_observation_window
        while len(self._motion_window)>2 and self._motion_window[1][0] <= cutoff+1e-9:
            self._motion_window.popleft()
        anchor_time,anchor_x,anchor_y = self._motion_window[0]
        window_velocity = (z-np.array([anchor_x,anchor_y]))/max(self._measurement_elapsed-anchor_time,1e-6)
        if self._regression_velocity and len(self._motion_window)>=3:
            samples = np.asarray(self._motion_window,dtype=float)
            centred_time = samples[:,0]-samples[:,0].mean()
            variance = float(centred_time@centred_time)
            if variance>1e-12:
                window_velocity = centred_time@samples[:,1:]/variance
        raw_speed = float(np.linalg.norm(window_velocity))
        innovation = z-self.H@self.x
        velocity_change = float(np.linalg.norm(raw_velocity-self.x[2:]))
        inconsistent = bool(self.motion_guard_enable and (
            raw_speed > self.max_speed+1e-9 or
            (self._measurement_count >= 2 and self.velocity_trusted
             and float(np.linalg.norm(innovation)) > self.measurement_position_slack
             and velocity_change > self.velocity_innovation_limit)))
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, (self.P @ self.H.T).T).T
        self.x += K @ innovation
        A = np.eye(4)-K@self.H
        self.P = A@self.P@A.T + K@self.R@K.T
        if inconsistent:
            # A clipped vector is not evidence for a measured velocity. Keep
            # the current detection position and use an isotropic reachable
            # envelope until real source observations establish motion again.
            self.x[:2],self.x[2:] = z,0.0
            self.velocity_trusted = False
            self._reconfirm_elapsed = 0.0
            self.motion_status = 'speed_bound' if raw_speed > self.max_speed else 'motion_innovation'
            self.moving = self.static_confirmed = False
            self._mv_duration = self._st_duration = 0.0
        elif self._measurement_count == 1:
            self.x[2:] = raw_velocity
            self.velocity_trusted = True
            self.motion_status = 'measured_velocity'
        elif not self.velocity_trusted:
            compatible = (elapsed <= self.velocity_reconfirm_time+1e-9 and
                self._last_raw_velocity is not None and
                float(np.linalg.norm(window_velocity-self._last_raw_velocity)) <= self.velocity_innovation_limit)
            self._reconfirm_elapsed = self._reconfirm_elapsed+elapsed if compatible else 0.0
            self.x[:2],self.x[2:] = z,0.0
            if self._reconfirm_elapsed+1e-9 >= self.velocity_reconfirm_time:
                self.x[2:] = window_velocity
                self.velocity_trusted = True
                self.motion_status = 'reconfirmed_velocity'
            else:
                self.motion_status = 'reconfirming_velocity'
        self._last_raw_velocity = window_velocity.copy()
        self._last_measurement = (float(zx),float(zy))
        self._measurement_count += 1
        speed = math.hypot(self.x[2],self.x[3])
        if speed > self.max_speed:
            self.x[2:] *= self.max_speed/speed
        self.miss = 0

    def position(self):
        return float(self.x[0]),float(self.x[1])

    def velocity(self):
        return float(self.x[2]),float(self.x[3])

    def speed_heading(self):
        vx,vy = self.velocity()
        speed = math.hypot(vx,vy)
        return (0.0,0.0) if speed < 0.05 else (min(speed,self.max_speed),math.atan2(vy,vx))

    def update_motion(self):
        if not self.velocity_trusted:
            return
        speed = math.hypot(self.x[2],self.x[3])
        if speed > self.dyn_speed_thr:
            self._mv_duration += self.dt
            self._st_duration = 0.0
            self.static_confirmed = False
        elif speed < self.static_speed_thr:
            self._st_duration += self.dt
            self._mv_duration = 0.0
        else:
            self._mv_duration = self._st_duration = 0.0
            # A slow start must release stationary confirmation even before the
            # dynamic hold timer completes. Preserve the measured velocity.
            self.static_confirmed = False
        if self._mv_duration >= self._mv_required:
            self.moving = True
        elif self._st_duration >= self._st_required:
            self.moving, self.static_confirmed = False, True


def _linear_assignment(cost):
    """Exact assignment, including a small dependency-free Hungarian fallback."""
    if _HAS_SCIPY:
        return linear_sum_assignment(cost)
    # Shortest augmenting path formulation. Caller supplies rows <= columns.
    n, m = cost.shape
    u, v = np.zeros(n+1), np.zeros(m+1)
    p, way = np.zeros(m+1, dtype=int), np.zeros(m+1, dtype=int)
    for row in range(1, n+1):
        p[0], column = row, 0
        minimum, used = np.full(m+1, np.inf), np.zeros(m+1, dtype=bool)
        while True:
            used[column] = True
            active = p[column]
            for j in range(1, m+1):
                if not used[j]:
                    candidate = cost[active-1, j-1] - u[active] - v[j]
                    if candidate < minimum[j]:
                        minimum[j], way[j] = candidate, column
            eligible = np.where(used, np.inf, minimum)
            next_column = int(np.argmin(eligible))
            delta = minimum[next_column]
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            column = next_column
            if p[column] == 0:
                break
        while column:
            previous = way[column]
            p[column] = p[previous]
            column = previous
    columns = np.full(n, -1, dtype=int)
    for column in range(1, m+1):
        if p[column]:
            columns[p[column]-1] = column-1
    return np.arange(n), columns


class HungarianMatcher:
    """Global assignment with physical gates and explicit miss/birth choices.

    A missing SciPy installation never silently changes to greedy association.
    ``costs`` may contain normalized covariance/shape costs; the distance gate
    always remains in metres and is checked independently.
    """

    def __init__(self, gate: float = 1.0):
        self.gate = float(gate)

    def match(self, pred: list, dets: list, allowed=None, costs=None,
              unmatched_cost=None) -> tuple:
        """pred: [(px, py)] 预测位置；dets: [(dx, dy)] 检测位置。

        返回 (matched, unmatched_det_idx)：
          matched: [(track_idx, det_idx)]，cost 均 <= gate
          unmatched_det_idx: 未匹配的检测下标
        """
        n = len(pred)
        m = len(dets)
        if n == 0 or m == 0:
            return [], list(range(m))
        delta = np.asarray(pred,dtype=np.float64)[:,None,:]-np.asarray(dets,dtype=np.float64)[None,:,:]
        distances = np.hypot(delta[:,:,0],delta[:,:,1])
        admissible = distances <= self.gate
        if allowed is not None:
            admissible &= np.asarray(allowed,dtype=bool)
        raw = distances if costs is None else np.asarray(costs, dtype=float)
        if raw.shape != (n, m) or not np.all(np.isfinite(raw)) or np.any(raw < 0):
            raise ValueError('Invalid association costs')
        miss = float(unmatched_cost if unmatched_cost is not None else (max(n,m)+1)*(self.gate+1))
        if not math.isfinite(miss) or miss <= 0:
            raise ValueError('Invalid unmatched cost')
        forbidden = max(1e6, (n+m+1)*(miss+float(np.max(raw))+1))
        cost = np.full((n+m, n+m), forbidden)
        cost[:n, :m] = np.where(admissible, raw, forbidden)
        cost[np.arange(n), m+np.arange(n)] = miss
        cost[n+np.arange(m), np.arange(m)] = miss
        cost[n:, m:] = 0.0
        rows, cols = _linear_assignment(cost)
        matched = [(int(ri), int(ci)) for ri,ci in zip(rows,cols)
                   if ri < n and ci < m and admissible[ri,ci]]
        matched_det = {d for _, d in matched}
        unmatched = [j for j in range(m) if j not in matched_det]
        return matched, unmatched


# ---------------------------------------------------------------------------
# 全局跟踪器
# ---------------------------------------------------------------------------
class GlobalKFTracker:
    """One update per source cloud. Unknown motion is not confirmed static."""
    STATIC_TYPES = ('static_large','block')

    def __init__(self, dt=0.1, gate=1.0, lost_ttl=10, q=0.3, r=0.15,
                 max_speed=1.5, dyn_speed_thr=0.3, dyn_hold_frames=5,
                 static_speed_thr=0.15, static_hold_frames=10,
                 static_small_thr=0.6, lost_ttl_seconds=None, motion_guard_enable=True,
                 measurement_position_slack=0.15, velocity_innovation_limit=0.8,
                 velocity_reconfirm_time=0.2, enhanced_association=True,
                 mahalanobis_gate=9.21, unmatched_cost=0.45,
                 dyn_confirm_seconds=None, static_confirm_seconds=None,
                 class_speed_limits=None, preserve_semantics=False,
                 motion_observation_window=None,velocity_max_age=None):
        self.dt, self.gate = float(dt), float(gate)
        self.lost_ttl = int(lost_ttl)
        self.lost_ttl_seconds = float(lost_ttl_seconds if lost_ttl_seconds is not None else lost_ttl*self.dt)
        self.q,self.r,self.max_speed = float(q),float(r),float(max_speed)
        self.motion_guard_enable = bool(motion_guard_enable)
        self.measurement_position_slack = float(measurement_position_slack)
        self.velocity_innovation_limit = float(velocity_innovation_limit)
        self.velocity_reconfirm_time = float(velocity_reconfirm_time)
        self.enhanced_association = bool(enhanced_association)
        self.mahalanobis_gate, self.unmatched_cost = float(mahalanobis_gate), float(unmatched_cost)
        self.dyn_confirm_seconds, self.static_confirm_seconds = dyn_confirm_seconds, static_confirm_seconds
        self.class_speed_limits = dict(class_speed_limits or {})
        self.preserve_semantics = bool(preserve_semantics)
        self.motion_observation_window = motion_observation_window
        self.velocity_max_age = None if velocity_max_age is None else float(velocity_max_age)
        values = (self.dt,self.gate,self.lost_ttl_seconds,self.r,self.max_speed,
                  self.measurement_position_slack,self.velocity_innovation_limit,self.velocity_reconfirm_time)
        if not all(math.isfinite(v) and v > 0.0 for v in values) or not math.isfinite(self.q) or self.q < 0.0:
            raise ValueError('Invalid tracking bounds or timing')
        for value in (self.mahalanobis_gate,self.unmatched_cost,*self.class_speed_limits.values(),
                      *[v for v in (dyn_confirm_seconds,static_confirm_seconds,motion_observation_window,velocity_max_age) if v is not None]):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError('Invalid association, class speed or confirmation duration')
        self.stats = {}
        self.dyn_speed_thr,self.dyn_hold_frames = dyn_speed_thr,dyn_hold_frames
        self.static_speed_thr,self.static_hold_frames = static_speed_thr,static_hold_frames
        self.static_small_thr = static_small_thr
        self.tracks,self._next_id = {},1
        self._matcher = HungarianMatcher(gate)
        self._now,self._last_stamp = 0.0,None
        self.last_detection_ids = {}

    def reset(self):
        self.tracks.clear()
        self._next_id,self._now,self._last_stamp = 1,0.0,None
        self.last_detection_ids = {}

    def step(self, detections, dt=None, stamp=None, now=None):
        if any(not math.isfinite(float(value)) for value in (dt,stamp,now) if value is not None):
            raise ValueError('Nonfinite tracker time')
        if stamp is not None and self._last_stamp is not None:
            if stamp <= self._last_stamp:
                return self._output()
            dt = float(stamp)-self._last_stamp
        elapsed = max(float(dt if dt is not None else self.dt),1e-6)
        self._now = (float(now) if now is not None else float(stamp) if stamp is not None else self._now+elapsed)
        if stamp is not None:
            self._last_stamp = float(stamp)
        # Expire before associating: a detection after TTL cannot revive an old ID.
        self._delete_missed()
        ids = sorted(self.tracks)
        for track in self.tracks.values():
            track.predict(elapsed)
        pred = [self.tracks[i].position() for i in ids]
        valid = [(i,d) for i,d in enumerate(detections)
                 if all(math.isfinite(float(v)) for v in (d[0],d[1],d[3],d[4]))]
        valid_indices = [item[0] for item in valid]
        dets = [item[1] for item in valid]
        merged = np.zeros(len(dets),dtype=bool)
        self.last_detection_ids = {}
        allowed = np.ones((len(ids),len(dets)), dtype=bool)
        if self.motion_guard_enable:
            anchors = np.asarray([self.tracks[tid]._last_measurement for tid in ids],dtype=float).reshape(-1,2)
            observations = np.asarray([(d[0],d[1]) for d in dets],dtype=float).reshape(-1,2)
            reachable = np.asarray([self.tracks[tid].max_speed*max(self._now-self.tracks[tid].last_seen,0.0)
                                    +2.0*self.measurement_position_slack for tid in ids])
            delta = anchors[:,None,:]-observations[None,:,:]
            allowed = np.hypot(delta[:,:,0],delta[:,:,1]) <= reachable[:,None]+1e-9
        costs = None
        physical_rejections = int(np.count_nonzero(~allowed))
        if self.enhanced_association and ids and dets:
            costs, covariance_allowed = self._association_costs(ids, dets)
            allowed &= covariance_allowed
            if self.preserve_semantics:
                for column,det in enumerate(dets):
                    if det[2] != 'unknown':
                        continue
                    contained = [row for row,tid in enumerate(ids)
                        if not self.tracks[tid].association_ambiguous and self.tracks[tid].hits>=3
                        and not self.tracks[tid].static_confirmed
                        and abs(pred[row][0]-det[0])<=det[3]/2+.05
                        and abs(pred[row][1]-det[1])<=det[4]/2+.05]
                    if (len(contained)>=2 and det[3]*det[4]>
                        1.5*np.median([self.tracks[ids[row]].observed_length*self.tracks[ids[row]].observed_width for row in contained])):
                        merged[column] = True
                        allowed[contained,column] = False
                for row,tid in enumerate(ids):
                    if self.tracks[tid].association_ambiguous:
                        costs[row,~merged] += .8
        matched,unmatched = self._matcher.match(pred,[(d[0],d[1]) for d in dets],allowed=allowed,
            costs=costs, unmatched_cost=self.unmatched_cost if costs is not None else None)
        self.stats = {'association_bound_rejections':physical_rejections,
                      'association_rejections':int(np.count_nonzero(~allowed)),
                      'solver':'scipy_hungarian' if _HAS_SCIPY else 'python_hungarian',
                      'matched':len(matched),'new':len(unmatched),'merged_observations':int(merged.sum())}

        matched_ids = set()
        for ti,di in matched:
            tr = self.tracks[ids[ti]]
            if dets[di][2] in self.class_speed_limits and tr.semantic_class in ('unknown',dets[di][2]):
                tr.max_speed = float(self.class_speed_limits[dets[di][2]])
            if merged[di]:
                tr.x[:2],tr.x[2:] = dets[di][:2],0.
                tr.velocity_trusted,tr.association_ambiguous = False,True
                tr.motion_status = 'merged_observation'
            else:
                tr.update(dets[di][0],dets[di][1],measurement_dt=self._now-tr.last_seen)
                tr.association_ambiguous = False
            tr.last_seen = self._now
            self._refresh_attr(tr,dets[di])
            tr.update_motion()
            tr.hits += 1
            tr.lifecycle = 'confirmed' if tr.hits >= 3 else 'tentative'
            matched_ids.add(tr.id)
            self.last_detection_ids[valid_indices[di]] = tr.id
        for di in unmatched:
            dx,dy,otype,L,W = dets[di][:5]
            tid,self._next_id = self._next_id,self._next_id+1
            tr = KFTrack(tid,dx,dy,self.dt,q=self.q,r=self.r,obs_type=otype,
                         length=L,width=W,max_speed=self.class_speed_limits.get(otype,self.max_speed),dyn_speed_thr=self.dyn_speed_thr,
                         dyn_hold_frames=self.dyn_hold_frames,static_speed_thr=self.static_speed_thr,
                         static_hold_frames=self.static_hold_frames,
                         motion_guard_enable=self.motion_guard_enable,
                         measurement_position_slack=self.measurement_position_slack,
                         velocity_innovation_limit=self.velocity_innovation_limit,
                         velocity_reconfirm_time=self.velocity_reconfirm_time,
                         dyn_confirm_seconds=self.dyn_confirm_seconds,
                         static_confirm_seconds=self.static_confirm_seconds,
                         motion_observation_window=self.motion_observation_window)
            tr.semantic_score = float(dets[di][5]) if len(dets[di]) > 5 else 0.0
            tr.association_ambiguous = bool(merged[di])
            if merged[di]:
                tr.motion_status = 'merged_observation'
            tr.last_seen = self._now
            self.tracks[tid] = tr
            self.last_detection_ids[valid_indices[di]] = tid
        for tid in ids:
            if tid not in matched_ids:
                self.tracks[tid].miss += 1
                self.tracks[tid].lifecycle = 'occluded'
        self._delete_missed()
        return self._output()

    def _refresh_attr(self,tr,det):
        residual = math.hypot(det[0]-tr.x[0],det[1]-tr.x[1])
        tr.length = max(tr.length,float(det[3])+2*residual)
        tr.width = max(tr.width,float(det[4])+2*residual)
        tr.obs_type = det[2]
        tr.observed_length,tr.observed_width = float(det[3]),float(det[4])
        if det[2] in ('pedestrian','vehicle','bicycle'):
            score = float(det[5]) if len(det)>5 else 0.0
            if tr.semantic_class in ('unknown',det[2]):
                tr.semantic_class = det[2]
                tr.semantic_score = score
                tr._pending_class,tr._pending_class_hits = '',0
            else:
                tr._pending_class_hits = tr._pending_class_hits+1 if tr._pending_class==det[2] else 1
                tr._pending_class = det[2]
                if tr._pending_class_hits>=3:
                    tr.semantic_class,tr.semantic_score = det[2],score
                    tr._pending_class,tr._pending_class_hits = '',0
            tr.max_speed = float(self.class_speed_limits.get(tr.semantic_class,self.max_speed))
            if float(np.linalg.norm(tr.x[2:]))>tr.max_speed+1e-9:
                tr.x[2:],tr.velocity_trusted = 0.,False
                tr.motion_status = 'reclassified_speed_bound'

    def _association_costs(self,ids,dets):
        positions = np.asarray([(d[0],d[1]) for d in dets])
        sizes = np.asarray([(d[3],d[4]) for d in dets])
        costs,allowed = np.zeros((len(ids),len(dets))),np.ones((len(ids),len(dets)),dtype=bool)
        for row,tid in enumerate(ids):
            tr = self.tracks[tid]
            delta = positions-tr.x[:2]
            S = tr.H@tr.P@tr.H.T+tr.R
            d2 = np.einsum('ij,ji->i',delta,np.linalg.solve(S,delta.T))
            allowed[row] = d2 <= self.mahalanobis_gate
            position_cost = .5*np.linalg.norm(delta,axis=1)/self.gate+.5*np.sqrt(np.maximum(d2,0)/self.mahalanobis_gate)
            # Use observation sizes, never the growing collision envelope.
            shape = np.maximum([tr.observed_length,tr.observed_width],.05)
            safe_sizes = np.maximum(sizes,.05)
            lo = np.maximum(tr.x[:2]-shape/2,positions-safe_sizes/2)
            hi = np.minimum(tr.x[:2]+shape/2,positions+safe_sizes/2)
            intersection = np.prod(np.maximum(hi-lo,0),axis=1)
            iou = intersection/(np.prod(shape)+np.prod(safe_sizes,axis=1)-intersection+1e-9)
            size_cost = np.minimum(np.mean(np.abs(np.log(safe_sizes/shape)),axis=1),1.0)
            costs[row] = .65*position_cost+.20*(1-iou)+.15*size_cost
            if self.preserve_semantics and tr.semantic_class != 'unknown':
                for column,det in enumerate(dets):
                    if det[2] in ('pedestrian','vehicle','bicycle') and det[2] != tr.semantic_class:
                        costs[row,column] += .35
        return costs,allowed

    def _delete_missed(self):
        for tid in list(self.tracks):
            if self._now-self.tracks[tid].last_seen > self.lost_ttl_seconds+1e-6:
                del self.tracks[tid]

    def snapshot(self,stamp):
        """Predict to a publication time without advancing the measurement KF."""
        if not math.isfinite(float(stamp)) or float(stamp)<self._now:
            raise ValueError('Snapshot precedes the tracking source state')
        return self._output(at_time=float(stamp))

    def _output(self,at_time=None):
        result = []
        publication_time = self._now if at_time is None else float(at_time)
        extrapolation = max(publication_time-self._now,0.)
        for tid in sorted(self.tracks):
            tr = self.tracks[tid]
            age = max(publication_time-tr.last_seen,0.0)
            if age>self.lost_ttl_seconds+1e-6:
                continue
            x,y = tr.position()
            x += float(tr.x[2])*extrapolation
            y += float(tr.x[3])*extrapolation
            if tr.moving or not tr.static_confirmed:
                speed,heading = tr.speed_heading()
                # Preserve the first-observation velocity uncertainty through
                # legacy ROS type, even when empty cloud frames keep predicting
                # this object. A predicted miss must not masquerade as a second
                # real position measurement at the downstream adapter.
                otype = 'unknown' if not tr.velocity_trusted else 'pedestrian'
                if not tr.velocity_trusted:
                    speed,heading = 0.0,0.0
            else:
                speed,heading = 0.0,0.0
                otype = 'block' if max(tr.length,tr.width)<=self.static_small_thr else 'static_large'
            if self.preserve_semantics and tr.velocity_trusted and tr.semantic_class != 'unknown':
                otype = tr.semantic_class
            elif self.preserve_semantics and not tr.static_confirmed:
                otype = 'unknown'
            stale_velocity = self.velocity_max_age is not None and age>self.velocity_max_age
            estimated_speed = float(np.linalg.norm(tr.x[2:]))
            if stale_velocity:
                otype,speed,heading = 'unknown',0.,0.
            # Array headers advance even when a particular object is occluded.
            # Carry its first-observation reachable bound in the published shape
            # so downstream refreshing the array cannot reset that uncertainty.
            unknown_rate = tr.max_speed+estimated_speed if not tr.velocity_trusted or tr.miss or extrapolation or stale_velocity else 0.0
            growth = (0.4+2*unknown_rate)*age
            result.append((tid,x,y,speed,heading,otype,tr.length+growth,tr.width+growth))
        return result
