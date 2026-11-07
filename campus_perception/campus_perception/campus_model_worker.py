"""One in-flight model job; no inference backlog or repeated KF measurements."""
from concurrent.futures import ThreadPoolExecutor
import time


class ModelWorker:
    def __init__(self,directory,backend='tensorrt',score_threshold=.35,max_range=5.,
                 enabled=True,model_factory=None):
        self.enabled = bool(enabled)
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='campus_model') if enabled else None
        self.future,self.model,self.boxes = None,None,[]
        self.error = ''
        self.stats = {'status':'initializing' if enabled else 'disabled','skipped_busy':0}
        self.directory,self.backend = directory,backend
        self.score_threshold,self.max_range = score_threshold,max_range
        self.factory = model_factory
        if enabled:
            self.future = self.executor.submit(self._initialize)

    def _initialize(self):
        if self.factory is None:
            from .centerpoint_model import CenterPointModel
            self.factory = CenterPointModel
        self.model = self.factory(self.directory,self.backend,
                                  score_threshold=self.score_threshold,max_range=self.max_range)
        return None

    def collect(self):
        if self.future is None or not self.future.done():
            return
        future,self.future = self.future,None
        try:
            value = future.result()
            if value is not None:
                self.boxes,duration,stamp,metadata = value
                self.stats.update(metadata)
                self.stats.update(inference_seconds=duration,source_stamp=stamp)
            self.stats['status'] = 'ready'
            self.error = ''
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.stats['status'] = 'unavailable' if self.model is None else 'inference_failed'

    def submit(self,xyz,stamp,world_transform,metadata=None):
        self.collect()
        if not self.enabled or self.model is None:
            return False
        if self.future is not None:
            self.stats['skipped_busy'] += 1
            return False
        self.future = self.executor.submit(self._infer,xyz.copy(),stamp,world_transform.copy(),dict(metadata or {}))
        return True

    def _infer(self,xyz,stamp,world_transform,metadata):
        start = time.perf_counter()
        boxes = self.model.infer(xyz,stamp,world_transform)
        return boxes,time.perf_counter()-start,stamp,metadata

    def current_boxes(self,stamp,max_age=.25,generation=None):
        self.collect()
        if generation is not None and self.stats.get('fusion_generation',0) != generation:
            return []
        return [box for box in self.boxes if 0<=stamp-box['stamp']<=max_age]

    def close(self):
        if self.executor is not None:
            # Free CUDA resources on the same worker after any pending job.
            self.executor.submit(self._close_model)
            self.executor.shutdown(wait=False,cancel_futures=False)

    def _close_model(self):
        if self.model is not None:
            self.model.close()
