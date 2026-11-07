#!/usr/bin/env python3
"""Run real ONNX inference on synthetic inputs; not a vehicle accuracy benchmark."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'campus_perception')]
from campus_perception.centerpoint_model import CenterPointModel,pillar_features
from campus_perception.campus_cloud import LocalGroundFilter,model_frame_from_body
from campus_perception.kf_tracker import GlobalKFTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory',type=Path,default=ROOT/'campus_perception/models/centerpoint/base')
    parser.add_argument('--backend',choices=('onnxruntime','tensorrt'),default='onnxruntime')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = {'host':platform.platform(),'backend':args.backend,'vehicle_benchmark':False,
              'accuracy_measured':False,'synthetic_input_only':True,'graphs':[],'fusion_enabled':False}
    if args.backend=='onnxruntime':
        import onnx
        for name in ('pts_voxel_encoder.onnx','pts_backbone_neck_head.onnx'):
            graph = onnx.load(args.model_directory/name);onnx.checker.check_model(graph,full_check=True)
            result['graphs'].append({'file':name,'checker':'passed',
                'operators':sorted({node.op_type for node in graph.graph.node}),
                'inputs':{value.name:[d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]
                          for value in graph.graph.input}})
    start = time.perf_counter()
    model = CenterPointModel(args.model_directory,backend=args.backend)
    result['model_classes'] = [name.lower() for name in model.params['class_names']]
    result['load_seconds'] = time.perf_counter()-start
    x,y = np.meshgrid(np.linspace(-10,10,100),np.linspace(-8,8,80))
    road = np.column_stack((x.ravel(),y.ravel(),np.full(x.size,-.45))).astype(np.float32)
    rng = np.random.default_rng(20261107)
    object_points = rng.uniform([2.8,.8,-.4],[3.2,1.2,1.0],size=(500,3)).astype(np.float32)
    samples = [np.empty((0,3),dtype=np.float32),np.concatenate((road,object_points))]
    import yaml
    campus = yaml.safe_load((ROOT/'campus_perception/config/perception_tracking.yaml').read_text())['perception_tracking_node']['ros__parameters']
    result['body_height'] = campus['body_height']
    result['model_reference_height'] = campus['centerpoint_reference_height']
    result['inference'] = []
    for i,cloud in enumerate(samples):
        body = cloud+np.array([0.,0.,campus['sensor_z']],dtype=np.float32)
        model_points,world_transform = model_frame_from_body(body,np.eye(4),campus['body_height'],campus['centerpoint_reference_height'])
        start = time.perf_counter();boxes = model.infer(model_points,100+i*.1,world_transform)
        result['inference'].append({'points':len(cloud),'seconds':time.perf_counter()-start,
            'returned_boxes':len(boxes),'classes':sorted({box['class'] for box in boxes}),
            'output_finite':all(np.isfinite([box['x'],box['y'],box['length'],box['width'],box['score']]).all() for box in boxes)})
    base = np.concatenate((road+np.array([0,0,.45]),object_points+np.array([0,0,.45])))
    ground = LocalGroundFilter(anchor_tolerance=.03)
    start=time.perf_counter();mask=ground.nonground_mask(base)
    result['geometry'] = {**ground.stats,'seconds':time.perf_counter()-start,
                          'object_points_retained':int(mask[len(road):].sum()),'object_points_total':len(object_points)}
    noise_results = []
    for window in (None,.5):
        t = GlobalKFTracker(max_speed=1.,class_speed_limits={'pedestrian':.6},preserve_semantics=True,
                            motion_observation_window=window)
        rng = np.random.default_rng(20261107);unknown,identities = 0,set()
        for i in range(100):
            point=np.array([2+i*.04,1])+rng.normal(0,.025,2)
            t.step([(*point,'pedestrian',.3,.3,.9)],stamp=100+i*.1)
            tid=t.last_detection_ids[0];identities.add(tid)
            out=next(value for value in t._output() if value[0]==tid)
            unknown += out[5]=='unknown' or out[3]<0
        noise_results.append({'observation_window':window,'unknown_frames':unknown,'frames':100,
                              'unique_ids':len(identities),'noise_std_m':.025,'true_speed_mps':.4})
    result['source_window_comparison'] = noise_results
    model.close();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
