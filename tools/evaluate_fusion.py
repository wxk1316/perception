#!/usr/bin/env python3
"""Paired synthetic A/B probe; never a campus accuracy or Orin FPS claim."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'campus_perception'))
from campus_perception.temporal_cloud import TwoFrameCloudFusion
from campus_perception.campus_cloud import pose_matrix,transform_points,model_frame_from_body,LocalGroundFilter
from campus_perception.centerpoint_model import CenterPointModel,pillar_features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory',type=Path,default=ROOT/'campus_perception/models/centerpoint/base')
    parser.add_argument('--backend',choices=('onnxruntime','tensorrt'),default='onnxruntime')
    parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.repeats<2:raise ValueError('At least two repeat sources required')
    model=CenterPointModel(args.model_directory,backend=args.backend,max_range=5.)
    rng=np.random.default_rng(20261107)
    gx,gy=np.meshgrid(np.linspace(-10,10,100),np.linspace(-8,8,80))
    ground=np.column_stack((gx.ravel(),gy.ravel(),np.full(gx.size,-.17)))
    actor=rng.uniform([2.8,.8,-.12],[3.2,1.2,1.5],size=(500,3))
    sources=[]
    for i in range(args.repeats+1):
        yaw=i*.02
        world_body=pose_matrix((i*.04,0.,0.),(0.,0.,np.sin(yaw/2),np.cos(yaw/2)))
        # Alternate sparse beam subsets; actor moves independently by 6cm/frame.
        points=np.concatenate((ground,actor[i%2::2]+[i*.06,0.,0.]))
        body=transform_points(points,np.linalg.inv(world_body)).astype(np.float32)
        sources.append((body,100.+i*.1,world_body))
    cases=[]
    for radius,enabled in ((12.,False),(5.,False),(5.,True)):
        cache=TwoFrameCloudFusion();prepare_times=[];infer_times=[];counts=[];pillars=[];boxes=[]
        for i,(xyz,stamp,world_body) in enumerate(sources):
            start=time.perf_counter()
            xyzt=cache.fuse(xyz,stamp,world_body,enabled=enabled,radius=radius)
            normalized,model_world=model_frame_from_body(xyzt[:,:3],world_body,.17,1.8)
            features=np.column_stack((normalized,xyzt[:,3]))
            prep=time.perf_counter()-start
            _,coords=pillar_features(features,model.params)
            start=time.perf_counter();detected=model.infer(features,stamp,model_world);duration=time.perf_counter()-start
            if i:  # exclude the cold/first source from timing statistics
                prepare_times.append(prep);infer_times.append(duration)
                counts.append(dict(current=cache.stats['current_points'],history=cache.stats['history_points']))
                pillars.append(len(coords));boxes.append(dict(count=len(detected),classes=sorted({b['class'] for b in detected})))
        cases.append(dict(input_radius=radius,fusion_enabled=enabled,point_counts=counts,
            occupied_pillars=pillars,preparation_median_seconds=float(np.median(prepare_times)),
            inference_median_seconds=float(np.median(infer_times)),
            inference_p95_seconds=float(np.quantile(infer_times,.95)),synthetic_detections=boxes))
    current=sources[-1][0];current=current[np.linalg.norm(current[:,:2],axis=1)<=5.]
    base=current+[0.,0.,.17];ground_filter=LocalGroundFilter(anchor_tolerance=.03)
    nonground=ground_filter.nonground_mask(base)
    result=dict(date='2026-11-07',backend=args.backend,synthetic_only=True,
        campus_accuracy_measured=False,vehicle_parallel_load_measured=False,
        geometry_uses_current_frame=True,model_classes=[n.lower() for n in model.params['class_names']],
        actor_motion_per_scan_m=.06,ego_translation_per_scan_m=.04,
        nominal_source_interval_s=.1,cases=cases,
        geometry_ground_filter={**ground_filter.stats,'input_points':len(current),'retained_points':int(nonground.sum())},
        notes=['Detection counts on synthetic surfaces are not AP, recall, or ID metrics.',
               'Point ROI changes occupied pillars; fixed model BEV/head size stays unchanged.',
               'Source transforms align ego motion; moving actor history remains separated by its real displacement.'])
    model.close();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
