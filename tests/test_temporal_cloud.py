"""Full label head, exact two-frame alignment and near-footprint regression."""
from pathlib import Path
import sys
import math
import numpy as np
import pytest
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'campus_perception'))
from campus_perception.temporal_cloud import TwoFrameCloudFusion, rectangle_distance, nearby_obstacles
from campus_perception.campus_cloud import pose_matrix, transform_points
from campus_perception.semantics import MODEL_CLASSES, planner_obstacles
from campus_perception.kf_tracker import GlobalKFTracker
from campus_perception.centerpoint_model import decode_predictions, CenterPointModel


def pose(x=0.,yaw=0.):
    return pose_matrix((x,0.,0.),(0.,0.,math.sin(yaw/2),math.cos(yaw/2)))


@pytest.mark.parametrize('yaw',[0.,.08,math.pi/2])
def test_stored_source_pose_aligns_history_to_current_body(yaw):
    cache=TwoFrameCloudFusion()
    previous=np.array([[2.,1.,.1]],dtype=np.float32)
    cache.fuse(previous,100.,pose(),radius=5.)
    current=cache.fuse(previous,100.1,pose(.04,yaw),radius=5.)
    np.testing.assert_allclose(transform_points(current[1:,:3],pose(.04,yaw)),previous,atol=2e-6)
    np.testing.assert_allclose(current[:,3],[0.,.1],atol=1e-6)


def test_only_last_raw_frame_is_cached_not_recursive_fusion():
    cache=TwoFrameCloudFusion()
    for i in range(5):
        out=cache.fuse([[1+i*.1,0.,0.]],100+i*.1,np.eye(4))
        assert len(out)==(1 if i==0 else 2)
    np.testing.assert_allclose(out[:,0],[1.4,1.3])
    assert cache.stats['previous_source_stamp']==pytest.approx(100.3)


def test_gap_empty_and_switch_never_reintroduce_stale_measurements():
    cache=TwoFrameCloudFusion()
    cache.fuse([[1.,0.,0.]],100.,pose())
    assert len(cache.fuse([[2.,0.,0.]],100.2,pose()))==1
    assert cache.stats['mode']=='gap_single_frame'
    assert len(cache.fuse(np.empty((0,3)),100.3,pose()))==0
    cache.fuse([[1.,0.,0.]],100.4,pose(),enabled=False)
    assert len(cache.fuse([[2.,0.,0.]],100.5,pose(),enabled=True))==1
    assert len(cache.fuse([[3.,0.,0.]],100.6,pose(),enabled=True))==2


def test_history_is_cropped_after_alignment_in_current_radius():
    cache=TwoFrameCloudFusion()
    cache.fuse([[5.05,0.,0.],[4.95,0.,0.]],100.,pose())
    out=cache.fuse([[1.,0.,0.]],100.1,pose(.1),radius=5.)
    assert len(out)==3  # the previous 5.05m return enters the current 5m ROI
    assert np.all(np.linalg.norm(out[:,:2],axis=1)<=5.)


def test_ego_alignment_does_not_remove_actor_motion():
    cache=TwoFrameCloudFusion()
    cache.fuse([[2.,0.,0.]],100.,pose())
    out=cache.fuse([[2.06,0.,0.]],100.1,pose())
    assert out[0,0]-out[1,0]==pytest.approx(.06,abs=1e-6)


@pytest.mark.parametrize('stamp',[100.,99.,float('nan')])
def test_invalid_source_time_is_rejected(stamp):
    cache=TwoFrameCloudFusion()
    cache.fuse([[1.,0.,0.]],100.,pose())
    with pytest.raises(ValueError):cache.fuse([[1.,0.,0.]],stamp,pose())


def test_near_range_uses_footprint_and_rotation():
    assert rectangle_distance(6.,0.,4.,1.)==4.
    assert rectangle_distance(6.,0.,4.,1.,yaw=math.pi/2)==pytest.approx(5.5)
    items=[(1,6.,0.,0.,0.,'bus',4.,1.),(2,6.,0.,0.,0.,'car',1.,1.)]
    assert [x[0] for x in nearby_obstacles(items,(0.,0.),5.)]==[1]


def params():
    return yaml.safe_load((ROOT/'campus_perception/models/centerpoint/base/ml_package.param.yaml').read_text())['/**']['ros__parameters']['model_params']


def heads():
    out={name:np.zeros((1,size,480,480),np.float32) for name,size in
         [('heatmap',5),('reg',2),('height',1),('dim',3),('rot',2),('vel',2)]}
    out['heatmap'].fill(-20);out['rot'][:,1]=1
    return out


def test_all_five_trained_classes_are_decoded_separately():
    outputs=heads()
    for kind,column in enumerate((227,234,240,246,253)):
        outputs['heatmap'][0,kind,240,column]=6.
    boxes=decode_predictions(outputs,params(),100.,pose(),max_range=5.)
    assert {b['class'] for b in boxes}==set(MODEL_CLASSES)
    assert 'vehicle' not in {b['class'] for b in boxes}


def test_decoder_does_not_drop_large_near_box_based_on_center():
    out=heads();out['heatmap'][0,2,240,260]=6.
    out['dim'][0,:,240,260]=np.log([1.,4.,2.])
    boxes=decode_predictions(out,params(),100.,pose(),max_range=5.)
    assert len(boxes)==1 and boxes[0]['class']=='bus' and boxes[0]['x']>5.


@pytest.mark.parametrize('kind',MODEL_CLASSES)
def test_tracker_keeps_semantics_and_planner_adapter_separates_motion(kind):
    tracker=GlobalKFTracker(class_speed_limits={name:1. for name in MODEL_CLASSES},preserve_semantics=True)
    out=tracker.step([(2.,1.,kind,.5,.5,.9)],stamp=100.)[0]
    assert out[5]==kind and out[3]<0
    assert planner_obstacles([out])[0][5]=='unknown'
    trusted=(1,2.,1.,.3,0.,kind,.5,.5)
    assert planner_obstacles([trusted])[0][5]==('vehicle' if kind in ('car','truck','bus') else kind)


def test_inference_accepts_external_age_without_hidden_history():
    model=CenterPointModel.__new__(CenterPointModel)
    model.params=params();model.spatial=np.zeros((1,32,480,480),np.float32)
    model.score_threshold=.35;model.max_range=5.
    class Encoder:
        def infer(self,features):
            self.features=features.copy()
            return {'pillar_features':np.zeros((len(features),32),np.float32)}
    class Head:
        def infer(self,_):return heads()
    model.encoder,model.head=Encoder(),Head()
    model.infer(np.array([[2.,0.,.1]],np.float32),100.,pose())
    model.infer(np.array([[3.,0.,.1]],np.float32),100.1,pose())
    assert len(model.encoder.features)==1  # no previous inference cloud appended
    model.infer(np.array([[3.,0.,.1,0.],[2.,0.,.1,.1]],np.float32),100.2,pose())
    assert np.max(model.encoder.features[:,:,3])==pytest.approx(.1)
