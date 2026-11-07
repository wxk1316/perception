"""Campus regression scenarios: identity, coverage and source-time contracts."""
from pathlib import Path
from types import SimpleNamespace as NS
import math
import struct
import sys
import time
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'campus_perception')]
from campus_perception.campus_cloud import (DecodedCloud,decode_cloud,deskew_cloud,
    sensor_matrix,transform_points,LocalGroundFilter,semantic_clusters,pose_matrix,model_frame_from_body)
from campus_perception.centerpoint_model import pillar_features,decode_predictions
from campus_perception.kf_tracker import GlobalKFTracker,HungarianMatcher
from campus_perception.obstacle_fusion import Clusterer,WorldObstacleRepresenter,VoxelCluster
from campus_perception.campus_model_worker import ModelWorker
from campus_perception.campus_fusion import fuse_coverage,expand_publication_delay,MODEL_ID_OFFSET

SPEEDS = {'pedestrian':.6,'vehicle':1.,'bicycle':1.}


def tracker(**kwargs):
    options = dict(max_speed=1.,class_speed_limits=SPEEDS,preserve_semantics=True,
                   dyn_confirm_seconds=.3,static_confirm_seconds=1.2,motion_observation_window=.5,
                   lost_ttl_seconds=.6,velocity_max_age=.25)
    options.update(kwargs)
    return GlobalKFTracker(**options)


def cloud_message(points,stamp=100.,field='timestamp',big=False):
    endian = '>' if big else '<'
    fields = [NS(name=name,offset=i*4,datatype=7,count=1) for i,name in enumerate(('x','y','z'))]
    fields.append(NS(name=field,offset=12,datatype=8,count=1))
    raw = b''.join(struct.pack(endian+'fffd',*point)+b'PAD!' for point in points)
    return NS(fields=fields,point_step=20,row_step=24,width=1,height=len(points),
              data=raw,is_bigendian=big,header=NS(stamp=NS(sec=int(stamp),nanosec=round((stamp-int(stamp))*1e9))))


class CampusTrackingTests(unittest.TestCase):
    def test_semantic_classes_survive_motion_and_stationary_confirmation(self):
        for kind,speed in (('pedestrian',.45),('vehicle',.8),('bicycle',.75)):
            t = tracker()
            for i in range(20):
                out = t.step([(2+i*.1*speed,1,kind,.3,.3,.9)],stamp=100+i*.1)[0]
            self.assertEqual(out[5],kind)
            self.assertAlmostEqual(out[3],speed,delta=.03)
            stopped_x,stopped_y = out[1],out[2]
            for i in range(35):
                out = t.step([(stopped_x,stopped_y,kind,.3,.3,.9)],stamp=102+i*.1)[0]
            self.assertEqual(out[5],kind)
            self.assertTrue(t.tracks[1].static_confirmed)

    def test_pedestrian_above_bound_is_unknown_and_still_present(self):
        t = tracker()
        for i in range(10):
            out = t.step([(2+i*.09,1,'pedestrian',.3,.3,.9)],stamp=100+i*.1)[0]
        self.assertEqual(out[5],'pedestrian')
        self.assertLess(out[3],0.)
        self.assertFalse(t.tracks[1].velocity_trusted)
        self.assertAlmostEqual(out[1],2.81)

    def test_equal_centres_different_classes_keep_identity(self):
        t = tracker()
        t.step([(2,1,'vehicle',2.,1.,.9),(2,1,'pedestrian',.3,.3,.9)],stamp=100)
        t.step([(2,1,'pedestrian',.3,.3,.9),(2,1,'vehicle',2.,1.,.9)],stamp=100.1)
        self.assertEqual(t.last_detection_ids,{0:2,1:1})

    def test_one_classification_flip_does_not_change_id_or_class(self):
        t = tracker()
        for i in range(5):
            t.step([(2+i*.04,1,'bicycle',.8,.4,.9)],stamp=100+i*.1)
        out = t.step([(2.2,1,'pedestrian',.8,.4,.8)],stamp=100.5)[0]
        self.assertEqual(t.last_detection_ids,{0:1})
        self.assertEqual(out[5],'bicycle')

    def test_stationary_person_starting_slowly_releases_static_state(self):
        t=tracker(dyn_speed_thr=.12,static_speed_thr=.06)
        for i in range(20):t.step([(2.,1.,'pedestrian',.3,.3,.9)],stamp=100+i*.1)
        self.assertTrue(t.tracks[1].static_confirmed)
        for i in range(20):out=t.step([(2.+(i+1)*.01,1.,'pedestrian',.3,.3,.9)],stamp=102+i*.1)[0]
        self.assertFalse(t.tracks[1].static_confirmed)
        self.assertGreater(out[3],.06)
        self.assertEqual(out[5],'pedestrian')

    def test_two_pedestrians_cross_without_ids_swapping(self):
        t = tracker()
        for i in range(31):
            dets = [(2-.6+i*.04,1,'pedestrian',.25,.25,.9),
                    (2+.6-i*.04,1.1,'pedestrian',.35,.3,.9)]
            if i%2:
                dets.reverse()
            t.step(dets,stamp=100+i*.1)
            self.assertEqual(t.last_detection_ids,{0:2,1:1} if i%2 else {0:1,1:2})

    def test_merged_geometry_holds_parent_motion_and_recovers_ids(self):
        t = tracker()
        for i in range(9):
            t.step([(-.5+i*.04,2,'unknown',.2,.2),(.5-i*.04,2,'unknown',.2,.2)],stamp=100+i*.1)
        before = [t.tracks[i].hits for i in (1,2)]
        t.step([(0,2,'unknown',.7,.25)],stamp=100.9)
        self.assertEqual(t.stats['merged_observations'],1)
        self.assertEqual([t.tracks[i].hits for i in (1,2)],before)
        self.assertEqual(t.tracks[3].motion_status,'merged_observation')
        t.step([(-.1,2,'unknown',.2,.2),(.1,2,'unknown',.2,.2)],stamp=101.)
        self.assertEqual(t.last_detection_ids,{0:1,1:2})

    def test_expired_track_cannot_be_revived(self):
        t = tracker(lost_ttl_seconds=.5)
        t.step([(2,1,'unknown',.3,.3)],stamp=100)
        t.step([(2,1,'unknown',.3,.3)],stamp=100.6)
        self.assertEqual(t.last_detection_ids,{0:2})

    def test_confirmation_duration_is_independent_of_nominal_dt(self):
        t = tracker(dt=.05,dyn_speed_thr=.1)
        for i in range(6):
            t.step([(2+i*.04,1,'pedestrian',.3,.3)],stamp=100+i*.1)
        self.assertTrue(t.tracks[1].moving)
        self.assertEqual(t.tracks[1]._mv_required,.3)

    def test_missing_scipy_uses_global_optimum_and_rejection(self):
        import campus_perception.kf_tracker as module
        previous = module._HAS_SCIPY
        try:
            module._HAS_SCIPY = False
            pairs,_ = HungarianMatcher(1).match([(0,0),(0,0)],[(0,0),(0,0)],
                                                costs=np.array([[.1,.11],[.11,.8]]),unmatched_cost=1.)
            self.assertEqual(set(pairs),{(0,1),(1,0)})
            pairs,unmatched = HungarianMatcher(1).match([(0,0)],[(0,0)],costs=np.array([[.95]]),unmatched_cost=.4)
            self.assertEqual((pairs,unmatched),([],[0]))
        finally:
            module._HAS_SCIPY = previous

    def test_nonfinite_time_is_rejected(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                tracker().step([],stamp=value)

    def test_delayed_model_snapshot_does_not_repeat_kf_update(self):
        t = tracker()
        for i in range(10):
            t.step([(2+i*.08,1,'vehicle',4.,1.8,.9)],stamp=100+i*.1,now=100+i*.1)
        before=t.tracks[1].x.copy();hits=t.tracks[1].hits
        snapshot=t.snapshot(101.1)[0]
        np.testing.assert_array_equal(t.tracks[1].x,before)
        self.assertEqual(t.tracks[1].hits,hits)
        self.assertGreater(snapshot[1],float(before[0]))
        self.assertEqual(snapshot[5],'vehicle')
        # Includes the opposite-direction reachable position after an occlusion.
        self.assertLessEqual(snapshot[1]-snapshot[6]/2,2+.72-1.*.2-2.)
        self.assertEqual(t.snapshot(101.3)[0][5],'vehicle')
        self.assertLess(t.snapshot(101.3)[0][3],0.)
        self.assertEqual(t.snapshot(101.6),[])

    def test_low_speed_walking_with_centroid_noise_keeps_id_and_velocity(self):
        t = tracker();rng = np.random.default_rng(20261107)
        ids,unknown_velocity = [],[]
        for i in range(100):
            point = np.array([2+i*.04,1])+rng.normal(0,.025,2)
            t.step([(*point,'pedestrian',.3,.3,.9)],stamp=100+i*.1)
            tid = t.last_detection_ids[0]
            ids.append(tid);unknown_velocity.append(next(out[3]<0 for out in t._output() if out[0]==tid))
        self.assertEqual(len(set(ids)),1)
        self.assertLess(sum(unknown_velocity[20:]),8)


class CampusCloudTests(unittest.TestCase):
    def test_organized_padding_big_endian_and_absolute_time(self):
        for big in (False,True):
            msg = cloud_message([(1,0,-.4,99.95),(2,0,-.4,100.)],big=big)
            result = decode_cloud(msg)
            np.testing.assert_allclose(result.xyz,[[1,0,-.4],[2,0,-.4]])
            np.testing.assert_allclose(result.point_times,[99.95,100.])

    def test_relative_units_require_explicit_configuration(self):
        msg = cloud_message([(1,0,0,.05)],field='time')
        self.assertIsNone(decode_cloud(msg).point_times)
        np.testing.assert_allclose(decode_cloud(msg,time_mode='relative_seconds').point_times,[99.95])

    def test_clock_mismatch_never_fabricates_point_time(self):
        result = decode_cloud(cloud_message([(1,0,0,300.)]))
        self.assertIsNone(result.point_times)
        self.assertEqual(result.time_status,'time_outside_source_clock')

    def test_truncated_cloud_is_error_not_empty_free_space(self):
        msg = cloud_message([(1,0,0,100.)]);msg.data=msg.data[:8]
        with self.assertRaises(ValueError):decode_cloud(msg)

    def test_deskew_removes_ego_translation_and_rotation(self):
        poses = [(100.,(0,0,0)),(100.1,(.05,0,.08))]
        ext = dict(x=.1,y=.05,height=.45,roll=0.,pitch=0.,yaw=.1)
        world = np.array([[4.,1.,.5]])
        raw = np.concatenate([transform_points(world,np.linalg.inv(sensor_matrix(pose,**ext))) for _,pose in poses])
        corrected,status = deskew_cloud(DecodedCloud(raw,np.array([100.,100.1]),'validated'),100.1,poses,ext)
        expected = transform_points(world,np.linalg.inv(sensor_matrix(poses[-1][1],**ext)))
        np.testing.assert_allclose(corrected,np.repeat(expected,2,axis=0),atol=1e-6)
        self.assertEqual(status,'planar_compensated')
        _,status = deskew_cloud(DecodedCloud(raw,np.array([99.9,100.1]),'validated'),100.1,poses,ext)
        self.assertEqual(status,'pose_history_does_not_cover_scan')

    def test_full_pose_deskew_compensates_ramp_pitch_and_height(self):
        from scipy.spatial.transform import Rotation
        poses = [(100.,pose_matrix((0,0,0),(0,0,0,1))),
                 (100.1,pose_matrix((.05,0,.02),Rotation.from_euler('xyz',[.03,.10,.08]).as_quat()))]
        ext = dict(x=.1,y=.05,height=.45,roll=0.,pitch=.05,yaw=.1)
        world = np.array([[4.,1.,.5]])
        raw = np.concatenate([transform_points(world,np.linalg.inv(sensor_matrix(pose,**ext))) for _,pose in poses])
        corrected,status = deskew_cloud(DecodedCloud(raw,np.array([100.,100.1]),'validated'),100.1,[],ext,poses)
        expected = transform_points(world,np.linalg.inv(sensor_matrix(poses[-1][1],**ext)))
        np.testing.assert_allclose(corrected,np.repeat(expected,2,axis=0),atol=1e-6)
        self.assertEqual(status,'pose_3d_compensated')

    def test_height_normalization_keeps_physical_world_points_on_a_ramp(self):
        from scipy.spatial.transform import Rotation
        world = pose_matrix((2,3,.5),Rotation.from_euler('xyz',[.1,.2,.3]).as_quat())
        points = np.array([[3.,1.,-.12],[4.,2.,.5]],dtype=np.float32)
        model_points,model_world = model_frame_from_body(points,world,.17,1.8)
        np.testing.assert_allclose(transform_points(model_points,model_world),transform_points(points,world),atol=1e-6)

    def test_sloping_road_removed_low_curb_and_tree_retained(self):
        x,y = np.meshgrid(np.linspace(1,6,50),np.linspace(-3,3,40))
        road = np.column_stack((x.ravel(),y.ravel(),.02*x.ravel()))
        curb = np.array([(3.+i*.02,1.,.02*(3.+i*.02)+.07) for i in range(20)])
        tree = np.array([(4,0,.15+i*.05) for i in range(20)])
        mask = LocalGroundFilter(anchor_tolerance=.03).nonground_mask(np.concatenate((road,curb,tree)))
        self.assertLess(mask[:len(road)].mean(),.05)
        self.assertTrue(mask[len(road):].all())

    def test_elevated_platform_and_sparse_patch_remain_unknown(self):
        points = np.array([(x,y,.1) for x in np.linspace(1,2,10) for y in np.linspace(1,2,10)])
        self.assertTrue(LocalGroundFilter(anchor_tolerance=.03).nonground_mask(points).all())
        self.assertTrue(LocalGroundFilter().nonground_mask(points[:2]).all())

    def test_nonfinite_ground_parameters_cannot_remove_obstacles(self):
        for values in ({'anchor_tolerance':float('nan')},{'cell_size':float('inf')},{'max_slope_deg':float('nan')}):
            with self.assertRaises(ValueError):LocalGroundFilter(**values)

    def test_semantic_partition_preserves_residual_far_from_map_origin(self):
        points = [(100+i*.02,1) for i in range(5)]+[(103+i*.02,1) for i in range(5)]
        box = dict(x=100.,y=1.,yaw=0.,length=.4,width=.4,score=.9,stamp=100.,**{'class':'pedestrian'})
        groups = semantic_clusters(points,[box],Clusterer(),100.1,origin=(100.,0.))
        self.assertEqual(len(groups),2)
        self.assertEqual(groups[0].semantic_class,'pedestrian')
        self.assertFalse(hasattr(groups[1],'semantic_class'))
        old = semantic_clusters(points,[box],Clusterer(),101.,origin=(100.,0.))
        self.assertTrue(all(not hasattr(group,'semantic_class') for group in old))

    def test_known_vehicle_has_one_stable_id_across_tile_boundaries(self):
        representer = WorldObstacleRepresenter(tracker=tracker())
        ids=set()
        for i in range(10):
            points = [(2+i*.05+j*.05,1) for j in range(60)]
            group = VoxelCluster(points,0.)
            group.semantic_class,group.semantic_score = 'vehicle',.9
            out = representer.step([group],100+i*.1)
            self.assertEqual(len(out),1);ids.add(out[0][0])
        self.assertTrue(all(item[5]=='vehicle' for item in out))
        self.assertEqual(len(ids),1)

    def test_fusion_keeps_all_geometry_outside_model_envelope(self):
        geometry=[(1,2.,1.,0.,0.,'unknown',.3,.3),(2,5.,1.,0.,0.,'block',.3,.3)]
        models=[(1,2.,1.,0.,0.,'vehicle',4.,1.8)]
        result=fuse_coverage(geometry,models)
        self.assertEqual({item[0] for item in result},{2,MODEL_ID_OFFSET+1})

    def test_processing_delay_covers_source_age_and_revokes_stale_velocity(self):
        original=(MODEL_ID_OFFSET+1,2.,1.,.8,0.,'vehicle',4.,1.8)
        result=expand_publication_delay([original],.1,{original[0]:.2},SPEEDS)[0]
        self.assertEqual(result[1:3],original[1:3]) # source coordinate time is preserved
        self.assertEqual(result[3:6],(-1.,0.,'vehicle'))
        self.assertGreaterEqual(result[6],4.+2*1.*.1)


class CampusModelTests(unittest.TestCase):
    def setUp(self):
        import yaml
        self.params = yaml.safe_load((ROOT/'campus_perception/models/centerpoint/base/ml_package.param.yaml').read_text())['/**']['ros__parameters']['model_params']

    def test_features_use_history_age_and_correct_residuals(self):
        points = np.array([[.01,.02,-.45,0.],[.11,.12,-.35,.1]],dtype=np.float32)
        features,coords = pillar_features(points,self.params)
        self.assertEqual(features.shape,(1,32,9))
        np.testing.assert_allclose(features[0,:2,:4],points,atol=1e-6)
        np.testing.assert_allclose(features[0,:2,4:7],points[:,:3]-points[:,:3].mean(axis=0),atol=1e-6)
        self.assertEqual(float(features[0,1,3]),float(np.float32(.1)))
        self.assertTrue((features[0,2:]==0).all())
        np.testing.assert_equal(coords,[[240,240]])

    def test_pillar_dense_sampling_and_range_boundary(self):
        points = np.array([[.01+i*.001,.02,-.45,i*.001] for i in range(60)]+[[76.8,0,0,0]],dtype=np.float32)
        features,coords = pillar_features(points,self.params)
        self.assertEqual(len(coords),1)
        self.assertGreater(features[0,-1,3],.05)

    def test_head_decode_class_mapping_and_velocity_is_not_trusted(self):
        outputs = {name:np.zeros((1,size,480,480),dtype=np.float32) for name,size in
                   [('heatmap',5),('reg',2),('height',1),('dim',3),('rot',2),('vel',2)]}
        outputs['heatmap'].fill(-20)
        outputs['heatmap'][0,0,240,245] = 5
        outputs['dim'][0,:,240,245] = np.log([1.,2.,1.])
        outputs['rot'][0,1,240,245] = 1
        outputs['vel'].fill(100)
        boxes = decode_predictions(outputs,self.params,100.,np.eye(4))
        self.assertEqual(len(boxes),1)
        self.assertEqual(boxes[0]['class'],'car')
        self.assertAlmostEqual(boxes[0]['x'],1.6,places=5)
        self.assertNotIn('speed',boxes[0])

    def test_background_head_produces_no_objects(self):
        outputs = {name:np.zeros((1,size,480,480),dtype=np.float32) for name,size in
                   [('heatmap',5),('reg',2),('height',1),('dim',3),('rot',2),('vel',2)]}
        outputs['heatmap'].fill(-20)
        self.assertEqual(decode_predictions(outputs,self.params,100.,np.eye(4)),[])

    def test_worker_has_one_job_and_preserves_model_source_stamp(self):
        import threading
        release = threading.Event()
        class FakeModel:
            def __init__(self,*args,**kwargs):pass
            def infer(self,xyz,stamp,transform):
                release.wait(2)
                return [{'stamp':stamp,'class':'pedestrian'}]
            def close(self):pass
        worker = ModelWorker('',model_factory=FakeModel)
        try:
            deadline=time.monotonic()+2
            while worker.model is None and time.monotonic()<deadline:
                worker.collect();time.sleep(.001)
            self.assertTrue(worker.submit(np.zeros((1,3)),100.,np.eye(4)))
            self.assertFalse(worker.submit(np.zeros((1,3)),100.1,np.eye(4)))
            self.assertEqual(worker.stats['skipped_busy'],1)
            release.set()
            deadline=time.monotonic()+2
            while worker.future is not None and time.monotonic()<deadline:
                worker.collect();time.sleep(.001)
            self.assertEqual(worker.current_boxes(100.1)[0]['stamp'],100.)
            self.assertEqual(worker.current_boxes(100.5),[])
        finally:
            release.set();worker.close()


if __name__=='__main__':
    unittest.main()
