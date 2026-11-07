---
license: apache-2.0
pipeline_tag: object-detection
tags:
  - autoware
  - lidar
  - point-cloud
  - 3d-object-detection
  - centerpoint
  - pointpillars
  - tensorrt
  - onnx
  - ros2
  - autonomous-driving
---

# CenterPoint for Autoware (`lidar_centerpoint`)

3D object detection models for LiDAR point clouds, used by the
[`autoware_lidar_centerpoint`](https://github.com/autowarefoundation/autoware_universe/tree/main/perception/autoware_lidar_centerpoint)
node in [Autoware](https://github.com/autowarefoundation/autoware).

The models follow the **CenterPoint** [1] architecture with a **PointPillars** [2] voxel encoder and run with
TensorRT inside Autoware. They are exported as ONNX so they can be deployed across hardware; Autoware builds the
TensorRT engine from the ONNX file on first launch.

## Model overview

| | |
| --- | --- |
| Task | 3D object detection (oriented bounding boxes) from a LiDAR point cloud |
| Architecture | CenterPoint detection head on a PointPillars-style voxel encoder |
| Detected classes | `CAR`, `TRUCK`, `BUS`, `BICYCLE`, `PEDESTRIAN` |
| Runtime | TensorRT (FP16 by default, FP32 selectable) via the `autoware_lidar_centerpoint` ROS 2 node |
| Format | ONNX (Autoware builds the TensorRT engine locally on first launch) |
| License | Apache-2.0 (see Legal Notice for training-data terms) |

The network is split into two ONNX sub-models, matching how the node consumes them:

1. **Voxel encoder** — `pts_voxel_encoder.onnx`
   - input: `input_features` → output: `pillar_features`
2. **Backbone / neck / head** — `pts_backbone_neck_head.onnx`
   - input: `spatial_features` → outputs: `heatmap`, `reg`, `height`, `dim`, `rot`, `vel`

Pre-processing (voxelization, multi-frame densification) and post-processing (circle NMS, IoU NMS, yaw
normalization, distance-based score thresholding) run in the node, not in the ONNX graphs.

## Repository layout

One self-contained folder per variant, all folders carrying the same file set:

```
lidar_centerpoint/
├── base/            # model_name: centerpoint
├── tiny/            # model_name: centerpoint_tiny
├── sigma/           # model_name: centerpoint_sigma
└── short_range/     # model_name: centerpoint_short_range
    ├── pts_voxel_encoder.onnx               # voxel encoder
    ├── pts_backbone_neck_head.onnx          # backbone/neck/head
    ├── ml_package.param.yaml                # model manifest: parameters and artifact file names
    └── detection_class_remapper.param.yaml  # area-based class remapping
```

Autoware's perception launcher selects the variant folder from the model name; the
`autoware_lidar_centerpoint` node launch takes the folder directly as `model_path`
(e.g. `~/autoware_data/ml_models/lidar_centerpoint/tiny`).

## Variants in this repository

| Variant | Voxel size (x, y, z) [m] | Point cloud range [m] | Downsample factor | Encoder input features |
| --- | --- | --- | --- | --- |
| `centerpoint` (`base/`) | `0.32, 0.32, 10.0` | `[-76.8, -76.8, -4.0, 76.8, 76.8, 6.0]` | 1 | 9 |
| `centerpoint_tiny` (`tiny/`) | `0.32, 0.32, 10.0` | `[-76.8, -76.8, -4.0, 76.8, 76.8, 6.0]` | 2 | 9 |
| `centerpoint_sigma` (`sigma/`) | `0.32, 0.32, 10.0` | `[-76.8, -76.8, -4.0, 76.8, 76.8, 6.0]` | 1 | 9 |
| `centerpoint_short_range` (`short_range/`) | `0.16, 0.16, 8.0` | `[-51.2, -51.2, -3.0, 51.2, 51.2, 5.0]` | 4 | 10 |

Common model parameters: `point_feature_size: 4`. The default and tiny variants use a 480×480 BEV grid;
`centerpoint_tiny` halves the backbone output resolution via `downsample_factor: 2`, trading accuracy for
speed. `centerpoint_short_range` is a dedicated close-range model for redundant VRU detection.
`centerpoint_sigma` predicts per-regression-head variance and twist, so its head outputs carry twice the
channels of the other variants and its manifest sets `has_variance: true` and `has_twist: true`.

> **TensorRT engines are not distributed here.** TensorRT engines are specific to the GPU architecture and
> TensorRT version they are built on and are not portable, so Autoware builds them locally from the ONNX files
> on first launch (or via `build_only:=true`); the engine files are written into the variant folder.

## Inputs and outputs (as used by the node)

**Input** — `~/input/pointcloud` (`sensor_msgs/msg/PointCloud2`), with points of the form:

```cpp
struct InputPointType {
  float    x;
  float    y;
  float    z;
  uint8_t  intensity;
  uint8_t  return_type;
  uint16_t channel;
};
```

**Output** — `~/output/objects` (`autoware_perception_msgs/msg/DetectedObjects`): oriented 3D boxes with class
and score. `object.existence_probability` carries the DNN classification confidence (not a calibrated
probability).

## Usage in Autoware

Autoware's ansible artifacts role downloads this repository (pinned to a version tag) to
`~/autoware_data/ml_models/lidar_centerpoint/`. Launch with, e.g.:

```bash
ros2 launch autoware_lidar_centerpoint lidar_centerpoint.launch.xml \
  model_path:=$HOME/autoware_data/ml_models/lidar_centerpoint/tiny
```

`model_path` is the only model argument: it points at the variant folder, from which the node loads
`ml_package.param.yaml` as the manifest and resolves its relative entries against that same folder.

The manifest declares the bundle version, and the node checks it against the range it supports. A
bundle and a node from different layouts have to be updated together.

Add `build_only:=true` to build the TensorRT engine from the ONNX as a one-off pre-task.
See the [package README](https://github.com/autowarefoundation/autoware_universe/tree/main/perception/autoware_lidar_centerpoint)
for the full parameter reference and the training/deployment guide.

## Revisions

| Tag | Layout |
| --- | --- |
| `v4.1` | Manifests declare the bundle `version` and carry `model_params.yaw_norm_thresholds`; `deploy_metadata.yaml` dropped; required by `autoware_lidar_centerpoint` from this version on |
| `v4.0` | Per-variant folders (`base/`, `tiny/`, `sigma/`, `short_range/`) with uniform filenames; first publication of `sigma` and `short_range` weights |
| `v3.0` | Flat layout, filenames suffixed by variant; `sigma` parameter file only, no `short_range` |

## Training data

Models were trained with the Autoware fork of [MMDetection3D](https://github.com/autowarefoundation/mmdetection3d):

- **`centerpoint`** — nuScenes (~28k LiDAR frames) [3] + TIER IV internal data (~11k LiDAR frames), 60 epochs.
- **`centerpoint_tiny`** — Argoverse 2 (~110k LiDAR frames) [4] + TIER IV internal data (~11k LiDAR frames), 20 epochs.
- **`centerpoint_sigma`**, **`centerpoint_short_range`** — Training datasets, schedules, and evaluation metrics are not publicly documented.

A 600-frame sample dataset (T4 format, 5 classes; sensors: 1× Velodyne VLS128, 4× Velodyne VLP16, 1× Robosense
RS Bpearl) is available for evaluation and fine-tuning, as documented in the package README.

## Limitations

- Trained primarily on nuScenes / Argoverse 2 sensor configurations; accuracy on a different LiDAR setup
  (mounting position, beam count, concatenated clouds) can drop without fine-tuning.
- Only the five classes above are detected. Other road users fall outside the label set.

## Citation

```bibtex
@article{yin2021centerpoint,
  title   = {Center-based 3D Object Detection and Tracking},
  author  = {Yin, Tianwei and Zhou, Xingyi and Kr{\"a}henb{\"u}hl, Philipp},
  journal = {arXiv preprint arXiv:2006.11275},
  year    = {2020}
}
@inproceedings{lang2019pointpillars,
  title     = {PointPillars: Fast Encoders for Object Detection from Point Clouds},
  author    = {Lang, Alex H. and Vora, Sourabh and Caesar, Holger and Zhou, Lubing and Yang, Jiong and Beijbom, Oscar},
  booktitle = {CVPR},
  year      = {2019}
}
```

## References

- [1] Yin et al., "Center-based 3D Object Detection and Tracking", arXiv:2006.11275, 2020.
- [2] Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds", CVPR 2019.
- [3] nuScenes — <https://www.nuscenes.org/nuscenes>
- [4] Argoverse 2 — <https://www.argoverse.org/av2.html>

## Acknowledgment

Special thanks to [Deepen AI](https://www.deepen.ai/) for the 3D annotation tools used to create the sample dataset.

## Legal Notice

_The nuScenes dataset is released publicly for non-commercial use under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International Public License. Additional Terms of Use can be found at
<https://www.nuscenes.org/terms-of-use>. To inquire about a commercial license please contact
<nuscenes@motional.com>._
