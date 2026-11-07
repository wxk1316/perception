# Bundled pretrained models

`centerpoint/base` and `centerpoint/tiny` contain unchanged pretrained ONNX and
parameter files from AutowareFoundation/lidar_centerpoint, revision
`494c8171def40bd36cc2feb323e0a5acbfab132b` (v4.1).

Source and model card: https://huggingface.co/AutowareFoundation/lidar_centerpoint

The model card declares Apache-2.0. Each variant retains the model card, the
Apache License 2.0 text, the training-data legal notice and a SHA-256 download
manifest. Retain them when copying or distributing the model bundle.

The campus NumPy pre/post-processing, TensorRT Python integration and tracking
code are project code. Deterministic pillar sampling, class remapping, NMS and
geometric fusion differ from the full Autoware pipeline; Autoware benchmark
accuracy and speed do not certify this implementation or the Helios vehicle.
