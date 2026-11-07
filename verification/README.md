# 独立包验证记录

本独立包已迁移并运行47项离线测试，覆盖KF/匈牙利、合簇/交叉、速度与语义、源时间、地面、SE(3)补偿、模型特征/解码及ROS测试替身。包含扩展消息 shape_heading 与速度 heading 分离回归。

`pytest_results.xml` 是本独立包测试记录。`package_checks.json` 记录Python3.10语法、模型hash与wheel打包检查。base/tiny CPU合成模型探针见对应JSON。

本机 macOS ARM64，无ROS DDS、RViz GUI、Orin CUDA或真实车辆；wheel构建不等于ROS colcon车机构建。未声明已达到10Hz或已测校园精度。

可在有NumPy/SciPy/PyYAML/pytest的Python环境中，从仓库根目录运行：

```bash
python3 -m pytest -q tests
```

ROS测试使用替身，不需要启动ROS或车辆。真实CPU ONNX探针另需onnx和onnxruntime，仅做可加载/有限值合成验证：

```bash
python3 tools/validate_campus_model.py --backend onnxruntime \
  --model-directory campus_perception/models/centerpoint/base --output /tmp/base_model_check.json
```

车机TensorRT探针要求先生成engine：

```bash
python3 tools/validate_campus_model.py --backend tensorrt \
  --model-directory "$MODEL_DIR" --output /tmp/orin_model_check.json
```

预训练权重原始文件hash保存在每个变体的bundle.lock.json，源码/交付文件hash在仓库SOURCE_MANIFEST.json。
