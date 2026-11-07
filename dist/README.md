# 源码打包下载

campus_perception_20261107.zip 包含独立 ROS2 感知包、base/tiny ONNX、测试与文档。
它不包含定位/控制源码、已编译 TensorRT engine 或 Git 历史。
同目录 SHA256 用于下载校验。解压到新的感知工作区 src/perception 后按 docs/INSTALL_zh.md 构建。
本ZIP排除dist目录，避免归档递归包含自己。SOURCE_MANIFEST.json记录归档中的源码/权重/文档hash。
