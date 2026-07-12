# 工业级模型适配说明

本工程现在同时支持教学源码模型和工业级框架适配模型。

## 支持模型

```text
tiny_detector    本工程从零实现的轻量检测器
clip             本工程从零实现的 MiniCLIP 学习版
grounding_dino   本工程从零实现的 GroundingDINO-like 学习版
yolov5           Ultralytics YOLOv5 官方仓库适配
timm             timm 分类模型适配
mmdetection      OpenMMLab MMDetection 适配
detectron2       Meta Detectron2 适配
```

## 统一训练入口

```bash
bash run_train.sh tiny_detector configs/default.yaml
bash run_train.sh clip configs/default.yaml
bash run_train.sh grounding_dino configs/default.yaml
bash run_train.sh yolov5 configs/default.yaml
bash run_train.sh timm configs/default.yaml
bash run_train.sh mmdetection configs/default.yaml
bash run_train.sh detectron2 configs/default.yaml
```

## 统一推理入口

```bash
bash run_infer.sh tiny_detector configs/default.yaml input.mp4 outputs/inference/tiny.mp4
bash run_infer.sh grounding_dino configs/default.yaml input.mp4 outputs/inference/grounding.mp4 "cat,dog,fox"
bash run_infer.sh yolov5 configs/default.yaml input.mp4 outputs/inference/yolov5.mp4
bash run_infer.sh timm configs/default.yaml input.jpg outputs/inference/timm.txt
bash run_infer.sh mmdetection configs/default.yaml input.jpg outputs/inference/mmdet.jpg
bash run_infer.sh detectron2 configs/default.yaml input.jpg outputs/inference/detectron2.jpg
```

## YOLOv5

YOLOv5 使用官方仓库训练和推理。本工程负责：

1. 读取 `configs/default.yaml`；
2. 自动生成 YOLOv5 需要的 `configs/yolov5_animal.yaml`；
3. 调用官方 `train.py` / `detect.py`；
4. 统一日志输出到 `logs/train_yolov5.log`、`logs/infer_yolov5.log`。

准备：

```bash
mkdir -p third_party
git clone https://github.com/ultralytics/yolov5 third_party/yolov5
pip install -r third_party/yolov5/requirements.txt
```

数据格式沿用 YOLO 检测格式。

## timm

`timm` 用于动物分类迁移学习，不是目标检测。数据需要按类别文件夹组织：

```text
data/animal_classification/train/cat/*.jpg
data/animal_classification/train/dog/*.jpg
data/animal_classification/val/cat/*.jpg
data/animal_classification/val/dog/*.jpg
```

训练：

```bash
bash run_train.sh timm configs/default.yaml
```

推理输出 top-k 文本结果：

```bash
bash run_infer.sh timm configs/default.yaml samples/cat.jpg outputs/inference/timm_result.txt
```

## MMDetection

MMDetection 适合系统学习工业级检测框架。由于它本身是配置驱动，本工程只做统一入口适配。

安装参考：

```bash
pip install -U openmim
mim install mmengine mmcv mmdet
```

你需要准备自己的 MMDetection 配置文件，例如：

```text
configs/mmdet/animal_faster_rcnn.py
```

并在 `configs/default.yaml` 中设置：

```yaml
mmdetection:
  config: configs/mmdet/animal_faster_rcnn.py
  checkpoint: outputs/mmdetection/animal_mmdet/best.pth
```

训练：

```bash
bash run_train.sh mmdetection configs/default.yaml
```

## Detectron2

Detectron2 当前适配器使用 COCO 格式数据。

目录示例：

```text
data/animal_coco/annotations/instances_train.json
data/animal_coco/annotations/instances_val.json
data/animal_coco/train/*.jpg
data/animal_coco/val/*.jpg
```

安装请参考官方文档：

```text
https://detectron2.readthedocs.io/
```

训练：

```bash
bash run_train.sh detectron2 configs/default.yaml
```

推理：

```bash
bash run_infer.sh detectron2 configs/default.yaml samples/cat.jpg outputs/inference/detectron2.jpg
```

## 教学源码与工业框架的区别

- `tiny_detector`、`clip`、`grounding_dino`：核心网络和 loss 都在本工程源码中，适合学习原理。
- `yolov5`、`timm`、`mmdetection`、`detectron2`：适合学习工业级工程架构和实际落地流程，本工程通过 adapter 统一调用。

如果你的目标是先理解算法，建议顺序：

```text
tiny_detector -> clip -> grounding_dino -> yolov5 -> timm -> mmdetection/detectron2
```
