# Animal Detection / Recognition 工程使用指南

本文档整理当前工程的环境安装、数据准备、训练、推理、日志输出和检查命令。

## 1. 工程入口脚本

所有运行脚本都在 `script/` 目录下：

```text
script/
  setup_conda_env.sh
  run_train.sh
  run_train_detection.sh
  run_train_recognition.sh
  run_infer.sh
  run_infer_detection.sh
  run_infer_recognition.sh
  run_infer_specialized.sh
```

推荐优先使用按任务划分的脚本：

```text
检测模型训练: script/run_train_detection.sh
识别模型训练: script/run_train_recognition.sh
检测模型推理: script/run_infer_detection.sh
识别模型推理: script/run_infer_recognition.sh
专用模型推理: script/run_infer_specialized.sh
环境安装:     script/setup_conda_env.sh
```

## 2. 环境安装

### tiny detector 环境

```bash
bash script/setup_conda_env.sh tiny_detector
```

### timm / ViT 识别环境

```bash
bash script/setup_conda_env.sh timm
```

### OpenCLIP 环境

```bash
bash script/setup_conda_env.sh openclip
```

### 其他模型环境

```bash
bash script/setup_conda_env.sh clip
bash script/setup_conda_env.sh grounding_dino
bash script/setup_conda_env.sh yolov5
bash script/setup_conda_env.sh mmdetection
bash script/setup_conda_env.sh detectron2
bash script/setup_conda_env.sh superanimal
bash script/setup_conda_env.sh pytorch_wildlife
bash script/setup_conda_env.sh birder
```

安装全部环境：

```bash
bash script/setup_conda_env.sh all
```

## 3. 数据准备

### 检测模型数据格式

检测模型使用 YOLO 格式数据：

```text
data/animal_detection/images/train/*.jpg
data/animal_detection/images/val/*.jpg
data/animal_detection/labels/train/*.txt
data/animal_detection/labels/val/*.txt
```

每个 label 文件每行：

```text
class_id center_x center_y width height
```

坐标需要归一化到 `0-1`。

### 识别模型数据格式

识别模型使用分类目录格式：

```text
data/animal_classification/train/cat/*.jpg
data/animal_classification/train/dog/*.jpg
data/animal_classification/val/cat/*.jpg
data/animal_classification/val/dog/*.jpg
```

类别名需要和 `configs/default.yaml` 中的 `data.class_names` 对应。

## 4. 检测模型训练

推荐命令：

```bash
bash script/run_train_detection.sh tiny_detector configs/default.yaml
```

其他检测模型：

```bash
bash script/run_train_detection.sh grounding_dino configs/default.yaml
bash script/run_train_detection.sh yolov5 configs/default.yaml
bash script/run_train_detection.sh mmdetection configs/default.yaml
bash script/run_train_detection.sh detectron2 configs/default.yaml
```

## 5. 检测模型推理

推荐命令：

```bash
bash script/run_infer_detection.sh tiny_detector configs/default.yaml samples/demo.mp4 outputs/inference/tiny_result.mp4
```

GroundingDINO-like 可输入文本类别：

```bash
bash script/run_infer_detection.sh grounding_dino configs/default.yaml samples/demo.mp4 outputs/inference/grounding_result.mp4 "cat,dog,fox"
```

其他检测模型：

```bash
bash script/run_infer_detection.sh yolov5 configs/default.yaml samples/demo.mp4 outputs/inference/yolov5_result.mp4
bash script/run_infer_detection.sh mmdetection configs/default.yaml samples/demo.jpg outputs/inference/mmdet_result.jpg
bash script/run_infer_detection.sh detectron2 configs/default.yaml samples/demo.jpg outputs/inference/detectron2_result.jpg
```

## 6. 识别模型训练

推荐使用 Transformer 识别模型 `timm`，默认配置为 ViT：

```yaml
timm:
  model_name: vit_base_patch16_224
```

训练命令：

```bash
bash script/run_train_recognition.sh timm configs/default.yaml
```

CLIP 学习版训练：

```bash
bash script/run_train_recognition.sh clip configs/default.yaml
```

## 7. 识别模型推理

`timm` 图像识别：

```bash
bash script/run_infer_recognition.sh timm configs/default.yaml animal.jpg outputs/inference/timm.txt
```

CLIP / OpenCLIP 零样本识别：

```bash
bash script/run_infer_recognition.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "red panda,fox,bear"
```

或者使用专用入口：

```bash
bash script/run_infer_specialized.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "red panda,fox,bear"
```

## 8. 专用动物 Transformer / 行为分析模型

### OpenCLIP

```bash
bash script/run_infer_specialized.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "red panda,fox,bear"
```

### SuperAnimal / DeepLabCut

```bash
bash script/run_infer_specialized.sh superanimal configs/default.yaml animal_video.mp4 outputs/superanimal/result.mp4
```

### Microsoft CameraTraps / Pytorch-Wildlife

```bash
bash script/run_infer_specialized.sh pytorch_wildlife configs/default.yaml camera_trap_dir outputs/wildlife/result.json
```

### Birder-MViT

```bash
bash script/run_infer_specialized.sh birder configs/default.yaml bird.jpg outputs/inference/birder.txt
```

## 9. 日志和输出文件

训练和推理日志默认保存到：

```text
logs/
```

### 训练日志

例如 tiny detector 训练会输出：

```text
logs/train_tiny_detector.log
logs/train_tiny_detector_metrics.jsonl
logs/train_tiny_detector_metrics.csv
```

记录内容包括：

```text
epoch
phase
step
total loss
box loss
objectness loss
class loss
train/val 汇总指标
```

### 推理日志

例如 tiny detector 推理会输出：

```text
logs/infer_tiny_detector.log
logs/infer_tiny_detector_detections.jsonl
logs/infer_tiny_detector_detections.csv
```

检测结果 CSV 字段：

```text
source,frame_idx,det_id,class_id,class_name,score,x1,y1,x2,y2
```

其中 `x1,y1,x2,y2` 是原始视频帧或图片坐标系下的 2D box。

## 10. 训练输出

默认输出目录：

```text
outputs/
```

常见输出：

```text
outputs/checkpoints/best.pt
outputs/loss_curve.png
outputs/inference/*.mp4
outputs/inference/*.jpg
outputs/inference/*.txt
```

## 11. 工程检查命令

修改代码或脚本后建议执行：

```bash
chmod +x script/*.sh envs/*.sh
bash -n script/run_train.sh
bash -n script/run_infer.sh
bash -n script/run_train_detection.sh
bash -n script/run_train_recognition.sh
bash -n script/run_infer_detection.sh
bash -n script/run_infer_recognition.sh
bash -n script/run_infer_specialized.sh
bash -n script/setup_conda_env.sh
python3 -m compileall src
```

当前已检查通过：

```text
所有 shell 脚本语法检查通过
python3 -m compileall src 通过
linter 检查结果: No linter errors found
```

## 12. 推荐使用顺序

如果你刚开始跑工程，推荐顺序：

```bash
bash script/setup_conda_env.sh tiny_detector
bash script/run_train_detection.sh tiny_detector configs/default.yaml
bash script/run_infer_detection.sh tiny_detector configs/default.yaml samples/demo.mp4 outputs/inference/tiny_result.mp4
```

如果你要训练 Transformer 识别模型：

```bash
bash script/setup_conda_env.sh timm
bash script/run_train_recognition.sh timm configs/default.yaml
```

如果你要做开放词汇动物识别：

```bash
bash script/setup_conda_env.sh openclip
bash script/run_infer_specialized.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "red panda,fox,bear"
```
