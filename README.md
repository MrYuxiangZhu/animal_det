# Animal Video Detector

这是一个用于学习目标检测完整流程的轻量动物检测工程。代码从零实现了 CNN Backbone、检测头、YOLO 风格解码、损失函数、训练循环、视频/图片推理、日志系统和 loss 曲线保存；没有直接调用 YOLO、Faster R-CNN 等现成检测模型库。

## 工程结构

```text
animal_det/
  configs/                 # YAML 配置
  docs/                    # 数据集和使用说明
  logs/                    # 训练、推理日志输出目录
  outputs/                 # checkpoint、曲线、推理结果
  src/
    data/                  # 数据读取、预处理、公开数据整理
    models/                # Backbone、检测头、损失函数源码
    utils/                 # 配置、日志、框操作、可视化工具
    train.py               # 训练入口
    infer.py               # 图片/视频推理入口
  environment.yml          # conda 环境描述
  requirements.txt         # pip 依赖
  setup_conda_env.sh       # 自动化环境安装脚本
```

## 1. 创建环境

所有运行脚本现在统一放在 `script/` 目录中。完整使用方法见 `docs/USAGE.md`。

```bash
bash script/setup_conda_env.sh tiny_detector
bash script/setup_conda_env.sh timm
bash script/setup_conda_env.sh openclip
```

如果需要 GPU，请确保本机 CUDA 驱动与 PyTorch 版本匹配。脚本默认通过 pip 安装 PyTorch，必要时可按 [PyTorch 官网](https://pytorch.org/) 的命令替换安装。

## 2. 准备公开动物数据集

训练数据采用 YOLO 格式，详见 `docs/DATASET.md`。

目标目录示例：

```text
data/animal_detection/images/train/*.jpg
data/animal_detection/images/val/*.jpg
data/animal_detection/labels/train/*.txt
data/animal_detection/labels/val/*.txt
```

标注文件每行：

```text
class_id center_x center_y width height
```

`configs/default.yaml` 默认类别为：cat、dog、horse、cow、sheep、elephant、bear、zebra、giraffe。

## 3. 选择模型开始训练

现在支持 3 个模型入口：

```text
tiny_detector    从零实现的轻量 YOLO 风格检测器
clip             学习版 CLIP，用于图文对比学习和零样本动物分类
grounding_dino   学习版 Grounding DINO，用文本查询做开放词汇动物检测
```

训练命令可以按任务区分：

```bash
bash run_train_detection.sh tiny_detector configs/default.yaml
bash run_train_detection.sh grounding_dino configs/default.yaml
bash run_train_recognition.sh timm configs/default.yaml
bash run_train_recognition.sh clip configs/default.yaml
```

也可以继续使用统一入口：

```bash
bash run_train.sh tiny_detector configs/default.yaml
bash run_train.sh clip configs/default.yaml
bash run_train.sh grounding_dino configs/default.yaml
```

训练过程会生成：

- 日志：`logs/*.log`
- 最优模型：`outputs/checkpoints/**/best.pt`
- loss 曲线：`outputs/*loss_curve.png`

更多说明见 `docs/MULTI_MODEL.md`。

## 4. 视频或图片推理

普通检测器推理：

```bash
bash run_infer.sh tiny_detector configs/default.yaml /path/to/animal.mp4 outputs/inference/tiny_result.mp4
```

CLIP 图片分类，可指定候选动物文本：

```bash
bash run_infer.sh clip configs/default.yaml /path/to/animal.jpg "" "cat,dog,horse,fox"
```

GroundingDINO-like 文本条件检测：

```bash
bash run_infer.sh grounding_dino configs/default.yaml /path/to/animal.mp4 outputs/inference/grounding_result.mp4 "cat,dog,horse,fox"
```

## 5. 模型源码说明

核心模型在 `src/models/`：

- `backbone.py`：实现 `ConvBNAct`、`ResidualBlock`、`TinyBackbone`。
- `detector.py`：实现 `AnimalDetector` 检测网络和 `decode_predictions` 解码函数。
- `loss.py`：实现 anchor 匹配、框回归损失、objectness 损失和类别损失。

推理后处理在 `src/utils/box_ops.py`，包括 `xywh_to_xyxy`、IoU 和纯 PyTorch NMS。

## 6. 重要配置

常用可调参数在 `configs/default.yaml`：

- `data.image_size`：输入分辨率。
- `data.class_names` / `model.num_classes`：类别名和类别数，二者要一致。
- `model.anchors`：归一化 anchor 宽高，可以根据数据集目标大小统计后调整。
- `train.batch_size`、`train.epochs`、`train.learning_rate`：训练超参数。
- `infer.conf_threshold`、`infer.iou_threshold`：推理置信度和 NMS 阈值。

## 7. 学习建议

1. 先用少量公开动物图片跑通训练，确认 loss 能下降。
2. 可视化检查标注格式是否正确，尤其是类别 ID 和归一化坐标。
3. 根据目标尺度调整 anchors。小动物多时减小 anchor，大动物多时增大 anchor。
4. 如果视频推理漏检严重，可降低 `conf_threshold`，但误检会增加。
