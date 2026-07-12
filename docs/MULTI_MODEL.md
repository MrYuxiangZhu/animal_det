# 多模型训练与推理说明

工程现在支持 3 类模型，通过 `run_train.sh` 和 `run_infer.sh` 自由选择。

## 1. tiny_detector

这是工程原本从零实现的轻量 YOLO 风格动物检测器。

训练：

```bash
bash run_train.sh tiny_detector configs/default.yaml
```

推理：

```bash
bash run_infer.sh tiny_detector configs/default.yaml samples/demo.mp4 outputs/inference/tiny_result.mp4
```

特点：

- 需要检测框标注。
- 适合学习传统单阶段检测器的完整训练流程。
- 模型源码：`src/models/backbone.py`、`src/models/detector.py`、`src/models/loss.py`。

## 2. clip

这是一个学习版 MiniCLIP，不依赖 OpenAI CLIP 源码，保留 CLIP 的核心思想：图像编码器和文本编码器通过对比损失对齐。

训练数据需要分类目录：

```text
data/animal_classification/train/cat/*.jpg
data/animal_classification/train/dog/*.jpg
data/animal_classification/val/cat/*.jpg
```

训练：

```bash
bash run_train.sh clip configs/default.yaml
```

图片零样本/候选类别分类：

```bash
bash run_infer.sh clip configs/default.yaml samples/cat.jpg "" "cat,dog,horse,fox"
```

说明：

- CLIP 是分类/图文检索模型，本身不直接输出检测框。
- 如果要在视频中定位动物，需要先用检测器裁剪候选动物，再用 CLIP 识别类别。
- 本工程的 `clip` 入口主要用于学习图文对比学习、零样本分类推理逻辑。

核心源码：

- `src/models/clip_like.py`
- `src/train_clip.py`
- `src/infer_clip.py`

## 3. grounding_dino

这是 Grounding DINO 思想的学习版适配，不直接复制官方复杂实现，保留开放词汇检测的关键结构：

- 文本查询编码器；
- 图像空间特征；
- 图文相似度作为类别响应；
- 网格框回归；
- 文本条件检测损失。

训练：

```bash
bash run_train.sh grounding_dino configs/default.yaml
```

用文本指定动物类别进行检测：

```bash
bash run_infer.sh grounding_dino configs/default.yaml samples/demo.mp4 outputs/inference/grounding_result.mp4 "cat,dog,horse,fox"
```

特点：

- 需要检测框标注。
- 推理时可以传入文本候选动物名称。
- 如果传入训练时未见过的新动物名称，由于本工程是轻量学习版，泛化能力取决于训练数据和文本编码学习效果，不等同官方预训练 Grounding DINO 的开放词汇能力。

核心源码：

- `src/models/grounding_dino_like.py`
- `src/models/grounding_loss.py`
- `src/train_grounding.py`
- `src/infer_grounding.py`

## 与官方 CLIP / Grounding DINO 的关系

本工程目标是学习完整源码，因此默认实现为教学版：

- 不直接调用 OpenAI CLIP 或 Grounding DINO 的现成模型完成任务。
- 保留核心机制，代码更短、更容易逐行理解。
- 如果后续要接入官方仓库，可以在 `src/models/adapters/` 里新增官方模型 wrapper，并复用现有 `run_train.sh` / `run_infer.sh` 模型选择机制。

## 脚本参数

训练：

```bash
bash run_train.sh <模型名> <配置文件>
```

推理：

```bash
bash run_infer.sh <模型名> <配置文件> <输入路径> <输出路径> <文本类别>
```

可选模型名：

```text
tiny_detector
clip
grounding_dino
```
