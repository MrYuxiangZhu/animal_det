# TINY DETECTOR LOSS CURVE ANALYSIS

本文档重新分析 `tiny_detector` 训练 loss 曲线异常问题，并特别对比 `openclip` loss 曲线为什么更正常。

执行命令：

```bash
bash script/run_train_detection.sh tiny_detector configs/coco_animals_10cls.yaml
```

观察到的 Tiny Detector 曲线：

- `train_total` 从约 `0.45` 持续下降到接近 `0.01`；
- `train_box`、`train_obj`、`train_cls` 都快速下降；
- `val_total` 前期从约 `0.40` 降到约 `0.29`，随后持续上升到约 `0.62`；
- 训练集和验证集走势严重背离。

对比 OpenCLIP 曲线：

- `train_loss` 下降；
- `val_loss` 也整体下降并趋于稳定；
- 训练集和验证集走势一致；
- 泛化表现明显更正常。

因此，问题不能简单归因为“数据集整体有问题”。更准确地说：

```text
OpenCLIP 当前训练目标更容易泛化；
Tiny Detector 当前检测模型、loss 和训练策略存在明显短板。
```

---

## 1. Tiny Detector 有没有使用 backbone

有。

`tiny_detector` 使用了项目内自定义的 `TinyBackbone`。

核心代码在：

```text
src/models/detector.py
src/models/backbone.py
```

`AnimalDetector` 初始化时：

```python
self.backbone = TinyBackbone(width_mult=width_mult)
```

前向传播时：

```python
feats = self.neck(self.backbone(x))
pred = self.head(feats)
```

所以 Tiny Detector 的结构是：

```text
input image
    |
    v
TinyBackbone
    |
    v
neck
    |
    v
detection head
    |
    v
[B, A, H, W, 5 + C]
```

但是需要强调：

```text
Tiny Detector 用了 backbone，但这个 backbone 是从零训练的轻量 CNN，不是 ImageNet/CLIP 预训练 backbone。
```

这正是 Tiny Detector 和 OpenCLIP 曲线差异的重要原因之一。

---

## 2. OpenCLIP 曲线为什么比 Tiny Detector 好

OpenCLIP 的 loss 下降更稳定，不代表检测任务一定更简单，而是因为当前两个训练任务本质差别很大。

### 2.1 OpenCLIP 很可能使用了预训练视觉/文本编码器

配置中 OpenCLIP 相关字段包括：

```yaml
openclip:
  model_name: ViT-B-32
  checkpoint_path: weights/openclip/open_clip_model.safetensors
  freeze_encoder: true
```

这说明 OpenCLIP 更接近“基于预训练模型做轻量适配/分类对齐”。它的视觉特征不是从零学起，而是已经具备较强泛化能力。

因此 OpenCLIP 训练时通常是在一个较好的特征空间里优化分类或对比目标，loss 更容易稳定下降。

### 2.2 Tiny Detector 是从零训练检测器

Tiny Detector 的 backbone 是：

```text
ConvBNAct + ResidualBlock + stride=2 downsample
```

它没有加载预训练权重。训练一开始，模型不知道动物边缘、纹理、部件、形状等通用视觉特征，需要从 detection 数据里从零学习：

- 图像低级特征；
- 动物语义特征；
- 目标中心定位；
- 宽高回归；
- objectness；
- 类别预测。

这比 OpenCLIP 的分类/识别训练困难很多。

### 2.3 OpenCLIP 做的是识别，Tiny Detector 做的是检测

OpenCLIP 曲线对应的通常是图像级识别或图文对齐任务。每张图主要优化“这张图是什么类别”。

Tiny Detector 优化的是更复杂的检测目标：

```text
目标在哪 + 目标多大 + 是否有目标 + 是什么类别
```

也就是同时学习：

- localization；
- objectness；
- classification。

其中 localization 和 objectness 对训练策略非常敏感。即使类别学得很好，只要框回归或 objectness 不稳定，验证 loss 仍可能上升。

### 2.4 OpenCLIP 的验证 loss 更像语义泛化，Tiny Detector 的验证 loss 受标注框和 anchor 强影响

检测任务的 loss 受很多额外因素影响：

- anchor 是否适配；
- 小目标是否能匹配到合适网格；
- 一个 cell 内多个目标如何处理；
- objectness 正负样本比例；
- NMS 前候选框质量；
- 标签框质量；
- 训练/验证目标尺度分布。

所以 OpenCLIP 正常下降，并不能推出 Tiny Detector 也应该自然下降。

---

## 3. 重新判断：Tiny Detector 的主要问题不是“有没有 backbone”，而是 backbone 和检测训练设计太弱

当前 Tiny Detector 的问题更准确地说是：

```text
有 backbone，但 backbone 从零训练、单尺度、无预训练、无强增强、loss 简化，导致训练集可记住，验证集泛化差。
```

当前曲线不是“模型完全学不会”。相反，它在训练集上学得太好了：

```text
train loss 接近 0
```

这说明模型容量足够记忆训练集。

真正的问题是：

```text
学到的东西不能稳定迁移到验证集。
```

---

## 4. 最关键原因一：TinyBackbone 从零训练，泛化弱

### 4.1 当前 TinyBackbone 结构

输入 `416 x 416` 时，backbone 大约经过 5 次下采样：

```text
416 -> 208 -> 104 -> 52 -> 26 -> 13
```

最后只输出一个 `13 x 13` 特征图。

这个 backbone 的优点是简单、快、容易理解。

缺点是：

- 没有预训练；
- 语义表达能力有限；
- 小目标特征在多次下采样后容易消失；
- 只有最终单尺度特征，没有 FPN；
- 对 COCO animal 这种复杂场景泛化能力弱。

### 4.2 为什么 OpenCLIP 不容易出现这种问题

OpenCLIP 使用的 ViT-B/32 或类似模型已经通过大规模数据预训练，视觉特征更稳健。

TinyBackbone 则只靠当前数据训练。如果检测数据数量不够、标注有噪声、目标尺度复杂，训练集可以记住，验证集容易崩。

### 4.3 解决方案

优先级从低成本到高收益：

1. 给 Tiny Detector 使用预训练 backbone。
2. 或者接入 `timm` 的预训练轻量 CNN，比如：
   - `mobilenetv3_small_100`；
   - `efficientnet_b0`；
   - `resnet18`；
   - `convnext_tiny`。
3. 冻结前几层，只训练 neck/head，再逐步解冻。
4. 如果继续使用自定义 TinyBackbone，需要大幅增强数据增强和正则化。

---

## 5. 最关键原因二：单尺度 13x13 检测头不适合 COCO animal

### 5.1 当前结构限制

Tiny Detector 只在最后一层特征图上预测：

```text
13 x 13 grid
```

每个 grid 对应原图约：

```text
416 / 13 = 32 px
```

对于 COCO 中的动物，小目标非常常见，例如：

- 远处的 bird；
- 小猫小狗；
- 人群或背景中的动物；
- 被遮挡的动物；
- 宽高比例极端的 giraffe、zebra、horse。

单个 `13 x 13` 输出层对这些目标不友好。

### 5.2 训练曲线如何体现这个问题

训练集上的小目标、特殊姿态、特殊场景，模型可以靠记忆拟合。

但验证集中只要出现位置、尺度、场景变化，单尺度检测头就难以泛化，表现为：

```text
train loss 继续下降
val loss 开始上升
```

### 5.3 解决方案

最有效的结构改造是增加多尺度检测头：

```text
52 x 52: 小目标
26 x 26: 中目标
13 x 13: 大目标
```

如果暂时不改结构，短期可以：

```yaml
data:
  image_size: 640
```

这样最终特征图大约变为：

```text
640 / 32 = 20
```

虽然仍然单尺度，但空间分辨率比 `13 x 13` 好一些。

---

## 6. 最关键原因三：当前 DetectionLoss 过于简化

### 6.1 当前 loss 组成

当前 total loss：

```text
total = 5.0 * box_loss + obj_loss + cls_loss
```

其中：

- `box_loss`：SmoothL1；
- `obj_loss`：BCEWithLogits，加正负样本权重；
- `cls_loss`：BCEWithLogits。

这个实现可以跑通，但和成熟 YOLO loss 相比，缺少几个关键机制。

### 6.2 缺少 ignore mask

YOLO 类检测器通常不会把所有非 best-anchor 都当作负样本。

当前实现中，一个真实目标只会匹配一个 best anchor：

```text
best_anchor -> positive
其他 anchor -> negative
```

问题是：

```text
同一个目标附近的其他 anchor 可能也预测得不错，但被当成负样本惩罚。
```

这会让 objectness 学习不稳定，特别是在验证集上更容易出现高 loss。

### 6.3 box loss 不是 IoU loss

当前 box loss 比较的是：

```text
[sigmoid(tx), sigmoid(ty), exp(tw)*anchor_w, exp(th)*anchor_h]
```

和 target：

```text
[offset_x, offset_y, w, h]
```

这能训练框，但并不直接优化最终框和真实框之间的 IoU。

成熟检测器更常用：

- GIoU；
- DIoU；
- CIoU；
- SIoU。

这些 loss 更贴近检测指标。

### 6.4 objectness 正负样本极不平衡

以 `416 -> 13 x 13`、3 anchors 为例，每张图有：

```text
13 * 13 * 3 = 507 个候选位置
```

如果一张图只有 1 到 3 个动物，那么正样本只有几个，其余几百个都是负样本。

当前虽然用了权重：

```text
positive weight = 5.0
negative weight = 0.5
```

但仍然不如 Focal Loss 稳定。

### 6.5 解决方案

建议按顺序改：

1. 增加 ignore mask。
2. objectness 换成 Focal Loss 或 BCE + ignore mask。
3. box loss 换成 CIoU loss。
4. 输出和 target 构造统一到完整归一化 `xyxy` 或 `cxcywh` 空间。

---

## 7. 最关键原因四：anchor 与 COCO animal 10 类不匹配

当前 anchor：

```yaml
anchors:
  - [0.08, 0.10]
  - [0.16, 0.20]
  - [0.32, 0.40]
```

这些 anchor 是手工写死的，不一定适配 COCO animal。

COCO animal 中目标尺度和长宽比差异非常大：

- bird 可能很小；
- giraffe 很高很窄；
- zebra/horse 可能宽高比较大；
- elephant/cow 可能很大；
- bear/cat/dog 可能尺度跨度大。

3 个 anchor 很难覆盖这些情况。

### 7.1 解决方案

短期：先用更宽的 anchor 覆盖范围：

```yaml
anchors:
  - [0.04, 0.05]
  - [0.10, 0.13]
  - [0.22, 0.30]
```

或者增加为 5 个 anchor，需要同步改 `num_anchors`：

```yaml
num_anchors: 5
anchors:
  - [0.03, 0.04]
  - [0.07, 0.10]
  - [0.14, 0.20]
  - [0.28, 0.36]
  - [0.48, 0.60]
```

更推荐：写脚本对训练集 YOLO 标签做 k-means 聚类，自动得到 anchor。

---

## 8. 为什么不能只看 loss 判断 Tiny Detector 一定差

需要注意：检测 loss 和 OpenCLIP loss 不完全可比。

OpenCLIP loss 是识别/对比任务 loss。

Tiny Detector loss 是检测任务 loss，包含：

```text
box + objectness + classification
```

而且验证 loss 上升不一定等价于 mAP 完全下降，但当前曲线上升幅度较大，确实说明训练策略或模型设计有明显问题。

真正应该补充的评估指标是：

- Precision；
- Recall；
- mAP@0.5；
- mAP@0.5:0.95；
- 每类 AP；
- 小/中/大目标 AP。

如果只看 loss，容易误判检测效果。

---

## 9. 推荐解决方案：分三阶段做

## 阶段一：不改模型结构，先稳定训练

目标：验证是否主要是过拟合、学习率、anchor 问题。

推荐配置：

```yaml
data:
  image_size: 416

model:
  num_classes: 10
  width_mult: 0.5
  num_anchors: 3
  anchors:
    - [0.04, 0.05]
    - [0.10, 0.13]
    - [0.22, 0.30]

tiny_detector:
  num_classes: 10
  width_mult: 0.5
  num_anchors: 3
  anchors:
    - [0.04, 0.05]
    - [0.10, 0.13]
    - [0.22, 0.30]
  epochs: 20
  batch_size: 8
  learning_rate: 0.0003
  weight_decay: 0.002
  checkpoint_interval: 1
  resume: ""
  device: auto
```

理由：

- `width_mult: 0.5` 降低从零训练模型的记忆能力；
- `learning_rate: 0.0003` 避免快速过拟合；
- `weight_decay: 0.002` 增强正则；
- `epochs: 20` 避免 50 轮长时间过拟合；
- `checkpoint_interval: 1` 方便选最佳 epoch；
- anchor 更偏向覆盖小中目标。

同时增加 early stopping：

```text
patience = 5
monitor = val_total
```

如果第 8 到第 12 轮验证 loss 最低，就不要继续训练到 50 轮。

---

## 阶段二：改数据和 loss，让检测训练更像 YOLO

目标：修复训练机制，而不是只靠调参。

### 9.1 数据增强

训练集加入：

- random horizontal flip；
- random HSV；
- random brightness/contrast；
- random scale；
- mild affine；
- 后续可加 Mosaic。

验证集必须保持无随机增强。

### 9.2 Anchor 聚类

新增脚本统计：

```text
train labels 中所有 bbox 的 w/h
```

用 k-means 生成适合当前数据集的 anchor。

### 9.3 Loss 改造

建议优先改：

1. ignore mask；
2. CIoU loss；
3. Focal objectness loss。

推荐目标 loss：

```text
total = box_gain * ciou_loss + obj_gain * focal_obj_loss + cls_gain * cls_loss
```

---

## 阶段三：升级 backbone 和检测头

目标：让 Tiny Detector 真正适合 COCO animal 检测。

### 10.1 用预训练 backbone 替换 TinyBackbone

建议接入 `timm`：

- `mobilenetv3_small_100`：轻量；
- `efficientnet_b0`：效果和速度平衡；
- `resnet18`：简单稳定；
- `convnext_tiny`：效果更强但更重。

推荐路径：

```text
timm pretrained backbone -> neck -> detection head
```

先冻结 backbone 前几层，只训练 neck/head。

### 10.2 增加多尺度检测头

从 backbone 取多个阶段特征：

```text
C3 -> 52x52
C4 -> 26x26
C5 -> 13x13
```

用 FPN/PAN 融合后输出多尺度预测。

最终结构更接近：

```text
input
  |
pretrained backbone
  |
FPN/PAN
  |------ P3 52x52 small objects
  |------ P4 26x26 medium objects
  |------ P5 13x13 large objects
```

这会比当前单尺度 head 更适合 COCO animal。

---

## 10. 最推荐的实际执行顺序

不要一次性全改。建议按下面顺序：

### Step 1：确认 Tiny Detector 确实在用 backbone

已确认：

```text
AnimalDetector -> TinyBackbone -> neck -> head
```

问题不是没用 backbone，而是 backbone 从零训练且能力有限。

### Step 2：先做配置修复

修改 `configs/coco_animals_10cls.yaml`：

```yaml
tiny_detector:
  num_classes: 10
  width_mult: 0.5
  num_anchors: 3
  anchors:
    - [0.04, 0.05]
    - [0.10, 0.13]
    - [0.22, 0.30]
  epochs: 20
  batch_size: 8
  learning_rate: 0.0003
  weight_decay: 0.002
  checkpoint_interval: 1
  resume: ""
  device: auto
```

### Step 3：增加 early stopping

防止验证 loss 已经变差还继续训练。

### Step 4：增加 train-only augmentation

先加水平翻转和颜色扰动，不要一开始上 Mosaic。

### Step 5：做 anchor k-means

用真实训练集标签统计 anchor，不要手工猜。

### Step 6：补 mAP 评估

不要只看 loss。每轮验证后计算：

```text
mAP@0.5
precision
recall
per-class AP
```

### Step 7：如果目标是明显提升效果，换预训练 backbone + 多尺度 head

这是长期最有效方案。

---

## 11. 最终结论

OpenCLIP 的 loss 比 Tiny Detector 好，不是因为 Tiny Detector 没有 backbone。

准确结论是：

```text
Tiny Detector 有 backbone，但它是从零训练的轻量 TinyBackbone；
OpenCLIP 通常依赖大规模预训练模型，天然泛化更好；
Tiny Detector 当前还是单尺度、anchor 简化、loss 简化、无强增强的教学型检测器；
所以 Tiny Detector train loss 能降到很低，但 val loss 会明显反升。
```

当前最优先解决的不是继续训练更久，而是：

1. 减小学习率；
2. 增强正则；
3. 降低模型容量或加预训练；
4. 加 early stopping；
5. 做 anchor 聚类；
6. 加数据增强；
7. 改进 loss；
8. 最后升级为预训练 backbone + 多尺度检测头。

如果只想快速让曲线不那么异常，先用：

```yaml
tiny_detector:
  width_mult: 0.5
  epochs: 20
  learning_rate: 0.0003
  weight_decay: 0.002
  checkpoint_interval: 1
  anchors:
    - [0.04, 0.05]
    - [0.10, 0.13]
    - [0.22, 0.30]
```

如果想让 Tiny Detector 真正接近 OpenCLIP 那种泛化稳定性，关键是引入预训练 backbone 和更成熟的检测训练机制。
