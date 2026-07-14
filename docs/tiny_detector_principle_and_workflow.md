# Tiny Detector 原理、实现过程与执行过程

本文档说明本项目中 `tiny_detector` 的整体设计思路、核心原理、代码实现结构、训练流程、推理流程以及运行方式。对应核心代码主要位于：

- `src/models/backbone.py`：轻量 CNN 主干网络与基础卷积模块
- `src/models/detector.py`：`AnimalDetector` 检测模型与预测解码逻辑
- `src/models/loss.py`：YOLO 风格检测损失 `DetectionLoss`
- `src/data/dataset.py`：检测数据集读取与 YOLO 标签解析
- `src/trainers/tiny_detector.py`：训练入口
- `src/trainers/detection_engine.py`：单轮训练/验证循环与 checkpoint 保存
- `src/inferencers/tiny_detector.py`：推理入口
- `src/inferencers/tiny_detector_core.py`：模型加载、单帧推理、后处理与绘制
- `configs/default.yaml`：数据、模型、训练、推理配置

---

## 1. Tiny Detector 是什么

`tiny_detector` 是项目中从零实现的轻量级动物目标检测模型。它采用类似 YOLO 的单阶段检测思想：

1. 输入一张图片；
2. 通过轻量 CNN 提取特征；
3. 在输出特征图的每个网格位置上，为多个 anchor 预测候选框；
4. 每个候选框同时预测：
   - 位置偏移；
   - 宽高；
   - 是否包含目标的 objectness；
   - 类别概率；
5. 推理阶段对所有候选框进行置信度筛选和 NMS，得到最终检测框。

它不是调用 YOLOv5、Detectron2、MMDetection 等外部工业框架，而是项目内自定义实现的一个轻量检测器，适合用于理解目标检测完整链路，也适合在小规模动物检测数据集上快速实验。

---

## 2. 整体架构

### 2.1 数据流总览

训练阶段的数据流如下：

```text
YOLO 格式图片和标签
        |
        v
AnimalDetectionDataset
        |
        v
letterbox 等比例缩放 + padding
        |
        v
image_to_tensor 转为 [C, H, W]
        |
        v
DataLoader 组成 batch
        |
        v
AnimalDetector 前向传播
        |
        v
DetectionLoss 构造 target 并计算 loss
        |
        v
反向传播 + AdamW 更新参数
        |
        v
保存 checkpoint / loss 曲线 / 日志
```

推理阶段的数据流如下：

```text
图片或视频帧
        |
        v
OpenCV 读取 BGR 图像
        |
        v
BGR 转 RGB
        |
        v
letterbox 到固定输入尺寸
        |
        v
image_to_tensor + batch 维度
        |
        v
AnimalDetector 输出 raw prediction
        |
        v
decode_predictions 解码 cxcywh/objectness/class_probs
        |
        v
conf_threshold 过滤低置信度框
        |
        v
按类别执行 NMS
        |
        v
坐标从 letterbox 图还原到原图
        |
        v
OpenCV 绘制检测框并保存结果
```

---

## 3. 配置文件

默认配置位于 `configs/default.yaml`。

### 3.1 数据配置

```yaml
data:
  dataset_name: animals10
  root: data/coco_animals/detection
  train_images: train/images
  val_images: val/images
  train_labels: train/labels
  val_labels: val/labels
  class_names: [dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel]
  image_size: 416
  num_workers: 4
```

含义：

- `root`：检测数据集根目录；
- `train_images` / `val_images`：训练集和验证集图片目录；
- `train_labels` / `val_labels`：训练集和验证集标签目录；
- `class_names`：类别 ID 到类别名称的映射；
- `image_size`：模型输入尺寸，默认 416，即输入被处理为 `416 x 416`；
- `num_workers`：DataLoader 读取数据的进程数。

标签采用 YOLO 格式，每张图片对应一个同名 `.txt` 文件，每行：

```text
class_id center_x center_y width height
```

其中 `center_x`、`center_y`、`width`、`height` 都是相对于原图宽高归一化到 `[0, 1]` 的值。

### 3.2 模型配置

```yaml
model:
  num_classes: 10
  width_mult: 0.75
  num_anchors: 3
  anchors:
    - [0.08, 0.10]
    - [0.16, 0.20]
    - [0.32, 0.40]
```

含义：

- `num_classes`：类别数；
- `width_mult`：网络宽度倍率，用于控制通道数，越小模型越轻；
- `num_anchors`：每个网格位置预测的 anchor 数量；
- `anchors`：anchor 宽高，使用归一化尺度。

### 3.3 训练配置

```yaml
train:
  epochs: 50
  batch_size: 8
  learning_rate: 0.001
  weight_decay: 0.0005
  checkpoint_interval: 5
  resume: ""
  device: auto
```

含义：

- `epochs`：训练轮数；
- `batch_size`：批大小；
- `learning_rate`：AdamW 学习率；
- `weight_decay`：权重衰减；
- `checkpoint_interval`：每隔多少 epoch 保存一次 checkpoint；
- `resume`：如果不为空，则从指定 checkpoint 恢复训练；
- `device`：运行设备，通常可设为 `auto`、`cuda`、`cuda:0` 或 `cpu`。

### 3.4 推理配置

```yaml
infer:
  checkpoint: outputs/checkpoints/best.pt
  source: samples/demo.mp4
  output: outputs/inference/demo_result.mp4
  conf_threshold: 0.35
  iou_threshold: 0.45
```

含义：

- `checkpoint`：推理加载的权重路径；
- `source`：输入图片、视频或其他支持的数据源；
- `output`：输出路径；
- `conf_threshold`：置信度阈值；
- `iou_threshold`：NMS IoU 阈值。

注意：训练脚本默认会创建类似 `outputs/tiny_detector/<timestamp>/checkpoints/best.pt` 的目录结构；如果推理配置中的 `infer.checkpoint` 仍指向 `outputs/checkpoints/best.pt`，需要根据实际训练输出路径手动修改配置或在命令行传入对应配置文件。

---

## 4. 模型原理

### 4.1 单阶段检测思想

Tiny Detector 属于单阶段检测器。单阶段检测器不先生成 proposal，而是在特征图上直接预测目标框和类别。

假设输入图片尺寸为 `416 x 416`，主干网络经过 5 次 stride=2 的下采样：

```text
416 -> 208 -> 104 -> 52 -> 26 -> 13
```

所以最终特征图大约是 `13 x 13`。模型在每个网格位置预测 `A` 个 anchor，本项目默认 `A = 3`。

如果类别数为 `C = 10`，那么每个 anchor 的预测维度是：

```text
5 + C = tx, ty, tw, th, objectness, class_logits...
```

因此模型最终输出形状为：

```text
[B, A, H, W, 5 + C]
```

默认情况下大致为：

```text
[B, 3, 13, 13, 15]
```

其中：

- `B`：batch size；
- `A`：anchor 数量；
- `H, W`：输出网格高宽；
- `5 + C`：每个 anchor 的预测向量。

### 4.2 Backbone：TinyBackbone

`TinyBackbone` 是轻量 CNN 主干，位于 `src/models/backbone.py`。

它由以下模块组成：

- `ConvBNAct`：卷积 + BatchNorm + SiLU；
- `ResidualBlock`：轻量残差块；
- 多个 stride=2 的卷积层：逐步降低分辨率、扩大感受野。

结构大致为：

```text
Input [3, 416, 416]
        |
ConvBNAct 3 -> 32, stride=2
        |
ConvBNAct 32 -> 64, stride=2
        |
ResidualBlock
        |
ConvBNAct 64 -> 128, stride=2
        |
ResidualBlock
        |
ConvBNAct 128 -> 256, stride=2
        |
ResidualBlock
        |
ConvBNAct 256 -> 512, stride=2
        |
ResidualBlock
        |
Output feature map [C_out, 13, 13]
```

由于使用了 `width_mult`，实际通道数会乘以该倍率。例如 `width_mult=0.75` 时，基础的 512 通道会变为约 384 通道。

### 4.3 Neck

`AnimalDetector` 中的 neck 是一个简单的卷积特征变换模块：

```text
ConvBNAct(out_channels, out_channels, 3, 1)
ConvBNAct(out_channels, out_channels // 2, 1, 1)
ConvBNAct(out_channels // 2, out_channels, 3, 1)
```

作用：

1. 进一步融合 backbone 输出的语义特征；
2. 通过 `1x1` 卷积压缩通道再恢复，降低计算量；
3. 为最后的检测 head 提供更适合预测框和类别的特征。

### 4.4 Detection Head

检测头是一个 `1x1` 卷积：

```python
nn.Conv2d(out_channels, num_anchors * (5 + num_classes), kernel_size=1)
```

它把每个网格位置的特征转换为 anchor 预测向量。

前向传播中的形状变化：

```text
head 输出: [B, A * (5 + C), H, W]
reshape : [B, A, 5 + C, H, W]
permute : [B, A, H, W, 5 + C]
```

这样后续 loss 和 decode 都可以按 anchor、网格位置和预测向量进行处理。

---

## 5. 预测框解码原理

模型直接输出的是 raw prediction，不是最终框。`decode_predictions` 负责把 raw 输出转换为可解释的框、目标置信度和类别概率。

### 5.1 中心点解码

对每个网格位置 `(grid_x, grid_y)`，模型输出 `tx, ty`。解码公式：

```text
cx = (sigmoid(tx) + grid_x) / W
cy = (sigmoid(ty) + grid_y) / H
```

解释：

- `sigmoid(tx)` 和 `sigmoid(ty)` 限制在 `[0, 1]`；
- 表示目标中心相对于当前网格左上角的偏移；
- 加上网格坐标后，再除以网格宽高，得到归一化中心点。

### 5.2 宽高解码

模型输出 `tw, th`，结合 anchor 解码：

```text
bw = exp(tw) * anchor_w
bh = exp(th) * anchor_h
```

解释：

- anchor 提供基础宽高；
- `exp` 让模型以乘法比例调整 anchor；
- 输出宽高仍是归一化尺度。

### 5.3 Objectness 和类别概率

```text
objectness = sigmoid(raw_objectness)
class_probs = sigmoid(class_logits)
```

最终每个类别的分数：

```text
score[class] = objectness * class_probs[class]
```

推理阶段会取分数最高的类别作为该框类别：

```text
score, label = max(score_per_class)
```

---

## 6. Loss 原理

`DetectionLoss` 位于 `src/models/loss.py`，包含三部分：

```text
total_loss = 5.0 * box_loss + obj_loss + cls_loss
```

### 6.1 Target 构造

输入标签格式为：

```text
[class_id, cx, cy, w, h]
```

这些坐标已经在 `AnimalDetectionDataset` 中转换成了相对于 letterbox 后 `image_size x image_size` 图像的归一化坐标。

对每个真实框：

1. 根据 `cx, cy` 找到它落在哪个网格：

```text
gx = int(cx * W)
gy = int(cy * H)
```

2. 计算真实框宽高与每个 anchor 的 IoU；
3. 选择 IoU 最大的 anchor 作为正样本 anchor；
4. 在 `(batch_index, best_anchor, gy, gx)` 位置写入：
   - `obj_target = 1`；
   - `box_target = [cx * W - gx, cy * H - gy, w, h]`；
   - `cls_target[class_id] = 1`；
   - `positive = True`。

没有匹配到目标的位置默认为负样本，即 `obj_target = 0`。

### 6.2 Box Loss

只在正样本位置计算框回归损失：

```text
box_loss = SmoothL1(pred_box, target_box)
```

其中：

- `pred_box` 包含预测的网格内中心偏移和 anchor 缩放后的宽高；
- `target_box` 是构造 target 时写入的真实框参数。

### 6.3 Objectness Loss

objectness 使用 `BCEWithLogitsLoss`，也就是对 raw logit 直接计算二分类交叉熵。

为了缓解正负样本极度不均衡，代码对 objectness loss 加权：

```text
正样本权重: 5.0
负样本权重: 0.5
```

这样模型会更重视真正包含目标的位置。

### 6.4 Class Loss

类别损失只在正样本位置计算：

```text
cls_loss = BCEWithLogits(class_logits, one_hot_class_target)
```

虽然每个框最终通常只属于一个类别，但这里使用 BCE 而不是 CrossEntropy，因此形式上是多标签分类损失。

---

## 7. 数据读取与预处理

### 7.1 图片读取

`AnimalDetectionDataset.__getitem__` 使用 OpenCV 读取图片：

```text
cv2.imread -> BGR
cv2.cvtColor -> RGB
```

模型训练和推理内部均使用 RGB 输入。

### 7.2 Letterbox

为了保持原图宽高比，项目使用 `letterbox` 进行等比例缩放和 padding，而不是直接拉伸。

例如原图为 `1280 x 720`，输入尺寸为 `416`：

1. 计算缩放比例：

```text
scale = min(416 / h, 416 / w)
```

2. 将原图等比例缩放；
3. 创建 `416 x 416` 的灰色画布，填充值为 114；
4. 把缩放后的图像放到画布中心；
5. 返回：
   - letterbox 后图像；
   - `scale`；
   - `(pad_x, pad_y)`。

训练时需要同步变换标签框；推理后需要用 `scale` 和 `pad` 把预测框还原回原图坐标。

### 7.3 Tensor 转换

`image_to_tensor` 做两件事：

1. `uint8 [0, 255]` 转为 `float32 [0, 1]`；
2. 维度从 `[H, W, C]` 转为 `[C, H, W]`。

最终 DataLoader 返回：

```text
images:  [B, 3, image_size, image_size]
targets: List[Tensor], 每张图一个 [N, 5] 的标签张量
metas:   图片路径等元信息
```

因为每张图目标数量不同，所以 `detection_collate` 不会把 targets 堆叠成一个大 tensor，而是保留为 list。

---

## 8. 训练实现过程

训练入口是：

```text
src/trainers/tiny_detector.py
```

### 8.1 启动命令

默认配置训练：

```bash
python -m src.trainers.tiny_detector --config configs/default.yaml
```

如果使用自定义配置：

```bash
python -m src.trainers.tiny_detector --config configs/coco_animals_10cls.yaml
```

### 8.2 训练入口 main 流程

`main()` 主要步骤：

1. 解析命令行参数；
2. 加载 YAML 配置；
3. 设置随机种子；
4. 创建训练输出目录；
5. 创建 checkpoint 目录；
6. 初始化 logger 和 metric tracker；
7. 选择运行设备；
8. 构建训练集和验证集 DataLoader；
9. 构建模型、loss、optimizer；
10. 如果配置了 `resume`，从 checkpoint 恢复；
11. 按 epoch 循环训练和验证；
12. 保存 loss 曲线；
13. 保存 best checkpoint 和周期 checkpoint。

### 8.3 DataLoader 构建

`build_dataloaders(cfg)` 创建两个数据集：

```text
train_set = AnimalDetectionDataset(...train...)
val_set   = AnimalDetectionDataset(...val...)
```

然后分别创建：

```text
train_loader: shuffle=True
val_loader:   shuffle=False
```

训练集打乱顺序，验证集保持稳定顺序。

### 8.4 模型、损失和优化器构建

`build_components(cfg, device)` 创建：

```text
model     = AnimalDetector(...)
criterion = DetectionLoss(...)
optimizer = torch.optim.AdamW(...)
```

其中模型和 loss 都会移动到指定设备。

### 8.5 单个 epoch 的执行

单轮训练和验证共用 `run_detection_epoch`。

训练阶段：

```text
model.train(True)
启用梯度
for images, targets, metas in train_loader:
    images -> device
    preds = model(images)
    loss, parts = criterion(preds, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(max_norm=10.0)
    optimizer.step()
    记录 loss
返回平均 loss
```

验证阶段：

```text
model.train(False)
禁用梯度 torch.no_grad()
for images, targets, metas in val_loader:
    images -> device
    preds = model(images)
    loss, parts = criterion(preds, targets)
    只记录 loss，不反向传播
返回平均 loss
```

### 8.6 日志、曲线与 checkpoint

每个 epoch 会记录：

- train total loss；
- train box loss；
- train obj loss；
- train cls loss；
- val total loss；
- val box loss；
- val obj loss；
- val cls loss。

同时会保存 loss 曲线：

```text
<run_dir>/tiny_detector_loss_curve.png
```

checkpoint 保存逻辑：

- 如果当前 `val_total` 小于历史最好值，保存 `best.pt`；
- 如果 epoch 能被 `checkpoint_interval` 整除，保存 `epoch_xxx.pt`。

checkpoint 内容包括：

```text
epoch
model state_dict
optimizer state_dict
best_val
config
```

---

## 9. 推理实现过程

推理入口是：

```text
src/inferencers/tiny_detector.py
```

### 9.1 启动命令

使用配置中的 source 和 output：

```bash
python -m src.inferencers.tiny_detector --config configs/default.yaml
```

命令行覆盖输入输出：

```bash
python -m src.inferencers.tiny_detector \
  --config configs/default.yaml \
  --source samples/demo.mp4 \
  --output outputs/inference/demo_result.mp4
```

如果是图片：

```bash
python -m src.inferencers.tiny_detector \
  --config configs/default.yaml \
  --source samples/demo.jpg \
  --output outputs/inference/demo_result.jpg
```

### 9.2 推理入口 main 流程

`main()` 主要步骤：

1. 解析 `--config`、`--source`、`--output`；
2. 加载配置；
3. 初始化 logger；
4. 选择设备：如果 CUDA 可用则使用 CUDA，否则使用 CPU；
5. 读取 source 和 output；
6. 创建推理 tracker；
7. 调用 `load_tiny_detector` 加载模型和 anchor；
8. 调用 `build_tiny_frame_inferencer` 构造单帧推理函数；
9. 调用 `run_image_or_video` 对图片或视频逐帧处理；
10. 输出推理统计汇总。

### 9.3 模型加载

`load_tiny_detector(cfg, device)` 做以下事情：

```text
checkpoint = torch.load(cfg["infer"]["checkpoint"])
model = AnimalDetector(...)
model.load_state_dict(checkpoint["model"])
model.eval()
anchors = torch.tensor(cfg["model"]["anchors"])
return model, anchors
```

注意：

- 模型结构必须和训练时配置一致；
- `num_classes`、`num_anchors`、`width_mult`、`anchors` 不应随意改动；
- 如果改了类别数或 anchor，旧 checkpoint 可能无法加载。

### 9.4 单帧推理过程

`build_tiny_frame_inferencer` 返回一个闭包函数 `infer_frame(frame)`。

对每一帧：

```text
frame: OpenCV BGR 图像
        |
        v
cv2.cvtColor(frame, BGR2RGB)
        |
        v
letterbox(rgb, image_size)
        |
        v
image_to_tensor(inp).unsqueeze(0).to(device)
        |
        v
with torch.no_grad(): raw = model(tensor)
        |
        v
postprocess_tiny(...)
        |
        v
tracker.log_detections(...)
        |
        v
draw_detections(frame, detections, class_names)
```

最终返回的是已经画好框的 BGR frame，用于保存图片或写入视频。

---

## 10. 后处理过程

后处理函数是：

```text
postprocess_tiny(raw, anchors, conf_threshold, iou_threshold, image_size, original_shape, scale, pad)
```

### 10.1 解码 raw prediction

首先调用：

```text
boxes, obj, cls_probs = decode_predictions(raw, anchors)
```

得到：

- `boxes`：归一化 `cx, cy, w, h`；
- `obj`：目标置信度；
- `cls_probs`：类别概率。

### 10.2 计算类别分数

```text
scores_per_cls = obj.unsqueeze(-1) * cls_probs
scores, labels = scores_per_cls.max(dim=-1)
```

也就是说，一个框的最终类别是分数最高的类别。

### 10.3 置信度过滤

```text
mask = scores[0] > conf_threshold
```

默认 `conf_threshold = 0.35`，低于该阈值的候选框会被丢弃。

### 10.4 坐标转换

模型输出的框是归一化 `cxcywh`，需要先变成 `xyxy` 并乘以输入尺寸：

```text
xyxy = xywh_to_xyxy(boxes) * image_size
```

此时坐标仍然是在 letterbox 后的 `416 x 416` 图像上。

### 10.5 按类别 NMS

为了去除重复框，代码按类别分别做 NMS：

```text
for cls_id in labels.unique():
    cls_idx = torch.where(labels == cls_id)[0]
    keep = nms(xyxy[cls_idx], scores[cls_idx], iou_threshold)
```

NMS 的作用：

1. 按置信度从高到低排序；
2. 保留最高分框；
3. 删除与该框 IoU 高于阈值的其他框；
4. 重复直到没有框可处理。

默认 `iou_threshold = 0.45`。

### 10.6 坐标还原到原图

因为推理输入经过 letterbox，所以输出框需要反变换：

```text
x = (x - pad_x) / scale
y = (y - pad_y) / scale
```

然后裁剪到原图范围内：

```text
x in [0, original_w - 1]
y in [0, original_h - 1]
```

最终每个检测结果格式为：

```text
(box_xyxy, score, class_id)
```

其中 `box_xyxy` 是原图坐标上的整数框。

---

## 11. 可视化绘制

`draw_detections` 使用 OpenCV 绘制：

- 矩形框；
- 类别名称；
- 置信度分数。

显示文本格式：

```text
{name} {score:.2f}
```

例如：

```text
dog 0.87
cat 0.76
```

---

## 12. 训练与推理的关键文件关系

```text
configs/default.yaml
        |
        +--> src/trainers/tiny_detector.py
        |          |
        |          +--> src/data/dataset.py
        |          +--> src/models/detector.py
        |          +--> src/models/loss.py
        |          +--> src/trainers/detection_engine.py
        |
        +--> src/inferencers/tiny_detector.py
                   |
                   +--> src/inferencers/tiny_detector_core.py
                   +--> src/models/detector.py
                   +--> src/data/transforms.py
                   +--> src/utils/box_ops.py
```

---

## 13. 完整训练执行过程示例

### 13.1 准备数据

数据目录需要符合配置：

```text
data/coco_animals/detection/
  train/
    images/
      xxx.jpg
    labels/
      xxx.txt
  val/
    images/
      yyy.jpg
    labels/
      yyy.txt
```

每个标签文件内容示例：

```text
0 0.512 0.438 0.233 0.310
2 0.701 0.522 0.180 0.260
```

### 13.2 启动训练

```bash
python -m src.trainers.tiny_detector --config configs/default.yaml
```

### 13.3 训练时发生的事情

1. 创建输出目录，例如：

```text
outputs/tiny_detector/20260714_xxxxxx/
```

2. 创建 checkpoint 目录：

```text
outputs/tiny_detector/20260714_xxxxxx/checkpoints/
```

3. 加载数据；
4. 初始化模型；
5. 进入 epoch 循环；
6. 每个 epoch 先训练再验证；
7. 保存曲线；
8. 如果验证 loss 最优，保存：

```text
outputs/tiny_detector/20260714_xxxxxx/checkpoints/best.pt
```

9. 每隔 `checkpoint_interval` 保存：

```text
outputs/tiny_detector/20260714_xxxxxx/checkpoints/epoch_005.pt
outputs/tiny_detector/20260714_xxxxxx/checkpoints/epoch_010.pt
...
```

---

## 14. 完整推理执行过程示例

### 14.1 修改 checkpoint 路径

将配置中的：

```yaml
infer:
  checkpoint: outputs/checkpoints/best.pt
```

改成真实训练得到的路径，例如：

```yaml
infer:
  checkpoint: outputs/tiny_detector/20260714_xxxxxx/checkpoints/best.pt
```

### 14.2 启动视频推理

```bash
python -m src.inferencers.tiny_detector \
  --config configs/default.yaml \
  --source samples/demo.mp4 \
  --output outputs/inference/demo_result.mp4
```

### 14.3 启动图片推理

```bash
python -m src.inferencers.tiny_detector \
  --config configs/default.yaml \
  --source samples/demo.jpg \
  --output outputs/inference/demo_result.jpg
```

### 14.4 推理时发生的事情

1. 加载 checkpoint；
2. 构建模型；
3. 加载权重；
4. 读取图片或视频；
5. 对每一帧执行：
   - BGR 转 RGB；
   - letterbox；
   - 转 tensor；
   - 前向传播；
   - 解码预测；
   - 置信度过滤；
   - NMS；
   - 坐标还原；
   - 绘制检测框；
6. 保存输出图片或视频；
7. 记录检测统计。

---

## 15. 代码中的几个关键设计点

### 15.1 为什么 targets 是 list

目标检测中，每张图的目标数量不同。例如：

```text
image_1: 2 个目标
image_2: 0 个目标
image_3: 7 个目标
```

如果强行堆叠成 `[B, N, 5]`，需要 padding 和 mask。当前实现使用 list，结构更简单：

```text
targets = [Tensor[N1, 5], Tensor[N2, 5], ...]
```

### 15.2 为什么使用 letterbox

直接 resize 会改变目标长宽比，可能让动物形状变形，影响检测质量。letterbox 保留比例，只在边缘填充，因此更适合检测任务。

### 15.3 为什么 objectness 要加权

检测任务中大部分网格和 anchor 都是背景，负样本远多于正样本。如果不加权，模型可能倾向于预测所有位置都没有目标。当前实现：

```text
正样本 objectness 权重 5.0
负样本 objectness 权重 0.5
```

可以让模型更重视包含真实目标的 anchor。

### 15.4 为什么按类别 NMS

不同类别的目标即使重叠，也不一定应该互相抑制。例如一只动物和另一个类别目标可能在图中重叠。按类别 NMS 可以避免不同类别之间错误删除。

### 15.5 width_mult 的作用

`width_mult` 控制通道数：

```text
实际通道数 = int(基础通道数 * width_mult)
```

例如：

- `width_mult = 1.0`：模型更大，表达能力更强；
- `width_mult = 0.75`：默认配置，速度和效果折中；
- `width_mult = 0.5`：模型更小，速度更快，但精度可能下降。

---

## 16. 可能的改进方向

当前 `tiny_detector` 简洁清晰，但仍有不少可以增强的地方。

### 16.1 多尺度检测

当前模型只使用单尺度输出，大致是 `13 x 13`。这对大目标较友好，但对小目标不够友好。可以增加 `26 x 26`、`52 x 52` 等多尺度检测头。

### 16.2 更强的数据增强

目前主要是 letterbox 和归一化。可以加入：

- 随机水平翻转；
- HSV 颜色扰动；
- 随机缩放；
- Mosaic；
- MixUp；
- 随机裁剪。

### 16.3 更好的 anchor 聚类

当前 anchor 是手工配置：

```text
[0.08, 0.10]
[0.16, 0.20]
[0.32, 0.40]
```

可以使用训练集标注框做 k-means 聚类，得到更贴合当前数据集的 anchor。

### 16.4 更完整的评估指标

目前训练主要记录 loss。实际检测任务还应评估：

- Precision；
- Recall；
- F1；
- AP；
- mAP@0.5；
- mAP@0.5:0.95。

### 16.5 更稳定的 loss 设计

可以考虑：

- GIoU / DIoU / CIoU loss；
- Focal Loss；
- label smoothing；
- objectness ignore mask。

### 16.6 推理部署优化

可以进一步支持：

- TorchScript；
- ONNX；
- TensorRT；
- 半精度 FP16；
- 批量图片推理；
- 摄像头实时推理。

---

## 17. 常见问题

### 17.1 checkpoint 加载失败

常见原因：

1. `num_classes` 和训练时不一致；
2. `width_mult` 和训练时不一致；
3. `num_anchors` 或 `anchors` 和训练时不一致；
4. 配置文件使用错了；
5. checkpoint 路径不是 tiny_detector 训练得到的权重。

### 17.2 推理没有框

可能原因：

1. 模型没有训练好；
2. `conf_threshold` 太高；
3. checkpoint 路径错误；
4. 类别配置和训练时不一致；
5. 输入图像和训练数据分布差异太大。

可以临时降低阈值：

```yaml
infer:
  conf_threshold: 0.15
```

### 17.3 检测框位置偏移

可能原因：

1. 标签格式不是 YOLO 格式；
2. 标签坐标不是归一化坐标；
3. 图片和标签文件不匹配；
4. letterbox 后标签转换或推理坐标还原逻辑被修改；
5. 输入图片在标注后被裁剪或缩放过。

### 17.4 小目标检测不好

当前模型单尺度输出对小目标不够友好。可以考虑：

1. 增大输入尺寸；
2. 增加多尺度检测头；
3. 调整 anchor；
4. 加强小目标数据增强；
5. 使用更强的工业检测框架如 YOLOv5、MMDetection、Detectron2。

---

## 18. 总结

`tiny_detector` 是一个清晰的 YOLO 风格轻量目标检测器，核心特点是：

- 使用自定义轻量 CNN backbone；
- 使用单尺度 anchor-based 检测头；
- 输出 `[B, A, H, W, 5 + C]`；
- 使用 objectness、box regression、classification 三部分 loss；
- 训练阶段支持日志、曲线、checkpoint 和 resume；
- 推理阶段支持图片/视频输入、置信度过滤、NMS、坐标还原和结果绘制。

它非常适合理解目标检测从数据读取、模型前向、loss 构造、训练循环、checkpoint 保存，到推理后处理和可视化输出的完整闭环。
