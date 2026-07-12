# OpenCLIP 训练与推理设计框架

本文档用于学习本工程中 OpenCLIP 的完整训练和推理过程，包括：

```text
数据组织
配置文件
模型加载
训练流程
零样本推理流程
日志与指标
调试入口
常见问题
```

本工程中的 OpenCLIP 主要用于 **动物识别 Recognition**，不是目标检测模型。

如果输入是整张视频或包含多个动物的图片，推荐先用检测模型得到动物 2D box，再裁剪动物区域送入 OpenCLIP 做识别。

---

## 1. OpenCLIP 在工程中的定位

OpenCLIP 属于识别模型：

```text
输入：动物图片或动物裁剪图
输出：动物类别和置信度
```

它支持两种学习/使用方式：

```text
1. 零样本推理 Zero-shot Inference
   不训练分类器，直接比较 image feature 和 text feature。

2. 线性头训练 Linear Probe Training
   冻结 OpenCLIP 图像编码器，只训练一个 Linear 分类头。
```

当前工程已经实现：

```text
零样本推理: src/inferencers/openclip.py
线性头训练: src/trainers/openclip.py
OpenCLIP 适配: src/specialized/openclip_adapter.py
```

---

## 2. 相关文件结构

```text
configs/default.yaml
src/
  specialized/
    openclip_adapter.py        # OpenCLIP 模型加载、本地权重加载、零样本分类
  trainers/
    openclip.py                # OpenCLIP linear classifier 训练入口
  inferencers/
    openclip.py                # OpenCLIP 零样本推理入口
  data/
    classification_dataset.py  # 分类数据集读取
  utils/
    logger.py                  # 运行日志目录和 .log 文件
    tracker.py                 # metrics jsonl/csv 记录
    visualization.py           # loss 曲线保存
script/
  run_train_recognition.sh     # 识别模型训练统一入口
  run_infer_recognition.sh     # 识别模型推理统一入口
  run_infer_specialized.sh     # 专用模型推理入口
docs/
  OPENCLIP_TRAIN_INFER.md      # OpenCLIP 使用流程
  OPENCLIP_DESIGN.md           # 本设计文档
```

---

## 3. 数据目录设计

OpenCLIP 训练使用分类目录格式。

推荐数据目录：

```text
data/animals10/recognition/
  train/
    dog/*.jpg
    horse/*.jpg
    elephant/*.jpg
    butterfly/*.jpg
    chicken/*.jpg
    cat/*.jpg
    cow/*.jpg
    sheep/*.jpg
    spider/*.jpg
    squirrel/*.jpg
  val/
    dog/*.jpg
    horse/*.jpg
    elephant/*.jpg
    butterfly/*.jpg
    chicken/*.jpg
    cat/*.jpg
    cow/*.jpg
    sheep/*.jpg
    spider/*.jpg
    squirrel/*.jpg
```

类别目录名必须和配置中的类别名一致。

当前 Animals-10 推荐类别：

```yaml
data:
  class_names: [dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel]
```

类别顺序非常重要：

```text
class_names[0] -> label 0 -> dog
class_names[1] -> label 1 -> horse
...
```

训练时模型输出维度必须和类别数量一致：

```yaml
model:
  num_classes: 10
```

---

## 4. 配置文件设计

OpenCLIP 配置在：

```text
configs/default.yaml
```

核心配置：

```yaml
openclip:
  data_root: data/animals10/recognition
  model_name: ViT-B-32
  pretrained: ""
  checkpoint_path: weights/openclip/open_clip_model.safetensors
  prompt_template: "a photo of a {name}, a wild animal"
  topk: 5
  freeze_encoder: true
  epochs: 20
  batch_size: 16
  num_workers: 0
  learning_rate: 0.001
  loss_curve: outputs/openclip_loss_curve.png
```

字段含义：

```text
data_root       OpenCLIP 训练数据根目录，目录下需要有 train/val。
model_name      OpenCLIP 模型结构，例如 ViT-B-32。
pretrained      在线预训练权重标签。离线训练时设为 ""。
checkpoint_path 本地离线权重路径，推荐使用 open_clip_model.safetensors。
prompt_template 零样本推理时的文本模板。
topk            推理输出前 K 个类别。
freeze_encoder  true 表示冻结 OpenCLIP，只训练线性分类头。
epochs          训练轮数。
batch_size      批大小。
num_workers     DataLoader worker 数。WSL 下建议 0，避免 Tkinter/Tcl worker 崩溃。
learning_rate   线性分类头学习率。
loss_curve      loss 曲线保存路径。
```

---

## 5. OpenCLIP 模型加载设计

模型加载逻辑在：

```text
src/specialized/openclip_adapter.py
```

核心函数：

```python
load_openclip_model(model_name, pretrained, device, checkpoint_path="")
```

设计目标：

```text
1. 支持联网下载 OpenCLIP 权重；
2. 支持本地离线 safetensors / pt / bin 权重；
3. 返回 model、preprocess、tokenizer；
4. 训练和推理共用同一套加载逻辑。
```

### 5.1 在线模式

如果配置：

```yaml
openclip:
  pretrained: laion2b_s34b_b79k
  checkpoint_path: ""
```

则会调用 OpenCLIP 自动下载权重。

缺点：需要能访问 HuggingFace。

### 5.2 离线模式

如果配置：

```yaml
openclip:
  pretrained: ""
  checkpoint_path: weights/openclip/open_clip_model.safetensors
```

则不会联网，而是：

```text
1. 创建 ViT-B-32 模型结构；
2. 读取本地 safetensors 权重；
3. load_state_dict 加载权重；
4. 返回模型和 preprocess。
```

这是当前推荐方式。

---

## 6. OpenCLIP 训练设计

训练入口：

```text
src/trainers/openclip.py
```

运行命令：

```bash
bash script/run_train_recognition.sh openclip configs/default.yaml
```

等价于：

```bash
python -m src.trainers.openclip --config configs/default.yaml
```

---

## 7. 训练整体流程

训练流程如下：

```text
读取 configs/default.yaml
  -> 设置随机种子
  -> 创建 logger 和 MetricTracker
  -> 选择 device
  -> 加载 OpenCLIP 模型和 preprocess
  -> 构建 OpenCLIPLinearClassifier
  -> 读取 train / val 分类数据
  -> 创建 DataLoader
  -> 定义 CrossEntropyLoss
  -> 定义 AdamW optimizer
  -> epoch 循环训练
  -> 保存 best_linear.pt
  -> 保存 loss curve
  -> 写入 logs
```

---

## 8. 训练模型结构

当前训练方式不是全量微调 OpenCLIP，而是 **Linear Probe**。

结构：

```text
image
  -> OpenCLIP preprocess
  -> OpenCLIP image encoder ViT-B-32
  -> normalized image feature
  -> Linear classifier
  -> class logits
  -> CrossEntropyLoss
```

代码中的模型类：

```python
class OpenCLIPLinearClassifier(nn.Module):
    clip_model
    classifier = nn.Linear(embed_dim, num_classes)
```

如果：

```yaml
openclip:
  freeze_encoder: true
```

则：

```text
OpenCLIP 图像编码器不更新参数
只训练 Linear classifier
```

优点：

```text
训练快
显存低
适合 RTX 2070
不破坏预训练语义能力
适合小数据集
```

---

## 9. 训练数据流

数据集类：

```python
OpenCLIPImageDataset
```

它内部复用：

```python
AnimalClassificationDataset
```

读取流程：

```text
data/animals10/recognition/train/dog/xxx.jpg
  -> PIL.Image.open(...).convert("RGB")
  -> OpenCLIP preprocess
  -> tensor [3, 224, 224]
  -> label id
```

DataLoader 输出：

```text
images: [batch_size, 3, 224, 224]
labels: [batch_size]
```

---

## 10. 训练损失和指标

损失函数：

```python
nn.CrossEntropyLoss()
```

指标：

```text
train loss
train acc
val loss
val acc
```

每个 step 记录：

```text
epoch
phase
step
loss
acc
```

每个 epoch 记录：

```text
epoch
phase=train/val/epoch
train_loss
train_acc
val_loss
val_acc
```

---

## 11. 训练输出

训练过程中保存：

```text
outputs/checkpoints/openclip/best_linear.pt
outputs/openclip_loss_curve.png
```

日志目录按时间创建：

```text
logs/YYYYMMDD_HHMMSS_train_openclip/
  train_openclip.log
  train_openclip_metrics.jsonl
  train_openclip_metrics.csv
```

示例：

```text
logs/20260712_165341_train_openclip/
  train_openclip.log
  train_openclip_metrics.jsonl
  train_openclip_metrics.csv
```

---

## 12. best checkpoint 内容

`best_linear.pt` 内容：

```python
{
    "epoch": epoch,
    "model": model.classifier.state_dict(),
    "class_names": class_names,
    "embed_dim": embed_dim,
    "config": cfg,
}
```

注意：当前保存的是线性分类头，不是完整 OpenCLIP 权重。

原因：

```text
OpenCLIP 主干来自本地预训练权重
训练过程中只更新 classifier
```

---

## 13. 零样本推理设计

推理入口：

```text
src/inferencers/openclip.py
```

运行命令：

```bash
bash script/run_infer_specialized.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "dog,horse,cat"
```

等价于：

```bash
python -m src.inferencers.openclip \
  --config configs/default.yaml \
  --source animal.jpg \
  --output outputs/inference/openclip.txt \
  --text "dog,horse,cat"
```

---

## 14. 零样本推理数据流

流程：

```text
输入图片 animal.jpg
  -> PIL RGB
  -> OpenCLIP preprocess
  -> image encoder
  -> image feature normalize

输入文本 dog,horse,cat
  -> prompt_template
  -> tokenizer
  -> text encoder
  -> text feature normalize

image feature @ text feature.T
  -> softmax
  -> top-k category scores
  -> 写入 txt
```

---

## 15. prompt 设计

配置：

```yaml
openclip:
  prompt_template: "a photo of a {name}, a wild animal"
```

如果输入类别是：

```text
dog
horse
cat
```

实际 prompt：

```text
a photo of a dog, a wild animal
a photo of a horse, a wild animal
a photo of a cat, a wild animal
```

可以根据数据场景修改：

```yaml
prompt_template: "a camera trap photo of a {name}"
```

或者：

```yaml
prompt_template: "a photo of a {name}, an animal species"
```

---

## 16. 推理输出

输出文件示例：

```text
dog: 0.8421
cat: 0.0912
horse: 0.0667
```

输出路径：

```text
outputs/inference/openclip.txt
```

---

## 17. 当前训练和推理的区别

### 17.1 训练

```text
训练 Linear 分类头
输入来自 data/animals10/recognition/train 和 val
输出 best_linear.pt
适合固定类别闭集分类
```

### 17.2 零样本推理

```text
不使用 best_linear.pt
使用 OpenCLIP 图文特征匹配
可以识别 text 中指定的任意候选类别
适合开放类别识别
```

这意味着：

```text
训练 linear head 和零样本推理是两条路线
```

当前工程默认推理走的是零样本路线。

---

## 18. 为什么训练后推理不加载 best_linear.pt

因为 `best_linear.pt` 是闭集分类头，只能识别训练时的固定类别。

而 OpenCLIP 零样本推理可以识别任意文本候选，例如：

```text
red panda
snow leopard
raccoon
fox
```

如果后续需要闭集线性头推理，可以新增：

```text
src/inferencers/openclip_linear.py
```

流程是：

```text
image -> OpenCLIP image encoder -> classifier -> class logits
```

---

## 19. 调试入口

VS Code / Cursor 调试配置：

```text
.vscode/launch.json
```

可选择：

```text
Train | openclip
Infer | openclip zero-shot image
```

建议断点位置：

```text
src/trainers/openclip.py
  main()
  run_epoch()
  OpenCLIPImageDataset.__getitem__()
  OpenCLIPLinearClassifier.forward()

src/specialized/openclip_adapter.py
  load_openclip_model()
  classify_image()

src/inferencers/openclip.py
  main()
```

---

## 20. 常见问题

### 20.1 HuggingFace 无法下载权重

报错：

```text
Network is unreachable
Failed to download weights
```

解决：使用离线权重。

配置：

```yaml
openclip:
  pretrained: ""
  checkpoint_path: weights/openclip/open_clip_model.safetensors
```

确认文件存在：

```bash
ls -lh weights/openclip/open_clip_model.safetensors
```

### 20.2 No pretrained weights loaded

如果看到：

```text
No pretrained weights loaded for model 'ViT-B-32'. Model initialized randomly.
```

说明没有正确加载本地权重。

检查：

```yaml
openclip:
  pretrained: ""
  checkpoint_path: weights/openclip/open_clip_model.safetensors
```

### 20.3 没找到训练图片

报错：

```text
没有在 data/animals10/recognition/train 找到分类图片
```

检查目录：

```bash
find data/animals10/recognition/train -maxdepth 2 -type f | head
```

正确结构：

```text
data/animals10/recognition/train/dog/xxx.jpg
data/animals10/recognition/train/cat/xxx.jpg
```

错误结构：

```text
data/animals10/recognition/train/xxx.jpg
```

### 20.4 DataLoader worker 崩溃

如果出现：

```text
Tcl_AsyncDelete
DataLoader worker exited unexpectedly
```

使用：

```yaml
openclip:
  num_workers: 0
```

工程已经默认这样配置。

### 20.5 CUDA 显存不足

降低 batch size：

```yaml
openclip:
  batch_size: 8
```

---

## 21. 推荐学习顺序

建议按这个顺序看代码：

```text
1. configs/default.yaml
2. src/inferencers/openclip.py
3. src/specialized/openclip_adapter.py
4. src/trainers/openclip.py
5. src/data/classification_dataset.py
6. src/utils/logger.py
7. src/utils/tracker.py
```

先理解零样本推理，再理解 linear probe 训练。

---

## 22. 推荐运行顺序

### 22.1 确认权重

```bash
ls -lh weights/openclip/open_clip_model.safetensors
```

### 22.2 确认数据

```bash
find data/animals10/recognition/train -maxdepth 2 -type f | head
find data/animals10/recognition/val -maxdepth 2 -type f | head
```

### 22.3 训练 OpenCLIP linear head

```bash
bash script/run_train_recognition.sh openclip configs/default.yaml
```

### 22.4 零样本推理

```bash
bash script/run_infer_specialized.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "dog,horse,elephant,butterfly,chicken,cat,cow,sheep,spider,squirrel"
```

---

## 23. 总结

本工程 OpenCLIP 设计可以概括为：

```text
预训练 OpenCLIP ViT-B-32
  -> 零样本图文匹配推理
  -> 或冻结图像编码器训练 Linear 分类头
```

训练适合固定类别分类：

```text
Animals-10 -> OpenCLIP image encoder -> Linear head -> 10 类动物分类
```

推理适合开放类别识别：

```text
animal image -> image feature
candidate animal texts -> text features
similarity -> top-k result
```

对于视频动物识别，推荐两阶段：

```text
检测模型得到动物 2D box
  -> 裁剪动物图像
  -> OpenCLIP 识别动物类别
```
