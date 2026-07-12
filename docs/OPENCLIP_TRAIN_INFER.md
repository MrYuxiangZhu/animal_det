# OpenCLIP 训练与推理流程

本工程的 OpenCLIP 分成两种用法：

```text
1. 零样本推理：不训练，直接用图像特征和文本特征匹配。
2. 线性头训练：冻结 OpenCLIP 图像编码器，只训练一个动物分类头。
```

OpenCLIP 推荐用于动物识别任务，特别适合：

```text
动物裁剪图 -> 识别动物类别
```

如果输入是完整视频，建议先用检测模型得到动物 2D box，再裁剪动物区域送入 OpenCLIP。

## 1. 安装环境

```bash
bash script/setup_conda_env.sh openclip
conda activate animal-det-openclip
```

如果只想手动安装依赖：

```bash
pip install open_clip_torch torch torchvision pillow opencv-python pyyaml tqdm matplotlib
```

## 2. 数据准备

OpenCLIP 线性头训练使用分类目录格式：

```text
data/animal_classification/train/cat/*.jpg
data/animal_classification/train/dog/*.jpg
data/animal_classification/train/horse/*.jpg

data/animal_classification/val/cat/*.jpg
data/animal_classification/val/dog/*.jpg
data/animal_classification/val/horse/*.jpg
```

类别名需要和 `configs/default.yaml` 中一致：

```yaml
data:
  class_names: [cat, dog, horse, cow, sheep, elephant, bear, zebra, giraffe]
```

如果你的分类数据只有 `cat` 和 `dog`，需要同步改成：

```yaml
data:
  class_names: [cat, dog]

model:
  num_classes: 2
```

## 3. OpenCLIP 配置

配置在 `configs/default.yaml`：

```yaml
openclip:
  data_root: data/animal_classification
  model_name: ViT-B-32
  pretrained: laion2b_s34b_b79k
  prompt_template: "a photo of a {name}, a wild animal"
  topk: 5
  freeze_encoder: true
  epochs: 20
  batch_size: 16
  learning_rate: 0.001
  loss_curve: outputs/openclip_loss_curve.png
```

说明：

```text
model_name       OpenCLIP 模型结构，默认 ViT-B-32
pretrained       预训练权重名称
prompt_template  零样本文本模板
freeze_encoder   true 表示冻结 OpenCLIP，只训练分类头
batch_size       RTX 2070 建议 8 或 16
```

如果显存不够，把 batch size 改小：

```yaml
openclip:
  batch_size: 8
```

## 4. 零样本推理，不需要训练

这是 OpenCLIP 最核心的用法。

命令：

```bash
bash script/run_infer_specialized.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "red panda,fox,bear,cat,dog"
```

也可以走识别入口：

```bash
bash script/run_infer_recognition.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "red panda,fox,bear,cat,dog"
```

输出文件示例：

```text
red panda: 0.7341
fox: 0.1432
cat: 0.0718
dog: 0.0391
bear: 0.0118
```

## 5. 训练 OpenCLIP 线性分类头

当前工程实现的是：

```text
冻结 OpenCLIP 图像编码器 + 训练一个 Linear 分类头
```

优点：

```text
1. 训练快；
2. 显存占用低；
3. 适合 RTX 2070；
4. 不破坏 OpenCLIP 预训练能力；
5. 适合小规模动物分类数据。
```

训练命令：

```bash
bash script/run_train_recognition.sh openclip configs/default.yaml
```

或者：

```bash
bash script/run_train.sh openclip configs/default.yaml
```

训练输出：

```text
outputs/checkpoints/openclip/best_linear.pt
outputs/openclip_loss_curve.png
logs/train_openclip.log
logs/train_openclip_metrics.jsonl
logs/train_openclip_metrics.csv
```

日志中会记录：

```text
step loss
step acc
epoch train_loss
epoch train_acc
epoch val_loss
epoch val_acc
```

## 6. 训练后如何推理

注意：当前 `openclip_adapter.py` 的推理入口默认是 OpenCLIP 零样本推理，不加载 `best_linear.pt`。

也就是说：

```bash
bash script/run_infer_specialized.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "cat,dog,horse"
```

走的是：

```text
image feature <-> text prompt feature
```

如果你想使用训练好的线性头 `best_linear.pt` 做闭集分类，后续可以继续扩展一个入口：

```text
src/inferencers/openclip_linear.py
```

当前推荐优先使用零样本推理，因为它能识别配置外的新动物文本，例如：

```text
red panda
snow leopard
raccoon
```

## 7. 推荐流程

### 开放类别识别

```bash
bash script/setup_conda_env.sh openclip
bash script/run_infer_specialized.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "red panda,fox,bear,cat,dog"
```

### 有训练数据时增强固定类别识别

```bash
bash script/setup_conda_env.sh openclip
bash script/run_train_recognition.sh openclip configs/default.yaml
```

### 视频动物识别建议

OpenCLIP 本身是图片识别模型，不直接检测视频目标。

推荐两阶段：

```text
视频帧
  -> 检测模型得到动物 2D box
  -> 裁剪动物区域
  -> OpenCLIP 识别动物类别
```

例如：

```bash
bash script/run_infer_detection.sh tiny_detector configs/default.yaml samples/demo.mp4 outputs/inference/tiny_result.mp4
```

然后根据：

```text
logs/infer_tiny_detector_detections.csv
```

裁剪动物区域，再送入 OpenCLIP。

## 8. 常见问题

### 1. 第一次运行很慢

第一次运行会下载 OpenCLIP 预训练权重，速度取决于网络。

### 2. CUDA out of memory

降低 batch size：

```yaml
openclip:
  batch_size: 8
```

### 3. 类别不匹配

检查：

```yaml
data:
  class_names: [...]
```

分类目录名必须和 `class_names` 对应。

### 4. 零样本识别效果不好

尝试修改 prompt：

```yaml
openclip:
  prompt_template: "a camera trap photo of a {name}"
```

或者：

```yaml
openclip:
  prompt_template: "a photo of a {name}, an animal species"
```
