# 检测模型与识别模型划分

本工程把动物视频理解拆成两个任务：

## 1. 检测模型 Detection

检测模型回答：动物在哪里，是什么大类。

输入通常是图片或视频，输出是：

```text
[x1, y1, x2, y2, score, class_name]
```

检测模型列表：

```text
tiny_detector    从零实现的轻量检测器，适合学习检测原理
grounding_dino   文本条件开放词汇检测，带 Transformer 文本查询编码器
yolov5           工业级单阶段检测器
mmdetection      工业级检测框架，可配置 Faster R-CNN、DETR、DINO 等
detectron2       Meta 工业级检测框架，可配置 Faster R-CNN、ViTDet 等
```

训练检测模型：

```bash
bash script/run_train_detection.sh tiny_detector configs/default.yaml
bash script/run_train_detection.sh grounding_dino configs/default.yaml
bash script/run_train_detection.sh yolov5 configs/default.yaml
bash script/run_train_detection.sh mmdetection configs/default.yaml
bash script/run_train_detection.sh detectron2 configs/default.yaml
```

推理检测模型：

```bash
bash script/run_infer_detection.sh tiny_detector configs/default.yaml input.mp4 outputs/inference/tiny.mp4
bash script/run_infer_detection.sh grounding_dino configs/default.yaml input.mp4 outputs/inference/grounding.mp4 "cat,dog,fox"
bash script/run_infer_detection.sh yolov5 configs/default.yaml input.mp4 outputs/inference/yolov5.mp4
```

## 2. 识别模型 Recognition

识别模型回答：这张裁剪出来的动物图属于哪一个类别，或者和哪些文本类别最匹配。

输入通常是单张动物图片、检测框裁剪图，输出是类别概率 Top-K。

识别模型列表：

```text
timm    默认使用 ViT Transformer: vit_base_patch16_224
clip    图文对比识别，文本编码器使用 Transformer
```

训练识别模型：

```bash
bash script/run_train_recognition.sh timm configs/default.yaml
bash script/run_train_recognition.sh clip configs/default.yaml
```

推理识别模型：

```bash
bash script/run_infer_recognition.sh timm configs/default.yaml animal_crop.jpg outputs/inference/timm.txt
bash script/run_infer_recognition.sh clip configs/default.yaml animal_crop.jpg outputs/inference/clip.txt "cat,dog,horse,fox"
```

## 3. 推荐流水线

实际动物视频项目建议使用两阶段流水线：

```text
视频帧 -> 检测模型定位动物 -> 裁剪动物区域 -> 识别模型细分类
```

检测阶段可以用：

```text
yolov5 / grounding_dino / mmdetection / detectron2
```

识别阶段建议以 Transformer 为主：

```text
timm: vit_base_patch16_224 / swin_tiny_patch4_window7_224 / deit_base_patch16_224
clip: 图文对比识别，可做开放类别识别
```

## 4. 为什么识别模型以 Transformer 为主

动物细粒度识别经常依赖局部纹理、花纹、姿态和长距离上下文，例如猫科动物、犬科动物、鸟类、鹿科动物之间的细微差异。Transformer/ViT 的 patch attention 更适合建模这些全局和局部关系。

`timm` 默认配置已改为：

```yaml
timm:
  model_name: vit_base_patch16_224
```

如果显存不够，可以换成更小的 Transformer：

```yaml
timm:
  model_name: deit_tiny_patch16_224
```

或者：

```yaml
timm:
  model_name: swin_tiny_patch4_window7_224
```
