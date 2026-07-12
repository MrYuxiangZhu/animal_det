# 数据目录规范：按数据集分文件夹

为了避免不同公开数据集、不同任务的数据混在一起，工程统一采用：

```text
data/<dataset_name>/<task_type>/<split>/...
```

其中：

```text
dataset_name  数据集名称，例如 animals10、coco_animals、cub200、snapshot_serengeti
task_type     任务类型，例如 detection、recognition、coco、pose
split         数据划分，例如 train、val、test
```

## 1. 检测数据 Detection

检测数据使用 YOLO 格式，推荐结构：

```text
data/coco_animals/detection/
  train/
    images/*.jpg
    labels/*.txt
  val/
    images/*.jpg
    labels/*.txt
```

每个 label 文件格式：

```text
class_id center_x center_y width height
```

坐标为 0-1 归一化坐标。

对应配置：

```yaml
data:
  dataset_name: coco_animals
  root: data/coco_animals/detection
  train_images: train/images
  val_images: val/images
  train_labels: train/labels
  val_labels: val/labels
```

训练：

```bash
bash script/run_train_detection.sh tiny_detector configs/default.yaml
```

推理：

```bash
bash script/run_infer_detection.sh tiny_detector configs/default.yaml samples/demo.mp4 outputs/inference/tiny_result.mp4
```

## 2. 识别数据 Recognition

识别数据使用按类别分文件夹的分类格式，推荐结构：

```text
data/animals10/recognition/
  train/
    cat/*.jpg
    dog/*.jpg
    horse/*.jpg
  val/
    cat/*.jpg
    dog/*.jpg
    horse/*.jpg
```

对应配置：

```yaml
clip:
  data_root: data/animals10/recognition

timm:
  data_root: data/animals10/recognition

openclip:
  data_root: data/animals10/recognition
```

训练 OpenCLIP：

```bash
bash script/run_train_recognition.sh openclip configs/default.yaml
```

训练 timm ViT：

```bash
bash script/run_train_recognition.sh timm configs/default.yaml
```

## 3. COCO 格式数据

如果使用 Detectron2 或 MMDetection 的 COCO 标注，推荐单独放在：

```text
data/coco_animals/coco/
  annotations/
    instances_train.json
    instances_val.json
  train/*.jpg
  val/*.jpg
```

对应配置：

```yaml
detectron2:
  train_json: data/coco_animals/coco/annotations/instances_train.json
  val_json: data/coco_animals/coco/annotations/instances_val.json
  train_images: data/coco_animals/coco/train
  val_images: data/coco_animals/coco/val
```

## 4. 鸟类细粒度数据

推荐：

```text
data/cub200/recognition/
  train/<bird_species>/*.jpg
  val/<bird_species>/*.jpg
```

如果使用 Birder 官方仓库，也可以保留其原始结构：

```text
data/cub200/birder_raw/...
```

## 5. 红外相机 / 野生动物数据

推荐：

```text
data/snapshot_serengeti/camera_trap/
  images/...
  metadata/...
```

或：

```text
data/lila_camera_traps/camera_trap/
  images/...
  annotations/...
```

## 6. 姿态关键点数据

推荐：

```text
data/superanimal_quadruped/pose/
  videos/...
  annotations/...
```

SuperAnimal 推理可以直接使用视频路径，不一定需要训练集。

## 7. 多数据集并存示例

```text
data/
  animals10/
    recognition/
      train/cat/*.jpg
      val/cat/*.jpg
  coco_animals/
    detection/
      train/images/*.jpg
      train/labels/*.txt
      val/images/*.jpg
      val/labels/*.txt
    coco/
      annotations/instances_train.json
      annotations/instances_val.json
      train/*.jpg
      val/*.jpg
  cub200/
    recognition/
      train/001.Black_footed_Albatross/*.jpg
      val/001.Black_footed_Albatross/*.jpg
  snapshot_serengeti/
    camera_trap/
      images/...
```

这样每个数据集互不混淆，也方便同一个数据集同时保留不同任务格式。

## 8. 迁移旧目录

如果你以前使用：

```text
data/coco_animals/detection
```

建议迁移为：

```bash
mkdir -p data/coco_animals
mv data/coco_animals/detection data/coco_animals/detection
```

如果你以前使用：

```text
data/animals10/recognition
```

建议迁移为：

```bash
mkdir -p data/animals10
mv data/animals10/recognition data/animals10/recognition
```

迁移后确认 `configs/default.yaml` 中路径已经对应新结构。
