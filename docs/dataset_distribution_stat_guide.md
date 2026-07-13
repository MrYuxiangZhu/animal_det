# 数据集分布统计教程

本文说明如何统计工程中的训练数据类别分布，并把统计结果保存为图片和 JSON 文件。

## 1. 脚本位置

Python 统计脚本：

```bash
src/data/stat_dataset_distribution.py
```

便捷 Shell 脚本：

```bash
script/stat_dataset_distribution.sh
```

## 2. 支持的数据格式

当前统计脚本支持两种数据格式。

### 2.1 识别数据 recognition

目录格式：

```text
data/<dataset>/recognition/
  train/
    dog/
    cat/
  val/
    dog/
    cat/
```

统计含义：每个类别文件夹下的图片数量。

### 2.2 检测数据 detection

目录格式：

```text
data/<dataset>/detection/
  train/
    images/
    labels/
  val/
    images/
    labels/
```

其中 `labels/*.txt` 是 YOLO 格式：

```text
class_id center_x center_y width height
```

统计含义：每个类别的目标框数量，而不是图片数量。

## 3. 快速统计 COCO 10 类动物数据

如果你已经执行过：

```bash
bash script/prepare_coco_animals.sh coco10-copy data/coco data/coco_animals_10cls
```

那么可以直接运行：

```bash
bash script/stat_dataset_distribution.sh coco10
```

这会同时统计：

```text
data/coco_animals_10cls/recognition
data/coco_animals_10cls/detection
```

默认输出目录：

```text
outputs/dataset_stats
```

输出文件包括：

```text
outputs/dataset_stats/coco_animals_10cls_recognition_by_split.png
outputs/dataset_stats/coco_animals_10cls_recognition_total.png
outputs/dataset_stats/coco_animals_10cls_recognition_summary.json
outputs/dataset_stats/coco_animals_10cls_detection_by_split.png
outputs/dataset_stats/coco_animals_10cls_detection_total.png
outputs/dataset_stats/coco_animals_10cls_detection_summary.json
```

## 4. 统计 COCO 6 类动物数据

如果你使用的是默认 6 类 COCO animal 子集：

```bash
bash script/stat_dataset_distribution.sh coco6
```

默认统计目录：

```text
data/coco_animals/recognition
data/coco_animals/detection
```

类别顺序：

```text
dog, horse, elephant, cat, cow, sheep
```

## 5. 统计 animals10 识别数据

```bash
bash script/stat_dataset_distribution.sh animals10
```

默认统计目录：

```text
data/animals10/recognition
```

类别顺序：

```text
dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel
```

## 6. 自定义输出目录

第二个参数可以指定输出目录：

```bash
bash script/stat_dataset_distribution.sh coco10 outputs/my_dataset_stats
```

输出将保存到：

```text
outputs/my_dataset_stats
```

## 7. 直接调用 Python 脚本

### 7.1 统计 recognition

```bash
python3 -m src.data.stat_dataset_distribution \
  --type recognition \
  --root data/coco_animals_10cls/recognition \
  --classes bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe \
  --output-dir outputs/dataset_stats \
  --name coco_animals_10cls
```

### 7.2 统计 detection

```bash
python3 -m src.data.stat_dataset_distribution \
  --type detection \
  --root data/coco_animals_10cls/detection \
  --classes bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe \
  --output-dir outputs/dataset_stats \
  --name coco_animals_10cls
```

## 8. 输出图片说明

每次统计会生成两类图。

### 8.1 by_split 图

文件名类似：

```text
coco_animals_10cls_detection_by_split.png
```

含义：展示 `train` 和 `val` 中每个类别的数量。

适合检查：

- 训练集和验证集类别是否都存在
- 是否有某些类别在验证集中缺失
- train/val 分布是否严重不一致

### 8.2 total 图

文件名类似：

```text
coco_animals_10cls_detection_total.png
```

含义：统计所有 split 加起来的类别总量。

适合检查：

- 类别整体是否长尾
- 哪些类别样本最多
- 哪些类别样本太少

## 9. 输出 JSON 说明

文件名类似：

```text
coco_animals_10cls_detection_summary.json
```

JSON 中包含：

- 数据集名称
- 数据类型
- 数据根目录
- 类别列表
- 每个 split 的类别数量
- 总类别数量

适合后续做自动分析或写报告。

## 10. 结果解读建议

如果发现某些类别数量明显少，可以考虑：

- 降低该类别 crop 的过滤阈值
- 对少样本类别做数据增强
- 训练时使用 class weight 或 balanced sampler
- 只训练样本量充足的类别子集

如果 detection 和 recognition 数量差异很大，这是正常的：

- detection 统计的是目标框数量
- recognition 统计的是从 bbox crop 出来的目标图片数量
- 小目标可能会因为 `min_crop_size` 被过滤掉

## 11. 常见问题

### 11.1 找不到图片

确认是否已经运行 COCO 预处理：

```bash
bash script/prepare_coco_animals.sh coco10-copy data/coco data/coco_animals_10cls
```

### 11.2 detection 某些类别为 0

检查 YOLO 标签里的 `class_id` 是否和 `--classes` 的顺序一致。

### 11.3 recognition 某些类别为 0

检查对应目录是否存在：

```text
data/coco_animals_10cls/recognition/train/<class_name>
data/coco_animals_10cls/recognition/val/<class_name>
```

### 11.4 输出图片为空或字体异常

确认环境中已安装：

```bash
pip install matplotlib
```
