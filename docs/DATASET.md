# 公开动物视频数据集准备说明

本工程训练代码使用 YOLO 文本标注格式，便于把不同公开数据集统一到同一种结构。

## 推荐公开数据来源

1. **Open Images V7**：包含大量动物类别的图片级和检测框标注，可筛选 Cat、Dog、Horse、Cow、Sheep、Elephant、Bear、Zebra、Giraffe 等类别。
2. **YouTube-VOS / AnimalTrack / TAO**：包含视频目标跟踪数据，可以按帧导出检测框。
3. **Kaggle / Roboflow Universe 动物检测数据集**：很多已经提供 YOLO 格式标注，适合快速训练。

由于公开数据集下载方式经常变化，本仓库不硬编码某个平台的私有下载 API。建议先把数据转换成下面的标准格式，再运行训练。

## 标准目录结构

```text
data/animal_detection/
  images/
    train/
      xxx.jpg
    val/
      yyy.jpg
  labels/
    train/
      xxx.txt
    val/
      yyy.txt
```

每个标注文件和图片同名，内容为 YOLO 格式：

```text
class_id center_x center_y width height
```

坐标都是相对原图宽高归一化到 0-1 的浮点数。

## 从视频数据集生成训练帧

如果公开数据集提供的是视频和逐帧标注，你可以按固定间隔抽帧，保存为 jpg，并把对应帧的标注框写成 YOLO txt。类别 ID 必须与 `configs/default.yaml` 的 `data.class_names` 顺序一致。

## 整理已经是 YOLO 格式的数据

如果你下载的数据目录已经包含 `images/` 和 `labels/`，可以用脚本划分训练集和验证集：

```bash
python -m src.data.prepare_public_dataset \
  --source-root /path/to/downloaded_yolo_dataset \
  --output-root data/animal_detection \
  --val-ratio 0.2
```
