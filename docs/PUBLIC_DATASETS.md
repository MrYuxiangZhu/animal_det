# 公开动物训练集推荐与工程使用方法

本文档汇总适合当前工程的公开动物数据集，并说明下载后如何整理成工程需要的格式。

当前工程主要分为两类训练任务：

```text
1. 检测 Detection：图片/视频帧 -> 动物 2D box + 类别
2. 识别 Recognition：动物裁剪图/单张动物图 -> 动物类别
```

还有一些专用任务：

```text
3. 红外相机野生动物识别
4. 鸟类细粒度识别
5. 动物姿态关键点/行为分析
```

## 1. 推荐优先级

如果你想最快跑通工程，推荐顺序：

```text
第一步：用 Animals-10 / Oxford-IIIT Pet 跑通识别模型 openclip/timm
第二步：用 COCO animal subset / Roboflow animal detection 跑通检测模型 tiny_detector/yolov5
第三步：用 iNaturalist / CUB-200 / CameraTrap 数据集做进阶细粒度或野生动物项目
```

## 2. 识别模型推荐数据集

识别模型包括：

```text
timm
clip
openclip
birder
```

识别模型需要分类目录格式：

```text
data/animal_classification/
  train/
    cat/*.jpg
    dog/*.jpg
  val/
    cat/*.jpg
    dog/*.jpg
```

### 2.1 Animals-10

适合：快速跑通动物分类训练。

地址：

```text
https://www.kaggle.com/datasets/alessiocorrado99/animals10
```

特点：

```text
10 类动物
数据量适中
适合 timm / OpenCLIP linear 训练
不适合检测，因为没有 2D box
```

常见类别包括：

```text
dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant
```

下载方式：

```bash
pip install kaggle
kaggle datasets download -d alessiocorrado99/animals10 -p data/raw/animals10 --unzip
```

整理到工程：

```text
data/animal_classification/train/<class_name>/*.jpg
data/animal_classification/val/<class_name>/*.jpg
```

训练 OpenCLIP 线性头：

```bash
bash script/setup_conda_env.sh openclip
bash script/run_train_recognition.sh openclip configs/default.yaml
```

训练 timm ViT：

```bash
bash script/setup_conda_env.sh timm
bash script/run_train_recognition.sh timm configs/default.yaml
```

### 2.2 Oxford-IIIT Pet Dataset

适合：猫狗品种识别、细粒度识别入门。

地址：

```text
https://www.robots.ox.ac.uk/~vgg/data/pets/
```

特点：

```text
37 个猫狗品种
有分类标签
有 segmentation mask
适合识别模型
可扩展做实例分割学习
```

下载：

```bash
mkdir -p data/raw/oxford_pets
cd data/raw/oxford_pets
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
cd -
```

整理方式：

```text
根据 annotations/list.txt 中的类别，把 images/*.jpg 拆分到：
data/animal_classification/train/<breed>/*.jpg
data/animal_classification/val/<breed>/*.jpg
```

适合训练：

```bash
bash script/run_train_recognition.sh timm configs/default.yaml
bash script/run_train_recognition.sh openclip configs/default.yaml
```

### 2.3 iNaturalist

适合：大规模动物物种识别、长尾类别学习。

地址：

```text
https://github.com/visipedia/inat_comp
https://www.kaggle.com/c/inaturalist-2019-fgvc6
```

特点：

```text
类别非常多
覆盖鸟类、哺乳类、昆虫、爬行动物等
适合训练 Transformer 识别模型
数据量较大，下载和训练成本较高
```

推荐用途：

```text
timm ViT / Swin 动物细粒度识别
OpenCLIP linear probe
物种级识别
```

建议：先抽取你需要的动物子集，不建议一开始训练全量。

### 2.4 CUB-200-2011 鸟类数据集

适合：鸟类细粒度分类。

地址：

```text
https://www.vision.caltech.edu/datasets/cub_200_2011/
```

特点：

```text
200 种鸟类
细粒度分类经典数据集
适合 birder / timm / OpenCLIP 对比测试
```

下载：

```bash
mkdir -p data/raw/cub200
cd data/raw/cub200
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz
cd -
```

整理后可放到：

```text
data/animal_classification/train/<bird_species>/*.jpg
data/animal_classification/val/<bird_species>/*.jpg
```

## 3. 检测模型推荐数据集

检测模型包括：

```text
tiny_detector
grounding_dino
yolov5
mmdetection
detectron2
```

检测模型需要 YOLO 格式：

```text
data/animal_detection/
  images/
    train/*.jpg
    val/*.jpg
  labels/
    train/*.txt
    val/*.txt
```

每个 label 文件：

```text
class_id center_x center_y width height
```

坐标为 0-1 归一化坐标。

### 3.1 COCO Animal Subset

适合：最标准的动物检测入门数据。

地址：

```text
https://cocodataset.org/#download
```

COCO 中常见动物类别：

```text
bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
```

优点：

```text
标注质量高
有 2D box
适合 tiny_detector / YOLOv5 / MMDetection / Detectron2
```

缺点：

```text
原始格式是 COCO JSON，需要转换成 YOLO 格式
```

下载：

```bash
mkdir -p data/raw/coco
cd data/raw/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
cd -
```

整理方式：

```text
COCO JSON -> 过滤动物类别 -> 转 YOLO label -> 放入 data/animal_detection
```

训练：

```bash
bash script/run_train_detection.sh tiny_detector configs/default.yaml
bash script/run_train_detection.sh yolov5 configs/default.yaml
```

### 3.2 Open Images Animal Subset

适合：类别更丰富的动物检测。

地址：

```text
https://storage.googleapis.com/openimages/web/index.html
```

特点：

```text
检测类别多
动物类别丰富
适合开放动物检测训练
数据规模很大
```

推荐工具：

```text
OIDv4_ToolKit
FiftyOne
```

FiftyOne 示例：

```bash
pip install fiftyone
```

用 Python 下载指定类别，例如：

```python
import fiftyone.zoo as foz

classes = ["Cat", "Dog", "Horse", "Cow", "Sheep", "Elephant", "Bear", "Zebra", "Giraffe"]
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=classes,
    max_samples=5000,
)
```

之后需要导出为 YOLO 格式。

### 3.3 Roboflow Animal Detection Datasets

适合：最快拿到 YOLO 格式检测数据。

地址：

```text
https://universe.roboflow.com/search?q=animal%20detection
```

优点：

```text
很多数据集可以直接导出 YOLO 格式
适合快速训练 tiny_detector / YOLOv5
不用自己写 COCO -> YOLO 转换
```

使用方式：

```text
1. 打开 Roboflow Universe
2. 搜索 animal detection / wildlife detection / cattle detection / pet detection
3. 选择数据集
4. 导出 YOLOv5 / YOLOv8 格式
5. 解压后整理到 data/animal_detection
```

训练：

```bash
bash script/run_train_detection.sh tiny_detector configs/default.yaml
```

### 3.4 Animals Detection Images Dataset / Kaggle

Kaggle 上有多个动物检测数据集。

搜索：

```text
https://www.kaggle.com/search?q=animal+detection+dataset
```

推荐关键词：

```text
animal detection
wildlife detection
cattle detection
camera trap detection
pet detection
```

注意：不同 Kaggle 数据集格式不统一，可能需要转成 YOLO 格式。

## 4. 红外相机 / 野生动物数据集

适合模型：

```text
pytorch_wildlife
openclip
timm
```

### 4.1 Snapshot Serengeti

地址：

```text
https://lila.science/datasets/snapshot-serengeti
```

特点：

```text
红外相机野生动物
物种识别
适合野生动物普查项目
```

适合：

```text
Pytorch-Wildlife 流水线测试
OpenCLIP 零样本识别
ViT/Swin 分类训练
```

### 4.2 LILA BC Camera Trap Datasets

地址：

```text
https://lila.science/datasets
```

特点：

```text
大量野生动物相机陷阱数据集
包含不同地区、不同动物
适合真实野生动物识别项目
```

推荐用途：

```text
Pytorch-Wildlife
CameraTraps
OpenCLIP
动物出现频率统计
```

### 4.3 Caltech Camera Traps

地址：

```text
https://beerys.github.io/CaltechCameraTraps/
```

特点：

```text
经典 camera trap 数据集
适合跨域泛化研究
```

## 5. 动物姿态 / 行为分析数据集

适合模型：

```text
superanimal
DeepLabCut
```

### 5.1 DeepLabCut Model Zoo / SuperAnimal

地址：

```text
https://github.com/DeepLabCut/DeepLabCut
https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html
```

特点：

```text
跨物种关键点模型
可直接做猫狗、四足动物姿态估计
不一定需要你自己训练
```

用途：

```text
动物关键点
动物行为分析
步态分析
姿态估计
```

推理：

```bash
bash script/run_infer_specialized.sh superanimal configs/default.yaml animal_video.mp4 outputs/superanimal/result.mp4
```

## 6. 当前工程推荐数据集组合

### 快速跑通识别

推荐：

```text
Animals-10
Oxford-IIIT Pet
```

命令：

```bash
bash script/setup_conda_env.sh openclip
bash script/run_train_recognition.sh openclip configs/default.yaml
bash script/run_infer_specialized.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "cat,dog,horse"
```

### 快速跑通检测

推荐：

```text
Roboflow animal detection YOLO format
COCO animal subset
```

命令：

```bash
bash script/setup_conda_env.sh tiny_detector
bash script/run_train_detection.sh tiny_detector configs/default.yaml
bash script/run_infer_detection.sh tiny_detector configs/default.yaml samples/demo.mp4 outputs/inference/tiny_result.mp4
```

### 工业检测落地

推荐：

```text
COCO animal subset
Open Images animal subset
Roboflow wildlife detection
```

模型：

```text
yolov5
mmdetection
detectron2
```

### 鸟类细粒度识别

推荐：

```text
CUB-200-2011
iNaturalist bird subset
```

模型：

```text
birder
timm ViT/Swin
openclip
```

### 野生动物红外相机

推荐：

```text
Snapshot Serengeti
LILA BC camera trap datasets
Caltech Camera Traps
```

模型：

```text
pytorch_wildlife
openclip
timm
```

## 7. 数据整理后的配置检查

### 检测配置

检查 `configs/default.yaml`：

```yaml
data:
  root: data/animal_detection
  train_images: images/train
  val_images: images/val
  train_labels: labels/train
  val_labels: labels/val
  class_names: [cat, dog, horse, cow, sheep, elephant, bear, zebra, giraffe]

model:
  num_classes: 9
```

`class_names` 数量必须等于 `model.num_classes`。

### 识别配置

```yaml
openclip:
  data_root: data/animal_classification

timm:
  data_root: data/animal_classification
```

分类目录名必须和 `data.class_names` 对应。

## 8. 是否自动下载

当前工程：

```text
不会自动下载训练数据集
```

会自动下载的通常只是：

```text
OpenCLIP / timm / YOLOv5 / Detectron2 等预训练模型权重
```

训练图片和标注需要你手动下载并整理。

## 9. 最推荐你先下载哪个？

如果你当前主要想跑 OpenCLIP / timm 识别：

```text
优先下载 Animals-10
```

如果你当前主要想跑动物检测：

```text
优先下载 Roboflow 上的 YOLO 格式 animal detection 数据集
```

如果你想做标准化检测实验：

```text
下载 COCO，然后过滤 animal subset
```

如果你想做鸟类识别：

```text
下载 CUB-200-2011
```

如果你想做野生动物红外相机：

```text
下载 Snapshot Serengeti 或 LILA BC 数据集
```
