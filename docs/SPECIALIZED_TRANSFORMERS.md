# 专用动物 Transformer / 行为分析框架适配

本工程把通用检测、通用识别和专用动物 Transformer 框架分开管理。

## 分组

```text
通用检测: tiny_detector, grounding_dino, yolov5, mmdetection, detectron2
通用识别: timm, clip
专用动物 Transformer/行为分析: openclip, superanimal, pytorch_wildlife, birder
```

## OpenCLIP

用途：图文对比学习、零样本动物识别。

```bash
bash setup_conda_env.sh openclip
bash run_infer_specialized.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "red panda,snow leopard,fox"
```

配置：

```yaml
openclip:
  model_name: ViT-B-32
  pretrained: laion2b_s34b_b79k
  prompt_template: "a photo of a {name}, a wild animal"
```

## SuperAnimal / DeepLabCut

用途：跨物种动物关键点、姿态估计、行为分析。

```bash
bash setup_conda_env.sh superanimal
bash run_infer_specialized.sh superanimal configs/default.yaml animal_video.mp4 outputs/superanimal/result.mp4
```

配置：

```yaml
superanimal:
  superanimal_name: superanimal_quadruped
  model_name: hrnet_w32
  detector_name: fasterrcnn_resnet50_fpn_v2
```

## Microsoft CameraTraps / Pytorch-Wildlife

用途：红外相机、野生动物检测和分类流水线。

```bash
bash setup_conda_env.sh pytorch_wildlife
mkdir -p third_party
git clone https://github.com/microsoft/CameraTraps third_party/CameraTraps
bash run_infer_specialized.sh pytorch_wildlife configs/default.yaml camera_trap_dir outputs/wildlife/result.json
```

注意：不同版本仓库入口脚本可能不同，需要根据官方仓库实际脚本调整：

```yaml
pytorch_wildlife:
  entry_script: demo/pytorch_wildlife_infer.py
```

## Birder-MViT

用途：鸟类细粒度 Transformer 识别。

```bash
bash setup_conda_env.sh birder
mkdir -p third_party
git clone https://github.com/birder-project/birder third_party/birder
bash run_infer_specialized.sh birder configs/default.yaml bird.jpg outputs/inference/birder.txt
```

配置：

```yaml
birder:
  model_name: mvit_v2_t
  topk: 5
```

## 推荐流水线

野外视频或红外相机：

```text
Pytorch-Wildlife 检测动物
  -> OpenCLIP / Birder 细粒度识别
  -> SuperAnimal 姿态关键点
  -> 行为分析
```

开放类别普通动物识别：

```text
检测模型裁剪动物区域
  -> OpenCLIP 文本候选匹配
```

鸟类项目：

```text
检测模型定位鸟
  -> Birder-MViT 细粒度分类
```
