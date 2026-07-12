# 本地 OpenCLIP 官方源码接入说明

工程已把官方 OpenCLIP 源码作为本地第三方源码接入：

```text
third_party/open_clip
```

本工程的 OpenCLIP 训练和推理会优先导入：

```text
third_party/open_clip/src/open_clip
```

如果该目录不存在，才会回退到 conda/pip 环境中的 `open_clip` 包。

## 1. 关键代码

本地源码导入工具：

```text
src/utils/third_party.py
```

核心逻辑：

```text
PROJECT_ROOT/third_party/open_clip/src
  -> sys.path.insert(0, ...)
  -> import open_clip
```

OpenCLIP 模型加载入口：

```text
src/specialized/openclip_adapter.py
```

训练入口：

```text
src/trainers/openclip.py
```

推理入口：

```text
src/inferencers/openclip.py
```

## 2. 如何确认使用的是本地源码

运行训练或推理时，会打印：

```text
[INFO] OpenCLIP module: /home/yuxiangzhu/volume/animal_det/third_party/open_clip/src/open_clip/__init__.py
```

如果看到的是：

```text
site-packages/open_clip
```

说明本地源码目录不存在或结构不正确。

## 3. OpenCLIP 源码学习重点

### ViT patch embedding

位置：

```text
third_party/open_clip/src/open_clip/transformer.py
```

重点类：

```text
VisionTransformer
```

重点层：

```python
self.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=width,
    kernel_size=patch_size,
    stride=patch_size,
    bias=False,
)
```

对于 `ViT-B-32`：

```text
image_size = 224
patch_size = 32
width = 768
```

张量变化：

```text
[B, 3, 224, 224]
  -> Conv2d(3, 768, kernel=32, stride=32)
[B, 768, 7, 7]
  -> flatten
[B, 768, 49]
  -> permute
[B, 49, 768]
  -> prepend class token
[B, 50, 768]
```

## 4. Patch embedding 调试脚本

新增脚本：

```text
src/debug/openclip_patch_embedding.py
```

运行：

```bash
python -m src.debug.openclip_patch_embedding --config configs/default.yaml
```

会打印：

```text
visual class
conv1 层结构
input shape
after conv1 shape
after flatten shape
after permute patch tokens shape
class token shape
positional embedding shape
```

## 5. 如何断点调试源码

在 Cursor / VS Code 中，可以直接对以下文件打断点：

```text
third_party/open_clip/src/open_clip/factory.py
third_party/open_clip/src/open_clip/model.py
third_party/open_clip/src/open_clip/transformer.py
third_party/open_clip/src/open_clip/pretrained.py
third_party/open_clip/src/open_clip/tokenizer.py
```

推荐断点：

```text
factory.py -> create_model_and_transforms
factory.py -> create_model
transformer.py -> VisionTransformer.__init__
transformer.py -> VisionTransformer.forward
model.py -> CLIP.encode_image
model.py -> CLIP.encode_text
```

## 6. 保持兼容性

当前设计兼容两种模式：

```text
1. 有 third_party/open_clip/src/open_clip
   使用本地源码，便于学习和断点调试。

2. 没有 third_party/open_clip/src/open_clip
   使用 pip/conda 安装的 open_clip_torch。
```

所以不会破坏原来的训练和推理命令。

## 7. 常用命令

训练 OpenCLIP linear head：

```bash
bash script/run_train_recognition.sh openclip configs/default.yaml
```

OpenCLIP 零样本推理：

```bash
bash script/run_infer_specialized.sh openclip configs/default.yaml animal.jpg outputs/inference/openclip.txt "dog,cat,horse"
```

调试 patch embedding：

```bash
python -m src.debug.openclip_patch_embedding --config configs/default.yaml
```
