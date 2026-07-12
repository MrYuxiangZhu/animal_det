# OpenCLIP 训练与推理分支简图

## 1. 训练阶段

```text
bash script/run_train_recognition.sh openclip configs/default.yaml
  -> bash script/run_train.sh openclip configs/default.yaml
    -> python -m src.trainers.openclip --config configs/default.yaml
      -> main()
        -> load_openclip_model(...)
          -> import_open_clip()
            -> 优先导入 third_party/open_clip/src/open_clip
        -> OpenCLIPLinearClassifier(clip_model, ...)
        -> DataLoader
          -> OpenCLIPImageDataset
            -> 只读取图片
            -> preprocess(image)
        -> run_epoch(...)
          -> logits = model(images)
            -> OpenCLIPLinearClassifier.forward()
              -> clip_model.encode_image(images)
                -> CLIP._encode_image()
                  -> self.visual(image)
                    -> VisionTransformer / ModifiedResNet / TimmModel
                      -> 特征向量 [B, D]
              -> classifier(features)
            -> 输出 logits [B, num_classes]
```

### 训练阶段关键点

- 只使用图片输入。
- 不走文本编码器 `encode_text()`。
- `model(images)` 实际上是分类器前向：
  - 先提图像特征
  - 再接线性分类头

---

## 2. 推理阶段

```text
python -m src.inferencers.openclip --config configs/default.yaml
  -> main()
    -> load_openclip_model(...)
      -> import_open_clip()
        -> 优先导入 third_party/open_clip/src/open_clip
    -> class_names = cfg["data"]["class_names"]
    -> prompts = [prompt_template.format(name=name) for name in class_names]
    -> text_tokens = tokenizer(prompts)
    -> image_tensor = preprocess(image)
    -> model.encode_image(image_tensor)
      -> CLIP._encode_image()
        -> self.visual(image)
    -> model.encode_text(text_tokens)
      -> CLIP._encode_text()
        -> token_embedding + positional_embedding
        -> text transformer
        -> text pooling
    -> image_features @ text_features.T
    -> softmax
    -> topk 结果
```

### 推理阶段关键点

- 同时使用图片和文本。
- 文本输入来自：
  - `class_names`
  - `prompt_template`
- 这是标准的 zero-shot CLIP 相似度推理。

---

## 3. 两条分支对比

```text
训练阶段
图片 -> image encoder -> 分类头 -> logits

推理阶段
图片 -> image encoder --\
                         -> 相似度计算 -> 预测类别
文本 -> text encoder  --/
```

---

## 4. 你这份工程里最核心的入口

### 训练入口

- `script/run_train_recognition.sh`
- `script/run_train.sh`
- `src/trainers/openclip.py`

### 推理入口

- `src/inferencers/openclip.py`
- `src/specialized/openclip_adapter.py`

### OpenCLIP 本体

- `third_party/open_clip/src/open_clip/model.py`
- `third_party/open_clip/src/open_clip/transformer.py`
- `third_party/open_clip/src/open_clip/factory.py`

---

## 5. 一句话总结

- **训练**：只走 `image encoder -> classifier`
- **推理**：走 `image encoder + text encoder -> 相似度`

