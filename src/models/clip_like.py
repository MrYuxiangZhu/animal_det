import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.backbone import ConvBNAct, ResidualBlock


class SimpleTokenizer:
    """面向学习用途的字符级 tokenizer，避免依赖外部 CLIP tokenizer。"""

    def __init__(self, context_length: int = 32) -> None:
        vocab_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.,:/")
        self.stoi = {ch: idx + 2 for idx, ch in enumerate(vocab_chars)}
        self.pad_id = 0
        self.unk_id = 1
        self.context_length = context_length
        self.vocab_size = len(self.stoi) + 2

    def encode(self, texts: List[str]) -> torch.Tensor:
        tokens = torch.full((len(texts), self.context_length), self.pad_id, dtype=torch.long)
        for i, text in enumerate(texts):
            ids = [self.stoi.get(ch, self.unk_id) for ch in text[: self.context_length]]
            if ids:
                tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return tokens


class VisionEncoder(nn.Module):
    """CLIP 风格图像编码器，将图片映射到统一语义向量空间。"""

    def __init__(self, embed_dim: int = 256, width_mult: float = 0.75) -> None:
        super().__init__()

        def c(channels: int) -> int:
            return max(8, int(channels * width_mult))

        self.net = nn.Sequential(
            ConvBNAct(3, c(64), 7, 2),
            ConvBNAct(c(64), c(128), 3, 2),
            ResidualBlock(c(128)),
            ConvBNAct(c(128), c(256), 3, 2),
            ResidualBlock(c(256)),
            ConvBNAct(c(256), c(512), 3, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(c(512), embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.net(images).flatten(1)
        return F.normalize(self.proj(feats), dim=-1)


class TextEncoder(nn.Module):
    """CLIP 风格文本编码器，使用 TransformerEncoder 汇聚类别描述。"""

    def __init__(self, vocab_size: int, context_length: int = 32, embed_dim: int = 256, width: int = 256, layers: int = 4, heads: int = 4) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.pos_embedding = nn.Parameter(torch.empty(context_length, width))
        encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=width * 4, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.ln = nn.LayerNorm(width)
        self.proj = nn.Linear(width, embed_dim)
        nn.init.normal_(self.pos_embedding, std=0.01)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.eq(0)
        x = self.token_embedding(tokens) + self.pos_embedding.unsqueeze(0)
        x = self.transformer(x, src_key_padding_mask=mask)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.proj(self.ln(pooled)), dim=-1)


class MiniCLIP(nn.Module):
    """从零实现的 CLIP 学习版，训练目标是图像和文本描述的对比对齐。"""

    def __init__(self, vocab_size: int, context_length: int = 32, embed_dim: int = 256, width_mult: float = 0.75) -> None:
        super().__init__()
        self.visual = VisionEncoder(embed_dim=embed_dim, width_mult=width_mult)
        self.text = TextEncoder(vocab_size=vocab_size, context_length=context_length, embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text(tokens)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(images)
        text_features = self.encode_text(tokens)
        scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2
