from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.backbone import ConvBNAct, TinyBackbone


class TextQueryEncoder(nn.Module):
    """把动物类别文本编码成检测查询，模拟 Grounding DINO 的文本查询思想。"""

    def __init__(self, vocab_size: int, context_length: int, hidden_dim: int, num_layers: int = 2) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``TextQueryEncoder``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            vocab_size: vocab_size 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            context_length: context_length 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            hidden_dim: hidden_dim 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            num_layers: num_layers 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.empty(context_length, hidden_dim))
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.pos_embedding, std=0.01)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """定义模块的前向传播逻辑，将输入张量转换为模型输出。
        
        所属类: ``TextQueryEncoder``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            tokens: tokens 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        mask = tokens.eq(0)
        x = self.token_embedding(tokens) + self.pos_embedding.unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=mask)
        valid = (~mask).float().unsqueeze(-1)
        x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.norm(x)


class GroundingDINOAnimal(nn.Module):
    """学习版开放词汇检测器。

    它不是官方 Grounding DINO 代码的复制，而是保留核心学习思想：
    1. 图像 backbone 产生空间特征；
    2. 文本编码器产生类别查询；
    3. 每个网格位置与文本查询计算相似度，实现文本条件检测；
    4. box head 预测每个网格的候选框。
    """

    def __init__(self, vocab_size: int, context_length: int, num_text_queries: int, hidden_dim: int = 256, width_mult: float = 0.75) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``GroundingDINOAnimal``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            vocab_size: vocab_size 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            context_length: context_length 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            num_text_queries: num_text_queries 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            hidden_dim: hidden_dim 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            width_mult: width_mult 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        super().__init__()
        self.backbone = TinyBackbone(width_mult=width_mult)
        self.visual_proj = ConvBNAct(self.backbone.out_channels, hidden_dim, kernel_size=1)
        self.text_encoder = TextQueryEncoder(vocab_size, context_length, hidden_dim)
        self.box_head = nn.Sequential(
            ConvBNAct(hidden_dim, hidden_dim, 3),
            nn.Conv2d(hidden_dim, 4, kernel_size=1),
        )
        self.objectness_head = nn.Sequential(
            ConvBNAct(hidden_dim, hidden_dim, 3),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * 10.0)
        self.num_text_queries = num_text_queries

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """定义模块的前向传播逻辑，将输入张量转换为模型输出。
        
        所属类: ``GroundingDINOAnimal``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            images: 输入图像张量批次，形状通常为 [B, C, H, W]。
            text_tokens: 文本 prompt 经 tokenizer 编码后的 token 张量。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        visual = self.visual_proj(self.backbone(images))
        b, c, h, w = visual.shape
        text_features = F.normalize(self.text_encoder(text_tokens), dim=-1)
        visual_flat = F.normalize(visual.flatten(2).transpose(1, 2), dim=-1)
        class_logits = self.logit_scale.clamp(max=100) * torch.einsum("bnc,qc->bnq", visual_flat, text_features)
        class_logits = class_logits.view(b, h, w, self.num_text_queries).permute(0, 3, 1, 2).contiguous()
        box_raw = self.box_head(visual).permute(0, 2, 3, 1).contiguous()
        objectness = self.objectness_head(visual).squeeze(1)
        return box_raw, objectness, class_logits


def decode_grounding_boxes(box_raw: torch.Tensor) -> torch.Tensor:
    """把网络输出从训练/预测空间解码到归一化几何空间。
    
    Args:
        box_raw: GroundingDINO-like 模型输出的原始框回归张量。
    
    Returns:
        函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
    """
    b, h, w, _ = box_raw.shape
    device = box_raw.device
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, h, w, 2).float()
    xy = (box_raw[..., 0:2].sigmoid() + grid) / torch.tensor([w, h], device=device)
    wh = box_raw[..., 2:4].sigmoid().clamp(min=1e-4)
    return torch.cat((xy, wh), dim=-1)
