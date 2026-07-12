import torch
from torch import nn


class ConvBNAct(nn.Module):
    """卷积、批归一化和 SiLU 激活的基础模块。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``ConvBNAct``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            in_channels: in_channels 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            out_channels: out_channels 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            kernel_size: kernel_size 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            stride: stride 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """定义模块的前向传播逻辑，将输入张量转换为模型输出。
        
        所属类: ``ConvBNAct``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            x: x 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        return self.block(x)


class ResidualBlock(nn.Module):
    """轻量残差块，用于增强特征表达并缓解深层网络梯度问题。"""

    def __init__(self, channels: int) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``ResidualBlock``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            channels: channels 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        super().__init__()
        hidden = max(channels // 2, 8)
        self.conv1 = ConvBNAct(channels, hidden, kernel_size=1)
        self.conv2 = ConvBNAct(hidden, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """定义模块的前向传播逻辑，将输入张量转换为模型输出。
        
        所属类: ``ResidualBlock``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            x: x 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        return x + self.conv2(self.conv1(x))


class TinyBackbone(nn.Module):
    """从零实现的轻量 CNN 主干，输出 13x13 或 26x26 网格特征。"""

    def __init__(self, width_mult: float = 0.75) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``TinyBackbone``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            width_mult: width_mult 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        super().__init__()

        def c(channels: int) -> int:
            """封装该模块中的一个可复用业务步骤，供训练、推理或工具流程调用。
            
            Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
                channels: channels 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            
            Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
                函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
            """
            return max(8, int(channels * width_mult))

        self.layers = nn.Sequential(
            ConvBNAct(3, c(32), 3, 2),
            ConvBNAct(c(32), c(64), 3, 2),
            ResidualBlock(c(64)),
            ConvBNAct(c(64), c(128), 3, 2),
            ResidualBlock(c(128)),
            ConvBNAct(c(128), c(256), 3, 2),
            ResidualBlock(c(256)),
            ConvBNAct(c(256), c(512), 3, 2),
            ResidualBlock(c(512)),
        )
        self.out_channels = c(512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """定义模块的前向传播逻辑，将输入张量转换为模型输出。
        
        所属类: ``TinyBackbone``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            x: x 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        return self.layers(x)
