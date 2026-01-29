# ==============================================================================
# Python Script: ResNet-18 Convolutional Neural Network Implementation
# Author: [Original Author/Your Name]
# Date: [Current Date]
# Description:
# This script defines the ResNet-18 architecture using PyTorch. It includes the
# definition of the fundamental Residual block, which utilizes skip connections
# (shortcuts) to solve the vanishing gradient problem in deep networks.
# The ResNet18 class assembles these blocks into four main stages following
# an initial convolutional layer.
# The script also demonstrates model instantiation, device placement (GPU/CPU),
# and printing a detailed summary of the model's architecture and parameters
# using `torchsummary`.
# ==============================================================================

# 1. Import Necessary Libraries
# ------------------------------------------------------------------------------
import torch  # 导入PyTorch库，它是构建深度学习模型的核心。
from torch import (
    nn,
)  # 从PyTorch中导入神经网络模块，包含了构建层（如卷积层、线性层）和激活函数等。
from torchsummary import (
    summary,
)  # 导入torchsummary库，用于方便地打印模型的结构概览和参数数量。


# 2. Residual Block Definition
# ------------------------------------------------------------------------------
class Residual(nn.Module):
    """
    残差模块 (Residual Block) 的定义。
    残差网络的核心组件。通过引入跳跃连接（Skip Connection），将输入直接加到卷积层的输出上。
    这使得网络可以学习恒等映射，从而能够训练更深的网络而不会出现梯度消失问题。
    """

    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        """
        残差模块的初始化函数。

        Args:
            input_channels (int): 输入特征图的通道数。
            num_channels (int): 输出特征图的通道数（也是模块内部卷积层的通道数）。
            use_1conv (bool): 是否在跳跃连接路径上使用1x1卷积。
                              通常用于改变通道数或特征图尺寸（当stride > 1时），以便输入和输出能相加。
            strides (int): 第一个卷积层的步长。如果大于1，则进行下采样。
        """
        super(Residual, self).__init__()  # 调用父类nn.Module的构造函数进行初始化。
        self.ReLU = nn.ReLU()  # 定义ReLU激活函数。

        # 主路径第一层：3x3卷积 -> 批量归一化 (BatchNorm)。
        # 如果 strides > 1，这里会进行下采样，特征图尺寸减半。
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
            stride=strides,
        )
        self.bn1 = nn.BatchNorm2d(num_channels)

        # 主路径第二层：3x3卷积 -> 批量归一化 (BatchNorm)。
        # 这一层保持特征图尺寸不变。
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(num_channels)

        # 跳跃连接路径（Shortcut Path）。
        # 如果输入输出形状不一致（通道数不同 或 进行了下采样），则需要通过1x1卷积调整输入的形状。
        if use_1conv:
            self.conv3 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=num_channels,
                kernel_size=1,
                stride=strides,
            )
        else:
            self.conv3 = None

    def forward(self, x):
        """
        残差模块的前向传播函数。

        Args:
            x (torch.Tensor): 输入特征图张量。

        Returns:
            torch.Tensor: 加上残差连接后的输出特征图张量。
        """
        # 主路径计算：Conv1 -> BN -> ReLU -> Conv2 -> BN
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        # 跳跃连接路径计算：
        # 如果定义了conv3（即形状需要调整），则对输入x应用conv3；否则直接使用x。
        if self.conv3:
            x = self.conv3(x)

        # 核心步骤：将主路径输出 y 和跳跃连接路径 x 相加。
        # 然后再次经过ReLU激活。
        y = self.ReLU(y + x)
        return y


# 3. ResNet-18 Model Definition
# ------------------------------------------------------------------------------
class ResNet18(nn.Module):
    """
    ResNet-18 模型的定义。
    该模型由初始卷积层、四个残差层（包含多个残差块）和最终的分类层组成。
    适用于图像分类任务。
    """

    def __init__(self, Residual):
        """
        ResNet-18 模型的初始化函数。

        Args:
            Residual (class): 残差模块的类定义，用于构建模型内部的残差块。
        """
        super(ResNet18, self).__init__()  # 调用父类nn.Module的构造函数进行初始化。

        # 第一个处理块 (Block 1): 初始特征提取。
        # 结构：7x7卷积 -> ReLU -> BatchNorm -> 3x3最大池化。
        self.b1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 第二个处理块 (Block 2): 对应ResNet的 conv2_x。
        # 包含2个残差块，通道数维持在64，不进行下采样。
        self.b2 = nn.Sequential(
            Residual(64, 64, use_1conv=False, strides=1),
            Residual(64, 64, use_1conv=False, strides=1),
        )

        # 第三个处理块 (Block 3): 对应ResNet的 conv3_x。
        # 包含2个残差块。第一个块将通道数从64提升到128，并进行下采样 (stride=2)。
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1conv=True, strides=2),
            Residual(128, 128, use_1conv=False, strides=1),
        )

        # 第四个处理块 (Block 4): 对应ResNet的 conv4_x。
        # 包含2个残差块。第一个块将通道数从128提升到256，并进行下采样 (stride=2)。
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1conv=True, strides=2),
            Residual(256, 256, use_1conv=False, strides=1),
        )

        # 第五个处理块 (Block 5): 对应ResNet的 conv5_x。
        # 包含2个残差块。第一个块将通道数从256提升到512，并进行下采样 (stride=2)。
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1conv=True, strides=2),
            Residual(512, 512, use_1conv=False, strides=1),
        )

        # 第六个处理块 (Block 6): 分类器。
        # 结构：全局平均池化 -> 展平 -> 全连接层。
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 将任意尺寸特征图变为 1x1。
            nn.Flatten(),  # 展平为向量。
            nn.Linear(512, 10),  # 全连接层，输出10个类别的logits。
        )

    def forward(self, x):
        """
        定义模型的前向传播路径。
        数据 x 将按照定义的层顺序进行处理。

        Args:
            x (torch.Tensor): 输入张量，通常是图像数据。
                              期望形状为 (batch_size, 1, H, W)。

        Returns:
            torch.Tensor: 模型的输出，形状为 (batch_size, 10)。
        """
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


# 4. Model Instantiation and Summary (Entry Point)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    当脚本直接运行时，执行此代码块。
    用于模型的实例化、设备设置和打印模型摘要。
    """
    # 检查是否有可用的CUDA GPU，如果有则使用GPU，否则使用CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # 打印当前使用的设备。

    # 实例化ResNet18模型，并将Residual模块的类作为参数传入。
    # 然后将模型移动到选定的设备上（GPU或CPU）。
    model = ResNet18(Residual).to(device)
    print("ResNet-18 Model Initialized.")

    # 使用torchsummary打印模型的详细概览。
    # 输入尺寸假设为单通道 224x224 图像。
    print(summary(model, (1, 224, 224)))
