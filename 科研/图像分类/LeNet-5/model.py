# ==============================================================================
# Python Script: LeNet-5 Convolutional Neural Network Implementation
# Author: [Original Author/Your Name]
# Date: [Current Date]
# Description:
# This script defines a classic LeNet-5 convolutional neural network architecture
# using PyTorch. It includes the model definition with convolutional layers,
# activation functions, pooling layers, and fully connected layers.
# The script also demonstrates how to instantiate the model, move it to an
# available device (GPU/CPU), and print a summary of its architecture,
# output shapes, and parameter counts using `torchsummary`.
# ==============================================================================

# 1. Import Necessary Libraries
# ------------------------------------------------------------------------------
import torch  # 导入PyTorch库，它是构建深度学习模型的核心。
from torch import nn  # 从PyTorch中导入神经网络模块，包含了构建层（如卷积层、线性层）和激活函数等。
from torchsummary import summary  # 导入torchsummary库，用于方便地打印模型的结构概览和参数数量。


# 2. LeNet Model Definition
# ------------------------------------------------------------------------------
class LeNet(nn.Module):
    """
    LeNet-5 卷积神经网络模型定义。
    这是一个经典的CNN架构，通常用于手写数字识别（如MNIST）。
    本实现针对单通道输入（例如灰度图像），并调整了第一层卷积的padding以适应28x28输入。
    """

    def __init__(self):
        """
        LeNet 模型的初始化函数，定义了网络的各个层。
        """
        super(LeNet, self).__init__()  # 调用父类nn.Module的构造函数进行初始化。

        # 卷积层 C1: 1输入通道, 6输出通道, 5x5卷积核, 步长1, 填充2。
        # 填充2使得28x28的输入经过5x5卷积后仍保持28x28的尺寸。
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sig = nn.Sigmoid()  # Sigmoid激活函数，LeNet-5原论文中使用的激活函数。

        # 平均池化层 S2: 2x2池化核, 步长2。
        # 将28x28的特征图下采样为14x14。
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 卷积层 C3: 6输入通道, 16输出通道, 5x5卷积核, 无填充。
        # 将14x14的特征图经过5x5卷积后变为10x10。
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # 平均池化层 S4: 2x2池化核, 步长2。
        # 将10x10的特征图下采样为5x5。
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()  # 展平层，将多维特征图展平为一维向量，以便输入全连接层。
        # 经过S4层后，特征图尺寸为16x5x5，展平后为16 * 5 * 5 = 400个特征。

        # 全连接层 F5: 400输入特征, 120输出特征。
        self.f5 = nn.Linear(400, 120)
        # 全连接层 F6: 120输入特征, 84输出特征。
        self.f6 = nn.Linear(120, 84)
        # 输出层 F7: 84输入特征, 10输出特征。
        # 10个输出对应FashionMNIST数据集的10个类别。
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        """
        定义模型的前向传播路径。
        数据 x 将按照定义的层顺序进行处理。
        Args:
            x (torch.Tensor): 输入张量，通常是图像数据。
                              期望形状为 (batch_size, 1, H, W)，例如 (batch_size, 1, 28, 28)。
        Returns:
            torch.Tensor: 模型的输出，通常是每个类别的对数几率（logits）。
                          形状为 (batch_size, 10)。
        """
        # 1. 卷积层 C1 -> Sigmoid激活 -> 平均池化 S2
        x = self.sig(self.c1(x))  # 对C1层的输出应用Sigmoid激活函数。
        x = self.s2(x)  # 对激活后的特征图进行S2平均池化。

        # 2. 卷积层 C3 -> Sigmoid激活 -> 平均池化 S4
        x = self.sig(self.c3(x))  # 对C3层的输出应用Sigmoid激活函数。
        x = self.s4(x)  # 对激活后的特征图进行S4平均池化。

        # 3. 展平特征图
        x = self.flatten(x)  # 将多维特征图展平为一维向量。

        # 4. 全连接层 F5 -> F6 -> F7 (输出层)
        x = self.f5(x)  # 经过第一个全连接层。
        x = self.f6(x)  # 经过第二个全连接层。
        x = self.f7(x)  # 经过输出层，得到最终的分类预测。
        return x


# 3. Model Instantiation and Summary (Entry Point)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    当脚本直接运行时，执行此代码块。
    用于模型的实例化、设备设置和打印模型摘要。
    """
    # 检查是否有可用的CUDA GPU，如果有则使用GPU，否则使用CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # 打印当前使用的设备。

    # 实例化LeNet模型，并将其移动到选定的设备上（GPU或CPU）。
    model = LeNet().to(device)

    # 使用torchsummary打印模型的详细概览。
    # 第一个参数是模型实例。
    # 第二个参数是输入张量的尺寸（不包含batch_size）。
    # 对于FashionMNIST，输入是单通道灰度图像，尺寸为28x28。
    print(summary(model, (1, 28, 28)))