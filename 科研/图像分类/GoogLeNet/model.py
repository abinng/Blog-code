# ==============================================================================
# Python Script: GoogLeNet (InceptionV1) Convolutional Neural Network Implementation
# Author: [Original Author/Your Name]
# Date: [Current Date]
# Description:
# This script defines the GoogLeNet (also known as InceptionV1) convolutional
# neural network architecture using PyTorch. It includes the definition of
# the fundamental Inception module, which processes inputs through parallel
# convolutional and pooling layers, and then concatenates their outputs.
# The GoogLeNet class then assembles these Inception modules along with
# standard convolutional and pooling layers to form the complete network.
# The script also demonstrates model instantiation, device placement (GPU/CPU),
# and printing a detailed summary of the model's architecture and parameters
# using `torchsummary`.
# ==============================================================================

# 1. Import Necessary Libraries
# ------------------------------------------------------------------------------
import torch  # 导入PyTorch库，它是构建深度学习模型的核心。
from torch import nn  # 从PyTorch中导入神经网络模块，包含了构建层（如卷积层、线性层）和激活函数等。
from torchsummary import summary  # 导入torchsummary库，用于方便地打印模型的结构概览和参数数量。


# 2. Inception Module Definition
# ------------------------------------------------------------------------------
class Inception(nn.Module):
    """
    Inception模块的定义。
    Inception模块是GoogLeNet的核心组成部分，它通过并行处理不同尺度的卷积核和池化操作，
    然后将它们的输出拼接在一起，以捕获多尺度特征并减少计算量。
    """
    def __init__(self, in_channels, c1, c2, c3, c4):
        """
        Inception模块的初始化函数。

        Args:
            in_channels (int): 输入特征图的通道数。
            c1 (int): 路径1（1x1卷积）的输出通道数。
            c2 (tuple): 路径2（1x1卷积后接3x3卷积）的输出通道数，(1x1卷积输出, 3x3卷积输出)。
            c3 (tuple): 路径3（1x1卷积后接5x5卷积）的输出通道数，(1x1卷积输出, 5x5卷积输出)。
            c4 (int): 路径4（3x3最大池化后接1x1卷积）的输出通道数。
        """
        super(Inception, self).__init__()  # 调用父类nn.Module的构造函数进行初始化。
        self.ReLU = nn.ReLU()  # 定义ReLU激活函数，将在各个路径的卷积操作后使用。

        # 路径1：单1x1卷积层。
        # 作用：保持特征图尺寸不变，进行通道数的调整和跨通道信息融合。
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)

        # 路径2：1x1卷积层后接3x3卷积层。
        # 作用：1x1卷积用于降维（bottleneck），减少3x3卷积的计算量，然后3x3卷积捕获局部特征。
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1) # padding=1保持特征图尺寸。

        # 路径3：1x1卷积层后接5x5卷积层。
        # 作用：类似路径2，1x1卷积降维，5x5卷积捕获更大范围的局部特征。
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2) # padding=2保持特征图尺寸。

        # 路径4：3x3最大池化层后接1x1卷积层。
        # 作用：最大池化捕获最显著特征，1x1卷积调整通道数并融合信息。
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1) # stride=1保持特征图尺寸。
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x):
        """
        Inception模块的前向传播函数。
        将输入 x 分别送入四个并行路径，然后将各路径的输出在通道维度上拼接。

        Args:
            x (torch.Tensor): 输入特征图张量。

        Returns:
            torch.Tensor: 拼接后的输出特征图张量。
        """
        # 路径1的计算：1x1卷积 -> ReLU激活。
        p1 = self.ReLU(self.p1_1(x))
        # 路径2的计算：1x1卷积 -> ReLU -> 3x3卷积 -> ReLU。
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        # 路径3的计算：1x1卷积 -> ReLU -> 5x5卷积 -> ReLU。
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        # 路径4的计算：3x3最大池化 -> 1x1卷积 -> ReLU。
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))
        # 将四个路径的输出在通道维度 (dim=1) 上拼接起来。
        # 拼接后的通道数 = c1 + c2[1] + c3[1] + c4。
        return torch.cat((p1, p2, p3, p4), dim=1)


# 3. GoogLeNet Model Definition
# ------------------------------------------------------------------------------
class GoogLeNet(nn.Module):
    """
    GoogLeNet (InceptionV1) 模型的定义。
    该模型由多个Inception模块、标准卷积层和池化层组成，用于图像分类任务。
    """
    def __init__(self, Inception):
        """
        GoogLeNet 模型的初始化函数。

        Args:
            Inception (class): Inception模块的类定义，用于构建模型内部的Inception块。
        """
        super(GoogLeNet, self).__init__()  # 调用父类nn.Module的构造函数进行初始化。

        # 第一个卷积块 (Block 1): 初始特征提取层。
        self.b1 = nn.Sequential(
            # 7x7卷积层：大卷积核，步长2，用于快速下采样和捕获粗粒度特征。
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),  # ReLU激活函数。
            # 3x3最大池化层：步长2，进一步下采样。
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第二个卷积块 (Block 2): 包含两个卷积层和一个最大池化层。
        self.b2 = nn.Sequential(
            # 1x1卷积层：进行通道调整。
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),  # ReLU激活函数。
            # 3x3卷积层：捕获局部特征。
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 3x3最大池化层：步长2，下采样。
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第三个卷积块 (Block 3): 包含两个Inception模块和一个最大池化层。
        self.b3 = nn.Sequential(
            # 第一个Inception模块：输入192通道，输出 64 + 128 + 32 + 32 = 256 通道。
            Inception(192, 64, (96, 128), (16, 32), 32),
            # 第二个Inception模块：输入256通道，输出 128 + 192 + 96 + 64 = 480 通道。
            Inception(256, 128, (128, 192), (32, 96), 64),
            # 3x3最大池化层：步长2，下采样。
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第四个卷积块 (Block 4): 包含五个Inception模块和一个最大池化层。
        self.b4 = nn.Sequential(
            # 第三个Inception模块：输入480通道，输出 192 + 208 + 48 + 64 = 512 通道。
            Inception(480, 192, (96, 208), (16, 48), 64),
            # 第四个Inception模块：输入512通道，输出 160 + 224 + 64 + 64 = 512 通道。
            Inception(512, 160, (112, 224), (24, 64), 64),
            # 第五个Inception模块：输入512通道，输出 128 + 256 + 64 + 64 = 512 通道。
            Inception(512, 128, (128, 256), (24, 64), 64),
            # 第六个Inception模块：输入512通道，输出 112 + 288 + 64 + 64 = 528 通道。
            Inception(512, 112, (128, 288), (32, 64), 64),
            # 第七个Inception模块：输入528通道，输出 256 + 320 + 128 + 128 = 832 通道。
            Inception(528, 256, (160, 320), (32, 128), 128),
            # 3x3最大池化层：步长2，下采样。
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第五个卷积块 (Block 5) 和分类器：包含两个Inception模块，全局平均池化和全连接层。
        self.b5 = nn.Sequential(
            # 第八个Inception模块：输入832通道，输出 256 + 320 + 128 + 128 = 832 通道。
            Inception(832, 256, (160, 320), (32, 128), 128),
            # 第九个Inception模块：输入832通道，输出 384 + 384 + 128 + 128 = 1024 通道。
            Inception(832, 384, (192, 384), (48, 128), 128),
            # 自适应平均池化层：将每个特征图的尺寸变为1x1，实现全局平均池化。
            # 这将 1024x7x7 (假设输入224x224) 变为 1024x1x1。
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # 展平层，将 1024x1x1 的特征图展平为 1024 维的向量。
            nn.Linear(1024, 10)  # 全连接层，将 1024 维特征映射到 10 个类别输出（例如FashionMNIST）。
        )

        # 权重初始化。
        # 遍历模型中的所有模块 (层)。
        for m in self.modules():
            # 如果模块是二维卷积层 (nn.Conv2d)。
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming正态分布初始化卷积层的权重，适用于ReLU激活函数，mode="fan_out"。
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                # 如果卷积层有偏置项。
                if m.bias is not None:
                    # 将偏置项初始化为常数0。
                    nn.init.constant_(m.bias, 0)
            # 如果模块是全连接层 (nn.Linear)。
            elif isinstance(m, nn.Linear): # 注意：这里修正了原始代码中的缩进错误。
                # 使用正态分布初始化全连接层的权重，均值为0，标准差为0.01。
                nn.init.normal_(m.weight, 0, 0.01)
                # 如果全连接层有偏置项。
                if m.bias is not None:
                    # 将偏置项初始化为常数0。
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        定义模型的前向传播路径。
        数据 x 将按照定义的层顺序进行处理。

        Args:
            x (torch.Tensor): 输入张量，通常是图像数据。
                              期望形状为 (batch_size, 1, H, W)，例如 (batch_size, 1, 224, 224)。

        Returns:
            torch.Tensor: 模型的输出，通常是每个类别的对数几率（logits）。
                          形状为 (batch_size, 10)。
        """
        # 输入数据 x 依次通过GoogLeNet的五个主要块。
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
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

    # 实例化GoogLeNet模型，并将Inception模块的类作为参数传入。
    # 然后将模型移动到选定的设备上（GPU或CPU）。
    model = GoogLeNet(Inception).to(device)
    print("GoogLeNet Model Initialized.")

    # 使用torchsummary打印模型的详细概览。
    # 第一个参数是模型实例。
    # 第二个参数是输入张量的尺寸（不包含batch_size）。
    # 对于GoogLeNet，通常输入是单通道灰度图像，尺寸为224x224。
    print(summary(model, (1, 224, 224)))