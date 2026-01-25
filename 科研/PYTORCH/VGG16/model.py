# ==============================================================================
# Python Script: VGG16 Convolutional Neural Network Implementation
# Author: [Original Author/Your Name]
# Date: [Current Date]
# Description:
# This script defines a VGG16 convolutional neural network architecture
# using PyTorch. It includes the model definition with multiple convolutional
# blocks, activation functions, pooling layers, and fully connected layers.
# The script also demonstrates how to instantiate the model, move it to an
# available device (GPU/CPU), and print a summary of its architecture,
# output shapes, and parameter counts using `torchsummary`.
# ==============================================================================

# 1. Import Necessary Libraries
# ------------------------------------------------------------------------------
import torch  # 导入PyTorch库，它是构建深度学习模型的核心。
from torch import nn  # 从PyTorch中导入神经网络模块，包含了构建层（如卷积层、线性层）和激活函数等。
from torchsummary import summary  # 导入torchsummary库，用于方便地打印模型的结构概览和参数数量。


# 2. VGG16 Model Definition
# ------------------------------------------------------------------------------
class VGG16(nn.Module):
    """
    VGG16 卷积神经网络模型定义。
    VGG16是一种经典的深度CNN架构，以其简单而重复的卷积层块结构著称，
    通常用于图像分类任务。本实现针对单通道输入（例如灰度图像），
    并调整了全连接层的输入尺寸以适应224x224的输入图像。
    """

    def __init__(self):
        """
        VGG16 模型的初始化函数，定义了网络的各个层。
        """
        super(VGG16, self).__init__()  # 调用父类nn.Module的构造函数进行初始化。

        # VGG16的第一个卷积块 (Block 1): 两个卷积层，一个最大池化层。
        # 输入通道为1 (灰度图像)，输出通道为64。
        self.block1 = nn.Sequential(
            # 卷积层: 1输入通道, 64输出通道, 3x3卷积核, 填充1 (保持尺寸)。
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 卷积层: 64输入通道, 64输出通道, 3x3卷积核, 填充1。
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 最大池化层: 2x2池化核, 步长2 (特征图尺寸减半)。
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # VGG16的第二个卷积块 (Block 2): 两个卷积层，一个最大池化层。
        # 输入通道为64，输出通道为128。
        self.block2 = nn.Sequential(
            # 卷积层: 64输入通道, 128输出通道, 3x3卷积核, 填充1。
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 卷积层: 128输入通道, 128输出通道, 3x3卷积核, 填充1。
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 最大池化层: 2x2池化核, 步长2。
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # VGG16的第三个卷积块 (Block 3): 三个卷积层，一个最大池化层。
        # 输入通道为128，输出通道为256。
        self.block3 = nn.Sequential(
            # 卷积层: 128输入通道, 256输出通道, 3x3卷积核, 填充1。
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 卷积层: 256输入通道, 256输出通道, 3x3卷积核, 填充1。
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 卷积层: 256输入通道, 256输出通道, 3x3卷积核, 填充1。
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 最大池化层: 2x2池化核, 步长2。
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # VGG16的第四个卷积块 (Block 4): 三个卷积层，一个最大池化层。
        # 输入通道为256，输出通道为512。
        self.block4 = nn.Sequential(
            # 卷积层: 256输入通道, 512输出通道, 3x3卷积核, 填充1。
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 卷积层: 512输入通道, 512输出通道, 3x3卷积核, 填充1。
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 卷积层: 512输入通道, 512输出通道, 3x3卷积核, 填充1。
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 最大池化层: 2x2池化核, 步长2。
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # VGG16的第五个卷积块 (Block 5): 三个卷积层，一个最大池化层。
        # 输入通道为512，输出通道为512。
        self.block5 = nn.Sequential(
            # 卷积层: 512输入通道, 512输出通道, 3x3卷积核, 填充1。
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 卷积层: 512输入通道, 512输出通道, 3x3卷积核, 填充1。
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 卷积层: 512输入通道, 512输出通道, 3x3卷积核, 填充1。
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU激活函数。
            # 最大池化层: 2x2池化核, 步长2。
            # 对于224x224的输入，经过5个Max Pooling层后，特征图尺寸变为 224 / (2^5) = 224 / 32 = 7x7。
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # VGG16的第六个全连接层块 (Block 6): 展平层和三个全连接层。
        self.block6 = nn.Sequential(
            nn.Flatten(),  # 展平层，将多维特征图展平为一维向量。
            # 经过block5后，特征图尺寸为512x7x7，展平后为 512 * 7 * 7 = 25088 个特征。
            # 第一个全连接层: 25088输入特征, 256输出特征 (原VGG16通常是4096，这里简化)。
            # 这里其实原始是4096个输出的，但是若显存太小了，就可以改小点改成256
            # 因为VGG当时是1000分类，这里是10分类，改小一点也可以的
            # nn.Linear(7 * 7 * 512, 4096), 
            nn.Linear(7 * 7 * 512, 256), 
            nn.ReLU(),  # ReLU激活函数。
            # 第二个全连接层: 256输入特征, 128输出特征。
            # nn.Linear(4096, 4096),
            nn.Linear(256, 128),
            nn.ReLU(),  # ReLU激活函数。
            # 输出层: 128输入特征, 10输出特征。
            # 10个输出对应分类任务的10个类别 (例如MNIST或FashionMNIST)。
            nn.Linear(4096, 10),
            nn.Linear(128, 10),
        )

        # 权重初始化。
        # 遍历模型中的所有模块 (层)。
        for m in self.modules():
            # 如果模块是二维卷积层 (nn.Conv2d)。
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming正态分布初始化卷积层的权重，适用于ReLU激活函数。
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                # 如果卷积层有偏置项。
                if m.bias is not None:
                    # 将偏置项初始化为常数0。
                    nn.init.constant_(m.bias, 0)
            # 如果模块是全连接层 (nn.Linear)。
            elif isinstance(m, nn.Linear):
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
        # 输入数据 x 依次通过VGG16的五个卷积块。
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # 经过卷积块后，数据 x 进入全连接层块进行分类。
        x = self.block6(x)
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

    # 实例化VGG16模型，并将其移动到选定的设备上（GPU或CPU）。
    model = VGG16().to(device)

    # 使用torchsummary打印模型的详细概览。
    # 第一个参数是模型实例。
    # 第二个参数是输入张量的尺寸（不包含batch_size）。
    # 对于VGG16，通常输入是单通道灰度图像，尺寸为224x224。
    print(summary(model, (1, 224, 224)))