# ==============================================================================
# Python Script: AlexNet Model Definition
# Author: [Original Author/Your Name]
# Date: [Current Date]
# Description:
# This script defines the AlexNet convolutional neural network architecture
# using PyTorch. It includes the class definition for AlexNet, specifying
# its layers and the forward pass logic. It also demonstrates how to
# instantiate the model and print a summary of its structure and parameters
# using `torchsummary`.
# ==============================================================================

# 1. Import Necessary Libraries
# ------------------------------------------------------------------------------
import torch  # 导入PyTorch库，它是构建和运行深度学习模型的核心。
from torch import nn  # 导入torch.nn模块，包含了构建神经网络层和模块的类。
from torchsummary import summary  # 导入torchsummary库，用于打印模型的详细结构和参数信息。
import torch.nn.functional as F  # 导入torch.nn.functional模块，包含了常用函数（如激活函数、池化函数）的函数式实现。


# 2. AlexNet Model Class Definition
# ------------------------------------------------------------------------------
class AlexNet(nn.Module):
    """
    AlexNet模型的PyTorch实现。
    AlexNet是一个经典的深度卷积神经网络，以其在2012年ImageNet竞赛中的胜利而闻名。
    此实现针对单通道输入（如灰度图像或FashionMNIST）和10个输出类别进行了调整。
    """
    def __init__(self):
        """
        构造函数：定义AlexNet模型的所有层。
        """
        super(AlexNet, self).__init__()  # 调用父类nn.Module的构造函数，进行必要的初始化。

        # 定义激活函数ReLU，可以在forward方法中重复使用。
        self.ReLU = nn.ReLU()

        # 第一层卷积层：
        # in_channels=1: 输入图像是单通道（灰度图）。
        # out_channels=96: 输出96个特征图。
        # kernel_size=11: 卷积核大小为11x11。
        # stride=4: 步长为4，大幅度缩小特征图尺寸。
        self.c1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)

        # 第一层最大池化层：
        # kernel_size=3: 池化窗口大小为3x3。
        # stride=2: 步长为2。
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 第二层卷积层：
        # in_channels=96: 匹配上一层的输出通道数。
        # out_channels=256: 输出256个特征图。
        # kernel_size=5: 卷积核大小为5x5。
        # padding=2: 填充2像素，以保持特征图的尺寸。
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)

        # 第二层最大池化层：
        # 配置同s2。
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 第三层卷积层：
        # in_channels=256: 匹配上一层的输出通道数。
        # out_channels=384: 输出384个特征图。
        # kernel_size=3: 卷积核大小为3x3。
        # padding=1: 填充1像素，以保持特征图的尺寸。
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)

        # 第四层卷积层：
        # 配置同c5，通道数保持不变。
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)

        # 第五层卷积层：
        # in_channels=384: 匹配上一层的输出通道数。
        # out_channels=256: 输出256个特征图。
        # 配置同c5。
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)

        # 第三层最大池化层：
        # 配置同s2。
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Flatten层：
        # 将多维的特征图展平为一维向量，以便输入到全连接层。
        self.flatten = nn.Flatten()

        # 全连接层1 (f1)：
        # 输入特征数为 6*6*256。这个尺寸是根据输入图像227x227经过所有卷积和池化层后的特征图大小计算得出的。
        # 输出特征数为4096。
        self.f1 = nn.Linear(6 * 6 * 256, 4096)

        # 全连接层2 (f2)：
        # 输入特征数为4096。
        # 输出特征数为4096。
        self.f2 = nn.Linear(4096, 4096)

        # 全连接层3 (f3，输出层)：
        # 输入特征数为4096。
        # 输出特征数为10，对应FashionMNIST的10个类别。
        self.f3 = nn.Linear(4096, 10)

    def forward(self, x):
        """
        前向传播函数：定义数据流经模型的路径。

        Args:
            x (torch.Tensor): 输入张量，通常是图像数据。

        Returns:
            torch.Tensor: 模型的输出，通常是每个类别的预测分数（logits）。
        """
        # 卷积层c1 -> ReLU激活。
        x = self.ReLU(self.c1(x))
        # 最大池化层s2。
        x = self.s2(x)
        # 卷积层c3 -> ReLU激活。
        x = self.ReLU(self.c3(x))
        # 最大池化层s4。
        x = self.s4(x)
        # 卷积层c5 -> ReLU激活。
        x = self.ReLU(self.c5(x))
        # 卷积层c6 -> ReLU激活。
        x = self.ReLU(self.c6(x))
        # 卷积层c7 -> ReLU激活。
        x = self.ReLU(self.c7(x))
        # 最大池化层s8。
        x = self.s8(x)

        # 将卷积层输出展平为一维向量。
        x = self.flatten(x)
        # 全连接层f1 -> ReLU激活。
        x = self.ReLU(self.f1(x))
        # Dropout层：以0.5的概率随机丢弃神经元，防止过拟合。
        # F.dropout是函数式接口，不带可学习参数。
        x = F.dropout(x, 0.5)
        # 全连接层f2 -> ReLU激活。
        x = self.ReLU(self.f2(x))
        # Dropout层：配置同上。
        x = F.dropout(x, 0.5)
        # 最后一层全连接层f3：输出原始的类别分数（logits），通常不在此处应用激活函数，
        # 因为交叉熵损失函数内部会包含Softmax操作。
        x = self.f3(x)
        return x  # 返回模型的最终输出。


# 3. Main Execution Block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    当脚本直接运行时，执行此代码块。
    负责实例化AlexNet模型并打印其结构摘要。
    """
    # 设定训练所用到的设备：如果有CUDA GPU则用GPU，否则用CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") # 打印当前使用的设备。

    # 实例化AlexNet模型，并将其移动到指定的设备上（GPU或CPU）。
    model = AlexNet().to(device)
    print("AlexNet model instantiated and moved to device.")

    # 使用torchsummary打印模型的详细结构和参数信息。
    # (1, 227, 227) 表示输入张量的形状：1个通道（灰度图），227x227像素。
    # 这个输入尺寸是AlexNet原始设计所期望的，用于计算全连接层输入维度。
    print("\nModel Summary:")
    summary(model, (1, 227, 227))
    print("\nModel summary printed.")