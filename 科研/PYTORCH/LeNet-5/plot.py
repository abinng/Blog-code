# ==============================================================================
# Python Script: FashionMNIST Data Loading and Visualization
# Author: [Original Author/Your Name]
# Date: [Current Date]
# Description:
# This script demonstrates how to load the FashionMNIST dataset using PyTorch's
# torchvision library, apply transformations to resize and convert images to
# tensors, and then visualize a batch of these processed images using Matplotlib.
# It's a common first step in setting up a deep learning project with image data.
# ==============================================================================

# 1. Import Necessary Libraries
# ------------------------------------------------------------------------------
from torchvision.datasets import FashionMNIST  # 导入FashionMNIST数据集类，用于加载时尚MNIST数据集。
from torchvision import transforms  # 导入transforms模块，用于对图像进行预处理操作。
import torch.utils.data as Data  # 导入torch.utils.data模块，并将其别名为Data，主要用于数据加载器（DataLoader）。
import numpy as np  # 导入NumPy库，通常用于科学计算和数组操作。
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，并将其别名为plt，用于绘制图表。


# 2. Load and Preprocess FashionMNIST Training Data
# ------------------------------------------------------------------------------
# 加载FashionMNIST训练数据集。
train_data = FashionMNIST(
    root="./data",  # 指定数据集的存储路径。如果该路径下没有数据，则会下载。
    train=True,  # 表示加载训练集（True）而不是测试集（False）。
    transform=transforms.Compose(
        [  # 定义图像预处理操作的组合，这些操作将按顺序应用。
            transforms.Resize(size=224),  # 将原始28x28像素的图像尺寸调整为224x224像素，以适应一些预训练模型的要求。
            transforms.ToTensor(),  # 将PIL图像或NumPy数组转换为PyTorch张量，同时将像素值从[0, 255]归一化到[0.0, 1.0]范围。
        ]
    ),
    download=True,  # 如果数据集在指定root路径下不存在，则自动从网上下载。
)


# 3. Create a DataLoader for Batch Processing
# ------------------------------------------------------------------------------
# 创建训练数据加载器，它负责按批次（batch）加载数据。
train_loader = Data.DataLoader(
    dataset=train_data,  # 指定要加载的数据集对象。
    batch_size=64,  # 设置每个批次（batch）中包含的样本数量。这里设置为64张图像。
    shuffle=True,  # 在每个epoch（遍历整个数据集一次）开始时打乱数据，有助于提高模型的泛化能力，避免模型学习到数据的顺序。
    num_workers=0,  # 设置用于数据加载的子进程数。0表示在主进程中加载数据，这在调试时比较方便，但在生产环境中通常会设置为大于0以加速数据加载。
)


# 4. Extract a Single Batch of Data
# ------------------------------------------------------------------------------
# 遍历训练数据加载器，获取批次数据。
for step, (b_x, b_y) in enumerate(train_loader):
    # 仅获取第一个批次的数据进行后续处理和可视化，然后跳出循环。
    # 这样可以避免遍历整个数据集，只处理一个示例批次。
    if step > 0:
        break

# 将图像数据从PyTorch张量转换为NumPy数组。
# b_x 的原始形状是 [batch_size, channels, height, width]，例如 [64, 1, 224, 224]。
# .squeeze() 方法会移除所有维度为1的维度（在这里是通道维度），使其变为 [batch_size, height, width]，例如 [64, 224, 224]。
# .numpy() 方法将PyTorch张量转换为NumPy数组，以便与Matplotlib兼容。
batch_x = b_x.squeeze().numpy()

# 将标签数据从PyTorch张量转换为NumPy数组。
# b_y 的原始形状是 [batch_size]，例如 [64]。
batch_y = b_y.numpy()

# 获取FashionMNIST数据集中所有类别的名称列表，用于在可视化时显示标签。
class_label = train_data.classes

# 打印当前批次图像数据的形状，验证数据维度。
# 输出将显示批次大小、图像高度和图像宽度。
print("The size of batch in train data:", batch_x.shape)  # 预期输出类似: (64, 224, 224)


# 5. Visualize the Extracted Batch of Images
# ------------------------------------------------------------------------------
# 创建一个新的Matplotlib图形，并设置其尺寸。
# figsize=(12, 5) 表示图形的宽度为12英寸，高度为5英寸。
plt.figure(figsize=(12, 5))

# 循环遍历当前批次中的每一张图像及其对应的标签。
# np.arange(len(batch_y)) 生成一个从0到 batch_size-1 的整数序列。
for ii in np.arange(len(batch_y)):
    # 在4行16列的网格中创建一个子图。
    # ii + 1 表示当前子图的索引，从1开始。由于batch_size是64，4*16正好是64个子图。
    plt.subplot(4, 16, ii + 1)

    # 显示当前图像。
    # batch_x[ii, :, :] 选取批次中的第ii张图像的所有像素数据。
    # cmap=plt.cm.gray 指定使用灰度颜色映射，因为FashionMNIST图像是单通道灰度图。
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)

    # 设置子图的标题为图像的类别名称。
    # class_label[batch_y[ii]] 根据图像的数字标签获取对应的文字类别名称。
    # size=10 设置标题的字体大小。
    plt.title(class_label[batch_y[ii]], size=10)

    # 关闭子图的坐标轴，使图像显示更简洁，不显示刻度线和标签。
    plt.axis("off")

    # 调整子图之间的宽度间距，使其更紧凑。
    # wspace=0.05 表示子图之间的宽度空间占子图平均宽度的5%。
    plt.subplots_adjust(wspace=0.05)

# 6. Display and Save the Plot
# ------------------------------------------------------------------------------
# 将绘制好的图形保存为PNG文件。
# 建议在plt.show()之前保存，以确保保存的是未被用户交互改变的初始状态。
plt.savefig("fashion_mnist_batch_plot.png")

# 显示所有绘制的图像。
# 这会打开一个交互式窗口显示图形。在脚本执行完毕前，该窗口会一直保持打开状态。
plt.show()

# 关闭图形，释放内存资源。
# 这在脚本中尤其重要，可以避免在循环或大量绘图操作时耗尽内存。
plt.close()