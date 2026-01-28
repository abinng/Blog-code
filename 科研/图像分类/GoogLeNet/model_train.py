# ==============================================================================
# Python Script: GoogLeNet Model Training and Evaluation on FashionMNIST
# Author: [Original Author/Your Name] # 请在此处填写作者姓名
# Date: 2025-12-21 # 已更新为当前日期
# Description:
# This script implements the training and validation process for a GoogLeNet
# convolutional neural network on the FashionMNIST dataset. It includes:
# 1. Data loading, preprocessing (resizing and tensor conversion), and splitting
#    into training and validation sets using PyTorch's DataLoader.
# 2. A comprehensive training loop that iterates through epochs, calculates
#    loss and accuracy for both training and validation sets, and saves
#    the best performing model based on validation accuracy.
# 3. Visualization of training and validation loss and accuracy over epochs
#    using Matplotlib.
# The GoogLeNet and Inception module definitions are assumed to be in a
# separate 'model.py' file.
# ==============================================================================

# 1. Import Necessary Libraries
# ------------------------------------------------------------------------------
import copy  # 用于深度复制模型参数，以便保存最佳模型。
import time  # 用于计算训练过程的耗时。

import torch  # PyTorch库，构建深度学习模型的核心。
from torchvision.datasets import FashionMNIST  # 从torchvision导入FashionMNIST数据集。
from torchvision import transforms  # 从torchvision导入transforms模块，用于数据预处理和增强。
import torch.utils.data as Data  # 从torch.utils.data导入Data模块，用于处理数据集和数据加载器。
import numpy as np  # NumPy库，用于数值计算（在此脚本中未直接使用，但常用于数据处理）。
import matplotlib.pyplot as plt  # Matplotlib的pyplot模块，用于数据可视化。
from model import GoogLeNet, Inception  # 从自定义的model.py文件中导入GoogLeNet模型类和其核心Inception模块。
import torch.nn as nn  # PyTorch神经网络模块，包含损失函数等。
import pandas as pd  # Pandas库，用于创建和处理数据表格，这里用于保存训练过程的指标。


# 2. Data Processing Function
# ------------------------------------------------------------------------------
def train_val_data_process():
    """
    加载并预处理FashionMNIST训练数据集，并将其划分为训练集和验证集。
    然后为这两个数据集创建数据加载器。

    Returns:
        tuple: 包含训练数据加载器和验证数据加载器的元组。
               (train_dataloader, val_dataloader)
    """
    # 加载FashionMNIST训练数据集。
    train_data = FashionMNIST(
        root="./data",  # 数据集存储的根目录。
        train=True,  # 指定加载训练集。
        transform=transforms.Compose(
            [  # 定义数据转换操作序列。
                transforms.Resize(
                    size=224
                ),  # 将图片尺寸调整为224x224，以适应GoogLeNet的输入要求。
                transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为Tensor，并归一化到[0.0, 1.0]。
            ]
        ),
        download=True,  # 如果数据集不存在，则自动下载。
    )

    # 将训练数据集随机分割为训练集和验证集。
    # 按照80%训练集和20%验证集的比例进行分割。
    train_data, val_data = Data.random_split(
        train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))]
    )

    # 创建训练数据加载器。
    train_dataloader = Data.DataLoader(
        dataset=train_data,  # 指定数据集。
        batch_size=128,  # 每个批次的样本数量。
        shuffle=True,  # 每个epoch开始时打乱数据。
        num_workers=2,  # 用于数据加载的子进程数量，提高加载效率。
    )

    # 创建验证数据加载器。
    val_dataloader = Data.DataLoader(
        dataset=val_data,  # 指定数据集。
        batch_size=128,  # 每个批次的样本数量。
        shuffle=True,  # 每个epoch开始时打乱数据。
        num_workers=2,  # 用于数据加载的子进程数量。
    )

    return train_dataloader, val_dataloader  # 返回训练和验证数据加载器。


# 3. Model Training Function
# ------------------------------------------------------------------------------
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    """
    执行模型的训练和验证过程。

    Args:
        model (nn.Module): 要训练的神经网络模型。
        train_dataloader (DataLoader): 训练数据加载器。
        val_dataloader (DataLoader): 验证数据加载器。
        num_epochs (int): 训练的总epoch数量。

    Returns:
        pd.DataFrame: 包含每个epoch的训练和验证损失及准确率的DataFrame。
    """
    # 设定训练所用到的设备，优先使用GPU（CUDA），否则使用CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")  # 打印当前使用的设备。

    # 使用Adam优化器，对模型的参数进行优化。
    # lr=0.001是学习率。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数采用交叉熵函数，适用于多分类任务。
    criterion = nn.CrossEntropyLoss()
    # 将模型移动到指定的训练设备上（GPU或CPU）。
    model = model.to(device)
    # 深度复制当前模型的参数（state_dict），用于在训练过程中保存最佳模型参数。
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化训练过程中的各项指标。
    best_acc = 0.0  # 记录验证集上的最高准确度。
    train_loss_all = []  # 存储每个epoch的训练集损失。
    val_loss_all = []  # 存储每个epoch的验证集损失。
    train_acc_all = []  # 存储每个epoch的训练集准确度。
    val_acc_all = []  # 存储每个epoch的验证集准确度。
    since = time.time()  # 记录训练开始时间，用于计算总耗时。

    # 开始训练循环，遍历所有epoch。
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")  # 打印当前epoch进度。
        print("-" * 10)  # 打印分隔线。

        # 每个epoch开始时，初始化本epoch的指标。
        train_loss = 0.0  # 训练集损失累计。
        train_corrects = 0  # 训练集正确预测数量累计。
        val_loss = 0.0  # 验证集损失累计。
        val_corrects = 0  # 验证集正确预测数量累计。
        train_num = 0  # 训练集样本数量累计。
        val_num = 0  # 验证集样本数量累计。

        # --------------------- 训练阶段 ---------------------
        # 遍历训练数据加载器中的每一个mini-batch。
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征数据移动到训练设备上。
            b_x = b_x.to(device)
            # 将标签数据移动到训练设备上。
            b_y = b_y.to(device)
            # 设置模型为训练模式。
            # 这会启用Dropout层和Batch Normalization层的训练行为（如更新统计量）。
            model.train()

            # 前向传播过程：输入一个batch的数据，得到模型的预测输出。
            output = model(b_x)
            # 查找模型输出中每一行（每个样本）最大值对应的索引，即预测的类别。
            pre_lab = torch.argmax(output, dim=1)
            # 计算当前batch的损失函数值。
            loss = criterion(output, b_y)

            # 将优化器中所有参数的梯度清零。
            optimizer.zero_grad()
            # 执行反向传播，计算每个可学习参数的梯度。
            loss.backward()
            # 根据计算出的梯度更新模型的参数。
            optimizer.step()
            # 累加当前batch的损失，乘以batch大小以获得总损失贡献。
            train_loss += loss.item() * b_x.size(0)
            # 累加正确预测的数量。
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 累加当前batch的样本数量。
            train_num += b_x.size(0)

        # --------------------- 验证阶段 ---------------------
        # 遍历验证数据加载器中的每一个mini-batch。
        # 在验证阶段，通常不需要计算梯度，可以节省内存和计算。
        with torch.no_grad():  # 禁用梯度计算。
            for step, (b_x, b_y) in enumerate(val_dataloader):
                # 将特征数据移动到验证设备上。
                b_x = b_x.to(device)
                # 将标签数据移动到验证设备上。
                b_y = b_y.to(device)
                # 设置模型为评估模式。
                # 这会禁用Dropout层，并使用Batch Normalization层的运行统计量。
                model.eval()
                # 前向传播过程：输入一个batch的数据，得到模型的预测输出。
                output = model(b_x)
                # 查找模型输出中每一行（每个样本）最大值对应的索引，即预测的类别。
                pre_lab = torch.argmax(output, dim=1)
                # 计算每一个batch的损失函数。
                loss = criterion(output, b_y)

                # 对损失函数进行累加。
                val_loss += loss.item() * b_x.size(0)
                # 如果预测正确，则准确度val_corrects累加。
                val_corrects += torch.sum(pre_lab == b_y.data)
                # 当前用于验证的样本数量。
                val_num += b_x.size(0)

        # 计算并保存当前epoch的训练集和验证集指标。
        train_loss_all.append(train_loss / train_num)  # 计算平均训练损失。
        train_acc_all.append(
            train_corrects.double().item() / train_num
        )  # 计算平均训练准确率。

        val_loss_all.append(val_loss / val_num)  # 计算平均验证损失。
        val_acc_all.append(
            val_corrects.double().item() / val_num
        )  # 计算平均验证准确率。

        # 打印当前epoch的训练和验证结果。
        print(
            f"Epoch {epoch} train loss:{train_loss_all[-1]:.4f} train acc: {train_acc_all[-1]:.4f}"
        )
        print(
            f"Epoch {epoch} val loss:{val_loss_all[-1]:.4f} val acc: {val_acc_all[-1]:.4f}"
        )

        # 如果当前epoch的验证准确率高于历史最高准确率。
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]  # 更新最高准确度。
            best_model_wts = copy.deepcopy(
                model.state_dict()
            )  # 深度复制当前模型的参数作为最佳模型。

        # 计算当前epoch的训练和验证总耗时。
        time_use = time.time() - since
        print(f"训练和验证耗费的时间{int(time_use // 60)}m{int(time_use % 60)}s")

    # 训练结束后，加载在验证集上表现最佳的模型参数。
    model.load_state_dict(best_model_wts)
    # 保存最佳模型的参数到指定路径。
    # 已将硬编码的绝对路径更改为相对路径，并使用特定文件名。
    torch.save(best_model_wts, "best_model_GoogLeNet.pth")

    # 将训练过程中的指标整理成一个pandas DataFrame。
    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all,
        }
    )

    return train_process  # 返回包含训练过程指标的DataFrame。


# 4. Plotting Function
# ------------------------------------------------------------------------------
def matplot_acc_loss(train_process):
    """
    绘制训练和验证过程中的损失和准确率曲线图。

    Args:
        train_process (pd.DataFrame): 包含训练过程指标的DataFrame。
    """
    # 创建一个图形，包含两个子图（一行两列）。
    plt.figure(figsize=(12, 4))

    # 第一个子图：损失曲线。
    plt.subplot(1, 2, 1)  # 1行2列的第一个子图。
    plt.plot(
        train_process["epoch"], train_process.train_loss_all, "ro-", label="Train loss"
    )  # 绘制训练损失曲线。
    plt.plot(
        train_process["epoch"], train_process.val_loss_all, "bs-", label="Val loss"
    )  # 绘制验证损失曲线。
    plt.legend()  # 显示图例。
    plt.xlabel("Epoch")  # 设置X轴标签。
    plt.ylabel("Loss")  # 设置Y轴标签。
    plt.title("Training and Validation Loss (GoogLeNet)")  # 设置子图标题。

    # 第二个子图：准确率曲线。
    plt.subplot(1, 2, 2)  # 1行2列的第二个子图。
    plt.plot(
        train_process["epoch"], train_process.train_acc_all, "ro-", label="Train acc"
    )  # 绘制训练准确率曲线。
    plt.plot(
        train_process["epoch"], train_process.val_acc_all, "bs-", label="Val acc"
    )  # 绘制验证准确率曲线。
    plt.xlabel("Epoch")  # 设置X轴标签。
    plt.ylabel("Accuracy")  # 设置Y轴标签。
    plt.legend()  # 显示图例。
    plt.title("Training and Validation Accuracy (GoogLeNet)")  # 设置子图标题。

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域。
    # 保存图形，确保在plt.show()之前调用。文件名已更新为实际使用的批次大小和epoch数量。
    plt.savefig("fashion_GoogLeNet_128bs_20ep.png")
    plt.show()  # 显示图形。


# 5. Main Execution Block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    脚本的入口点。当脚本直接运行时，此代码块将被执行。
    它负责实例化模型、加载数据、启动训练和绘制结果。
    """
    # 实例化GoogLeNet模型，并将Inception模块的类作为参数传入。
    model_googlenet = GoogLeNet(Inception)
    print("GoogLeNet Model Initialized.")

    # 加载并处理数据集，获取训练和验证数据加载器。
    train_dataloader, val_dataloader = train_val_data_process()
    print("Dataset Loaded and Processed.")

    # 利用GoogLeNet模型、数据加载器和指定的epoch数量（20）进行模型训练。
    # 训练过程的指标将存储在train_process DataFrame中。
    train_process = train_model_process(
        model_googlenet, train_dataloader, val_dataloader, num_epochs=20
    )
    print("Model Training Completed.")

    # 绘制训练过程中的损失和准确率曲线。
    matplot_acc_loss(train_process)
    print("Training Progress Plots Displayed.")