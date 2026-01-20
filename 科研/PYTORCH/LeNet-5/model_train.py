# ==============================================================================
# Python Script: LeNet Model Training and Validation on FashionMNIST
# Author: [Original Author/Your Name]
# Date: [Current Date]
# Description:
# This script implements the training and validation pipeline for a LeNet-5
# convolutional neural network on the FashionMNIST dataset. It includes
# functions for data loading, preprocessing, model training with an optimizer
# and loss function, tracking performance metrics (loss and accuracy),
# saving the best performing model, and visualizing the training progress.
# ==============================================================================

# 1. Import Necessary Libraries
# ------------------------------------------------------------------------------
import copy  # 用于深拷贝Python对象，这里用于保存模型参数。
import time  # 用于计算代码执行时间。

import torch  # 导入PyTorch库，它是构建和运行深度学习模型的核心。
from torchvision.datasets import FashionMNIST  # 导入FashionMNIST数据集类。
from torchvision import transforms  # 导入transforms模块，用于图像预处理操作。
import torch.utils.data as Data  # 导入torch.utils.data模块，用于数据加载器（DataLoader）。
import numpy as np  # 导入NumPy库，用于数值计算，尽管在此脚本中直接使用不多，但常用于数据处理。
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于数据可视化（绘制图表）。
from model import LeNet  # 从名为'model.py'的文件中导入自定义的LeNet模型类。
import torch.nn as nn  # 导入torch.nn模块，包含了神经网络层、损失函数等。
import pandas as pd  # 导入Pandas库，用于数据结构（如DataFrame）和数据分析，这里用于存储训练过程数据。


# 2. Function to Process Training and Validation Data
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
        root='./data',  # 指定数据集的存储路径。
        train=True,  # 表示加载训练集。
        transform=transforms.Compose([  # 定义图像预处理操作的组合。
            transforms.Resize(size=28),  # 将图像尺寸调整为28x28像素，与LeNet模型期望的输入尺寸一致。
            transforms.ToTensor()  # 将PIL图像或NumPy数组转换为PyTorch张量，并归一化到[0.0, 1.0]范围。
        ]),
        download=True  # 如果数据集不存在，则自动下载。
    )

    # 将训练数据集划分为训练集和验证集。
    # 80%用于训练，20%用于验证。random_split函数会随机进行划分。
    train_data, val_data = Data.random_split(
        train_data,
        [round(0.8 * len(train_data)), round(0.2 * len(train_data))]
    )

    # 创建训练数据加载器。
    train_dataloader = Data.DataLoader(
        dataset=train_data,  # 指定训练数据集。
        batch_size=32,  # 设置每个批次（batch）中包含的样本数量为32。
        shuffle=True,  # 在每个epoch开始时打乱数据，以增加模型的泛化能力。
        num_workers=2  # 设置用于数据加载的子进程数，可以加速数据读取。
    )

    # 创建验证数据加载器。
    val_dataloader = Data.DataLoader(
        dataset=val_data,  # 指定验证数据集。
        batch_size=32,  # 设置每个批次（batch）中包含的样本数量为32。
        shuffle=True,  # 在每个epoch开始时打乱数据，尽管对于验证集通常不是必需的，但无害。
        num_workers=2  # 设置用于数据加载的子进程数。
    )

    return train_dataloader, val_dataloader  # 返回训练和验证数据加载器。


# 3. Function to Train the Model
# ------------------------------------------------------------------------------
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    """
    训练和验证LeNet模型。

    Args:
        model (nn.Module): 待训练的PyTorch模型实例。
        train_dataloader (torch.utils.data.DataLoader): 训练数据加载器。
        val_dataloader (torch.utils.data.DataLoader): 验证数据加载器。
        num_epochs (int): 训练的总轮数（epoch数量）。

    Returns:
        pandas.DataFrame: 包含每个epoch的训练和验证损失以及准确率的DataFrame。
    """
    # 设定训练所用的设备：如果有CUDA GPU则用GPU，否则用CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}") # 打印当前使用的设备。

    # 使用Adam优化器，它是一种自适应学习率的优化算法。
    # model.parameters() 返回模型中所有可学习的参数。
    # lr (learning rate) 设置为0.001。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 损失函数为交叉熵损失（CrossEntropyLoss），适用于多分类问题。
    # 它结合了LogSoftmax和NLLLoss。
    criterion = nn.CrossEntropyLoss()

    # 将模型移动到指定的设备上（GPU或CPU）。
    model = model.to(device)

    # 复制当前模型的参数（state_dict），作为初始的最佳模型参数。
    # 在训练过程中，如果遇到更好的验证准确率，将更新此副本。
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化用于记录训练过程的参数。
    best_acc = 0.0  # 记录迄今为止最高的验证准确率。
    train_loss_all = []  # 存储每个epoch的训练集损失。
    val_loss_all = []  # 存储每个epoch的验证集损失。
    train_acc_all = []  # 存储每个epoch的训练集准确率。
    val_acc_all = []  # 存储每个epoch的验证集准确率。
    since = time.time()  # 记录训练开始时间，用于计算总耗时。

    # 开始训练循环，迭代num_epochs次。
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")  # 打印当前epoch进度。
        print("-" * 10)  # 分隔线，提高可读性。

        # 初始化每个epoch的统计变量。
        train_loss = 0.0  # 当前epoch训练集累计损失。
        train_corrects = 0  # 当前epoch训练集正确预测数量。
        val_loss = 0.0  # 当前epoch验证集累计损失。
        val_corrects = 0  # 当前epoch验证集正确预测数量。
        train_num = 0  # 当前epoch训练集样本总数。
        val_num = 0  # 当前epoch验证集样本总数。

        # --- 训练阶段 ---
        # 遍历训练数据加载器中的每个批次。
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征数据（图像）移动到训练设备上。
            b_x = b_x.to(device)
            # 将标签数据（真实类别）移动到训练设备上。
            b_y = b_y.to(device)

            # 设置模型为训练模式。
            # 这会启用如Dropout层和Batch Normalization层在训练时的行为。
            model.train()

            # 前向传播过程：输入一个批次的数据，模型输出对应的预测值（logits）。
            output = model(b_x)
            # 从模型的输出中找到每个样本预测概率最高的类别索引。
            # torch.argmax(output, dim=1) 返回每一行（即每个样本）中最大值对应的列索引。
            pre_lab = torch.argmax(output, dim=1)
            # 计算当前批次的损失函数值。
            loss = criterion(output, b_y)

            # 将优化器中所有参数的梯度清零。
            # 这是因为PyTorch默认会累积梯度，在每次反向传播前需要清零。
            optimizer.zero_grad()
            # 反向传播：根据损失计算模型参数的梯度。
            loss.backward()
            # 根据计算出的梯度更新模型的参数。
            optimizer.step()

            # 累加当前批次的损失到总训练损失中。
            # loss.item() 获取标量损失值，b_x.size(0) 是当前批次的样本数量。
            train_loss += loss.item() * b_x.size(0)
            # 统计当前批次中预测正确的样本数量，并累加到总的正确数量中。
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 累加当前批次的样本数量到总的训练样本数量中。
            train_num += b_x.size(0)

        # --- 验证阶段 ---
        # 遍历验证数据加载器中的每个批次。
        # torch.no_grad() 上下文管理器：在此块内的所有计算都不会跟踪梯度。
        # 这可以节省内存并加速计算，因为在验证阶段不需要进行反向传播。
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(val_dataloader):
                # 将特征数据（图像）移动到验证设备上。
                b_x = b_x.to(device)
                # 将标签数据（真实类别）移动到验证设备上。
                b_y = b_y.to(device)

                # 设置模型为评估模式。
                # 这会关闭如Dropout层和Batch Normalization层在训练和评估阶段行为不同的层。
                model.eval()

                # 前向传播过程：输入一个批次的数据，模型输出对应的预测值（logits）。
                output = model(b_x)
                # 从模型的输出中找到每个样本预测概率最高的类别索引。
                pre_lab = torch.argmax(output, dim=1)
                # 计算当前批次的损失函数值。
                loss = criterion(output, b_y)

                # 累加当前批次的损失到总验证损失中。
                val_loss += loss.item() * b_x.size(0)
                # 统计当前批次中预测正确的样本数量，并累加到总的正确数量中。
                val_corrects += torch.sum(pre_lab == b_y.data)
                # 累加当前批次的样本数量到总的验证样本数量中。
                val_num += b_x.size(0)

        # 计算并保存当前epoch的平均损失和准确率。
        train_loss_all.append(train_loss / train_num)  # 训练集平均损失。
        train_acc_all.append(train_corrects.double().item() / train_num)  # 训练集平均准确率。

        val_loss_all.append(val_loss / val_num)  # 验证集平均损失。
        val_acc_all.append(val_corrects.double().item() / val_num)  # 验证集平均准确率。

        # 打印当前epoch的训练和验证结果。
        print(f"Epoch {epoch} train loss:{train_loss_all[-1]:.4f} train acc: {train_acc_all[-1]:.4f}")
        print(f"Epoch {epoch} val loss:{val_loss_all[-1]:.4f} val acc: {val_acc_all[-1]:.4f}")

        # 如果当前epoch的验证准确率优于之前的最佳准确率，则更新最佳准确率和保存最佳模型参数。
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]  # 更新最高准确度。
            best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝当前模型的参数。

        # 计算并打印当前epoch为止的训练和验证总耗时。
        time_use = time.time() - since
        print(f"训练和验证耗费的时间{int(time_use // 60)}m{int(time_use % 60)}s")

    # 训练结束后，将最佳模型的参数加载回模型中。
    # 这样，最终保存的模型就是验证集上表现最好的模型。
    model.load_state_dict(best_model_wts)
    # 将最佳模型的'参数'保存到文件中。
    # 注意：'C:/Users/86159/Desktop/LeNet/best_model.pth' 是一个硬编码的绝对路径，
    # 在实际应用中，建议使用相对路径或配置变量来指定保存位置。
    torch.save(best_model_wts, "best_model.pth") # 保存到当前工作目录

    # 将训练过程中的各项指标整理成一个Pandas DataFrame，方便后续分析和可视化。
    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all,
    })

    return train_process  # 返回包含训练过程数据的DataFrame。


# 4. Function to Plot Accuracy and Loss
# ------------------------------------------------------------------------------
def matplot_acc_loss(train_process):
    """
    绘制训练和验证过程中的损失和准确率曲线图。

    Args:
        train_process (pandas.DataFrame): 包含训练过程数据的DataFrame。
    """
    # 创建一个图形，包含两个子图（一个用于损失，一个用于准确率）。
    plt.figure(figsize=(12, 4))

    # 第一个子图：损失曲线。
    plt.subplot(1, 2, 1)  # 1行2列的第一个子图。
    # 绘制训练损失曲线（红色圆点实线）。
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    # 绘制验证损失曲线（蓝色方块实线）。
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()  # 显示图例。
    plt.xlabel("Epoch")  # 设置X轴标签。
    plt.ylabel("Loss")  # 设置Y轴标签。
    plt.title("Training and Validation Loss") # 设置子图标题

    # 第二个子图：准确率曲线。
    plt.subplot(1, 2, 2)  # 1行2列的第二个子图。
    # 绘制训练准确率曲线（红色圆点实线）。
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    # 绘制验证准确率曲线（蓝色方块实线）。
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.legend()  # 显示图例。
    plt.xlabel("Epoch")  # 设置X轴标签。
    plt.ylabel("Accuracy")  # 设置Y轴标签。
    plt.title("Training and Validation Accuracy") # 设置子图标题

    plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域。
    plt.show()  # 显示绘制的图表。
    plt.savefig("fashion_LeNet-5_32bs_50ep.png")


# 5. Main Execution Block
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    当脚本直接运行时，执行此代码块。
    负责模型的实例化、数据加载、模型训练和结果可视化。
    """
    # 实例化LeNet模型。
    model = LeNet()
    print("LeNet model instantiated.")

    # 调用函数加载并获取训练和验证数据加载器。
    train_dataloader, val_dataloader = train_val_data_process()
    print("FashionMNIST data loaded and split into training/validation sets.")

    # 调用函数进行模型的训练，并指定训练的总轮数为20。
    # train_process 将包含训练过程中的各项指标。
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs=50)
    print("Model training complete.")

    # 调用函数绘制训练过程中的损失和准确率曲线图。
    matplot_acc_loss(train_process)
    print("Training process plots displayed.")