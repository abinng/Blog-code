# ==============================================================================
# Python Script: AlexNet Model Training and Validation on FashionMNIST
# Author: [Original Author/Your Name]
# Date: [Current Date]
# Description:
# This script implements the training and validation pipeline for an AlexNet
# convolutional neural network on the FashionMNIST dataset. It includes
# functions for data loading, preprocessing (resizing to 227x227 for AlexNet),
# model training with an Adam optimizer and CrossEntropyLoss, tracking
# performance metrics (loss and accuracy) per epoch, saving the best performing
# model based on validation accuracy, and visualizing the training progress
# (loss and accuracy curves).
# ==============================================================================

# 1. 导入必要的库
# ------------------------------------------------------------------------------
import copy  # 用于深拷贝Python对象，这里主要用于保存模型在验证集上表现最佳时的参数。
import time  # 用于计算代码执行时间，例如训练一个epoch或整个训练过程的耗时。

import torch  # 导入PyTorch库，它是构建和运行深度学习模型的核心框架。
from torchvision.datasets import FashionMNIST  # 从torchvision库中导入FashionMNIST数据集类。
from torchvision import transforms  # 从torchvision库中导入transforms模块，用于图像预处理操作。
import torch.utils.data as Data  # 导入torch.utils.data模块，用于数据加载器（DataLoader）的创建和管理。
import numpy as np  # 导入NumPy库，用于进行科学计算，尤其是在处理数组和矩阵时。
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于数据可视化，如绘制损失和准确率曲线。
from model import AlexNet  # 从名为'model.py'的文件中导入自定义的AlexNet模型类。
import torch.nn as nn  # 导入torch.nn模块，包含了神经网络层、损失函数等构建块。
import pandas as pd  # 导入Pandas库，用于数据结构（如DataFrame）和数据分析，这里用于存储和处理训练过程数据。


# 2. 数据处理函数
# ------------------------------------------------------------------------------
def train_val_data_process():
    """
    加载并预处理FashionMNIST训练数据集，将其划分为训练集和验证集，
    并为它们创建数据加载器。

    Returns:
        tuple: 包含训练数据加载器和验证数据加载器的元组。
               (train_dataloader, val_dataloader)
    """
    # 加载FashionMNIST训练数据集。
    train_data = FashionMNIST(
        root='./data',  # 指定数据集的存储路径。如果不存在，数据将下载到此目录。
        train=True,  # 指定加载的是训练集。
        transform=transforms.Compose([  # 定义一系列图像预处理操作。
            transforms.Resize(size=227),  # 将图像尺寸调整为227x227像素，这是AlexNet模型通常期望的输入尺寸。
            transforms.ToTensor()  # 将PIL图像或NumPy数组转换为PyTorch张量，并自动将像素值归一化到[0.0, 1.0]范围。
        ]),
        download=True  # 如果数据集在指定路径不存在，则自动从网上下载。
    )

    # 将原始训练数据集随机划分为训练集和验证集。
    # 按照大约80%训练集、20%验证集的比例进行划分。
    train_data, val_data = Data.random_split(
        train_data,
        [round(0.8 * len(train_data)), round(0.2 * len(train_data))]
    )

    # 创建训练数据加载器。
    train_dataloader = Data.DataLoader(
        dataset=train_data,  # 指定要加载的数据集。
        batch_size=32,  # 每个批次（batch）中包含32个样本。
        shuffle=True,  # 在每个epoch开始时打乱数据，以提高模型的泛化能力。
        num_workers=2  # 使用2个子进程进行数据加载，可以加速数据读取过程。
    )

    # 创建验证数据加载器。
    val_dataloader = Data.DataLoader(
        dataset=val_data,  # 指定要加载的验证数据集。
        batch_size=32,  # 每个批次中包含32个样本。
        shuffle=True,  # 在每个epoch开始时打乱数据，对验证集通常不是必需的，但无害。
        num_workers=2  # 使用2个子进程进行数据加载。
    )

    return train_dataloader, val_dataloader  # 返回创建好的训练和验证数据加载器。


# 3. 模型训练函数
# ------------------------------------------------------------------------------
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    """
    执行模型的训练和验证过程。

    Args:
        model (nn.Module): 待训练的PyTorch模型实例（这里是AlexNet）。
        train_dataloader (torch.utils.data.DataLoader): 训练数据的数据加载器。
        val_dataloader (torch.utils.data.DataLoader): 验证数据的数据加载器。
        num_epochs (int): 训练的总轮数（epoch数量）。

    Returns:
        pandas.DataFrame: 包含每个epoch的训练和验证损失以及准确率的DataFrame。
    """
    # 设定训练所使用的设备：如果系统支持CUDA（即有NVIDIA GPU），则使用GPU，否则使用CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}") # 打印当前使用的设备。

    # 使用Adam优化器。Adam是一种自适应学习率的优化算法，通常在深度学习中表现良好。
    # model.parameters() 返回模型中所有需要训练的参数。
    # lr (learning rate) 设置为0.001。                      
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 损失函数采用交叉熵损失（CrossEntropyLoss）。
    # 交叉熵损失适用于多分类问题，它内部包含了Softmax激活函数和负对数似然损失。
    criterion = nn.CrossEntropyLoss()

    # 将模型移动到指定的设备上（GPU或CPU），以便进行计算。
    model = model.to(device)

    # 复制当前模型的参数（state_dict）。
    # 这个副本将用于保存验证集上表现最佳的模型状态。
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化用于记录训练过程中的各项指标。
    best_acc = 0.0  # 记录迄今为止在验证集上达到的最高准确率。
    train_loss_all = []  # 存储每个epoch的训练集平均损失。
    val_loss_all = []  # 存储每个epoch的验证集平均损失。
    train_acc_all = []  # 存储每个epoch的训练集平均准确率。
    val_acc_all = []  # 存储每个epoch的验证集平均准确率。
    since = time.time()  # 记录训练开始的精确时间，用于计算总耗时。

    # 开始训练循环，迭代num_epochs次。
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")  # 打印当前epoch的进度。
        print("-" * 10)  # 打印分隔线，提高输出的可读性。

        # 初始化每个epoch的统计变量。
        train_loss = 0.0  # 当前epoch训练集累计损失。
        train_corrects = 0  # 当前epoch训练集正确预测的样本数量。
        val_loss = 0.0  # 当前epoch验证集累计损失。
        val_corrects = 0  # 当前epoch验证集正确预测的样本数量。
        train_num = 0  # 当前epoch训练集处理的样本总数。
        val_num = 0  # 当前epoch验证集处理的样本总数。

        # --- 训练阶段 ---
        # 遍历训练数据加载器中的每一个批次（mini-batch）。
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征数据（图像）移动到指定的设备上。
            b_x = b_x.to(device)
            # 将标签数据（真实类别）移动到指定的设备上。
            b_y = b_y.to(device)

            # 设置模型为训练模式。
            # 这会启用模型中如Dropout层和Batch Normalization层在训练时的特定行为。
            model.train()

            # 前向传播过程：将一个批次的输入数据b_x送入模型，得到模型的预测输出。
            output = model(b_x)
            # 从模型的输出中，找到每一行（即每个样本）中最大值对应的列索引。
            # 这个索引代表了模型预测的类别。
            pre_lab = torch.argmax(output, dim=1)
            # 计算当前批次的损失函数值。
            loss = criterion(output, b_y)

            # 将优化器中所有参数的梯度清零。
            # PyTorch默认会累积梯度，因此在每次反向传播前需要清零，以避免梯度累积。
            optimizer.zero_grad()
            # 反向传播：根据损失函数计算模型所有可学习参数的梯度。
            loss.backward()
            # 根据网络反向传播计算出的梯度信息，更新模型的参数。
            optimizer.step()

            # 累加当前批次的损失到总训练损失中。
            # loss.item() 获取标量损失值，b_x.size(0) 是当前批次的样本数量。
            train_loss += loss.item() * b_x.size(0)
            # 统计当前批次中预测正确的样本数量，并累加到总的正确数量中。
            # pre_lab == b_y.data 比较预测标签和真实标签，得到布尔张量，torch.sum计算True的数量。
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 累加当前批次的样本数量到总的训练样本数量中。
            train_num += b_x.size(0)

        # --- 验证阶段 ---
        # 使用torch.no_grad()上下文管理器。
        # 在此块内，PyTorch将不会跟踪梯度，这可以节省内存并加速计算，因为验证阶段不需要反向传播。
        with torch.no_grad():
            # 遍历验证数据加载器中的每一个批次。
            for step, (b_x, b_y) in enumerate(val_dataloader):
                # 将特征数据（图像）移动到指定的设备上。
                b_x = b_x.to(device)
                # 将标签数据（真实类别）移动到指定的设备上。
                b_y = b_y.to(device)

                # 设置模型为评估模式。
                # 这会禁用模型中如Dropout层和Batch Normalization层在训练时的特定行为，使其在评估时表现稳定。
                model.eval()
                
                # 前向传播过程：将一个批次的输入数据b_x送入模型，得到模型的预测输出。
                output = model(b_x)
                # 从模型的输出中，找到每一行中最大值对应的列索引，作为预测类别。
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
        train_loss_all.append(train_loss / train_num)  # 计算并保存训练集的平均损失。
        train_acc_all.append(train_corrects.double().item() / train_num)  # 计算并保存训练集的平均准确率。
        
        val_loss_all.append(val_loss / val_num)  # 计算并保存验证集的平均损失。
        val_acc_all.append(val_corrects.double().item() / val_num)  # 计算并保存验证集的平均准确率。

        # 打印当前epoch的训练和验证结果。
        print(f"Epoch {epoch} train loss:{train_loss_all[-1]:.4f} train acc: {train_acc_all[-1]:.4f}")
        print(f"Epoch {epoch} val loss:{val_loss_all[-1]:.4f} val acc: {val_acc_all[-1]:.4f}")

        # 检查当前epoch的验证准确率是否是迄今为止最高的。
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]  # 更新最高准确度。
            best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝当前模型的参数，作为最佳模型参数。

        # 计算并打印当前epoch为止的训练和验证总耗时。
        time_use = time.time() - since
        print(f"训练和验证耗费的时间{int(time_use // 60)}m{int(time_use % 60)}s")

    # 训练结束后，将验证集上表现最佳的模型参数加载回模型中。
    model.load_state_dict(best_model_wts)
    # 将最佳模型的参数保存到指定路径的文件中。
    # 注意：'C:/Users/86159/Desktop/AlexNet/best_model.pth' 是一个硬编码的绝对路径。
    # 在实际应用中，建议使用相对路径或通过配置参数指定保存位置，以提高代码的可移植性。
    torch.save(best_model_wts, "C:/Users/86159/Desktop/AlexNet/best_model.pth")

    # 将训练过程中收集到的各项指标整理成一个Pandas DataFrame。
    # 方便后续的数据分析和可视化。
    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),  # epoch编号。
        "train_loss_all": train_loss_all,  # 训练集损失列表。
        "val_loss_all": val_loss_all,  # 验证集损失列表。
        "train_acc_all": train_acc_all,  # 训练集准确率列表。
        "val_acc_all": val_acc_all,  # 验证集准确率列表。
    })

    return train_process  # 返回包含训练过程数据的DataFrame。


# 4. 绘图函数
# ------------------------------------------------------------------------------
def matplot_acc_loss(train_process):
    """
    绘制训练和验证过程中的损失函数和准确率曲线图。

    Args:
        train_process (pandas.DataFrame): 包含训练过程数据的DataFrame。
    """
    # 创建一个图形，并设置其大小为12x4英寸。
    plt.figure(figsize=(12, 4))

    # 第一个子图：绘制损失曲线。
    plt.subplot(1, 2, 1)  # 将图形分为1行2列，这是第一个子图。
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")  # 绘制训练损失曲线，红色圆点实线。
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")  # 绘制验证损失曲线，蓝色方块实线。
    plt.legend()  # 显示图例，解释每条曲线的含义。
    plt.xlabel("Epoch")  # 设置X轴标签为"Epoch"。
    plt.ylabel("Loss")  # 设置Y轴标签为"Loss"。
    plt.title("Training and Validation Loss") # 设置子图标题

    # 第二个子图：绘制准确率曲线。
    plt.subplot(1, 2, 2)  # 这是第二个子图。
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")  # 绘制训练准确率曲线，红色圆点实线。
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")  # 绘制验证准确率曲线，蓝色方块实线。
    plt.xlabel("Epoch")  # 设置X轴标签为"Epoch"。
    plt.ylabel("Accuracy")  # 设置Y轴标签为"Accuracy"。
    plt.legend()  # 显示图例。
    plt.title("Training and Validation Accuracy") # 设置子图标题

    plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域，防止标签重叠。
    plt.show()  # 显示绘制的图表。


# 5. 主执行块
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    当脚本直接运行时，此代码块将被执行。
    它负责模型的实例化、数据加载、模型训练和结果可视化。
    """
    # 实例化AlexNet模型。
    model = AlexNet()
    print("AlexNet model instantiated.")

    # 调用数据处理函数，获取训练和验证数据加载器。
    train_dataloader, val_dataloader = train_val_data_process()
    print("FashionMNIST data loaded and split into training/validation sets.")

    # 调用模型训练函数，传入模型、数据加载器和训练轮数（这里设置为20个epoch）。
    # train_process DataFrame将包含整个训练过程的统计数据。
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs=20)
    print("Model training complete.")

    # 调用绘图函数，可视化训练过程中的损失和准确率变化。
    matplot_acc_loss(train_process)
    print("Training process plots displayed.")