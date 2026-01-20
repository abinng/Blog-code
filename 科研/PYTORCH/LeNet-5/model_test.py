# ==============================================================================
# Python Script: LeNet Model Testing on FashionMNIST
# Author: [Original Author/Your Name]
# Date: [Current Date]
# Description:
# This script is designed to evaluate a pre-trained LeNet model's performance
# on the FashionMNIST test dataset. It handles loading the test data,
# loading the trained model weights, and then performing an inference pass
# to calculate and report the model's accuracy.
# ==============================================================================

# 1. Import Necessary Libraries
# ------------------------------------------------------------------------------
import torch  # 导入PyTorch库，它是构建和运行深度学习模型的核心。
import torch.utils.data as Data  # 导入torch.utils.data模块，用于数据加载器（DataLoader）。
from torchvision import transforms  # 导入transforms模块，用于图像预处理操作。
from torchvision.datasets import FashionMNIST  # 导入FashionMNIST数据集类，用于加载数据集。
from model import LeNet  # 从名为'model.py'的文件中导入自定义的LeNet模型类。


# 2. Function to Process Test Data
# ------------------------------------------------------------------------------
def test_data_process():
    """
    加载并预处理FashionMNIST测试数据集，并创建一个数据加载器。

    Returns:
        torch.utils.data.DataLoader: 配置好的测试数据加载器。
    """
    # 加载FashionMNIST测试数据集。
    test_data = FashionMNIST(
        root='./data',  # 指定数据集的存储路径。
        train=False,  # 表示加载测试集（而不是训练集）。
        transform=transforms.Compose([  # 定义图像预处理操作的组合。
            transforms.Resize(size=28),  # 将图像尺寸调整为28x28像素，与LeNet模型期望的输入尺寸一致。
            transforms.ToTensor()  # 将PIL图像或NumPy数组转换为PyTorch张量，并归一化到[0.0, 1.0]范围。
        ]),
        download=True  # 如果数据集不存在，则自动下载。
    )

    # 创建测试数据加载器。
    test_dataloader = Data.DataLoader(
        dataset=test_data,  # 指定要加载的数据集。
        batch_size=1,  # 设置每个批次（batch）中包含的样本数量为1，以便逐个样本进行评估。
        shuffle=True,  # 在每个epoch开始时打乱数据，尽管对于测试集通常不是必需的，但无害。
        num_workers=0  # 设置用于数据加载的子进程数，0表示在主进程中加载数据。
    )
    return test_dataloader  # 返回创建的数据加载器。


# 3. Function to Test the Model
# ------------------------------------------------------------------------------
def test_model_process(model, test_dataloader):
    """
    在测试数据集上评估模型的性能。

    Args:
        model (nn.Module): 待评估的PyTorch模型实例。
        test_dataloader (torch.utils.data.DataLoader): 测试数据加载器。
    """
    # 设定测试所用的设备：如果有CUDA GPU则用GPU，否则用CPU。
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Using device for testing: {device}") # 打印当前使用的设备。

    # 将模型移动到指定的设备上（GPU或CPU）。
    model = model.to(device)

    # 初始化用于记录测试结果的参数。
    test_corrects = 0.0  # 记录正确预测的样本数量。使用浮点数以便后续计算准确率。
    test_num = 0  # 记录总的测试样本数量。

    # 禁用梯度计算。在测试阶段，我们不需要计算梯度，这可以节省内存并加速计算。
    with torch.no_grad():
        # 遍历测试数据加载器中的每个批次。
        for test_data_x, test_data_y in test_dataloader:
            # 将特征数据（图像）移动到测试设备上。
            test_data_x = test_data_x.to(device)
            # 将标签数据（真实类别）移动到测试设备上。
            test_data_y = test_data_y.to(device)

            # 设置模型为评估模式。
            # 这会关闭如Dropout和Batch Normalization等在训练和评估阶段行为不同的层。
            model.eval()

            # 执行模型的前向传播过程，输入为测试数据，输出为对每个样本的预测值（logits）。
            output = model(test_data_x)

            # 从模型的输出中找到每个样本预测概率最高的类别索引。
            # torch.argmax(output, dim=1) 返回每一行（即每个样本）中最大值对应的列索引。
            pre_lab = torch.argmax(output, dim=1)

            # 统计当前批次中预测正确的样本数量，并累加到总的正确数量中。
            # (pre_lab == test_data_y) 会生成一个布尔张量，torch.sum将其中的True计数。
            test_corrects += torch.sum(pre_lab == test_data_y.data)

            # 累加当前批次的样本数量到总的测试样本数量中。
            test_num += test_data_x.size(0)

    # 计算测试准确率。
    # .double() 确保计算在浮点数精度下进行。
    # .item() 从PyTorch张量中提取Python数字。
    test_acc = test_corrects.double().item() / test_num
    print(f"测试的准确率为: {test_acc:.4f}")  # 打印格式化后的测试准确率。


# 4. Main Execution Block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    当脚本直接运行时，执行此代码块。
    负责模型的加载、测试数据的准备和模型评估的启动。
    """
    # 实例化LeNet模型。
    model = LeNet()
    # 加载预训练的模型权重。
    # load_state_dict() 就是把训练好的模型参数"装载"到新创建的模型中，让模型具备训练时学到的知识。
    # 'best_model.pth' 应该是之前训练过程中保存的最佳模型状态字典文件。
    model.load_state_dict(torch.load('best_model.pth'))

    # 调用函数加载并获取测试数据加载器。
    test_dataloader = test_data_process()

    # 调用函数对加载的模型进行测试。
    test_model_process(model, test_dataloader)
    
    
    """
    # 查看预测后每一个预测值和真实值的对应
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # 列表，用于映射最后的结果
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        # b_x 是图像特征 b_y 是标签值
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 切换为评估模式
            model.eval()
            # 前向传播进行预测
            output = model(b_x)
            # 得到预测后最大的一个概率的下标，作为最后的预测值
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            # 真实值
            label = b_y.item()
            # 输出，用classes进行映射
            print("预测值：", classes[result], "------", "真实值", classes[label])
    """