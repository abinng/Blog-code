# ==============================================================================
# Python Script: GoogLeNet Model Testing on FashionMNIST
# Author: [Original Author/Your Name]
# Date: [Current Date]
# Description:
# This script is designed to evaluate a pre-trained GoogLeNet model on the
# FashionMNIST test dataset. It includes functions for loading and preprocessing
# the test data, running the model in evaluation mode to calculate overall
# accuracy, and demonstrating individual sample predictions. The script
# assumes a 'best_model.pth' file (containing the trained model's state_dict)
# is available in the same directory.
# ==============================================================================

# 1. Import Necessary Libraries
# ------------------------------------------------------------------------------
import torch  # 导入PyTorch库，它是构建深度学习模型的核心。
import torch.utils.data as Data  # 从torch.utils.data导入Data模块，用于处理数据集和数据加载器。
from torchvision import transforms  # 从torchvision导入transforms模块，用于数据预处理。
from torchvision.datasets import FashionMNIST  # 从torchvision导入FashionMNIST数据集。
from model import GoogLeNet, Inception  # 从自定义的model.py文件中导入GoogLeNet模型类和其核心Inception模块。


# 2. Data Processing Function for Test Set
# ------------------------------------------------------------------------------
def test_data_process():
    """
    处理测试数据集：加载FashionMNIST测试集，进行预处理，并创建数据加载器。

    Returns:
        torch.utils.data.DataLoader: 测试数据加载器。
    """
    # 加载FashionMNIST测试数据集。
    test_data = FashionMNIST(root='./data',  # 数据集存储的根目录。
                              train=False,  # 指定加载测试集。
                              transform=transforms.Compose([  # 定义数据转换操作序列。
                                  transforms.Resize(size=224),  # 将图片尺寸调整为224x224，以适应GoogLeNet的输入要求。
                                  transforms.ToTensor()  # 将PIL Image或numpy.ndarray转换为Tensor，并归一化到[0.0, 1.0]。
                              ]),
                              download=True)  # 如果数据集不存在，则自动下载。

    # 创建测试数据加载器。
    test_dataloader = Data.DataLoader(dataset=test_data,  # 指定数据集。
                                       batch_size=1,  # 每个批次的样本数量设置为1，方便逐个样本进行预测和展示。
                                       shuffle=True,  # 每个epoch开始时打乱数据，确保随机抽样。
                                       num_workers=0)  # 用于数据加载的子进程数量，0表示在主进程中加载。
    return test_dataloader  # 返回测试数据加载器。


# 3. Model Testing Function
# ------------------------------------------------------------------------------
def test_model_process(model, test_dataloader):
    """
    在测试集上评估模型的性能，计算并打印整体准确率。

    Args:
        model (nn.Module): 待测试的神经网络模型。
        test_dataloader (torch.utils.data.DataLoader): 测试数据加载器。
    """
    # 设定测试所用到的设备，优先使用GPU（CUDA），否则使用CPU。
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Using device for testing: {device}") # 打印当前使用的设备。

    # 将模型移动到指定的测试设备上（GPU或CPU）。
    model = model.to(device)

    # 初始化参数。
    test_corrects = 0.0  # 记录测试集上正确预测的数量。
    test_num = 0  # 记录测试集上总样本的数量。

    # 在推理阶段，无需计算梯度，从而节省内存并加快运行速度。
    with torch.no_grad():
        # 遍历测试数据加载器中的每一个mini-batch（这里是单个样本）。
        for test_data_x, test_data_y in test_dataloader:
            # 将特征数据移动到测试设备上。
            test_data_x = test_data_x.to(device)
            # 将标签数据移动到测试设备上。
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式。
            # 这会禁用Dropout层，并使用Batch Normalization层的运行统计量而不是批次统计量。
            model.eval()
            # 前向传播过程：输入测试数据，得到模型的预测输出。
            output = model(test_data_x)
            # 查找模型输出中每一行（每个样本）最大值对应的索引，即预测的类别。
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则正确预测的数量test_corrects累加。
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将当前batch的样本数量累加到总测试样本数。
            test_num += test_data_x.size(0)

    # 计算测试准确率。
    # .double().item() 将张量转换为Python浮点数。
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)  # 打印最终的测试准确率。


# 4. Main Execution Block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    脚本的入口点。当脚本直接运行时，此代码块将被执行。
    它负责加载模型、测试整体准确率和演示单个样本预测。
    """
    # 实例化GoogLeNet模型，并将Inception模块的类作为参数传入。
    model = GoogLeNet(Inception)
    # 加载预训练模型的参数。
    # 'best_model.pth' 是在训练阶段保存的最佳模型参数文件。
    model.load_state_dict(torch.load('best_model.pth'))
    print("Pre-trained GoogLeNet model loaded successfully.")

    # 获取测试数据加载器。
    test_dataloader = test_data_process()
    print("Test dataset loaded and preprocessed.")

    # 利用加载的模型进行整体测试，计算并打印测试集准确率。
    test_model_process(model, test_dataloader)
    print("\n--- Demonstrating individual predictions ---")

    # 重新设定测试所用到的设备，以防在test_model_process中模型被移动到其他设备。
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    # 再次将模型移动到指定的设备上。
    model = model.to(device)

    # 定义FashionMNIST的类别名称列表，用于将预测的数字标签转换为可读的字符串。
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 在推理阶段，无需计算梯度。
    with torch.no_grad():
        # 遍历测试数据加载器中的每个样本（由于batch_size=1）。
        # 这里仅用于演示，会遍历所有样本并打印预测结果。
        for i, (b_x, b_y) in enumerate(test_dataloader):
            # 将特征数据移动到设备上。
            b_x = b_x.to(device)
            # 将标签数据移动到设备上。
            b_y = b_y.to(device)

            # 设置模型为评估模式。
            model.eval()
            # 前向传播，获取模型输出。
            output = model(b_x)
            # 获取预测的类别索引。
            pre_lab = torch.argmax(output, dim=1)
            # 将预测结果（张量）转换为Python标量。
            result = pre_lab.item()
            # 将真实标签（张量）转换为Python标量。
            label = b_y.item()

            # 打印预测值和真实值。
            print(f"Sample {i+1}: Predicted: {classes[result]} | True: {classes[label]}")

            # 如果只想看几个示例，可以取消注释下面的break语句。
            # if i >= 4: # 例如，只显示前5个样本的预测。
            #     break