# ==============================================================================
# Python Script: AlexNet Model Testing on FashionMNIST
# Author: [Original Author/Your Name]
# Date: [Current Date]
# Description:
# This script defines the process for testing a pre-trained AlexNet model
# on the FashionMNIST test dataset. It includes functions for loading and
# preprocessing the test data, evaluating the model's overall accuracy,
# and demonstrating individual predictions for sample images.
# ==============================================================================

# 1. 导入必要的库
# ------------------------------------------------------------------------------
import torch  # 导入PyTorch库，用于构建和运行深度学习模型。
import torch.utils.data as Data  # 导入torch.utils.data模块，用于数据加载器（DataLoader）的创建和管理。
from torchvision import transforms  # 从torchvision库中导入transforms模块，用于图像预处理操作。
from torchvision.datasets import FashionMNIST  # 从torchvision库中导入FashionMNIST数据集类。
from model import AlexNet  # 从名为'model.py'的文件中导入自定义的AlexNet模型类。


# 2. 测试数据处理函数
# ------------------------------------------------------------------------------
def test_data_process():
    """
    加载并预处理FashionMNIST测试数据集，并为其创建数据加载器。

    Returns:
        torch.utils.data.DataLoader: FashionMNIST测试数据的数据加载器。
    """
    # 加载FashionMNIST测试数据集。
    test_data = FashionMNIST(
        root='./data',  # 指定数据集的存储路径。
        train=False,  # 指定加载的是测试集。
        transform=transforms.Compose([  # 定义一系列图像预处理操作。
            transforms.Resize(size=227),  # 将图像尺寸调整为227x227像素，以匹配AlexNet的输入要求。
            transforms.ToTensor()  # 将PIL图像或NumPy数组转换为PyTorch张量，并归一化到[0.0, 1.0]。
        ]),
        download=True  # 如果数据集在指定路径不存在，则自动从网上下载。
    )

    # 创建测试数据加载器。
    test_dataloader = Data.DataLoader(
        dataset=test_data,  # 指定要加载的测试数据集。
        batch_size=1,  # 每个批次（batch）中包含1个样本，便于逐个样本进行预测和观察。
        shuffle=True,  # 在每个epoch开始时打乱数据，对测试集通常不是必需的，但无害。
        num_workers=0  # 不使用子进程进行数据加载，因为batch_size较小，且通常在测试时不需要并行加载。
    )
    return test_dataloader  # 返回创建好的测试数据加载器。


# 3. 模型测试函数
# ------------------------------------------------------------------------------
def test_model_process(model, test_dataloader):
    """
    评估给定模型在测试数据集上的准确率。

    Args:
        model (nn.Module): 待评估的PyTorch模型实例（这里是AlexNet）。
        test_dataloader (torch.utils.data.DataLoader): 测试数据的数据加载器。
    """
    # 设定测试所使用的设备：如果系统支持CUDA，则使用GPU，否则使用CPU。
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Using device for testing: {device}") # 打印当前使用的设备。

    # 将模型移动到指定的设备上（GPU或CPU），以便进行计算。
    model = model.to(device)

    # 初始化用于记录测试结果的参数。
    test_corrects = 0.0  # 记录测试集中正确预测的样本总数。
    test_num = 0  # 记录测试集中处理的样本总数。

    # 使用torch.no_grad()上下文管理器。
    # 在此块内，PyTorch将不会跟踪梯度，这可以节省内存并加速计算，因为测试阶段不需要反向传播。
    with torch.no_grad():
        # 遍历测试数据加载器中的每一个批次（mini-batch）。
        for test_data_x, test_data_y in test_dataloader:
            # 将特征数据（图像）移动到指定的设备上。
            test_data_x = test_data_x.to(device)
            # 将标签数据（真实类别）移动到指定的设备上。
            test_data_y = test_data_y.to(device)

            # 设置模型为评估模式。
            # 这会禁用模型中如Dropout层和Batch Normalization层在训练时的特定行为，使其在评估时表现稳定。
            model.eval()

            # 前向传播过程：将一个批次的输入数据test_data_x送入模型，得到模型的预测输出。
            output = model(test_data_x)
            # 从模型的输出中，找到每一行（即每个样本）中最大值对应的列索引。
            # 这个索引代表了模型预测的类别。
            pre_lab = torch.argmax(output, dim=1)

            # 统计当前批次中预测正确的样本数量，并累加到总的正确数量中。
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 累加当前批次的样本数量到总的测试样本数量中。
            test_num += test_data_x.size(0)

    # 计算并打印测试集的整体准确率。
    test_acc = test_corrects.double().item() / test_num
    print(f"测试的准确率为: {test_acc:.4f}")


# 4. 主执行块
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    当脚本直接运行时，此代码块将被执行。
    它负责加载预训练模型、准备测试数据、评估模型并进行单样本预测展示。
    """
    # 实例化AlexNet模型。
    model = AlexNet()
    print("AlexNet model instantiated.")

    # 加载预训练模型的参数。
    # 假设'best_model.pth'文件存在于当前脚本的同一目录下。
    # 如果文件路径不同，需要修改为正确的路径。
    try:
        model.load_state_dict(torch.load('best_model.pth'))
        print("Pre-trained model weights loaded from 'best_model.pth'.")
    except FileNotFoundError:
        print("Error: 'best_model.pth' not found. Please ensure the pre-trained model file is in the correct directory.")
        exit() # 如果模型文件不存在，则退出程序。

    # 调用测试数据处理函数，获取测试数据加载器。
    test_dataloader = test_data_process()
    print("FashionMNIST test data loaded.")

    # 调用模型测试函数，评估模型在整个测试集上的表现。
    print("\n--- Evaluating model on the entire test set ---")
    test_model_process(model, test_dataloader)

    # 设定测试所使用的设备，与上面保持一致。
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device) # 再次确保模型在正确设备上，尽管之前已经设置。

    # 定义FashionMNIST数据集的类别名称，用于将预测结果转换为可读的标签。
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print("\n--- Demonstrating individual predictions ---")
    # 使用torch.no_grad()上下文管理器进行单样本预测，不计算梯度。
    with torch.no_grad():
        # 遍历测试数据加载器中的前几个批次（因为batch_size=1，所以是前几个样本）。
        # 这里只取前5个样本进行展示。
        for i, (b_x, b_y) in enumerate(test_dataloader):
            if i >= 5: # 仅展示前5个样本的预测结果。
                break

            # 将特征和标签移动到指定的设备上。
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为评估模式。
            model.eval()
            # 前向传播，获取模型输出。
            output = model(b_x)
            # 获取模型预测的类别（概率最高的类别索引）。
            pre_lab = torch.argmax(output, dim=1)

            # 从张量中提取标量值。
            predicted_class_idx = pre_lab.item()
            true_class_idx = b_y.item()

            # 打印预测值和真实值。
            print(f"样本 {i+1}: 预测值: {classes[predicted_class_idx]} ------ 真实值: {classes[true_class_idx]}")