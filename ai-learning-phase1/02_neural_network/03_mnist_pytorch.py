"""
神经网络 - PyTorch 实现 MNIST 手写数字识别

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 学会使用 PyTorch 构建神经网络
  - 理解 Dataset, DataLoader, Model, Optimizer 的关系
  - 完成一个真实的图像分类任务

对比上一个练习:
  - 上一个: 纯 NumPy，手动实现反向传播
  - 这一个: PyTorch 自动计算梯度，代码更简洁
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================
# 1. 准备数据
# ============================================================
print("=" * 50)
print("PyTorch MNIST 手写数字识别")
print("=" * 50)

# 数据预处理：转为张量 + 归一化
transform = transforms.Compose([
    transforms.ToTensor(),           # 图片 → 张量，值域 [0, 1]
    transforms.Normalize((0.5,), (0.5,)),  # 归一化到 [-1, 1]
])

# 下载 MNIST 数据集
print("\n加载 MNIST 数据集...")
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

# DataLoader: 自动分批、打乱数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集: {len(train_dataset)} 张图片")
print(f"测试集: {len(test_dataset)} 张图片")
print(f"图片大小: 28x28 = 784 像素")
print(f"类别数: 10 (数字 0-9)")


# ============================================================
# 2. 定义模型
# ============================================================

class MNISTNet(nn.Module):
    """手写数字识别网络

    结构: 784 → 256 → 128 → 10

    对比纯 NumPy 版本:
      - nn.Linear 自动管理权重和偏置
      - nn.ReLU 就是我们之前手写的 relu 函数
      - 不需要手动实现反向传播！
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),   # 输入层 → 隐藏层1
            nn.ReLU(),             # 激活函数
            nn.Dropout(0.2),       # 随机丢弃 20% 神经元（防过拟合）
            nn.Linear(256, 128),   # 隐藏层1 → 隐藏层2
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),    # 隐藏层2 → 输出层（10个数字）
        )

    def forward(self, x):
        # 将 28x28 的图片展平为 784 的向量
        x = x.view(x.size(0), -1)
        return self.network(x)


# ============================================================
# 3. 训练
# ============================================================

# 选择设备（有 GPU 用 GPU，没有用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n使用设备: {device}")

# 创建模型、损失函数、优化器
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（分类任务标配）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 打印模型结构
print(f"\n模型结构:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()  # 训练模式（启用 Dropout）
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播（PyTorch 自动计算梯度！）
        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新权重

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += len(target)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()  # 评估模式（关闭 Dropout）
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度（节省内存）
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)

    return total_loss / len(loader), correct / total


# 开始训练
print("\n开始训练...")
print("-" * 60)

epochs = 5  # MNIST 比较简单，5 轮就够了
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device,
    )
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(
        f"Epoch {epoch}/{epochs} | "
        f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2%} | "
        f"测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.2%}"
    )

# ============================================================
# 4. 测试一些具体样本
# ============================================================
print("\n" + "=" * 50)
print("具体预测示例")
print("=" * 50)

model.eval()
test_data = test_dataset.data[:10].float().unsqueeze(1) / 255.0
test_data = transforms.Normalize((0.5,), (0.5,))(test_data).to(device)
test_labels = test_dataset.targets[:10]

with torch.no_grad():
    predictions = model(test_data).argmax(dim=1)

for i in range(10):
    pred = predictions[i].item()
    true = test_labels[i].item()
    status = "✅" if pred == true else "❌"
    print(f"  样本 {i+1}: 预测={pred}, 实际={true} {status}")

# ============================================================
# 5. 保存模型
# ============================================================
model_path = "02_neural_network/mnist_model.pth"
torch.save(model.state_dict(), model_path)
print(f"\n模型已保存到: {model_path}")

print("\n💡 PyTorch vs 纯 NumPy 对比:")
print("  1. 自动求导: 不需要手动实现反向传播")
print("  2. GPU 加速: 一行代码切换到 GPU")
print("  3. 丰富的层: nn.Linear, nn.Conv2d, nn.LSTM 等")
print("  4. 优化器: Adam, SGD 等开箱即用")
print("  5. 数据加载: DataLoader 自动分批和打乱")

print("\n✅ PyTorch MNIST 练习完成！")
