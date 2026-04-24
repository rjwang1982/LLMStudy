"""
神经网络 - 纯 NumPy 实现手写数字识别

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 理解前向传播的完整过程
  - 理解反向传播和梯度下降
  - 不依赖框架，从零理解神经网络

网络结构:
  输入层(784) → 隐藏层(128) → 输出层(10)
"""

import numpy as np

# ============================================================
# 1. 准备数据（使用简化的模拟数据）
# ============================================================
print("=" * 50)
print("纯 NumPy 实现神经网络")
print("=" * 50)


def generate_simple_data(n_samples: int = 1000):
    """生成简化的模拟数据（模拟 MNIST 的结构）

    为了不依赖外部数据集，我们生成一些简单的模式数据。
    每个"图片"是 16 个像素（4x4），有 3 个类别。
    """
    np.random.seed(42)
    X = []
    y = []

    for _ in range(n_samples):
        label = np.random.randint(0, 3)

        if label == 0:
            # 类别0: 左半边亮
            img = np.zeros(16)
            img[:8] = np.random.uniform(0.7, 1.0, 8)
            img[8:] = np.random.uniform(0.0, 0.3, 8)
        elif label == 1:
            # 类别1: 右半边亮
            img = np.zeros(16)
            img[:8] = np.random.uniform(0.0, 0.3, 8)
            img[8:] = np.random.uniform(0.7, 1.0, 8)
        else:
            # 类别2: 全部中等亮度
            img = np.random.uniform(0.4, 0.6, 16)

        X.append(img)
        y.append(label)

    return np.array(X), np.array(y)


# ============================================================
# 2. 激活函数和工具函数
# ============================================================

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)


def relu_derivative(x):
    """ReLU 的导数"""
    return (x > 0).astype(float)


def softmax(x):
    """Softmax 函数 - 将输出转换为概率分布"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 减最大值防溢出
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def one_hot(y, n_classes):
    """将标签转换为 one-hot 编码"""
    result = np.zeros((len(y), n_classes))
    result[np.arange(len(y)), y] = 1
    return result


def cross_entropy_loss(y_pred, y_true):
    """交叉熵损失函数"""
    n = len(y_true)
    # 裁剪防止 log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.sum(y_true * np.log(y_pred)) / n
    return loss


# ============================================================
# 3. 神经网络类
# ============================================================

class SimpleNeuralNetwork:
    """简单的两层神经网络

    结构: 输入 → 隐藏层(ReLU) → 输出层(Softmax)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """初始化网络权重

        使用 He 初始化（适合 ReLU 激活函数）
        """
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """前向传播

        X → (W1, b1) → ReLU → (W2, b2) → Softmax → 输出
        """
        # 隐藏层
        self.z1 = X @ self.W1 + self.b1      # 线性变换
        self.a1 = relu(self.z1)                # 激活函数

        # 输出层
        self.z2 = self.a1 @ self.W2 + self.b2  # 线性变换
        self.a2 = softmax(self.z2)              # Softmax 得到概率

        return self.a2

    def backward(self, X, y_true, learning_rate=0.01):
        """反向传播 - 计算梯度并更新权重

        这是神经网络学习的核心！
        通过链式法则，从输出层往回计算每个权重的梯度。
        """
        n = len(X)

        # 输出层的梯度
        dz2 = self.a2 - y_true  # softmax + cross_entropy 的梯度简化
        dW2 = self.a1.T @ dz2 / n
        db2 = np.sum(dz2, axis=0, keepdims=True) / n

        # 隐藏层的梯度（链式法则）
        dz1 = (dz2 @ self.W2.T) * relu_derivative(self.z1)
        dW1 = X.T @ dz1 / n
        db1 = np.sum(dz1, axis=0, keepdims=True) / n

        # 梯度下降更新权重
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def predict(self, X):
        """预测类别"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


# ============================================================
# 4. 训练网络
# ============================================================
print("\n生成训练数据...")
X, y = generate_simple_data(n_samples=1000)

# 划分训练集和测试集
split = 800
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"训练集: {X_train.shape[0]} 个样本")
print(f"测试集: {X_test.shape[0]} 个样本")
print(f"输入维度: {X_train.shape[1]}")
print(f"类别数: {len(np.unique(y))}")

# 创建网络
nn = SimpleNeuralNetwork(
    input_size=16,
    hidden_size=32,
    output_size=3,
)

# one-hot 编码
y_train_oh = one_hot(y_train, 3)

# 训练
print("\n开始训练...")
print("-" * 40)

epochs = 200
learning_rate = 0.5

for epoch in range(epochs):
    # 前向传播
    output = nn.forward(X_train)

    # 计算损失
    loss = cross_entropy_loss(output, y_train_oh)

    # 反向传播
    nn.backward(X_train, y_train_oh, learning_rate)

    # 每 20 轮打印一次
    if (epoch + 1) % 20 == 0:
        train_pred = nn.predict(X_train)
        train_acc = np.mean(train_pred == y_train)
        print(f"Epoch {epoch + 1:3d} | Loss: {loss:.4f} | 训练准确率: {train_acc:.2%}")

# ============================================================
# 5. 测试
# ============================================================
print("\n" + "=" * 50)
print("测试结果")
print("=" * 50)

test_pred = nn.predict(X_test)
test_acc = np.mean(test_pred == y_test)
print(f"测试准确率: {test_acc:.2%}")

# 显示一些预测结果
print("\n前 10 个测试样本:")
for i in range(10):
    status = "✅" if test_pred[i] == y_test[i] else "❌"
    print(f"  样本 {i+1}: 预测={test_pred[i]}, 实际={y_test[i]} {status}")

print("\n💡 关键概念回顾:")
print("  1. 前向传播: 数据从输入层流向输出层")
print("  2. 损失函数: 衡量预测和真实值的差距")
print("  3. 反向传播: 计算每个权重对损失的贡献（梯度）")
print("  4. 梯度下降: 沿梯度反方向更新权重，减小损失")

print("\n✅ 纯 NumPy 神经网络练习完成！")
print("   下一步: 用 PyTorch 框架实现同样的功能，体验框架的便利。")
