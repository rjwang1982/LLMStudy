"""
神经网络 - 感知机：最简单的神经元

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 理解神经元的工作原理：加权求和 + 激活函数
  - 理解权重更新的过程（学习）
  - 用感知机解决简单的逻辑门问题
"""

import numpy as np


class Perceptron:
    """感知机 - 最简单的神经网络单元

    工作原理:
        输入 x1, x2, ... → 加权求和 → 激活函数 → 输出

        output = activation(w1*x1 + w2*x2 + ... + bias)
    """

    def __init__(self, n_inputs: int, learning_rate: float = 0.1):
        """初始化感知机

        Args:
            n_inputs: 输入特征数量
            learning_rate: 学习率（每次更新的步长）
        """
        # 随机初始化权重（很小的随机数）
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, x: np.ndarray) -> int:
        """前向传播：计算输出

        Args:
            x: 输入向量

        Returns:
            0 或 1
        """
        # 加权求和
        total = np.dot(self.weights, x) + self.bias
        # 阶跃激活函数：大于0输出1，否则输出0
        return 1 if total > 0 else 0

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> list[float]:
        """训练感知机

        Args:
            X: 训练数据，shape=(n_samples, n_features)
            y: 标签，shape=(n_samples,)
            epochs: 训练轮数

        Returns:
            每轮的错误率列表
        """
        errors_per_epoch = []

        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = yi - prediction

                if error != 0:
                    errors += 1
                    # 核心：权重更新规则
                    # 如果预测错了，调整权重让下次更准
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error

            error_rate = errors / len(y)
            errors_per_epoch.append(error_rate)

            if errors == 0:
                print(f"  第 {epoch + 1} 轮: 完美分类！训练完成 🎉")
                break
            elif (epoch + 1) % 10 == 0:
                print(f"  第 {epoch + 1} 轮: 错误率 = {error_rate:.2%}")

        return errors_per_epoch


# ============================================================
# 实验1: 学习 AND 逻辑门
# ============================================================
print("=" * 50)
print("实验1: 感知机学习 AND 门")
print("=" * 50)
print("AND 门真值表: (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1\n")

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron_and = Perceptron(n_inputs=2, learning_rate=0.1)
perceptron_and.train(X_and, y_and, epochs=100)

print("\n验证结果:")
for x, y_true in zip(X_and, y_and):
    y_pred = perceptron_and.predict(x)
    status = "✅" if y_pred == y_true else "❌"
    print(f"  输入 {x} → 预测 {y_pred} (期望 {y_true}) {status}")

print(f"\n学到的权重: {perceptron_and.weights}")
print(f"学到的偏置: {perceptron_and.bias:.3f}")

# ============================================================
# 实验2: 学习 OR 逻辑门
# ============================================================
print("\n" + "=" * 50)
print("实验2: 感知机学习 OR 门")
print("=" * 50)
print("OR 门真值表: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→1\n")

y_or = np.array([0, 1, 1, 1])

perceptron_or = Perceptron(n_inputs=2, learning_rate=0.1)
perceptron_or.train(X_and, y_or, epochs=100)

print("\n验证结果:")
for x, y_true in zip(X_and, y_or):
    y_pred = perceptron_or.predict(x)
    status = "✅" if y_pred == y_true else "❌"
    print(f"  输入 {x} → 预测 {y_pred} (期望 {y_true}) {status}")

# ============================================================
# 实验3: XOR 问题 - 感知机的局限性
# ============================================================
print("\n" + "=" * 50)
print("实验3: 感知机学习 XOR 门（会失败！）")
print("=" * 50)
print("XOR 门真值表: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0")
print("⚠️  XOR 不是线性可分的，单层感知机无法学会！\n")

y_xor = np.array([0, 1, 1, 0])

perceptron_xor = Perceptron(n_inputs=2, learning_rate=0.1)
perceptron_xor.train(X_and, y_xor, epochs=100)

print("\n验证结果:")
correct = 0
for x, y_true in zip(X_and, y_xor):
    y_pred = perceptron_xor.predict(x)
    status = "✅" if y_pred == y_true else "❌"
    if y_pred == y_true:
        correct += 1
    print(f"  输入 {x} → 预测 {y_pred} (期望 {y_true}) {status}")

print(f"\n准确率: {correct}/{len(y_xor)}")
print("💡 这就是为什么我们需要多层神经网络（深度学习）！")
print("   下一个练习我们将用多层网络解决这个问题。")

print("\n✅ 感知机练习完成！")
