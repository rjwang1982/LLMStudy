"""
Python 基础 - Matplotlib 数据可视化

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 掌握基本绑图方法
  - 可视化激活函数（为神经网络做准备）
  - 理解损失函数的变化趋势
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（macOS）
plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 可视化常见激活函数
# ============================================================
print("正在绘制激活函数图...")

x = np.linspace(-5, 5, 200)


# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)


fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("常见激活函数", fontsize=16)

# Sigmoid
axes[0, 0].plot(x, sigmoid(x), "b-", linewidth=2)
axes[0, 0].set_title("Sigmoid")
axes[0, 0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
axes[0, 0].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
axes[0, 0].set_ylim(-0.1, 1.1)
axes[0, 0].grid(True, alpha=0.3)

# Tanh
axes[0, 1].plot(x, tanh(x), "r-", linewidth=2)
axes[0, 1].set_title("Tanh")
axes[0, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[0, 1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
axes[0, 1].grid(True, alpha=0.3)

# ReLU
axes[1, 0].plot(x, relu(x), "g-", linewidth=2)
axes[1, 0].set_title("ReLU")
axes[1, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[1, 0].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
axes[1, 0].grid(True, alpha=0.3)

# Leaky ReLU
axes[1, 1].plot(x, leaky_relu(x), "m-", linewidth=2)
axes[1, 1].set_title("Leaky ReLU")
axes[1, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[1, 1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("01_python_basics/activation_functions.png", dpi=150)
plt.close()
print("✅ 激活函数图已保存: activation_functions.png")

# ============================================================
# 2. 模拟训练过程的损失曲线
# ============================================================
print("正在绘制损失曲线...")

epochs = np.arange(1, 51)
# 模拟一个逐渐下降的损失值（加一点噪声更真实）
np.random.seed(42)
train_loss = 2.5 * np.exp(-0.08 * epochs) + np.random.randn(50) * 0.05
val_loss = 2.5 * np.exp(-0.06 * epochs) + 0.1 + np.random.randn(50) * 0.08

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_loss, "b-", label="训练损失", linewidth=2)
ax.plot(epochs, val_loss, "r--", label="验证损失", linewidth=2)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("模型训练过程 - 损失曲线", fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("01_python_basics/training_loss.png", dpi=150)
plt.close()
print("✅ 损失曲线图已保存: training_loss.png")

# ============================================================
# 3. 可视化数据分布
# ============================================================
print("正在绘制数据分布图...")

np.random.seed(42)
# 模拟两类数据（二分类问题）
class_a = np.random.randn(100, 2) + np.array([2, 2])
class_b = np.random.randn(100, 2) + np.array([-2, -2])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(class_a[:, 0], class_a[:, 1], c="blue", label="类别 A", alpha=0.6)
ax.scatter(class_b[:, 0], class_b[:, 1], c="red", label="类别 B", alpha=0.6)

# 画一条简单的决策边界
x_line = np.linspace(-5, 5, 100)
ax.plot(x_line, -x_line, "k--", label="决策边界", linewidth=2)

ax.set_xlabel("特征 1", fontsize=12)
ax.set_ylabel("特征 2", fontsize=12)
ax.set_title("二分类问题可视化", fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("01_python_basics/classification.png", dpi=150)
plt.close()
print("✅ 分类图已保存: classification.png")

print("\n✅ Matplotlib 可视化练习完成！请查看生成的 PNG 图片。")
