"""
Python 基础 - NumPy 入门

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 理解 NumPy 数组和 Python 列表的区别
  - 掌握数组创建、索引、运算
  - 理解向量化运算（AI 的核心基础）
"""

import numpy as np

# ============================================================
# 1. 为什么需要 NumPy？
# ============================================================
print("=" * 50)
print("1. NumPy vs Python 列表")
print("=" * 50)

# Python 列表：逐个元素操作
py_list = [1, 2, 3, 4, 5]
# py_list * 2 只是重复列表，不是数学乘法！
print(f"Python 列表 * 2 = {py_list * 2}")  # [1,2,3,4,5,1,2,3,4,5]

# NumPy 数组：向量化运算
np_array = np.array([1, 2, 3, 4, 5])
print(f"NumPy 数组 * 2 = {np_array * 2}")  # [2, 4, 6, 8, 10] ✅

# ============================================================
# 2. 创建数组的常用方式
# ============================================================
print("\n" + "=" * 50)
print("2. 创建数组")
print("=" * 50)

# 全零数组（常用于初始化权重）
zeros = np.zeros((3, 4))
print(f"全零数组 (3x4):\n{zeros}\n")

# 全一数组
ones = np.ones((2, 3))
print(f"全一数组 (2x3):\n{ones}\n")

# 随机数组（模拟神经网络初始权重）
random_weights = np.random.randn(3, 4)  # 标准正态分布
print(f"随机权重 (3x4):\n{random_weights}\n")

# 等间隔数组（常用于绘图的 x 轴）
x = np.linspace(0, 10, 5)  # 0到10之间均匀取5个点
print(f"等间隔数组: {x}")

# ============================================================
# 3. 数组形状和维度（非常重要！）
# ============================================================
print("\n" + "=" * 50)
print("3. 形状 (Shape) - AI 中最常见的 Bug 来源")
print("=" * 50)

# 1D 向量
vector = np.array([1, 2, 3])
print(f"向量: {vector}, shape={vector.shape}")

# 2D 矩阵（一张灰度图片就是一个 2D 矩阵）
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"矩阵:\n{matrix}\nshape={matrix.shape}")

# 3D 张量（一批彩色图片就是 4D 张量）
tensor = np.random.randn(2, 3, 4)  # 2个样本，3行，4列
print(f"\n3D 张量 shape={tensor.shape}")

# reshape - 改变形状但不改变数据
flat = np.arange(12)  # [0, 1, 2, ..., 11]
reshaped = flat.reshape(3, 4)
print(f"\n原始: {flat}")
print(f"reshape(3,4):\n{reshaped}")

# ============================================================
# 4. 向量化运算（AI 的核心）
# ============================================================
print("\n" + "=" * 50)
print("4. 向量化运算")
print("=" * 50)

# 模拟：3 个样本，每个样本 4 个特征
X = np.array([
    [1.0, 2.0, 3.0, 4.0],   # 样本1
    [5.0, 6.0, 7.0, 8.0],   # 样本2
    [9.0, 10.0, 11.0, 12.0], # 样本3
])

# 模拟：权重矩阵 (4个输入特征 -> 2个输出)
W = np.random.randn(4, 2)

# 矩阵乘法 - 神经网络的核心操作！
output = X @ W  # 等价于 np.dot(X, W)
print(f"输入 X shape: {X.shape}")
print(f"权重 W shape: {W.shape}")
print(f"输出 shape: {output.shape}")  # (3, 2) - 3个样本，2个输出
print(f"输出:\n{output}")

# ============================================================
# 5. 常用数学函数
# ============================================================
print("\n" + "=" * 50)
print("5. 常用数学函数（AI 中的激活函数基础）")
print("=" * 50)

x = np.array([-2, -1, 0, 1, 2], dtype=float)


# Sigmoid 函数 - 经典激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU 函数 - 最常用的激活函数
def relu(x):
    return np.maximum(0, x)


print(f"x = {x}")
print(f"sigmoid(x) = {sigmoid(x)}")
print(f"relu(x) = {relu(x)}")

# ============================================================
# 6. 广播机制 (Broadcasting)
# ============================================================
print("\n" + "=" * 50)
print("6. 广播机制")
print("=" * 50)

# 给每个样本的每个特征减去均值（数据标准化）
data = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
], dtype=float)

mean = data.mean(axis=0)  # 每列的均值
print(f"原始数据:\n{data}")
print(f"每列均值: {mean}")
print(f"减去均值（标准化）:\n{data - mean}")  # 广播！

# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 50)
print("💡 练习题")
print("=" * 50)

# 练习1: 创建一个 5x5 的单位矩阵（提示: np.eye）
# identity = ...

# 练习2: 生成 100 个 0-1 之间的随机数，计算均值和标准差
# random_data = ...

# 练习3: 实现 softmax 函数（提示: exp(x) / sum(exp(x))）
# def softmax(x):
#     pass

print("\n✅ NumPy 基础练习完成！")
