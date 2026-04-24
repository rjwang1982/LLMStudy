"""
模型压缩 - 量化：用更少的位数表示权重

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 理解量化的基本原理
  - 对比 FP32、FP16、INT8 的精度和大小
  - 手动实现简单的量化和反量化
  - 理解量化对模型精度的影响
"""

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# 1. 什么是量化？
# ============================================================
print("=" * 50)
print("1. 什么是量化？")
print("=" * 50)

print("""
量化 = 用更少的位数来表示模型权重

类比: 照片质量
  - 原图 (FP32): 高清无损，文件很大
  - 压缩 (FP16): 肉眼几乎看不出区别，文件减半
  - 缩略图 (INT8): 有点模糊，但文件很小

数据类型对比:
  ┌──────────┬──────┬──────────┬──────────────┐
  │ 类型     │ 位数 │ 内存占比 │ 精度         │
  ├──────────┼──────┼──────────┼──────────────┤
  │ FP32     │ 32   │ 100%     │ 最高         │
  │ FP16     │ 16   │ 50%      │ 几乎无损     │
  │ INT8     │ 8    │ 25%      │ 轻微损失     │
  │ INT4     │ 4    │ 12.5%    │ 有一定损失   │
  └──────────┴──────┴──────────┴──────────────┘

一个 7B 参数的模型:
  FP32: 28 GB
  FP16: 14 GB
  INT8:  7 GB
  INT4:  3.5 GB  ← 可以在普通笔记本上运行！
""")

# ============================================================
# 2. 手动实现 INT8 量化
# ============================================================
print("=" * 50)
print("2. 手动实现 INT8 量化")
print("=" * 50)

# 模拟一组模型权重（FP32）
np.random.seed(42)
weights_fp32 = np.random.randn(1000).astype(np.float32)

print(f"原始权重 (FP32):")
print(f"  数量: {len(weights_fp32)}")
print(f"  范围: [{weights_fp32.min():.4f}, {weights_fp32.max():.4f}]")
print(f"  内存: {weights_fp32.nbytes} bytes")


def quantize_int8(weights: np.ndarray) -> tuple:
    """将 FP32 权重量化为 INT8

    量化公式: q = round(w / scale) + zero_point
    反量化:   w ≈ (q - zero_point) * scale
    """
    w_min = weights.min()
    w_max = weights.max()

    # 计算缩放因子和零点
    scale = (w_max - w_min) / 255  # INT8 范围: 0-255
    zero_point = round(-w_min / scale)

    # 量化
    quantized = np.round(weights / scale + zero_point).astype(np.uint8)

    return quantized, scale, zero_point


def dequantize_int8(quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """将 INT8 反量化回 FP32"""
    return (quantized.astype(np.float32) - zero_point) * scale


# 执行量化
quantized, scale, zero_point = quantize_int8(weights_fp32)

print(f"\n量化后 (INT8):")
print(f"  范围: [{quantized.min()}, {quantized.max()}]")
print(f"  内存: {quantized.nbytes} bytes")
print(f"  压缩比: {weights_fp32.nbytes / quantized.nbytes:.1f}x")
print(f"  scale: {scale:.6f}")
print(f"  zero_point: {zero_point}")

# 反量化
weights_recovered = dequantize_int8(quantized, scale, zero_point)

# 计算误差
error = np.abs(weights_fp32 - weights_recovered)
print(f"\n量化误差:")
print(f"  平均绝对误差: {error.mean():.6f}")
print(f"  最大绝对误差: {error.max():.6f}")
print(f"  相对误差: {(error / (np.abs(weights_fp32) + 1e-8)).mean():.4%}")

# ============================================================
# 3. PyTorch 动态量化
# ============================================================
print("\n" + "=" * 50)
print("3. PyTorch 动态量化")
print("=" * 50)


# 创建一个简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


model = SimpleModel()

# 计算原始模型大小
def get_model_size(model):
    """计算模型参数占用的内存（MB）"""
    total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 * 1024)

original_size = get_model_size(model)
print(f"原始模型大小: {original_size:.2f} MB")

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # 量化 Linear 层
    dtype=torch.qint8,
)

quantized_size = get_model_size(quantized_model)
print(f"量化后大小: {quantized_size:.2f} MB")
print(f"压缩比: {original_size / max(quantized_size, 0.01):.1f}x")

# 对比推理速度
import time

test_input = torch.randn(100, 784)

# 原始模型
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(test_input)
original_time = time.time() - start

# 量化模型
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = quantized_model(test_input)
quantized_time = time.time() - start

print(f"\n推理速度对比 (100次):")
print(f"  原始模型: {original_time*1000:.1f} ms")
print(f"  量化模型: {quantized_time*1000:.1f} ms")
print(f"  加速比: {original_time/quantized_time:.2f}x")

# 对比输出差异
with torch.no_grad():
    output_original = model(test_input)
    output_quantized = quantized_model(test_input)

diff = (output_original - output_quantized).abs().mean().item()
print(f"\n输出平均差异: {diff:.6f}")
print("→ 量化后输出几乎不变！")

# ============================================================
# 4. 量化方法总结
# ============================================================
print("\n" + "=" * 50)
print("4. 量化方法总结")
print("=" * 50)

print("""
常见量化方法:

1. 动态量化 (Dynamic Quantization)
   - 权重提前量化，激活值运行时量化
   - 最简单，一行代码搞定
   - 适合: CPU 推理

2. 静态量化 (Static Quantization)
   - 需要校准数据集
   - 权重和激活值都提前量化
   - 适合: 追求最大速度

3. 量化感知训练 (QAT)
   - 训练时模拟量化效果
   - 精度损失最小
   - 适合: 对精度要求高

实际应用:
  - GGUF 格式: llama.cpp 使用，支持 Q4/Q5/Q8
  - GPTQ: GPU 上的 4-bit 量化
  - AWQ: 激活感知的权重量化
  - 下载模型时看到 Q4_K_M、Q8_0 等就是量化版本
""")

print("✅ 量化练习完成！")
