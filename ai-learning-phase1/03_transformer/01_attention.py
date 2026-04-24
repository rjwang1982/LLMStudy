"""
Transformer - 注意力机制可视化

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 理解"注意力"的直觉：哪些词对当前词更重要
  - 理解注意力分数的计算过程
  - 可视化注意力权重矩阵
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 注意力的直觉
# ============================================================
print("=" * 50)
print("注意力机制 - 直觉理解")
print("=" * 50)

print("""
想象你在读这句话: "小猫坐在垫子上，它很开心"

当你理解"它"这个词时，你的大脑会自动关注"小猫"，
而不是"垫子"或"上"。

这就是注意力机制的核心思想：
  对于每个词，计算它应该"关注"其他哪些词，以及关注多少。
""")

# ============================================================
# 2. 手动计算注意力
# ============================================================
print("=" * 50)
print("手动计算注意力分数")
print("=" * 50)

# 假设我们有 4 个词，每个词用一个 3 维向量表示
words = ["我", "喜欢", "机器", "学习"]
# 这些向量通常由 Embedding 层生成，这里手动设定
embeddings = np.array([
    [1.0, 0.0, 1.0],   # 我
    [0.0, 1.0, 0.0],   # 喜欢
    [1.0, 1.0, 0.0],   # 机器
    [0.0, 1.0, 1.0],   # 学习
])

print("词向量:")
for word, emb in zip(words, embeddings):
    print(f"  {word}: {emb}")

# 计算注意力分数: 每个词和其他所有词的相似度
# 使用点积（dot product）衡量相似度
scores = embeddings @ embeddings.T
print(f"\n原始注意力分数（点积）:\n{scores}")

# Softmax 归一化：将分数转换为概率（每行加起来 = 1）
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scores)
print(f"\n注意力权重（softmax 后）:")
for i, word in enumerate(words):
    weights_str = ", ".join(f"{words[j]}:{attention_weights[i,j]:.3f}" for j in range(len(words)))
    print(f"  {word} 关注 → [{weights_str}]")

# ============================================================
# 3. 可视化注意力矩阵
# ============================================================
print("\n正在绘制注意力热力图...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 原始分数
im1 = axes[0].imshow(scores, cmap="YlOrRd")
axes[0].set_xticks(range(len(words)))
axes[0].set_yticks(range(len(words)))
axes[0].set_xticklabels(words, fontsize=14)
axes[0].set_yticklabels(words, fontsize=14)
axes[0].set_title("原始注意力分数（点积）", fontsize=14)
for i in range(len(words)):
    for j in range(len(words)):
        axes[0].text(j, i, f"{scores[i,j]:.1f}", ha="center", va="center", fontsize=12)
plt.colorbar(im1, ax=axes[0])

# Softmax 后的权重
im2 = axes[1].imshow(attention_weights, cmap="YlOrRd")
axes[1].set_xticks(range(len(words)))
axes[1].set_yticks(range(len(words)))
axes[1].set_xticklabels(words, fontsize=14)
axes[1].set_yticklabels(words, fontsize=14)
axes[1].set_title("注意力权重（Softmax 后）", fontsize=14)
for i in range(len(words)):
    for j in range(len(words)):
        axes[1].text(j, i, f"{attention_weights[i,j]:.2f}", ha="center", va="center", fontsize=12)
plt.colorbar(im2, ax=axes[1])

plt.suptitle("注意力机制可视化", fontsize=16)
plt.tight_layout()
plt.savefig("03_transformer/attention_heatmap.png", dpi=150)
plt.close()
print("✅ 注意力热力图已保存: attention_heatmap.png")

# ============================================================
# 4. 加权求和 - 注意力的输出
# ============================================================
print("\n" + "=" * 50)
print("注意力输出 = 加权求和")
print("=" * 50)

print("""
注意力的最终输出:
  对于每个词，用注意力权重对所有词的向量做加权求和。
  这样每个词的新表示就包含了它"关注"的其他词的信息。
""")

# 加权求和
output = attention_weights @ embeddings
print("注意力输出（每个词的新表示）:")
for word, out in zip(words, output):
    print(f"  {word}: {out.round(3)}")

print("\n💡 关键理解:")
print("  1. 注意力分数 = 两个词向量的点积（越大越相似）")
print("  2. Softmax 将分数归一化为概率")
print("  3. 用概率对所有词向量加权求和，得到新表示")
print("  4. 新表示融合了上下文信息！")

print("\n✅ 注意力机制可视化完成！")
