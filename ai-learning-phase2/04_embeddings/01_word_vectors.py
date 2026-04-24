"""
Embeddings - 词向量基础

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 理解为什么需要把文本变成数字（向量）
  - 手动实现简单的词向量
  - 理解余弦相似度
  - 可视化词向量空间
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 为什么需要 Embedding？
# ============================================================
print("=" * 50)
print("1. 为什么需要 Embedding？")
print("=" * 50)

print("""
计算机不认识文字，只认识数字。
我们需要把文字转换成数字向量，才能让模型处理。

最简单的方法: One-Hot 编码
  猫 → [1, 0, 0, 0]
  狗 → [0, 1, 0, 0]
  鱼 → [0, 0, 1, 0]
  鸟 → [0, 0, 0, 1]

问题:
  1. 维度太高（词汇表有多大，向量就有多长）
  2. 所有词之间的距离都一样（猫和狗的距离 = 猫和鱼的距离）
  3. 无法表达语义关系

Embedding 的解决方案:
  把每个词映射到一个低维的稠密向量，
  让语义相近的词在向量空间中距离也近。
  猫 → [0.8, 0.2, 0.9]  ← 和狗比较近
  狗 → [0.7, 0.3, 0.8]
  鱼 → [0.1, 0.9, 0.3]  ← 和猫比较远
""")

# ============================================================
# 2. 手动构造词向量，理解语义空间
# ============================================================
print("=" * 50)
print("2. 手动构造词向量")
print("=" * 50)

# 假设我们用 3 个维度来表示词:
# 维度0: 是否是动物 (0-1)
# 维度1: 体型大小 (0=小, 1=大)
# 维度2: 是否是宠物 (0-1)

words = {
    "猫":   np.array([1.0, 0.3, 0.9]),
    "狗":   np.array([1.0, 0.5, 0.9]),
    "老虎": np.array([1.0, 0.9, 0.1]),
    "金鱼": np.array([1.0, 0.1, 0.8]),
    "汽车": np.array([0.0, 0.7, 0.0]),
    "自行车": np.array([0.0, 0.3, 0.0]),
    "飞机": np.array([0.0, 0.9, 0.0]),
}

print("词向量 [是否动物, 体型, 是否宠物]:")
for word, vec in words.items():
    print(f"  {word}: {vec}")

# ============================================================
# 3. 余弦相似度
# ============================================================
print("\n" + "=" * 50)
print("3. 余弦相似度 - 衡量两个向量的方向是否一致")
print("=" * 50)

print("""
余弦相似度公式:
  cos(A, B) = (A · B) / (|A| × |B|)

  值域: [-1, 1]
    1  = 完全相同方向（最相似）
    0  = 正交（无关）
   -1  = 完全相反方向（最不相似）
""")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# 计算所有词对的相似度
word_list = list(words.keys())
print("词对相似度:")
pairs = [
    ("猫", "狗"),
    ("猫", "老虎"),
    ("猫", "汽车"),
    ("狗", "金鱼"),
    ("汽车", "自行车"),
    ("汽车", "飞机"),
]

for w1, w2 in pairs:
    sim = cosine_similarity(words[w1], words[w2])
    print(f"  {w1} ↔ {w2}: {sim:.4f}")

# ============================================================
# 4. 相似度矩阵热力图
# ============================================================
print("\n正在绘制相似度矩阵...")

n = len(word_list)
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = cosine_similarity(
            words[word_list[i]], words[word_list[j]]
        )

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(word_list, fontsize=12)
ax.set_yticklabels(word_list, fontsize=12)
ax.set_title("词向量余弦相似度矩阵", fontsize=14)

for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{sim_matrix[i,j]:.2f}",
                ha="center", va="center", fontsize=9)

plt.colorbar(im)
plt.tight_layout()
plt.savefig("04_embeddings/similarity_matrix.png", dpi=150)
plt.close()
print("✅ 相似度矩阵已保存: similarity_matrix.png")

# ============================================================
# 5. 词向量的数学运算（类比推理）
# ============================================================
print("\n" + "=" * 50)
print("5. 词向量的魔法：类比推理")
print("=" * 50)

print("""
经典例子: King - Man + Woman ≈ Queen

原理: 词向量可以做加减法！
  "国王"和"男人"的差 ≈ "皇后"和"女人"的差
  因为这个差向量编码了"皇室"这个概念。
""")

# 用我们的简单向量演示
# 猫 - 宠物属性 + 野生属性 ≈ 老虎？
cat = words["猫"]
dog = words["狗"]
tiger = words["老虎"]

# 猫和老虎的区别主要在"是否宠物"这个维度
diff = tiger - cat
print(f"老虎 - 猫 = {diff}")
print(f"这个差向量主要体现在: 体型变大(+0.6), 宠物属性降低(-0.8)")

# 狗 + (老虎 - 猫) = ?  应该得到一个"大型野生犬科动物"
result = dog + diff
print(f"\n狗 + (老虎 - 猫) = {result}")
print("含义: 一个体型大、非宠物的动物（类似狼）")

# 找最相似的词
print("\n与结果向量最相似的词:")
for word, vec in words.items():
    sim = cosine_similarity(result, vec)
    print(f"  {word}: {sim:.4f}")

# ============================================================
# 6. 可视化词向量空间（2D 投影）
# ============================================================
print("\n正在绘制词向量空间...")

# 用前两个维度做简单的 2D 可视化
fig, ax = plt.subplots(figsize=(8, 6))

colors = {"猫": "blue", "狗": "blue", "老虎": "red", "金鱼": "blue",
          "汽车": "green", "自行车": "green", "飞机": "green"}

for word, vec in words.items():
    ax.scatter(vec[0], vec[1], c=colors[word], s=100, zorder=5)
    ax.annotate(word, (vec[0], vec[1]), fontsize=14,
                xytext=(5, 5), textcoords="offset points")

ax.set_xlabel("维度0: 是否动物", fontsize=12)
ax.set_ylabel("维度1: 体型大小", fontsize=12)
ax.set_title("词向量空间可视化（前2维）", fontsize=14)
ax.legend(["蓝=动物/宠物", "红=动物/野生", "绿=交通工具"],
          loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("04_embeddings/word_vector_space.png", dpi=150)
plt.close()
print("✅ 词向量空间图已保存: word_vector_space.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 50)
print("💡 关键概念总结")
print("=" * 50)
print("""
1. Embedding 把离散的文字映射为连续的向量
2. 语义相近的词，向量也相近
3. 余弦相似度衡量两个向量的方向一致性
4. 词向量支持数学运算（加减法 = 语义推理）
5. 实际的 Embedding 维度通常是 384/768/1536 维

下一步: 用预训练模型生成真正的句子嵌入！
""")

print("✅ 词向量基础练习完成！")
