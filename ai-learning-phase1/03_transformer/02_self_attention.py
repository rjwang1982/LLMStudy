"""
Transformer - Self-Attention 完整实现

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 理解 Query, Key, Value 的含义
  - 实现 Scaled Dot-Product Attention
  - 实现 Multi-Head Attention
  - 理解为什么需要多头注意力
"""

import numpy as np

np.random.seed(42)

# ============================================================
# 1. Q, K, V 的直觉理解
# ============================================================
print("=" * 50)
print("Self-Attention: Query, Key, Value")
print("=" * 50)

print("""
想象你在图书馆找书:
  - Query (查询): 你想找什么？ → "我想找关于 Python 的书"
  - Key (键):     每本书的标签  → "Python入门", "Java编程", "Python进阶"
  - Value (值):   书的实际内容  → 每本书的具体内容

过程:
  1. 用你的 Query 和每本书的 Key 比较 → 得到相关度分数
  2. 用分数对每本书的 Value 加权求和 → 得到你需要的信息

在 Self-Attention 中:
  - 每个词同时扮演 Query、Key、Value 三个角色
  - 通过三个不同的权重矩阵 (Wq, Wk, Wv) 将词向量投影到不同空间
""")

# ============================================================
# 2. Scaled Dot-Product Attention
# ============================================================
print("=" * 50)
print("Scaled Dot-Product Attention 实现")
print("=" * 50)


def scaled_dot_product_attention(Q, K, V):
    """缩放点积注意力

    公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        Q: Query 矩阵, shape=(seq_len, d_k)
        K: Key 矩阵, shape=(seq_len, d_k)
        V: Value 矩阵, shape=(seq_len, d_v)

    Returns:
        output: 注意力输出
        weights: 注意力权重
    """
    d_k = K.shape[-1]

    # 1. 计算注意力分数
    scores = Q @ K.T

    # 2. 缩放（防止点积值过大导致 softmax 梯度消失）
    scores = scores / np.sqrt(d_k)

    # 3. Softmax 归一化
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. 加权求和
    output = weights @ V

    return output, weights


# 示例：4 个词，每个词 8 维
seq_len = 4
d_model = 8
d_k = d_v = 8

# 模拟输入（4 个词的嵌入向量）
X = np.random.randn(seq_len, d_model)

# 权重矩阵（实际中这些是可学习的参数）
Wq = np.random.randn(d_model, d_k) * 0.1
Wk = np.random.randn(d_model, d_k) * 0.1
Wv = np.random.randn(d_model, d_v) * 0.1

# 计算 Q, K, V
Q = X @ Wq
K = X @ Wk
V = X @ Wv

print(f"输入 X shape: {X.shape}")
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"\n注意力输出 shape: {output.shape}")
print(f"注意力权重:\n{weights.round(3)}")
print(f"每行权重之和: {weights.sum(axis=1).round(3)}")  # 应该都是 1

# ============================================================
# 3. Multi-Head Attention
# ============================================================
print("\n" + "=" * 50)
print("Multi-Head Attention（多头注意力）")
print("=" * 50)

print("""
为什么需要多头？
  - 单头注意力只能学习一种"关注模式"
  - 多头注意力可以同时学习多种关注模式:
    - 头1: 关注语法关系（主语-谓语）
    - 头2: 关注语义关系（同义词）
    - 头3: 关注位置关系（相邻词）
  - 最后把所有头的结果拼接起来
""")


class MultiHeadAttention:
    """多头注意力机制

    将 d_model 维的输入分成 n_heads 个头，
    每个头独立计算注意力，最后拼接。
    """

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 每个头有独立的 Q, K, V 权重
        self.Wq = np.random.randn(d_model, d_model) * 0.1
        self.Wk = np.random.randn(d_model, d_model) * 0.1
        self.Wv = np.random.randn(d_model, d_model) * 0.1
        self.Wo = np.random.randn(d_model, d_model) * 0.1  # 输出投影

    def forward(self, X):
        """前向传播

        Args:
            X: 输入, shape=(seq_len, d_model)

        Returns:
            output: 多头注意力输出, shape=(seq_len, d_model)
        """
        seq_len = X.shape[0]

        # 线性投影
        Q = X @ self.Wq  # (seq_len, d_model)
        K = X @ self.Wk
        V = X @ self.Wv

        # 分成多个头: (seq_len, d_model) → (n_heads, seq_len, d_k)
        Q = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        # 每个头独立计算注意力
        all_outputs = []
        all_weights = []
        for h in range(self.n_heads):
            out, w = scaled_dot_product_attention(Q[h], K[h], V[h])
            all_outputs.append(out)
            all_weights.append(w)

        # 拼接所有头的输出: (n_heads, seq_len, d_k) → (seq_len, d_model)
        concat = np.concatenate(all_outputs, axis=-1)

        # 最终线性投影
        output = concat @ self.Wo

        return output, all_weights


# 测试多头注意力
d_model = 8
n_heads = 2
seq_len = 4

X = np.random.randn(seq_len, d_model)
mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
output, head_weights = mha.forward(X)

print(f"\n输入 shape: {X.shape}")
print(f"输出 shape: {output.shape}")
print(f"头数: {n_heads}")
print(f"每个头的维度: {d_model // n_heads}")

for h in range(n_heads):
    print(f"\n头 {h+1} 的注意力权重:")
    print(np.array(head_weights[h]).round(3))

print("\n💡 观察: 不同的头学到了不同的注意力模式！")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 50)
print("Self-Attention 核心公式")
print("=" * 50)
print("""
1. 单头注意力:
   Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

2. 多头注意力:
   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · Wo
   其中 head_i = Attention(X·Wq_i, X·Wk_i, X·Wv_i)

3. 关键参数:
   - d_model: 模型维度（如 512, 768）
   - n_heads: 头数（如 8, 12）
   - d_k = d_model / n_heads: 每个头的维度
""")

print("✅ Self-Attention 实现完成！")
