"""
Transformer - 迷你 Transformer 实现

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 理解 Transformer 的完整架构
  - 实现位置编码 (Positional Encoding)
  - 用 PyTorch 搭建一个可训练的迷你 Transformer
  - 完成一个简单的序列预测任务
"""

import torch
import torch.nn as nn
import numpy as np
import math

# ============================================================
# 1. 位置编码 (Positional Encoding)
# ============================================================
print("=" * 50)
print("1. 位置编码")
print("=" * 50)

print("""
为什么需要位置编码？
  Self-Attention 是"无序"的 —— 它不知道词的顺序。
  "猫追狗" 和 "狗追猫" 在纯注意力看来是一样的！

  位置编码给每个位置一个独特的向量，加到词向量上，
  这样模型就能区分不同位置的词了。

  使用正弦/余弦函数生成:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
""")


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# 可视化位置编码
pe = PositionalEncoding(d_model=64, max_len=50)
pe_values = pe.pe[0].numpy()
print(f"位置编码 shape: {pe_values.shape} (50个位置, 64维)")
print(f"位置 0 的编码（前8维）: {pe_values[0, :8].round(3)}")
print(f"位置 1 的编码（前8维）: {pe_values[1, :8].round(3)}")


# ============================================================
# 2. Transformer Block
# ============================================================
print("\n" + "=" * 50)
print("2. Transformer Block")
print("=" * 50)

print("""
一个 Transformer Block 包含:
  1. Multi-Head Self-Attention
  2. Add & Norm (残差连接 + 层归一化)
  3. Feed-Forward Network (两层全连接)
  4. Add & Norm

残差连接: output = LayerNorm(x + SubLayer(x))
  - 让梯度更容易传播（解决深层网络训练困难）
""")


class TransformerBlock(nn.Module):
    """单个 Transformer Block"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Multi-Head Attention（PyTorch 内置实现）
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + 残差连接
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # FFN + 残差连接
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


# ============================================================
# 3. 完整的迷你 Transformer
# ============================================================
print("\n" + "=" * 50)
print("3. 迷你 Transformer 模型")
print("=" * 50)


class MiniTransformer(nn.Module):
    """迷你 Transformer - 用于序列预测

    任务: 给定一个数字序列，预测下一个数字
    例如: [1, 2, 3, 4] → 预测 5
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_len: int = 50,
    ):
        super().__init__()

        self.d_model = d_model

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 多层 Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # 通过多层 Transformer
        for layer in self.layers:
            x = layer(x)

        # 取最后一个位置的输出来预测
        x = x[:, -1, :]  # (batch, d_model)
        logits = self.output_layer(x)  # (batch, vocab_size)

        return logits


# ============================================================
# 4. 训练：简单的序列预测任务
# ============================================================
print("\n" + "=" * 50)
print("4. 训练序列预测任务")
print("=" * 50)

print("任务: 学习简单的数字规律")
print("  输入 [1,2,3,4] → 预测 5")
print("  输入 [3,4,5,6] → 预测 7")
print("  ...")

# 生成训练数据
def generate_sequence_data(n_samples=500, seq_len=4, max_num=20):
    """生成连续数字序列数据"""
    X = []
    y = []
    for _ in range(n_samples):
        start = np.random.randint(0, max_num - seq_len)
        seq = list(range(start, start + seq_len))
        target = start + seq_len
        X.append(seq)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


vocab_size = 30  # 数字范围 0-29
seq_len = 4

X_train, y_train = generate_sequence_data(500, seq_len, max_num=vocab_size - seq_len)
X_test, y_test = generate_sequence_data(100, seq_len, max_num=vocab_size - seq_len)

print(f"\n训练样本数: {len(X_train)}")
print(f"序列长度: {seq_len}")
print(f"词汇表大小: {vocab_size}")
print(f"示例: {X_train[0].tolist()} → {y_train[0].item()}")

# 创建模型
model = MiniTransformer(
    vocab_size=vocab_size,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=128,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数量: {total_params:,}")

# 训练
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n开始训练...")
print("-" * 50)

for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()

    output = model(X_train)
    loss = criterion(output, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train).argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean()
            test_pred = model(X_test).argmax(dim=1)
            test_acc = (test_pred == y_test).float().mean()

        print(
            f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
            f"训练准确率: {train_acc:.2%} | 测试准确率: {test_acc:.2%}"
        )

# ============================================================
# 5. 测试
# ============================================================
print("\n" + "=" * 50)
print("5. 预测结果")
print("=" * 50)

model.eval()
with torch.no_grad():
    test_output = model(X_test)
    predictions = test_output.argmax(dim=1)

print("测试样本预测:")
for i in range(15):
    seq = X_test[i].tolist()
    pred = predictions[i].item()
    true = y_test[i].item()
    status = "✅" if pred == true else "❌"
    print(f"  {seq} → 预测: {pred}, 实际: {true} {status}")

final_acc = (predictions == y_test).float().mean()
print(f"\n最终测试准确率: {final_acc:.2%}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 50)
print("Transformer 架构总结")
print("=" * 50)
print("""
完整的 Transformer 包含:

  输入 → Embedding → 位置编码 → [Transformer Block × N] → 输出

  每个 Transformer Block:
    ┌─────────────────────────┐
    │  Multi-Head Attention   │
    │  + 残差连接 + LayerNorm │
    ├─────────────────────────┤
    │  Feed-Forward Network   │
    │  + 残差连接 + LayerNorm │
    └─────────────────────────┘

  关键组件:
    1. Self-Attention: 捕捉序列中的依赖关系
    2. 位置编码: 提供位置信息
    3. 残差连接: 帮助梯度传播
    4. LayerNorm: 稳定训练
    5. FFN: 增加模型表达能力

  这就是 GPT、BERT、LLaMA 等大模型的基础架构！
""")

print("✅ 迷你 Transformer 练习完成！")
print("🎉 恭喜！阶段一全部完成！")
print("   你已经理解了从神经元到 Transformer 的完整脉络。")
print("   下一步: 阶段二 - Embeddings 和 RAG")
