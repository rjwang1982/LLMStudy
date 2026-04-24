"""
多模态模型 - CLIP 原理：图文对齐

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 理解多模态的含义（图像 + 文本）
  - 理解 CLIP 的对比学习思想
  - 手动模拟图文对齐的过程
  - 理解为什么图文可以在同一个向量空间
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. 什么是多模态？
# ============================================================
print("=" * 50)
print("1. 什么是多模态模型？")
print("=" * 50)

print("""
"模态" = 信息的类型
  - 文本模态: 文字、句子、文档
  - 图像模态: 照片、图片
  - 音频模态: 语音、音乐
  - 视频模态: 视频流

多模态模型 = 能同时理解多种模态的模型

代表性模型:
  - CLIP (OpenAI): 图文对齐，理解图片和文字的关系
  - GPT-4V: 能看图回答问题
  - LLaVA: 开源的视觉语言模型
  - Whisper: 语音转文字

核心挑战:
  如何让不同模态的数据在同一个向量空间中对齐？
  → 让"一只猫的图片"和"一只猫"这段文字的向量很接近
""")

# ============================================================
# 2. CLIP 的核心思想
# ============================================================
print("=" * 50)
print("2. CLIP 的核心思想：对比学习")
print("=" * 50)

print("""
CLIP (Contrastive Language-Image Pre-training):

  训练数据: 4 亿对 (图片, 文字描述) 配对
  
  训练目标: 对比学习
    ✅ 匹配的图文对 → 向量要接近
    ❌ 不匹配的图文对 → 向量要远离

  架构:
    图片 → 图像编码器 (ViT/ResNet) → 图像向量
    文字 → 文本编码器 (Transformer)  → 文本向量

    两个向量在同一个空间中！可以直接计算相似度。

  应用:
    - 图片搜索: 用文字搜图片
    - 图片分类: 零样本分类（不需要训练数据）
    - 图文匹配: 判断图片和文字是否匹配
""")

# ============================================================
# 3. 模拟 CLIP 的对比学习
# ============================================================
print("=" * 50)
print("3. 模拟对比学习过程")
print("=" * 50)

# 模拟 4 对图文数据
image_descriptions = ["一只橘猫", "海边日落", "城市夜景", "森林小路"]
text_descriptions = ["一只橘猫", "海边日落", "城市夜景", "森林小路"]

# 模拟编码器输出的向量（实际中由神经网络生成）
np.random.seed(42)
embed_dim = 8

# 图像向量（模拟图像编码器的输出）
image_embeddings = np.random.randn(4, embed_dim).astype(np.float32)
# 文本向量（模拟文本编码器的输出）
text_embeddings = np.random.randn(4, embed_dim).astype(np.float32)

# 让匹配的对更相似（模拟训练后的效果）
for i in range(4):
    text_embeddings[i] = image_embeddings[i] + np.random.randn(embed_dim) * 0.3

# 归一化
image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

# 计算相似度矩阵
similarity_matrix = image_embeddings @ text_embeddings.T

print("图文相似度矩阵:")
print(f"{'':>12}", end="")
for t in text_descriptions:
    print(f"{t:>10}", end="")
print()

for i, img in enumerate(image_descriptions):
    print(f"{img:>10}  ", end="")
    for j in range(4):
        sim = similarity_matrix[i, j]
        marker = " ✅" if i == j else ""
        print(f"{sim:>8.3f}{marker}", end="")
    print()

print("\n✅ 对角线上的值最大 = 匹配的图文对最相似！")

# ============================================================
# 4. 对比损失函数 (InfoNCE Loss)
# ============================================================
print("\n" + "=" * 50)
print("4. 对比损失函数")
print("=" * 50)


def contrastive_loss(image_emb, text_emb, temperature=0.07):
    """CLIP 使用的对比损失 (InfoNCE)

    目标: 让匹配的图文对相似度高，不匹配的相似度低

    Args:
        image_emb: 图像嵌入 (N, D)
        text_emb: 文本嵌入 (N, D)
        temperature: 温度参数（越小越"尖锐"）
    """
    image_emb = torch.tensor(image_emb)
    text_emb = torch.tensor(text_emb)

    # 计算相似度矩阵
    logits = image_emb @ text_emb.T / temperature

    # 标签: 对角线上的是正样本
    labels = torch.arange(len(image_emb))

    # 双向对比损失
    loss_i2t = F.cross_entropy(logits, labels)      # 图→文
    loss_t2i = F.cross_entropy(logits.T, labels)     # 文→图

    loss = (loss_i2t + loss_t2i) / 2
    return loss.item()


loss = contrastive_loss(image_embeddings, text_embeddings)
print(f"对比损失: {loss:.4f}")

print("""
损失函数的直觉:
  - 对于每张图片，在所有文本中找到匹配的那个（图→文）
  - 对于每段文本，在所有图片中找到匹配的那个（文→图）
  - 本质上是一个 N 路分类问题！
  - 温度参数控制分布的"尖锐度"
""")

# ============================================================
# 5. 零样本分类 (Zero-Shot Classification)
# ============================================================
print("=" * 50)
print("5. 零样本分类 - CLIP 的杀手级应用")
print("=" * 50)

print("""
传统分类: 需要收集训练数据 → 训练模型 → 才能分类
CLIP 分类: 直接用文字描述类别 → 计算相似度 → 完成分类！

步骤:
  1. 把类别名变成文本: "a photo of a cat", "a photo of a dog"
  2. 用文本编码器得到类别向量
  3. 用图像编码器得到图片向量
  4. 计算图片和每个类别的相似度
  5. 相似度最高的就是预测类别
""")

# 模拟零样本分类
categories = ["猫", "狗", "汽车", "飞机", "花"]

# 模拟类别的文本向量
np.random.seed(123)
category_vectors = np.random.randn(5, embed_dim).astype(np.float32)
category_vectors = category_vectors / np.linalg.norm(category_vectors, axis=1, keepdims=True)

# 模拟一张"猫"图片的向量（和"猫"类别向量接近）
cat_image = category_vectors[0] + np.random.randn(embed_dim).astype(np.float32) * 0.2
cat_image = cat_image / np.linalg.norm(cat_image)

# 计算与每个类别的相似度
similarities = cat_image @ category_vectors.T

print("图片与各类别的相似度:")
for cat, sim in sorted(zip(categories, similarities), key=lambda x: -x[1]):
    bar = "█" * int((sim + 1) * 15)  # 归一化到可视化范围
    print(f"  {cat}: {sim:.4f} {bar}")

predicted = categories[np.argmax(similarities)]
print(f"\n预测类别: {predicted} ✅")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 50)
print("💡 多模态模型总结")
print("=" * 50)
print("""
CLIP 的核心:
  1. 两个编码器: 图像编码器 + 文本编码器
  2. 对比学习: 匹配的图文对拉近，不匹配的推远
  3. 共享向量空间: 图片和文字可以直接比较
  4. 零样本能力: 不需要训练数据就能分类

多模态的发展:
  CLIP (2021) → 图文对齐
  DALL-E (2021) → 文字生成图片
  GPT-4V (2023) → 看图对话
  Sora (2024) → 文字生成视频

关键理解:
  多模态的本质是把不同类型的数据
  映射到同一个向量空间中，
  这样就可以跨模态进行检索、理解和生成。
""")

print("✅ CLIP 概念练习完成！")
