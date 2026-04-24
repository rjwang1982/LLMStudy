"""
Embeddings - 句子嵌入：用预训练模型编码文本

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 使用 sentence-transformers 生成句子向量
  - 理解预训练模型的强大之处
  - 对比不同句子的语义相似度
  - 体验真实的文本嵌入效果
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. 加载预训练模型
# ============================================================
print("=" * 50)
print("1. 加载预训练 Embedding 模型")
print("=" * 50)

print("正在加载模型 all-MiniLM-L6-v2（首次会下载约 80MB）...")
# 这是一个轻量级但效果很好的句子嵌入模型
# 输出 384 维的向量
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ 模型加载完成！")

# ============================================================
# 2. 生成句子嵌入
# ============================================================
print("\n" + "=" * 50)
print("2. 生成句子嵌入")
print("=" * 50)

sentences = [
    "我喜欢吃苹果",
    "苹果是我最爱的水果",
    "今天天气真好",
    "阳光明媚的一天",
    "Python 是一门编程语言",
    "我正在学习写代码",
]

# 一行代码生成所有句子的嵌入向量！
embeddings = model.encode(sentences)

print(f"句子数量: {len(sentences)}")
print(f"每个向量维度: {embeddings.shape[1]}")
print(f"嵌入矩阵 shape: {embeddings.shape}")

# 看看第一个句子的向量（只显示前 10 维）
print(f"\n'{sentences[0]}' 的向量（前10维）:")
print(f"  {embeddings[0][:10].round(4)}")

# ============================================================
# 3. 计算句子间的相似度
# ============================================================
print("\n" + "=" * 50)
print("3. 句子语义相似度")
print("=" * 50)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 计算所有句子对的相似度
print("句子对相似度（越接近 1 越相似）:\n")

pairs = [
    (0, 1),  # 苹果相关
    (2, 3),  # 天气相关
    (4, 5),  # 编程相关
    (0, 2),  # 苹果 vs 天气
    (0, 4),  # 苹果 vs 编程
    (2, 4),  # 天气 vs 编程
]

for i, j in pairs:
    sim = cosine_similarity(embeddings[i], embeddings[j])
    bar = "█" * int(sim * 20)
    print(f"  [{sim:.4f}] {bar}")
    print(f"    '{sentences[i]}'")
    print(f"    '{sentences[j]}'\n")

# ============================================================
# 4. 语义搜索：找最相关的句子
# ============================================================
print("=" * 50)
print("4. 语义搜索")
print("=" * 50)

# 知识库
knowledge_base = [
    "Python 是由 Guido van Rossum 创建的编程语言",
    "机器学习是人工智能的一个分支",
    "深度学习使用多层神经网络",
    "自然语言处理让计算机理解人类语言",
    "ChatGPT 是 OpenAI 开发的大语言模型",
    "向量数据库用于存储和检索嵌入向量",
    "Transformer 架构是现代 NLP 的基础",
    "RAG 结合了检索和生成两种能力",
    "PyTorch 是最流行的深度学习框架之一",
    "卷积神经网络常用于图像识别",
]

# 编码知识库
kb_embeddings = model.encode(knowledge_base)

# 用户查询
queries = [
    "什么是大模型？",
    "如何处理文本数据？",
    "推荐一个深度学习工具",
]

print("语义搜索结果:\n")
for query in queries:
    query_embedding = model.encode([query])[0]

    # 计算与所有知识的相似度
    similarities = [
        cosine_similarity(query_embedding, kb_emb)
        for kb_emb in kb_embeddings
    ]

    # 排序，取 Top 3
    top_indices = np.argsort(similarities)[::-1][:3]

    print(f"🔍 查询: '{query}'")
    for rank, idx in enumerate(top_indices, 1):
        print(f"   Top{rank} [{similarities[idx]:.4f}]: {knowledge_base[idx]}")
    print()

# ============================================================
# 5. 嵌入向量的聚类效果
# ============================================================
print("=" * 50)
print("5. 嵌入向量自动聚类")
print("=" * 50)

# 三组不同主题的句子
topic_sentences = [
    # 美食
    "我喜欢吃火锅",
    "寿司是日本料理",
    "意大利面很好吃",
    # 运动
    "足球是世界第一运动",
    "NBA 篮球赛很精彩",
    "游泳对身体很好",
    # 科技
    "人工智能改变世界",
    "量子计算是未来",
    "5G 网络速度很快",
]

topic_labels = ["美食"] * 3 + ["运动"] * 3 + ["科技"] * 3

topic_embeddings = model.encode(topic_sentences)

# 计算每对句子的相似度，看看同主题是否更相似
print("同主题 vs 跨主题的平均相似度:\n")

same_topic_sims = []
diff_topic_sims = []

for i in range(len(topic_sentences)):
    for j in range(i + 1, len(topic_sentences)):
        sim = cosine_similarity(topic_embeddings[i], topic_embeddings[j])
        if topic_labels[i] == topic_labels[j]:
            same_topic_sims.append(sim)
        else:
            diff_topic_sims.append(sim)

print(f"  同主题平均相似度: {np.mean(same_topic_sims):.4f}")
print(f"  跨主题平均相似度: {np.mean(diff_topic_sims):.4f}")
print(f"  差距: {np.mean(same_topic_sims) - np.mean(diff_topic_sims):.4f}")
print("\n  → 同主题的句子在向量空间中确实更接近！")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 50)
print("💡 关键概念总结")
print("=" * 50)
print("""
1. sentence-transformers 可以一行代码生成句子嵌入
2. 预训练模型已经学会了丰富的语义知识
3. 语义搜索 = 编码查询 + 计算相似度 + 排序
4. 同主题的文本在向量空间中自然聚集
5. 这就是 RAG 系统的检索基础！

常用模型:
  - all-MiniLM-L6-v2: 轻量级，384 维，速度快
  - all-mpnet-base-v2: 更准确，768 维
  - text-embedding-3-small: OpenAI 的，1536 维
""")

print("✅ 句子嵌入练习完成！")
