"""
Embeddings - 向量检索实战：FAISS

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 理解向量检索的原理（暴力搜索 vs 近似搜索）
  - 使用 FAISS 构建高效的向量索引
  - 体验大规模向量检索的速度
  - 为 RAG 的检索环节打基础
"""

import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. 为什么需要向量检索？
# ============================================================
print("=" * 50)
print("1. 为什么需要向量检索？")
print("=" * 50)

print("""
场景: 你有 100 万篇文档的嵌入向量，用户提了一个问题。
  - 暴力搜索: 逐个计算相似度 → 太慢！
  - 向量索引: 用特殊的数据结构加速 → 毫秒级返回！

FAISS (Facebook AI Similarity Search):
  - Meta 开源的向量检索库
  - 支持十亿级别的向量检索
  - 多种索引类型，平衡速度和精度
""")

# ============================================================
# 2. 暴力搜索 vs FAISS
# ============================================================
print("=" * 50)
print("2. 性能对比：暴力搜索 vs FAISS")
print("=" * 50)

# 生成模拟数据
np.random.seed(42)
n_vectors = 100_000  # 10 万个向量
dim = 384            # 384 维（和 MiniLM 一致）

print(f"生成 {n_vectors:,} 个 {dim} 维向量...")
database = np.random.randn(n_vectors, dim).astype("float32")
query = np.random.randn(1, dim).astype("float32")

# 方法1: 暴力搜索（NumPy）
print("\n--- 暴力搜索 ---")
start = time.time()
# 计算查询向量与所有向量的距离
distances = np.linalg.norm(database - query, axis=1)
top_k_brute = np.argsort(distances)[:5]
brute_time = time.time() - start
print(f"耗时: {brute_time*1000:.1f} ms")
print(f"Top 5 索引: {top_k_brute}")

# 方法2: FAISS 精确搜索
print("\n--- FAISS 精确搜索 (Flat) ---")
index_flat = faiss.IndexFlatL2(dim)  # L2 距离（欧氏距离）
index_flat.add(database)
print(f"索引中的向量数: {index_flat.ntotal:,}")

start = time.time()
distances_faiss, indices_faiss = index_flat.search(query, 5)
flat_time = time.time() - start
print(f"耗时: {flat_time*1000:.1f} ms")
print(f"Top 5 索引: {indices_faiss[0]}")

# 方法3: FAISS 近似搜索（IVF）
print("\n--- FAISS 近似搜索 (IVF) ---")
nlist = 100  # 聚类中心数
quantizer = faiss.IndexFlatL2(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
index_ivf.train(database)  # 需要先训练（聚类）
index_ivf.add(database)
index_ivf.nprobe = 10  # 搜索时检查 10 个聚类

start = time.time()
distances_ivf, indices_ivf = index_ivf.search(query, 5)
ivf_time = time.time() - start
print(f"耗时: {ivf_time*1000:.1f} ms")
print(f"Top 5 索引: {indices_ivf[0]}")

print(f"\n📊 速度对比:")
print(f"  暴力搜索:    {brute_time*1000:>8.1f} ms")
print(f"  FAISS Flat:  {flat_time*1000:>8.1f} ms")
print(f"  FAISS IVF:   {ivf_time*1000:>8.1f} ms")

# ============================================================
# 3. 真实文本的向量检索
# ============================================================
print("\n" + "=" * 50)
print("3. 真实文本向量检索")
print("=" * 50)

print("加载 Embedding 模型...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 模拟一个技术文档库
documents = [
    "Python 是一种解释型、面向对象的高级编程语言",
    "JavaScript 是 Web 开发中最常用的脚本语言",
    "Docker 是一个开源的容器化平台",
    "Kubernetes 用于自动化容器的部署和管理",
    "Git 是分布式版本控制系统",
    "Linux 是一个开源的操作系统内核",
    "MySQL 是最流行的关系型数据库之一",
    "Redis 是一个高性能的键值存储数据库",
    "Nginx 是一个高性能的 HTTP 和反向代理服务器",
    "TensorFlow 是 Google 开发的深度学习框架",
    "PyTorch 是 Facebook 开发的深度学习框架",
    "Transformer 是一种基于注意力机制的神经网络架构",
    "BERT 是 Google 提出的预训练语言模型",
    "GPT 是 OpenAI 开发的生成式预训练模型",
    "RAG 是检索增强生成技术，结合检索和生成",
    "向量数据库专门用于存储和检索高维向量",
    "LangChain 是一个用于构建 LLM 应用的框架",
    "Prompt Engineering 是设计有效提示词的技术",
    "Fine-tuning 是在预训练模型基础上进行微调",
    "RLHF 是基于人类反馈的强化学习方法",
    "AWS Lambda 是无服务器计算服务",
    "Amazon S3 是对象存储服务",
    "机器学习模型需要大量数据进行训练",
    "数据预处理是机器学习流程的重要步骤",
    "交叉验证用于评估模型的泛化能力",
]

# 编码所有文档
print(f"编码 {len(documents)} 篇文档...")
doc_embeddings = model.encode(documents).astype("float32")

# 构建 FAISS 索引
index = faiss.IndexFlatIP(doc_embeddings.shape[1])  # 内积（余弦相似度）
# 归一化向量，这样内积 = 余弦相似度
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)

print(f"索引构建完成，包含 {index.ntotal} 个向量")

# 搜索
queries = [
    "如何学习深度学习？",
    "数据库有哪些选择？",
    "怎么部署应用？",
    "大模型相关技术",
]

print("\n搜索结果:\n")
for query_text in queries:
    query_vec = model.encode([query_text]).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, 3)

    print(f"🔍 '{query_text}'")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        print(f"   Top{rank} [{score:.4f}]: {documents[idx]}")
    print()

# ============================================================
# 4. 索引类型选择指南
# ============================================================
print("=" * 50)
print("4. FAISS 索引类型选择指南")
print("=" * 50)

print("""
┌─────────────────┬──────────┬──────────┬──────────────┐
│ 索引类型        │ 速度     │ 精度     │ 适用场景     │
├─────────────────┼──────────┼──────────┼──────────────┤
│ IndexFlatL2     │ 慢       │ 100%     │ < 10万向量   │
│ IndexFlatIP     │ 慢       │ 100%     │ 余弦相似度   │
│ IndexIVFFlat    │ 快       │ ~95%     │ 10万-100万   │
│ IndexIVFPQ      │ 很快     │ ~90%     │ > 100万      │
│ IndexHNSWFlat   │ 很快     │ ~98%     │ 通用推荐     │
└─────────────────┴──────────┴──────────┴──────────────┘

实际项目中:
  - 小规模（< 10万）: 直接用 Flat，简单可靠
  - 中规模（10万-1000万）: 用 IVF 或 HNSW
  - 大规模（> 1000万）: 用 IVF + PQ 压缩
  - 或者直接用向量数据库（ChromaDB, Pinecone, Milvus）
""")

print("✅ 向量检索实战完成！")
print("   下一步: 学习 RAG，把检索和大模型生成结合起来！")
