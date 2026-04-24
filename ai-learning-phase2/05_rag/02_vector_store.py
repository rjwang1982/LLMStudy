"""
RAG - 向量数据库：ChromaDB 实战

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 使用 ChromaDB 存储和检索向量
  - 理解向量数据库的核心操作：增删改查
  - 学会使用元数据过滤
  - 构建一个简单的知识库
"""

import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. ChromaDB 简介
# ============================================================
print("=" * 50)
print("1. ChromaDB - 轻量级向量数据库")
print("=" * 50)

print("""
ChromaDB 特点:
  - 开源、轻量、易用
  - 内置 Embedding 支持
  - 支持元数据过滤
  - 支持持久化存储
  - 非常适合学习和原型开发

核心概念:
  - Collection: 类似数据库的表
  - Document: 原始文本
  - Embedding: 文本的向量表示
  - Metadata: 附加信息（来源、日期等）
  - ID: 唯一标识符
""")

# ============================================================
# 2. 创建集合并添加数据
# ============================================================
print("=" * 50)
print("2. 创建知识库")
print("=" * 50)

# 初始化 ChromaDB（内存模式，不持久化）
client = chromadb.Client()

# 创建一个集合
collection = client.create_collection(
    name="tech_knowledge",
    metadata={"description": "技术知识库"},
)

# 准备知识数据
documents = [
    "Python 是一种解释型编程语言，以简洁的语法著称。它广泛用于 Web 开发、数据分析、人工智能等领域。",
    "JavaScript 是 Web 前端开发的核心语言。Node.js 让 JavaScript 也能用于后端开发。",
    "Docker 是一个容器化平台，可以将应用及其依赖打包成容器，实现一致的运行环境。",
    "Kubernetes（K8s）是容器编排平台，用于自动化部署、扩展和管理容器化应用。",
    "Git 是分布式版本控制系统，GitHub 是基于 Git 的代码托管平台。",
    "MySQL 是开源的关系型数据库，使用 SQL 语言进行数据操作。",
    "Redis 是内存键值数据库，常用于缓存、会话管理和消息队列。",
    "Transformer 是一种基于自注意力机制的神经网络架构，是 GPT 和 BERT 的基础。",
    "RAG 检索增强生成技术，通过检索外部知识来增强大模型的回答质量。",
    "LangChain 是构建大模型应用的框架，提供了链式调用、记忆管理等功能。",
    "向量数据库专门用于存储高维向量，支持快速的相似度搜索。ChromaDB 和 Pinecone 是常见选择。",
    "微服务架构将应用拆分为多个小型服务，每个服务独立部署和扩展。",
    "RESTful API 是一种 Web API 设计风格，使用 HTTP 方法进行资源操作。",
    "GraphQL 是 Facebook 开发的 API 查询语言，允许客户端精确指定需要的数据。",
    "CI/CD 持续集成和持续部署，自动化代码构建、测试和部署流程。",
]

# 元数据
metadatas = [
    {"category": "编程语言", "difficulty": "入门"},
    {"category": "编程语言", "difficulty": "入门"},
    {"category": "DevOps", "difficulty": "中级"},
    {"category": "DevOps", "difficulty": "高级"},
    {"category": "工具", "difficulty": "入门"},
    {"category": "数据库", "difficulty": "入门"},
    {"category": "数据库", "difficulty": "中级"},
    {"category": "AI", "difficulty": "中级"},
    {"category": "AI", "difficulty": "中级"},
    {"category": "AI", "difficulty": "中级"},
    {"category": "AI", "difficulty": "中级"},
    {"category": "架构", "difficulty": "高级"},
    {"category": "架构", "difficulty": "中级"},
    {"category": "架构", "difficulty": "中级"},
    {"category": "DevOps", "difficulty": "中级"},
]

ids = [f"doc_{i}" for i in range(len(documents))]

# 添加到集合（ChromaDB 会自动生成 Embedding）
print("添加文档到知识库...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
)

print(f"✅ 知识库创建完成，共 {collection.count()} 篇文档")

# ============================================================
# 3. 语义搜索
# ============================================================
print("\n" + "=" * 50)
print("3. 语义搜索")
print("=" * 50)

queries = [
    "如何学习编程？",
    "怎么部署应用到服务器？",
    "大模型相关的技术有哪些？",
    "数据存储方案",
]

for query in queries:
    results = collection.query(
        query_texts=[query],
        n_results=3,
    )

    print(f"\n🔍 '{query}'")
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        preview = doc[:60] + "..." if len(doc) > 60 else doc
        print(f"   [{dist:.4f}] [{meta['category']}] {preview}")

# ============================================================
# 4. 元数据过滤
# ============================================================
print("\n" + "=" * 50)
print("4. 元数据过滤 - 精确控制搜索范围")
print("=" * 50)

# 只搜索 AI 类别的文档
print("\n只搜索 AI 类别:")
results = collection.query(
    query_texts=["深度学习框架"],
    n_results=3,
    where={"category": "AI"},
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    preview = doc[:60] + "..."
    print(f"  [{meta['category']}] {preview}")

# 只搜索入门难度的文档
print("\n只搜索入门难度:")
results = collection.query(
    query_texts=["推荐学习什么技术？"],
    n_results=3,
    where={"difficulty": "入门"},
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    preview = doc[:60] + "..."
    print(f"  [{meta['difficulty']}] {preview}")

# 组合过滤
print("\n组合过滤（AI + 中级）:")
results = collection.query(
    query_texts=["AI 应用开发"],
    n_results=3,
    where={"$and": [{"category": "AI"}, {"difficulty": "中级"}]},
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    preview = doc[:60] + "..."
    print(f"  [{meta['category']}/{meta['difficulty']}] {preview}")

# ============================================================
# 5. 更新和删除
# ============================================================
print("\n" + "=" * 50)
print("5. 更新和删除操作")
print("=" * 50)

# 更新文档
collection.update(
    ids=["doc_0"],
    documents=["Python 是最流行的编程语言之一，2026 年最新版本是 3.13。它以简洁优雅的语法著称。"],
    metadatas=[{"category": "编程语言", "difficulty": "入门", "updated": "true"}],
)
print("✅ 更新了 doc_0")

# 验证更新
result = collection.get(ids=["doc_0"])
print(f"   更新后: {result['documents'][0][:50]}...")

# 删除文档
collection.delete(ids=["doc_14"])
print(f"✅ 删除了 doc_14，剩余 {collection.count()} 篇文档")

# ============================================================
# 6. 持久化存储
# ============================================================
print("\n" + "=" * 50)
print("6. 持久化存储")
print("=" * 50)

print("""
内存模式（当前使用）:
  client = chromadb.Client()
  → 程序结束数据就没了

持久化模式:
  client = chromadb.PersistentClient(path="./chroma_db")
  → 数据保存到磁盘，下次启动还在

生产环境:
  client = chromadb.HttpClient(host="localhost", port=8000)
  → 连接独立运行的 ChromaDB 服务
""")

# 演示持久化
persistent_client = chromadb.PersistentClient(path="./05_rag/chroma_data")
persistent_collection = persistent_client.get_or_create_collection("demo")
persistent_collection.add(
    documents=["这是一条持久化的测试数据"],
    ids=["persistent_1"],
)
print(f"✅ 持久化存储测试完成，数据保存在 ./05_rag/chroma_data/")
print(f"   集合中有 {persistent_collection.count()} 条数据")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 50)
print("💡 ChromaDB 核心操作总结")
print("=" * 50)
print("""
创建: collection.add(documents, metadatas, ids)
查询: collection.query(query_texts, n_results, where)
获取: collection.get(ids)
更新: collection.update(ids, documents, metadatas)
删除: collection.delete(ids)

元数据过滤:
  where={"key": "value"}           # 精确匹配
  where={"key": {"$gt": 5}}        # 大于
  where={"$and": [{...}, {...}]}   # 组合条件
""")

print("✅ ChromaDB 向量数据库实战完成！")
print("   下一步: 构建完整的 RAG 管道！")
