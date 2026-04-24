"""
RAG - 完整 RAG 管道（oLMX 本地 LLM 版）

作者: RJ.Wang
时间: 2026-04-22

学习目标:
  - 理解 RAG 的完整流程
  - 实现: 文档加载 → 分块 → 编码 → 存储 → 检索 → 生成
  - 使用 oLMX 本地 LLM 作为生成后端（无需云端 API Key）
  - 理解 RAG 如何减少大模型幻觉

前置要求:
  - oLMX 服务已启动: http://127.0.0.1:8019
  - 至少加载了一个模型（如 Qwen3.5-4B 或 Qwen3.5-9B）
  - 管理后台: http://127.0.0.1:8019/admin/dashboard
"""

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

# ============================================================
# 0. 配置 oLMX 连接
# ============================================================

# oLMX 提供 OpenAI 兼容 API，直接用 openai 库连接
OMLX_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8019/v1")
OMLX_API_KEY = os.getenv("OPENAI_API_KEY", "omlx")
# 默认使用 9B 模型（更强），可改为 Qwen3.5-4B-MLX-4bit（更快）
OMLX_MODEL = os.getenv("OMLX_MODEL", "Qwen3.5-9B-MLX-4bit")

llm_client = OpenAI(base_url=OMLX_BASE_URL, api_key=OMLX_API_KEY)

# ============================================================
# 1. RAG 架构概览
# ============================================================
print("=" * 60)
print("RAG (Retrieval-Augmented Generation) 完整管道")
print("使用 oLMX 本地 LLM 作为生成后端")
print("=" * 60)

print(f"""
配置信息:
  oLMX API:  {OMLX_BASE_URL}
  使用模型:  {OMLX_MODEL}

RAG 的核心流程:

  ┌─────────────────────────────────────────────┐
  │              离线阶段（索引构建）              │
  │                                             │
  │  文档 → 分块 → Embedding → 存入向量数据库    │
  └─────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────┐
  │              在线阶段（问答）                  │
  │                                             │
  │  用户提问 → Embedding → 检索相关文档          │
  │         → 构造 Prompt → oLMX 生成回答        │
  └─────────────────────────────────────────────┘
""")

# 验证 oLMX 连接
print("验证 oLMX 连接...")
try:
    models = llm_client.models.list()
    available = [m.id for m in models.data]
    print(f"✅ oLMX 连接成功！可用模型: {available}")
    if OMLX_MODEL not in available:
        print(f"⚠️  指定模型 {OMLX_MODEL} 不在列表中，将使用第一个可用模型")
        OMLX_MODEL = available[0] if available else OMLX_MODEL
        print(f"   切换到: {OMLX_MODEL}")
except Exception as e:
    print(f"❌ oLMX 连接失败: {e}")
    print("   请确保 oLMX 已启动: http://127.0.0.1:8019/admin/dashboard")
    print("   将使用模拟生成器继续演示...")
    llm_client = None

# ============================================================
# 2. 准备知识文档
# ============================================================
print("\n" + "=" * 60)
print("Step 1: 准备知识文档")
print("=" * 60)

knowledge_document = """
# AI 学习指南

## 第一章：Python 基础

Python 是学习 AI 的首选编程语言。它语法简洁，拥有丰富的科学计算库。

学习 Python 的推荐路径：首先掌握基本语法，包括变量、函数、类和模块。然后学习 NumPy 进行数值计算，Pandas 进行数据处理，Matplotlib 进行数据可视化。

Python 的包管理工具推荐使用 uv，它比 pip 更快更可靠。虚拟环境建议使用 uv venv 创建，每个项目独立一个环境。

## 第二章：机器学习基础

机器学习分为三大类：监督学习、无监督学习和强化学习。

监督学习需要标注数据，常见算法包括线性回归、逻辑回归、决策树、随机森林和支持向量机。评估指标包括准确率、精确率、召回率和 F1 分数。

无监督学习不需要标注数据，常见方法包括 K-Means 聚类、PCA 降维和自编码器。

强化学习通过与环境交互来学习，核心概念包括状态、动作、奖励和策略。AlphaGo 就是强化学习的经典应用。

## 第三章：深度学习

深度学习使用多层神经网络处理复杂数据。PyTorch 和 TensorFlow 是两大主流框架，推荐初学者使用 PyTorch。

卷积神经网络（CNN）擅长处理图像，经典模型包括 ResNet、VGG 和 EfficientNet。

循环神经网络（RNN）和 LSTM 适合处理序列数据，但现在已经被 Transformer 架构取代。

Transformer 是现代 NLP 的基础架构，核心是自注意力机制。BERT 用于文本理解，GPT 用于文本生成。

## 第四章：大语言模型

大语言模型（LLM）是基于 Transformer 的超大规模模型。GPT-4、Claude、LLaMA 是代表性模型。

使用大模型的方式：API 调用（最简单）、微调（Fine-tuning）、RAG（检索增强生成）。

Prompt Engineering 是与大模型交互的关键技能。好的提示词可以显著提升模型输出质量。常用技巧包括：角色设定、少样本示例、思维链（Chain of Thought）。

## 第五章：RAG 技术

RAG（Retrieval-Augmented Generation）结合了检索和生成两种能力。

RAG 的优势：减少幻觉、访问最新知识、可追溯来源、无需重新训练模型。

RAG 的核心组件：文本分块器、Embedding 模型、向量数据库、大语言模型。

构建 RAG 系统的步骤：1）收集和清洗文档 2）文本分块 3）生成嵌入向量 4）存入向量数据库 5）实现检索逻辑 6）构造 Prompt 模板 7）调用 LLM 生成回答。

常见的向量数据库：ChromaDB（轻量级）、Pinecone（云服务）、Milvus（分布式）、Weaviate（全功能）。

## 第六章：部署与实践

模型部署方式：Flask/FastAPI 构建 API、Docker 容器化、云服务（AWS SageMaker、Azure ML）。

MLOps 是机器学习的运维实践，包括模型版本管理、自动化训练、监控和 A/B 测试。

推荐的学习项目：1）手写数字识别 2）文本情感分析 3）基于 RAG 的问答系统 4）AI Agent 开发。
""".strip()

print(f"文档长度: {len(knowledge_document)} 字符")

# ============================================================
# 3. 文本分块
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 文本分块")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)

chunks = splitter.split_text(knowledge_document)
print(f"分块数: {len(chunks)}")
print(f"平均块长: {sum(len(c) for c in chunks) / len(chunks):.0f} 字符")

for i, chunk in enumerate(chunks[:3]):
    print(f"\n  Chunk {i+1} ({len(chunk)} 字符): {chunk[:80]}...")

# ============================================================
# 4. 编码并存入向量数据库
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 编码 + 存入向量数据库")
print("=" * 60)

print("加载 Embedding 模型...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection(name="ai_guide")

print("编码文档块...")
embeddings = embed_model.encode(chunks).tolist()

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"chunk_index": i, "length": len(c)} for i, c in enumerate(chunks)],
)

print(f"✅ 已存入 {collection.count()} 个文档块")

# ============================================================
# 5. 检索
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 检索相关文档")
print("=" * 60)


def retrieve(query: str, n_results: int = 3) -> list[str]:
    """检索与查询最相关的文档块"""
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )
    return results["documents"][0]


test_query = "如何学习深度学习？"
retrieved_docs = retrieve(test_query)

print(f"查询: '{test_query}'")
print(f"检索到 {len(retrieved_docs)} 个相关文档块:\n")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"  📄 文档 {i}: {doc[:100]}...")
    print()

# ============================================================
# 6. 构造 Prompt + oLMX 生成
# ============================================================
print("=" * 60)
print("Step 5: 构造 Prompt + oLMX 生成回答")
print("=" * 60)

PROMPT_TEMPLATE = """你是一个 AI 学习助手。请根据以下参考资料回答用户的问题。
要求：简洁明了，直接回答，不要重复参考资料原文。
如果参考资料中没有相关信息，请诚实地说"我不确定"。

参考资料:
{context}

用户问题: {question}"""


def build_prompt(question: str, context_docs: list[str]) -> str:
    """构造 RAG Prompt"""
    context = "\n\n".join(f"[文档{i+1}] {doc}" for i, doc in enumerate(context_docs))
    return PROMPT_TEMPLATE.format(context=context, question=question)


def llm_generate(prompt: str) -> str:
    """调用 oLMX 本地 LLM 生成回答

    oLMX 提供 OpenAI 兼容 API，所以直接用 openai 库调用。
    这意味着你的代码可以无缝切换到 OpenAI、DeepSeek 等云端 API！
    """
    if llm_client is None:
        return _fallback_generate(prompt)

    try:
        response = llm_client.chat.completions.create(
            model=OMLX_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  ⚠️ oLMX 调用失败: {e}")
        return _fallback_generate(prompt)


def _fallback_generate(prompt: str) -> str:
    """备用：简单的关键词匹配生成（oLMX 不可用时使用）"""
    context_start = prompt.find("参考资料:") + len("参考资料:")
    context_end = prompt.find("用户问题:")
    context = prompt[context_start:context_end].strip()

    sentences = [s.strip() + "。" for s in context.split("。") if len(s.strip()) > 10]
    response = "（oLMX 不可用，使用备用生成器）\n\n"
    for s in sentences[:3]:
        s = s.replace("[文档1] ", "").replace("[文档2] ", "").replace("[文档3] ", "")
        response += f"• {s}\n"
    return response


# ============================================================
# 7. 完整的 RAG 问答
# ============================================================
print("\n" + "=" * 60)
print("Step 6: 完整 RAG 问答演示")
print("=" * 60)


def rag_answer(question: str) -> str:
    """完整的 RAG 问答流程"""
    print(f"\n{'─' * 50}")
    print(f"❓ 问题: {question}")

    # Step 1: 检索
    docs = retrieve(question, n_results=3)
    print(f"📚 检索到 {len(docs)} 个相关文档块")

    # Step 2: 构造 Prompt
    prompt = build_prompt(question, docs)

    # Step 3: 调用 oLMX 生成回答
    print(f"🤖 调用 oLMX ({OMLX_MODEL}) 生成中...")
    answer = llm_generate(prompt)
    print(f"\n💬 回答:\n{answer}")

    return answer


# 测试多个问题
questions = [
    "Python 学习应该从哪里开始？",
    "什么是 RAG？它有什么优势？",
    "推荐什么深度学习框架？",
]

for q in questions:
    rag_answer(q)

# ============================================================
# 8. 对比实验：有 RAG vs 无 RAG
# ============================================================
print("\n\n" + "=" * 60)
print("🔬 对比实验：有 RAG vs 无 RAG")
print("=" * 60)

comparison_question = "学习 AI 推荐用什么包管理工具？"

print(f"\n问题: '{comparison_question}'")

# 无 RAG：直接问 LLM
print(f"\n--- 无 RAG（直接问 LLM）---")
if llm_client:
    try:
        direct_response = llm_client.chat.completions.create(
            model=OMLX_MODEL,
            messages=[{"role": "user", "content": comparison_question}],
            temperature=0.7,
            max_tokens=300,
        )
        print(direct_response.choices[0].message.content)
    except Exception as e:
        print(f"调用失败: {e}")
else:
    print("（oLMX 不可用）")

# 有 RAG：检索后再问
print(f"\n--- 有 RAG（检索增强）---")
rag_answer(comparison_question)

print("""
💡 对比观察:
  - 无 RAG: LLM 可能给出通用回答（pip、conda 等）
  - 有 RAG: 基于知识库，准确回答 "uv"（因为文档里写了推荐 uv）
  - 这就是 RAG 的价值：让 LLM 基于你的私有知识回答！
""")

# ============================================================
# 9. 切换到云端 API（可选）
# ============================================================
print("=" * 60)
print("🔧 进阶：切换到云端 API")
print("=" * 60)

print("""
本练习默认使用 oLMX 本地 LLM，代码完全兼容 OpenAI API。
切换到云端只需修改环境变量：

  # 切换到 OpenAI
  export OPENAI_BASE_URL="https://api.openai.com/v1"
  export OPENAI_API_KEY="sk-..."
  export OMLX_MODEL="gpt-4o-mini"

  # 切换到 DeepSeek
  export OPENAI_BASE_URL="https://api.deepseek.com/v1"
  export OPENAI_API_KEY="sk-..."
  export OMLX_MODEL="deepseek-chat"

  # 切回 oLMX 本地
  unset OPENAI_BASE_URL OPENAI_API_KEY OMLX_MODEL

代码不需要改一行！这就是 OpenAI 兼容 API 的好处。
""")

# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("💡 RAG 管道总结")
print("=" * 60)
print(f"""
完整 RAG 流程:

  1. 文档加载    → 读取各种格式的文档
  2. 文本分块    → RecursiveCharacterTextSplitter
  3. 向量编码    → SentenceTransformer (all-MiniLM-L6-v2)
  4. 向量存储    → ChromaDB
  5. 语义检索    → 根据用户问题检索 Top-K 文档
  6. Prompt 构造 → 将检索结果和问题组合成提示词
  7. LLM 生成    → oLMX 本地 ({OMLX_MODEL})

RAG vs 纯 LLM:
  ┌──────────┬──────────────┬──────────────┐
  │          │ 纯 LLM      │ RAG          │
  ├──────────┼──────────────┼──────────────┤
  │ 知识范围 │ 训练数据截止 │ 可访问最新   │
  │ 幻觉     │ 容易产生     │ 有据可查     │
  │ 可追溯   │ 无法追溯     │ 可引用来源   │
  │ 成本     │ 微调很贵     │ 无需重训练   │
  └──────────┴──────────────┴──────────────┘
""")

print("✅ RAG 完整管道练习完成！")
