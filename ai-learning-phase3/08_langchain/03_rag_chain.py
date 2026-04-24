"""
LangChain - LangChain 版 RAG

作者: RJ.Wang
时间: 2026-04-22

学习目标:
  - 用 LangChain 重构阶段二的 RAG 管道
  - 体验框架带来的简洁性
  - 理解 Retriever + Chain 的组合模式
  - 对比手写 RAG vs LangChain RAG
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

OMLX_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8019/v1")
OMLX_API_KEY = os.getenv("OPENAI_API_KEY", "omlx")
OMLX_MODEL = os.getenv("OMLX_MODEL", "Qwen3.5-9B-MLX-4bit")

print("=" * 60)
print("LangChain 版 RAG 管道")
print("=" * 60)

# ============================================================
# 1. 对比：手写 RAG vs LangChain RAG
# ============================================================
print("""
阶段二手写 RAG 的步骤:
  1. 手动加载文档
  2. 手动调用 text_splitter.split_text()
  3. 手动调用 model.encode() 生成向量
  4. 手动存入 ChromaDB
  5. 手动编码查询 + 检索
  6. 手动拼接 Prompt
  7. 手动调用 LLM

LangChain RAG:
  用几行代码把上面全部串起来！
""")

# ============================================================
# 2. 准备知识文档
# ============================================================
print("=" * 60)
print("Step 1: 准备文档 + 分块")
print("=" * 60)

document = """
# oLMX 使用指南

## 什么是 oLMX
oLMX 是一个专为 Apple Silicon 优化的本地 LLM 推理服务器。它支持连续批处理和分层 KV 缓存，可以从 macOS 菜单栏直接管理。

## 安装方式
macOS App: 从 GitHub Releases 下载 .dmg 文件，拖入 Applications 即可。
Homebrew: 运行 brew tap jundot/omlx 然后 brew install omlx。
从源码: git clone 后 pip install -e . 安装。

## API 兼容性
oLMX 提供 OpenAI 兼容 API，端点为 http://127.0.0.1:8019/v1。
支持的端点包括: /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/models。
任何兼容 OpenAI API 的客户端都可以直接连接。

## 模型管理
oLMX 支持多模型同时加载。模型通过 LRU 策略自动管理内存。
可以在管理后台 /admin 中手动加载、卸载和固定模型。
支持的模型类型: LLM、VLM（视觉语言模型）、Embedding、Reranker。

## KV 缓存
oLMX 使用两层 KV 缓存: 热层（内存）和冷层（SSD）。
热层存放频繁访问的缓存块，冷层在内存不足时将缓存写入 SSD。
即使服务器重启，SSD 缓存仍然有效，无需重新计算。

## 管理后台
访问 http://127.0.0.1:8019/admin 可以看到实时监控面板。
功能包括: 模型管理、聊天测试、性能基准测试、日志查看。
支持中文、英文、日文、韩文界面。

## 与 Claude Code 集成
oLMX 支持上下文缩放，让小上下文模型也能配合 Claude Code 使用。
SSE keep-alive 机制防止长时间预填充时的超时问题。

## 性能优化
oLMX 支持 SSD 缓存预填充加速，对于重复的长上下文请求效果显著。
缓存命中时可以跳过重新计算，直接从磁盘恢复 KV 状态。
""".strip()

# 分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
chunks = splitter.split_text(document)
print(f"文档分成 {len(chunks)} 个块")

# ============================================================
# 3. 创建向量存储 + 检索器
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 创建向量存储")
print("=" * 60)

# 使用 HuggingFace 的 Embedding 模型
print("加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 一行代码：分块文本 → 向量化 → 存入 ChromaDB
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="omlx_guide",
)

# 创建检索器（默认返回 Top 3）
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(f"✅ 向量存储创建完成，包含 {len(chunks)} 个文档块")

# 测试检索
test_results = retriever.invoke("如何安装 oLMX？")
print(f"\n测试检索 '如何安装 oLMX？':")
for i, doc in enumerate(test_results, 1):
    print(f"  文档{i}: {doc.page_content[:80]}...")

# ============================================================
# 4. 构建 RAG 链
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 构建 RAG 链")
print("=" * 60)

llm = ChatOpenAI(
    base_url=OMLX_BASE_URL, api_key=OMLX_API_KEY,
    model=OMLX_MODEL, temperature=0.7, max_tokens=500,
)

# RAG Prompt 模板
rag_prompt = ChatPromptTemplate.from_template("""根据以下参考资料回答问题。简洁明了，不要重复原文。
如果资料中没有相关信息，请说"我不确定"。

参考资料:
{context}

问题: {question}""")


def format_docs(docs):
    """将检索到的文档格式化为字符串"""
    return "\n\n".join(doc.page_content for doc in docs)


# 核心：用 LCEL 构建 RAG 链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG 链构建完成！")
print("""
链的数据流:
  用户问题
    ↓
  retriever → 检索相关文档 → format_docs → context
    ↓
  rag_prompt → 填充模板
    ↓
  llm → 生成回答
    ↓
  StrOutputParser → 提取文本
""")

# ============================================================
# 5. 测试 RAG 问答
# ============================================================
print("=" * 60)
print("Step 4: RAG 问答测试")
print("=" * 60)

questions = [
    "oLMX 是什么？",
    "怎么安装 oLMX？",
    "oLMX 支持哪些模型类型？",
    "KV 缓存是怎么工作的？",
    "管理后台有什么功能？",
]

for q in questions:
    print(f"\n❓ {q}")
    answer = rag_chain.invoke(q)
    print(f"💬 {answer}")

# ============================================================
# 6. 对比：手写 vs LangChain
# ============================================================
print("\n\n" + "=" * 60)
print("📊 代码量对比")
print("=" * 60)

print("""
手写 RAG（阶段二）:
  - embed_model = SentenceTransformer(...)
  - embeddings = embed_model.encode(chunks)
  - collection.add(documents=..., embeddings=...)
  - query_emb = embed_model.encode([query])
  - results = collection.query(query_embeddings=...)
  - prompt = TEMPLATE.format(context=..., question=...)
  - response = client.chat.completions.create(...)
  → 约 30 行核心代码

LangChain RAG（本练习）:
  - vectorstore = Chroma.from_texts(texts, embeddings)
  - retriever = vectorstore.as_retriever()
  - chain = {"context": retriever | format_docs, ...} | prompt | llm | parser
  - answer = chain.invoke(question)
  → 约 5 行核心代码

LangChain 的价值:
  不是"更强"，而是"更快搭建原型"。
  理解底层原理（阶段二）+ 高效开发（阶段三）= 完整能力。
""")

print("✅ LangChain RAG 练习完成！")
