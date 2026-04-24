"""
综合项目 - 智能知识助手：RAG + Agent + Memory

作者: RJ.Wang
时间: 2026-04-22

学习目标:
  - 将阶段一到三的所有知识整合
  - 构建一个完整的智能知识助手
  - 具备：知识检索(RAG) + 工具调用(Agent) + 对话记忆(Memory)
  - 体验一个"真实"的 AI 应用是什么样的
"""

import os
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

OMLX_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8019/v1")
OMLX_API_KEY = os.getenv("OPENAI_API_KEY", "omlx")
OMLX_MODEL = os.getenv("OMLX_MODEL", "Qwen3.5-9B-MLX-4bit")

print("=" * 60)
print("🎓 智能知识助手")
print("RAG + Agent + Memory 综合项目")
print("=" * 60)

# ============================================================
# 1. 构建知识库（RAG 部分）
# ============================================================
print("\n📚 构建知识库...")

knowledge_docs = """
# AI 学习完整指南

## Python 基础
Python 是 AI 开发的首选语言。推荐使用 uv 作为包管理器，比 pip 更快。
核心库：NumPy（数值计算）、Pandas（数据处理）、Matplotlib（可视化）、PyTorch（深度学习）。
每个项目应该有独立的虚拟环境，使用 uv venv 创建。

## 神经网络
神经网络由神经元组成，核心操作是加权求和 + 激活函数。
训练过程：前向传播 → 计算损失 → 反向传播 → 更新权重。
常用激活函数：ReLU（最常用）、Sigmoid、Tanh。
PyTorch 是推荐的深度学习框架，自动求导让反向传播变得简单。

## Transformer
Transformer 的核心是 Self-Attention 机制。
关键组件：Q/K/V 投影、缩放点积注意力、多头注意力、位置编码、残差连接。
GPT（生成式）和 BERT（理解式）都基于 Transformer 架构。

## Embedding
Embedding 将离散的文本映射为连续的向量。语义相近的文本，向量也相近。
常用模型：all-MiniLM-L6-v2（384维，轻量）、text-embedding-3-small（OpenAI）。
余弦相似度用于衡量两个向量的方向一致性。

## RAG 技术
RAG = 检索增强生成。流程：文档分块 → 向量化 → 存入向量库 → 检索 → 构造 Prompt → LLM 生成。
优势：减少幻觉、访问最新知识、可追溯来源。
关键参数：chunk_size（建议 500-1000）、chunk_overlap（10-20%）。

## 模型压缩
量化：降低权重精度（FP32→INT8→INT4），模型更小更快。7B 模型 FP32=28GB，INT4=3.5GB。
蒸馏：大模型教小模型，通过软标签传递知识。温度参数控制软标签的"软度"。

## LangChain
LangChain 是构建 LLM 应用的框架。核心语法 LCEL：prompt | model | parser。
关键组件：ChatModel（连接 LLM）、PromptTemplate（模板）、Memory（记忆）、Tool（工具）。

## Agent
AI Agent = LLM + 工具 + 记忆 + 规划。
ReAct 模式：思考 → 行动 → 观察 → 循环。
Function Calling 让 LLM 能调用外部函数。工具描述的质量直接影响 Agent 决策。

## oLMX
oLMX 是 Apple Silicon 优化的本地 LLM 服务器。提供 OpenAI 兼容 API。
管理后台：http://127.0.0.1:8019/admin。支持多模型管理、KV 缓存、性能监控。
当前加载模型：Qwen3.5-4B（轻量快速）、Qwen3.5-9B（更强能力）。
""".strip()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
chunks = splitter.split_text(knowledge_docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(f"✅ 知识库构建完成，{len(chunks)} 个文档块")

# ============================================================
# 2. 定义工具集
# ============================================================
print("🔧 注册工具...")


@tool
def search_knowledge(query: str) -> str:
    """搜索 AI 学习知识库。可以查找 Python、神经网络、Transformer、RAG、LangChain、Agent 等相关知识。"""
    docs = retriever.invoke(query)
    if docs:
        return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    return "未找到相关知识"


@tool
def calculator(expression: str) -> str:
    """数学计算器。支持加减乘除、幂运算、开方、三角函数等。示例：'2**10', 'math.sqrt(144)'"""
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    now = datetime.now()
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]
    return now.strftime(f"%Y年%m月%d日 %H:%M 星期{weekdays[now.weekday()]}")


@tool
def save_study_note(topic: str, content: str) -> str:
    """保存学习笔记。topic 是主题，content 是内容。"""
    notes_dir = "10_project/study_notes"
    os.makedirs(notes_dir, exist_ok=True)
    filepath = os.path.join(notes_dir, f"{topic}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n")
        f.write(f"*{datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write(content + "\n")
    return f"笔记已保存: {filepath}"


@tool
def get_learning_progress() -> str:
    """查看学习进度，列出已完成的学习模块"""
    progress = {
        "阶段一 - 打基础": ["Python 基础 ✅", "神经网络 ✅", "Transformer ✅"],
        "阶段二 - 学核心": ["Embeddings ✅", "RAG ✅", "模型压缩 ✅", "多模态 ✅"],
        "阶段三 - 做应用": ["LangChain ✅", "Agent ✅", "综合项目 📌 进行中"],
    }
    result = "📊 学习进度:\n"
    for phase, modules in progress.items():
        result += f"\n{phase}:\n"
        for m in modules:
            result += f"  {m}\n"
    return result


tools = [search_knowledge, calculator, get_current_time, save_study_note, get_learning_progress]
print(f"✅ 已注册 {len(tools)} 个工具")

# ============================================================
# 3. 创建带记忆的 Agent
# ============================================================
print("🧠 创建智能助手...")

llm = ChatOpenAI(
    base_url=OMLX_BASE_URL, api_key=OMLX_API_KEY,
    model=OMLX_MODEL, temperature=0.7, max_tokens=800,
)

# 记忆：让 Agent 记住对话历史
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    prompt="""你是一个 AI 学习助手，名叫"小智"。你的职责是帮助用户学习 AI 相关知识。

你的能力：
1. 搜索知识库回答技术问题（优先使用 search_knowledge）
2. 数学计算
3. 查看当前时间
4. 保存学习笔记
5. 查看学习进度

回答规则：
- 用中文回答，简洁明了
- 技术问题优先从知识库检索，不要凭空编造
- 如果知识库没有相关信息，诚实说明
- 记住用户之前说过的话""",
)

print("✅ 智能知识助手创建完成！")

# ============================================================
# 4. 模拟多轮对话
# ============================================================
print("\n" + "=" * 60)
print("📱 智能知识助手 - 对话演示")
print("=" * 60)

# 使用固定的 thread_id 保持对话连续性
config = {"configurable": {"thread_id": "demo_session"}}


def chat(message: str):
    """与助手对话"""
    print(f"\n👤 用户: {message}")

    result = agent.invoke(
        {"messages": [("human", message)]},
        config=config,
    )

    # 找到最后一条 AI 消息
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            print(f"🤖 小智: {msg.content}")
            break

    # 显示工具调用（如果有）
    tool_calls = []
    for msg in result["messages"]:
        if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(tc["name"])
    if tool_calls:
        print(f"   [使用了工具: {', '.join(tool_calls)}]")


# 多轮对话演示
chat("你好！我叫小明，我刚学完阶段二的 RAG")

chat("帮我查一下 Transformer 的核心知识点")

chat("我叫什么？我学到哪了？")

chat("帮我查一下 Agent 是什么，然后保存成学习笔记")

chat("查看一下我的学习进度")

chat("2 的 20 次方是多少？")

# ============================================================
# 5. 架构总结
# ============================================================
print("\n\n" + "=" * 60)
print("🏗️ 智能知识助手架构")
print("=" * 60)

print("""
┌──────────────────────────────────────────────┐
│                 智能知识助手                    │
├──────────────────────────────────────────────┤
│                                              │
│  用户输入                                     │
│    ↓                                         │
│  Memory（对话记忆）                            │
│    ↓                                         │
│  LLM（oLMX Qwen3.5-9B）                      │
│    ↓                                         │
│  ReAct 决策：需要工具吗？                      │
│    ├─ 是 → 选择工具 → 执行 → 观察结果         │
│    │       ├─ search_knowledge (RAG)         │
│    │       ├─ calculator                     │
│    │       ├─ save_study_note                │
│    │       ├─ get_learning_progress          │
│    │       └─ get_current_time               │
│    └─ 否 → 直接回答                           │
│    ↓                                         │
│  生成最终回答                                  │
│                                              │
├──────────────────────────────────────────────┤
│  技术栈:                                      │
│  LangChain + LangGraph + ChromaDB + oLMX     │
└──────────────────────────────────────────────┘
""")

# ============================================================
# 6. 整个学习路线回顾
# ============================================================
print("=" * 60)
print("🎉 恭喜！整个学习路线完成！")
print("=" * 60)

print("""
你已经掌握了从零到 Agent 的完整技能链:

  阶段一（打基础）:
    Python → NumPy → 神经网络 → PyTorch → Transformer

  阶段二（学核心）:
    Embedding → RAG → 模型压缩 → 多模态

  阶段三（做应用）:
    LangChain → Function Calling → ReAct Agent → 综合项目

下一步建议:
  1. 用真实文档替换模拟数据，构建你自己的知识助手
  2. 尝试更多工具：网页搜索、数据库查询、代码执行
  3. 学习 LangGraph 构建更复杂的 Agent 工作流
  4. 部署你的应用：FastAPI + Docker
  5. 持续关注新模型和新技术
""")

print("✅ 综合项目完成！整个学习路线圆满结束！🎉")
