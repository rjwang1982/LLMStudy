"""
Agent - 多工具 Agent：搜索/计算/文件

作者: RJ.Wang
时间: 2026-04-22

学习目标:
  - 构建一个拥有多种能力的 Agent
  - 实现文件读写、知识搜索、计算等工具
  - 让 Agent 自主组合多个工具解决复杂问题
  - 理解工具描述对 Agent 决策的影响
"""

import os
import json
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

OMLX_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8019/v1")
OMLX_API_KEY = os.getenv("OPENAI_API_KEY", "omlx")
OMLX_MODEL = os.getenv("OMLX_MODEL", "Qwen3.5-9B-MLX-4bit")

print("=" * 60)
print("多工具 Agent")
print("=" * 60)

# ============================================================
# 1. 定义丰富的工具集
# ============================================================
print("\n定义工具集...")

# --- 信息类工具 ---

@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S 星期") + ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]


@tool
def search_tech_docs(query: str) -> str:
    """搜索技术文档知识库。可以查找编程语言、框架、AI 技术等相关知识。"""
    docs = {
        "python": "Python 是解释型语言，语法简洁。常用库：NumPy(数值计算)、Pandas(数据处理)、PyTorch(深度学习)。包管理推荐 uv。",
        "langchain": "LangChain 是构建 LLM 应用的框架。核心组件：Model、Prompt、Chain、Memory、Tool、Agent。使用 LCEL 语法：prompt | model | parser。",
        "rag": "RAG(检索增强生成)流程：文档分块→向量化→存入向量库→检索→构造Prompt→LLM生成。优势：减少幻觉、访问最新知识。",
        "transformer": "Transformer 核心是 Self-Attention。结构：Embedding→位置编码→[Multi-Head Attention + FFN]×N→输出。GPT/BERT/LLaMA 都基于此。",
        "agent": "AI Agent = LLM + 工具 + 记忆 + 规划。ReAct模式：思考→行动→观察→循环。可以自主调用工具解决复杂问题。",
        "omlx": "oLMX 是 Apple Silicon 优化的本地 LLM 服务器。支持 OpenAI 兼容 API、多模型管理、KV 缓存、管理后台。端口 8019。",
        "docker": "Docker 是容器化平台。核心概念：镜像(Image)、容器(Container)、Dockerfile、docker-compose。用于应用打包和部署。",
        "git": "Git 是分布式版本控制。常用命令：clone、add、commit、push、pull、branch、merge。GitHub 是代码托管平台。",
    }
    results = []
    for key, value in docs.items():
        if key in query.lower() or any(word in value.lower() for word in query.lower().split()):
            results.append(f"[{key}] {value}")
    return "\n".join(results) if results else "未找到相关文档"


# --- 计算类工具 ---

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持：加减乘除、幂运算(2**10)、开方(math.sqrt)、三角函数(math.sin)等。"""
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# --- 文件类工具 ---

NOTES_DIR = "09_agent/notes"
os.makedirs(NOTES_DIR, exist_ok=True)


@tool
def save_note(title: str, content: str) -> str:
    """保存一条笔记到文件。title 是笔记标题（用作文件名），content 是笔记内容。"""
    filepath = os.path.join(NOTES_DIR, f"{title}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"*创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write(content)
    return f"笔记已保存到 {filepath}"


@tool
def read_note(title: str) -> str:
    """读取一条已保存的笔记。title 是笔记标题。"""
    filepath = os.path.join(NOTES_DIR, f"{title}.md")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return f"未找到笔记: {title}"


@tool
def list_notes() -> str:
    """列出所有已保存的笔记标题"""
    notes = [f.replace(".md", "") for f in os.listdir(NOTES_DIR) if f.endswith(".md")]
    if notes:
        return "已保存的笔记:\n" + "\n".join(f"  - {n}" for n in notes)
    return "暂无笔记"


# --- 汇总 ---

all_tools = [
    get_current_time,
    search_tech_docs,
    calculator,
    save_note,
    read_note,
    list_notes,
]

print(f"✅ 已注册 {len(all_tools)} 个工具:")
for t in all_tools:
    print(f"  🔧 {t.name}: {t.description[:40]}...")

# ============================================================
# 2. 创建多工具 Agent
# ============================================================
print("\n" + "=" * 60)
print("2. 创建多工具 Agent")
print("=" * 60)

llm = ChatOpenAI(
    base_url=OMLX_BASE_URL, api_key=OMLX_API_KEY,
    model=OMLX_MODEL, temperature=0,
)

agent = create_react_agent(
    model=llm,
    tools=all_tools,
    prompt="""你是一个全能 AI 助手，拥有以下能力：
- 查询当前时间
- 搜索技术文档
- 数学计算
- 保存和读取笔记

请根据用户需求自主选择合适的工具。用中文简洁回答。
如果需要多步操作，请逐步完成。""",
)

print("✅ 多工具 Agent 创建完成！")


def ask(question: str):
    """向 Agent 提问"""
    print(f"\n{'─' * 50}")
    print(f"❓ {question}")

    result = agent.invoke({"messages": [("human", question)]})

    for msg in result["messages"]:
        if msg.type == "human":
            continue
        elif msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = json.dumps(tc["args"], ensure_ascii=False)
                print(f"  🔧 {tc['name']}({args_str})")
        elif msg.type == "tool":
            content = msg.content[:120] + "..." if len(msg.content) > 120 else msg.content
            print(f"  📋 {content}")
        elif msg.type == "ai" and msg.content:
            print(f"  💬 {msg.content}")


# ============================================================
# 3. 测试各种场景
# ============================================================
print("\n" + "=" * 60)
print("3. 测试各种场景")
print("=" * 60)

# 信息查询
ask("现在几点了？")

# 知识搜索
ask("帮我查一下 LangChain 是什么")

# 计算
ask("计算圆周率的平方，保留小数")

# 笔记功能：保存
ask("帮我保存一条笔记，标题是'学习计划'，内容是'今天学习 LangChain Agent，明天学习 RAG 优化'")

# 笔记功能：读取
ask("读一下我的'学习计划'笔记")

# 复合任务
ask("帮我查一下 RAG 技术的资料，然后保存成笔记，标题叫'RAG笔记'")

# ============================================================
# 4. 工具描述的重要性
# ============================================================
print("\n\n" + "=" * 60)
print("4. 工具描述的重要性")
print("=" * 60)

print("""
Agent 选择工具的依据是工具的 description！

好的描述:
  "搜索技术文档知识库。可以查找编程语言、框架、AI 技术等相关知识。"
  → Agent 知道什么时候该用这个工具

差的描述:
  "搜索"
  → Agent 不确定这个工具能搜什么

最佳实践:
  1. 描述要说明工具的用途和适用场景
  2. 说明参数的格式和示例
  3. 说明返回值的格式
  4. 避免模糊的描述
""")

# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("💡 多工具 Agent 总结")
print("=" * 60)
print("""
本练习构建了一个拥有 6 个工具的 Agent:
  - 时间查询、知识搜索、数学计算
  - 笔记保存、读取、列表

关键收获:
  1. Agent 能自主选择合适的工具
  2. Agent 能组合多个工具完成复杂任务
  3. 工具描述直接影响 Agent 的决策质量
  4. create_react_agent 让创建 Agent 变得简单

这就是 AI Agent 的核心模式！
""")

print("✅ 多工具 Agent 练习完成！")
