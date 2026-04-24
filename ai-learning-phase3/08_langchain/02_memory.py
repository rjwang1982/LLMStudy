"""
LangChain - 记忆与多轮对话

作者: RJ.Wang
时间: 2026-04-22

学习目标:
  - 理解为什么 LLM 需要"记忆"
  - 实现多轮对话（带上下文）
  - 掌握不同的记忆策略
  - 构建一个有记忆的聊天机器人
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

OMLX_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8019/v1")
OMLX_API_KEY = os.getenv("OPENAI_API_KEY", "omlx")
OMLX_MODEL = os.getenv("OMLX_MODEL", "Qwen3.5-9B-MLX-4bit")

llm = ChatOpenAI(
    base_url=OMLX_BASE_URL, api_key=OMLX_API_KEY,
    model=OMLX_MODEL, temperature=0.7, max_tokens=500,
)

print("=" * 60)
print("LangChain 记忆与多轮对话")
print("=" * 60)

# ============================================================
# 1. 为什么需要记忆？
# ============================================================
print("\n" + "=" * 60)
print("1. LLM 的记忆问题")
print("=" * 60)

print("""
LLM 本身是"无状态"的 —— 每次调用都是独立的。

  用户: 我叫小明
  LLM:  你好小明！
  用户: 我叫什么？
  LLM:  我不知道你叫什么。  ← 忘了！

解决方案: 把历史对话一起发给 LLM
  用户: 我叫小明
  LLM:  你好小明！
  [把上面的对话历史也发过去]
  用户: 我叫什么？
  LLM:  你叫小明。  ← 记住了！
""")

# 演示无记忆的问题
print("--- 无记忆（每次独立调用）---")
r1 = llm.invoke("我叫小明，请记住我的名字")
print(f"  第1轮: {r1.content[:80]}")

r2 = llm.invoke("我叫什么名字？")
print(f"  第2轮: {r2.content[:80]}")
print("  → LLM 不记得了！\n")

# ============================================================
# 2. 手动管理对话历史
# ============================================================
print("=" * 60)
print("2. 手动管理对话历史")
print("=" * 60)

print("--- 手动传递历史消息 ---")
messages = [
    SystemMessage(content="你是一个友好的助手，请简洁回答。"),
    HumanMessage(content="我叫小明，我正在学习 AI"),
]

r1 = llm.invoke(messages)
print(f"  第1轮: {r1.content[:100]}")

# 把 LLM 的回复加入历史
messages.append(AIMessage(content=r1.content))
messages.append(HumanMessage(content="我叫什么？我在学什么？"))

r2 = llm.invoke(messages)
print(f"  第2轮: {r2.content[:100]}")
print("  → 这次记住了！因为我们把历史对话一起发了过去。")

# ============================================================
# 3. 使用 LangChain 的 MessageHistory
# ============================================================
print("\n" + "=" * 60)
print("3. LangChain MessageHistory 自动管理")
print("=" * 60)

# 创建带历史消息占位符的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的 AI 学习助手，回答简洁。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

# 存储不同会话的历史
session_store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


# 创建带记忆的链
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 模拟多轮对话
config = {"configurable": {"session_id": "user_001"}}

conversations = [
    "你好！我叫小明，我是一个 Python 初学者",
    "我想学习机器学习，你有什么建议？",
    "我叫什么？我的编程水平如何？",
]

print("\n多轮对话演示:")
for msg in conversations:
    print(f"\n  👤 用户: {msg}")
    response = chain_with_history.invoke(
        {"input": msg},
        config=config,
    )
    print(f"  🤖 助手: {response.content[:150]}")

# 查看存储的历史
history = get_session_history("user_001")
print(f"\n  📝 历史消息数: {len(history.messages)}")

# ============================================================
# 4. 多会话隔离
# ============================================================
print("\n" + "=" * 60)
print("4. 多会话隔离")
print("=" * 60)

# 用户 A 的会话
config_a = {"configurable": {"session_id": "user_A"}}
chain_with_history.invoke({"input": "我叫张三，我喜欢 Java"}, config=config_a)

# 用户 B 的会话
config_b = {"configurable": {"session_id": "user_B"}}
chain_with_history.invoke({"input": "我叫李四，我喜欢 Rust"}, config=config_b)

# 验证隔离
r_a = chain_with_history.invoke({"input": "我叫什么？喜欢什么语言？"}, config=config_a)
r_b = chain_with_history.invoke({"input": "我叫什么？喜欢什么语言？"}, config=config_b)

print(f"  用户A 的回答: {r_a.content[:100]}")
print(f"  用户B 的回答: {r_b.content[:100]}")
print("  → 不同会话的记忆是隔离的！")

# ============================================================
# 5. 记忆策略对比
# ============================================================
print("\n" + "=" * 60)
print("5. 记忆策略对比")
print("=" * 60)

print("""
┌─────────────────────┬──────────────────────────────────┐
│ 策略                │ 说明                             │
├─────────────────────┼──────────────────────────────────┤
│ 全量历史            │ 保留所有消息（简单但 token 多）  │
│ 窗口记忆            │ 只保留最近 N 轮（节省 token）    │
│ 摘要记忆            │ 用 LLM 总结历史（压缩信息）      │
│ 向量记忆            │ 把历史存向量库，按相关性检索      │
└─────────────────────┴──────────────────────────────────┘

选择建议:
  - 简单对话: 全量历史或窗口记忆
  - 长对话:   摘要记忆
  - 知识密集: 向量记忆（结合 RAG）
""")

# 演示窗口记忆：只保留最近 3 轮
print("--- 窗口记忆演示（最近 3 轮）---")

window_store = {}


def get_window_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in window_store:
        window_store[session_id] = InMemoryChatMessageHistory()
    history = window_store[session_id]
    # 只保留最近 6 条消息（3 轮 = 3 human + 3 ai）
    while len(history.messages) > 6:
        history.messages.pop(0)
    return history


window_chain = RunnableWithMessageHistory(
    chain, get_window_history,
    input_messages_key="input",
    history_messages_key="history",
)

config_w = {"configurable": {"session_id": "window_test"}}

msgs = [
    "我叫小红",
    "我喜欢 Python",
    "我在学深度学习",
    "我住在北京",
    "我叫什么？",  # 可能已经被窗口滑出去了
]

for msg in msgs:
    r = window_chain.invoke({"input": msg}, config=config_w)
    print(f"  👤 {msg}")
    print(f"  🤖 {r.content[:80]}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("💡 记忆与对话总结")
print("=" * 60)
print("""
1. LLM 本身无状态，需要手动传递历史
2. LangChain 的 RunnableWithMessageHistory 自动管理
3. session_id 实现多用户会话隔离
4. 窗口记忆控制 token 消耗
5. 记忆是 Agent 的基础能力之一
""")

print("✅ 记忆与多轮对话练习完成！")
