"""
LangChain - 基础概念：Model + Prompt + Chain

作者: RJ.Wang
时间: 2026-04-22

学习目标:
  - 理解 LangChain 的核心组件
  - 学会使用 ChatModel 连接 oLMX
  - 掌握 PromptTemplate 模板化提示词
  - 理解 LCEL (LangChain Expression Language) 链式语法
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# 0. 配置 oLMX 连接
# ============================================================

OMLX_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8019/v1")
OMLX_API_KEY = os.getenv("OPENAI_API_KEY", "omlx")
OMLX_MODEL = os.getenv("OMLX_MODEL", "Qwen3.5-9B-MLX-4bit")

print("=" * 60)
print("LangChain 基础：Model + Prompt + Chain")
print(f"使用模型: {OMLX_MODEL} (oLMX 本地)")
print("=" * 60)

# ============================================================
# 1. ChatModel - 连接 LLM
# ============================================================
print("\n" + "=" * 60)
print("1. ChatModel - 连接 LLM")
print("=" * 60)

print("""
LangChain 的 ChatModel 是对 LLM API 的封装。
因为 oLMX 兼容 OpenAI API，直接用 ChatOpenAI 就行。
""")

# 创建 ChatModel（连接 oLMX）
llm = ChatOpenAI(
    base_url=OMLX_BASE_URL,
    api_key=OMLX_API_KEY,
    model=OMLX_MODEL,
    temperature=0.7,
    max_tokens=500,
)

# 最简单的调用方式
response = llm.invoke("用一句话解释什么是 Python")
print(f"直接调用 LLM:")
print(f"  回答: {response.content}")

# ============================================================
# 2. PromptTemplate - 模板化提示词
# ============================================================
print("\n" + "=" * 60)
print("2. PromptTemplate - 模板化提示词")
print("=" * 60)

print("""
硬编码提示词的问题:
  prompt = "请用简单的语言解释什么是机器学习"
  → 每次改问题都要改代码

PromptTemplate 的好处:
  template = "请用{style}的语言解释什么是{topic}"
  → 只需要传参数，模板复用
""")

# 创建提示词模板
prompt = ChatPromptTemplate.from_template(
    "你是一个{role}。请用{style}的语言，在100字以内解释什么是{topic}。"
)

# 填充模板
formatted = prompt.format_messages(
    role="AI 老师",
    style="通俗易懂",
    topic="神经网络",
)
print(f"格式化后的提示词:")
print(f"  {formatted[0].content}")

# 调用 LLM
response = llm.invoke(formatted)
print(f"\n回答: {response.content}")

# ============================================================
# 3. LCEL - 链式表达式语言
# ============================================================
print("\n" + "=" * 60)
print("3. LCEL - LangChain 的核心语法")
print("=" * 60)

print("""
LCEL (LangChain Expression Language) 用管道符 | 连接组件:

  chain = prompt | model | output_parser

这就像 Unix 管道:
  cat file.txt | grep "error" | wc -l

数据流向:
  输入参数 → Prompt 模板 → LLM → 输出解析器 → 最终结果
""")

# 创建一个完整的链
chain = prompt | llm | StrOutputParser()

# 一行代码完成：模板填充 → LLM 调用 → 解析输出
result = chain.invoke({
    "role": "编程导师",
    "style": "用类比的方式",
    "topic": "Transformer",
})

print(f"链式调用结果:")
print(f"  {result}")

# ============================================================
# 4. 多种 Prompt 模板
# ============================================================
print("\n" + "=" * 60)
print("4. 多种 Prompt 模板")
print("=" * 60)

# 带系统消息的模板
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{domain}专家，回答简洁精准。"),
    ("human", "{question}"),
])

chain_with_system = system_prompt | llm | StrOutputParser()

# 测试不同领域
domains = [
    ("Python 编程", "列表推导式有什么用？"),
    ("机器学习", "过拟合怎么解决？"),
    ("数据库", "索引的作用是什么？"),
]

for domain, question in domains:
    result = chain_with_system.invoke({
        "domain": domain,
        "question": question,
    })
    print(f"\n🔹 [{domain}] {question}")
    print(f"   {result[:150]}...")

# ============================================================
# 5. 输出解析器
# ============================================================
print("\n\n" + "=" * 60)
print("5. 输出解析器 - 结构化输出")
print("=" * 60)

from langchain_core.output_parsers import JsonOutputParser

# 让 LLM 输出 JSON 格式
json_prompt = ChatPromptTemplate.from_template(
    """请为以下编程概念生成一个学习卡片，以 JSON 格式输出。
只输出 JSON，不要其他内容。

概念: {concept}

JSON 格式:
{{"name": "概念名称", "definition": "一句话定义", "example": "简短代码示例", "difficulty": "入门/中级/高级"}}"""
)

json_chain = json_prompt | llm | JsonOutputParser()

try:
    card = json_chain.invoke({"concept": "列表推导式"})
    print(f"学习卡片 (JSON):")
    for key, value in card.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"JSON 解析失败（小模型可能格式不完美）: {e}")

# ============================================================
# 6. 批量调用
# ============================================================
print("\n" + "=" * 60)
print("6. 批量调用")
print("=" * 60)

simple_chain = ChatPromptTemplate.from_template(
    "用一句话解释: {term}"
) | llm | StrOutputParser()

terms = [
    {"term": "API"},
    {"term": "SDK"},
    {"term": "REST"},
]

print("批量调用结果:")
results = simple_chain.batch(terms)
for term, result in zip(terms, results):
    print(f"  {term['term']}: {result[:80]}...")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("💡 LangChain 基础总结")
print("=" * 60)
print("""
核心组件:
  1. ChatModel  → 连接 LLM（oLMX / OpenAI / DeepSeek）
  2. Prompt     → 模板化提示词，支持变量替换
  3. Chain      → 用 | 管道符连接组件
  4. Parser     → 解析 LLM 输出（字符串/JSON/自定义）

LCEL 核心语法:
  chain = prompt | model | parser
  result = chain.invoke({"key": "value"})

关键理解:
  LangChain 不是一个模型，而是一个"胶水框架"，
  它把 Prompt、LLM、工具、记忆等组件粘合在一起。
""")

print("✅ LangChain 基础练习完成！")
