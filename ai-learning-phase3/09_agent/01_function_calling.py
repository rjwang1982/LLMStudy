"""
Agent - Function Calling：让 LLM 调用函数

作者: RJ.Wang
时间: 2026-04-22

学习目标:
  - 理解 Function Calling 的原理
  - 定义工具函数并让 LLM 自主调用
  - 理解 LLM 如何"决定"调用哪个函数
  - 为构建 Agent 打基础
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OMLX_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8019/v1")
OMLX_API_KEY = os.getenv("OPENAI_API_KEY", "omlx")
OMLX_MODEL = os.getenv("OMLX_MODEL", "Qwen3.5-9B-MLX-4bit")

client = OpenAI(base_url=OMLX_BASE_URL, api_key=OMLX_API_KEY)

print("=" * 60)
print("Function Calling：让 LLM 调用函数")
print("=" * 60)

# ============================================================
# 1. 什么是 Function Calling？
# ============================================================
print("""
传统 LLM:
  用户: "北京现在几度？"
  LLM:  "我无法获取实时天气信息。"  ← 只能说不知道

Function Calling:
  用户: "北京现在几度？"
  LLM:  → 决定调用 get_weather("北京")
  系统: → 执行函数，返回 "25°C"
  LLM:  "北京现在 25°C。"  ← 有了外部能力！

核心思想:
  LLM 不直接执行函数，而是"告诉你"它想调用什么函数、传什么参数。
  你的代码负责实际执行，然后把结果返回给 LLM。
""")

# ============================================================
# 2. 定义工具函数
# ============================================================
print("=" * 60)
print("2. 定义工具函数")
print("=" * 60)


# 实际的函数实现
def get_weather(city: str) -> str:
    """获取城市天气（模拟）"""
    weather_data = {
        "北京": "晴天，25°C，湿度 40%",
        "上海": "多云，22°C，湿度 65%",
        "深圳": "小雨，28°C，湿度 80%",
        "成都": "阴天，20°C，湿度 55%",
    }
    return weather_data.get(city, f"未找到 {city} 的天气数据")


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全地计算数学表达式
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            result = eval(expression)
            return str(result)
        return "不支持的表达式"
    except Exception as e:
        return f"计算错误: {e}"


def search_knowledge(query: str) -> str:
    """搜索知识库（模拟）"""
    knowledge = {
        "python": "Python 是一种解释型编程语言，由 Guido van Rossum 创建。",
        "transformer": "Transformer 是一种基于自注意力机制的神经网络架构。",
        "rag": "RAG 是检索增强生成技术，结合检索和生成能力。",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "未找到相关信息"


# 函数注册表（名称 → 函数）
available_functions = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_knowledge": search_knowledge,
}

print("已注册的工具函数:")
for name in available_functions:
    print(f"  🔧 {name}")

# ============================================================
# 3. 定义工具 Schema（告诉 LLM 有哪些工具可用）
# ============================================================
print("\n" + "=" * 60)
print("3. 工具 Schema 定义")
print("=" * 60)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，支持加减乘除",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如：2 + 3 * 4",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索技术知识库，查找编程和 AI 相关知识",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

print("工具 Schema 已定义（OpenAI 格式）")
print(f"共 {len(tools)} 个工具")

# ============================================================
# 4. 完整的 Function Calling 流程
# ============================================================
print("\n" + "=" * 60)
print("4. Function Calling 完整流程")
print("=" * 60)


def chat_with_tools(user_message: str) -> str:
    """带工具调用的完整对话流程"""
    print(f"\n{'─' * 50}")
    print(f"👤 用户: {user_message}")

    messages = [
        {"role": "system", "content": "你是一个有用的助手，可以查天气、做计算、搜索知识。请简洁回答。"},
        {"role": "user", "content": user_message},
    ]

    # 第一次调用：LLM 决定是否需要调用工具
    response = client.chat.completions.create(
        model=OMLX_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # 让 LLM 自己决定
    )

    assistant_message = response.choices[0].message

    # 检查是否有工具调用
    if assistant_message.tool_calls:
        print(f"🤖 LLM 决定调用工具:")

        # 把 assistant 的消息加入历史
        messages.append(assistant_message)

        # 执行每个工具调用
        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"   🔧 {func_name}({func_args})")

            # 执行函数
            func = available_functions[func_name]
            result = func(**func_args)
            print(f"   📋 结果: {result}")

            # 把工具结果加入消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # 第二次调用：LLM 根据工具结果生成最终回答
        final_response = client.chat.completions.create(
            model=OMLX_MODEL,
            messages=messages,
        )
        answer = final_response.choices[0].message.content
    else:
        # 不需要工具，直接回答
        answer = assistant_message.content
        print(f"🤖 LLM 直接回答（不需要工具）")

    print(f"💬 回答: {answer}")
    return answer


# ============================================================
# 5. 测试各种场景
# ============================================================
print("\n" + "=" * 60)
print("5. 测试各种场景")
print("=" * 60)

# 需要查天气
chat_with_tools("北京今天天气怎么样？")

# 需要计算
chat_with_tools("帮我算一下 (15 + 27) * 3 等于多少")

# 需要搜索知识
chat_with_tools("什么是 RAG 技术？")

# 不需要工具
chat_with_tools("你好，请做个自我介绍")

# ============================================================
# 总结
# ============================================================
print("\n\n" + "=" * 60)
print("💡 Function Calling 总结")
print("=" * 60)
print("""
完整流程:
  1. 定义工具函数 + Schema
  2. 把工具列表传给 LLM
  3. LLM 分析用户意图，决定是否调用工具
  4. 如果需要：返回函数名 + 参数
  5. 你的代码执行函数，获取结果
  6. 把结果返回给 LLM
  7. LLM 基于结果生成最终回答

关键理解:
  - LLM 不执行函数，只"建议"调用
  - 你的代码负责实际执行（安全可控）
  - 这就是 Agent 的基础能力！
""")

print("✅ Function Calling 练习完成！")
