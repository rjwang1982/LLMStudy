"""
Agent - ReAct Agent：推理 + 行动循环

作者: RJ.Wang
时间: 2026-04-22

学习目标:
  - 理解 ReAct 模式：Reasoning + Acting
  - 用 LangChain 构建 ReAct Agent
  - 理解 Agent 的"思考→行动→观察"循环
  - 让 Agent 自主选择工具解决问题
"""

import os
import math
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

OMLX_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8019/v1")
OMLX_API_KEY = os.getenv("OPENAI_API_KEY", "omlx")
OMLX_MODEL = os.getenv("OMLX_MODEL", "Qwen3.5-9B-MLX-4bit")

print("=" * 60)
print("ReAct Agent：推理 + 行动循环")
print("=" * 60)

# ============================================================
# 1. ReAct 模式解释
# ============================================================
print("""
ReAct = Reasoning + Acting（推理 + 行动）

传统 LLM:
  问题 → 直接回答（可能错误）

ReAct Agent:
  问题 → 思考（我需要什么信息？）
       → 行动（调用工具获取信息）
       → 观察（工具返回了什么？）
       → 思考（信息够了吗？）
       → ...（循环直到有足够信息）
       → 最终回答

示例:
  问题: "北京和上海哪个更热？"

  思考: 我需要知道两个城市的温度
  行动: get_weather("北京") → 25°C
  观察: 北京 25°C
  思考: 还需要上海的温度
  行动: get_weather("上海") → 22°C
  观察: 上海 22°C
  思考: 25 > 22，北京更热
  回答: 北京（25°C）比上海（22°C）更热。
""")

# ============================================================
# 2. 定义工具
# ============================================================
print("=" * 60)
print("2. 定义工具")
print("=" * 60)


@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气信息，包括温度和天气状况"""
    weather_data = {
        "北京": "晴天，25°C，湿度 40%，空气质量良好",
        "上海": "多云，22°C，湿度 65%，有轻雾",
        "深圳": "小雨，28°C，湿度 80%，注意带伞",
        "成都": "阴天，20°C，湿度 55%，适合出行",
        "杭州": "晴天，24°C，湿度 50%，适合户外活动",
    }
    return weather_data.get(city, f"未找到 {city} 的天气数据")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持加减乘除、幂运算、开方等。
    示例: '2 + 3', '10 ** 2', 'math.sqrt(144)'"""
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """单位转换工具。支持温度(C/F)、长度(km/mile)、重量(kg/lb)"""
    conversions = {
        ("C", "F"): lambda x: x * 9 / 5 + 32,
        ("F", "C"): lambda x: (x - 32) * 5 / 9,
        ("km", "mile"): lambda x: x * 0.621371,
        ("mile", "km"): lambda x: x * 1.60934,
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x * 0.453592,
    }
    key = (from_unit, to_unit)
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    return f"不支持 {from_unit} → {to_unit} 的转换"


tools = [get_weather, calculator, unit_converter]

print("已注册的工具:")
for t in tools:
    print(f"  🔧 {t.name}: {t.description[:50]}...")

# ============================================================
# 3. 创建 ReAct Agent
# ============================================================
print("\n" + "=" * 60)
print("3. 创建 ReAct Agent")
print("=" * 60)

llm = ChatOpenAI(
    base_url=OMLX_BASE_URL, api_key=OMLX_API_KEY,
    model=OMLX_MODEL, temperature=0,
)

# 一行代码创建 ReAct Agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="你是一个有用的助手，可以查天气、做计算和单位转换。请用中文简洁回答。",
)

print("✅ ReAct Agent 创建完成！")

# ============================================================
# 4. 测试 Agent
# ============================================================
print("\n" + "=" * 60)
print("4. Agent 测试")
print("=" * 60)


def ask_agent(question: str):
    """向 Agent 提问并展示思考过程"""
    print(f"\n{'─' * 50}")
    print(f"❓ {question}")

    result = agent.invoke({"messages": [("human", question)]})

    # 展示 Agent 的思考和行动过程
    for msg in result["messages"]:
        if msg.type == "human":
            continue
        elif msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  🔧 调用: {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"  📋 结果: {msg.content[:100]}")
        elif msg.type == "ai":
            print(f"  💬 回答: {msg.content}")


# 简单问题：单工具
ask_agent("北京今天天气怎么样？")

# 计算问题
ask_agent("帮我算一下 2 的 10 次方是多少")

# 单位转换
ask_agent("100 公斤等于多少磅？")

# 复合问题：可能需要多步
ask_agent("深圳现在多少度？把温度转换成华氏度")

# ============================================================
# 5. ReAct 循环可视化
# ============================================================
print("\n\n" + "=" * 60)
print("5. ReAct 循环可视化")
print("=" * 60)

print("""
Agent 处理 "深圳多少度？转成华氏度" 的过程:

  ┌─────────────────────────────────────┐
  │ 思考: 需要先查深圳天气获取温度      │
  │ 行动: get_weather("深圳")           │
  │ 观察: 小雨，28°C，湿度 80%         │
  ├─────────────────────────────────────┤
  │ 思考: 得到 28°C，需要转成华氏度     │
  │ 行动: unit_converter(28, "C", "F")  │
  │ 观察: 28 C = 82.40 F               │
  ├─────────────────────────────────────┤
  │ 思考: 信息足够了，可以回答          │
  │ 回答: 深圳 28°C（82.4°F）          │
  └─────────────────────────────────────┘

关键: Agent 自主决定调用顺序和次数！
""")

# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("💡 ReAct Agent 总结")
print("=" * 60)
print("""
ReAct 的核心:
  1. Reasoning: LLM 分析问题，决定下一步
  2. Acting: 调用合适的工具
  3. Observing: 获取工具返回的结果
  4. 循环直到有足够信息回答

LangGraph create_react_agent:
  - 自动处理 ReAct 循环
  - 自动管理消息历史
  - 支持多工具组合
  - 一行代码创建 Agent
""")

print("✅ ReAct Agent 练习完成！")
