# AI 大模型学习 - 阶段三：做应用

**作者**: RJ.Wang
**邮箱**: wangrenjun@gmail.com
**创建时间**: 2026-04-22

---

## 前置条件

| 类别 | 要求 | 如何验证 |
|------|------|----------|
| ✅ 阶段一 | 已完成 Python/神经网络/Transformer | 能解释 Self-Attention |
| ✅ 阶段二 | 已完成 Embedding/RAG/模型压缩 | 能独立搭建 RAG 管道 |
| 🤖 oLMX | 已安装并运行 | 访问 `http://127.0.0.1:8019/admin` 确认 |
| 🌐 网络 | 可访问 HuggingFace | 首次运行需下载模型 |

---

## 学习路线总览

```mermaid
graph LR
    A["📌 阶段一<br/>打基础"] --> B["📌 阶段二<br/>学核心"]
    B --> C["📌 阶段三<br/>做应用"]

    style A fill:#86efac,stroke:#22c55e,color:#000
    style B fill:#86efac,stroke:#22c55e,color:#000
    style C fill:#fbbf24,stroke:#d97706,color:#000,stroke-width:2px
```

---

## 阶段三内部学习流程

```mermaid
graph TD
    subgraph M1["📘 模块八：LangChain"]
        A1["01 基础概念<br/>Model / Prompt / Chain"] --> A2["02 记忆与对话<br/>Memory + 多轮对话"]
        A2 --> A3["03 RAG 链<br/>LangChain 版 RAG"]
    end

    subgraph M2["📗 模块九：Agent"]
        B1["01 Function Calling<br/>让 LLM 调用函数"] --> B2["02 ReAct Agent<br/>推理 + 行动循环"]
        B2 --> B3["03 多工具 Agent<br/>搜索/计算/文件"]
    end

    subgraph M3["📙 模块十：综合项目"]
        C1["01 智能知识助手<br/>RAG + Agent 综合"]
    end

    M1 --> M2 --> M3

    style M1 fill:#dbeafe,stroke:#3b82f6
    style M2 fill:#dcfce7,stroke:#22c55e
    style M3 fill:#fef3c7,stroke:#f59e0b
```

---

## Agent 核心架构

```mermaid
graph TD
    USER["用户提问"] --> LLM["🤖 LLM 思考"]
    LLM -->|"需要外部信息"| TOOL["🔧 调用工具"]
    TOOL --> OBS["📋 获取结果"]
    OBS --> LLM
    LLM -->|"已有足够信息"| ANS["💬 生成回答"]

    subgraph tools["可用工具"]
        T1["🔍 搜索"]
        T2["🧮 计算"]
        T3["📂 文件读写"]
        T4["🗄️ 数据库查询"]
    end

    TOOL --> tools

    style LLM fill:#6366f1,stroke:#4f46e5,color:#fff
    style ANS fill:#34d399,stroke:#059669,color:#000
```

---

## LangChain 核心概念

```mermaid
graph LR
    subgraph core["LangChain 核心组件"]
        direction TB
        MODEL["Model<br/>LLM / ChatModel"]
        PROMPT["Prompt<br/>提示词模板"]
        CHAIN["Chain<br/>链式调用"]
        MEMORY["Memory<br/>对话记忆"]
        TOOL["Tool<br/>外部工具"]
        AGENT["Agent<br/>自主决策"]
    end

    PROMPT --> CHAIN
    MODEL --> CHAIN
    MEMORY --> CHAIN
    CHAIN --> AGENT
    TOOL --> AGENT

    style core fill:#eff6ff,stroke:#3b82f6
    style AGENT fill:#fbbf24,stroke:#d97706,color:#000
```

---

## 项目结构

```
ai-learning-phase3/
├── 08_langchain/                   # LangChain 框架
│   ├── 01_basics.py                    # 基础：Model + Prompt + Chain
│   ├── 02_memory.py                    # 记忆：多轮对话
│   └── 03_rag_chain.py                 # LangChain 版 RAG
├── 09_agent/                       # Agent 智能体
│   ├── 01_function_calling.py          # Function Calling 基础
│   ├── 02_react_agent.py              # ReAct 推理行动循环
│   └── 03_multi_tool_agent.py          # 多工具 Agent
├── 10_project/                     # 综合项目
│   └── 01_knowledge_assistant.py       # 智能知识助手
└── README.md
```

---

## 运行方式

**方式一：Jupyter Notebook（推荐）**
```bash
cd ai-learning-phase3
uv run jupyter notebook
```

**方式二：命令行**
```bash
cd ai-learning-phase3
uv run python 08_langchain/01_basics.py
```

所有练习默认使用本地 oLMX（`http://127.0.0.1:8019/v1`），无需云端 API Key。

---

## 每个练习的学习目标

| 模块 | 练习 | 你将学到 | 预计时间 |
|------|------|----------|----------|
| LangChain | 01 基础 | ChatModel、PromptTemplate、LCEL 链式语法 | 2h |
| LangChain | 02 记忆 | ConversationBufferMemory、多轮对话 | 2h |
| LangChain | 03 RAG 链 | 用 LangChain 重构阶段二的 RAG 管道 | 3h |
| Agent | 01 Function Calling | OpenAI 格式的函数调用、工具定义 | 2h |
| Agent | 02 ReAct | 思考→行动→观察循环、LangChain Agent | 3h |
| Agent | 03 多工具 | 组合多个工具、Agent 自主选择 | 3h |
| 综合项目 | 01 知识助手 | RAG + Agent + Memory 全部整合 | 4h |

---

## 完成标志

- [ ] 能用 LCEL 语法写 `prompt | model | parser` 链
- [ ] 理解 Memory 如何让 LLM 记住上下文
- [ ] 能定义 Tool 并让 LLM 自主调用
- [ ] 理解 ReAct 的"思考→行动→观察"循环
- [ ] 能独立搭建一个 RAG + Agent 的智能助手

全部打勾，恭喜你完成了整个学习路线！🎉
