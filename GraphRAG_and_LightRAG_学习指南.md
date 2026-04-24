# 🕸️ GraphRAG & LightRAG 进阶学习指南

**作者**: RJ.Wang
**邮箱**: wangrenjun@gmail.com
**创建时间**: 2026-04-24
**内容来源**: 小林coding 公众号文章整理
**前置知识**: 建议先阅读 [RAG & Fine-tuning 学习指南](RAG_and_Fine-tuning_学习指南.md)

---

## 📖 前言：传统 RAG 够用吗？

上一篇学习指南讲了传统 RAG 的完整流程：文档切块 → 向量化 → 入库 → 检索 → 生成。这套方案在 Demo 场景下跑得很好，但一旦拿到企业的复杂场景，就会撞上几堵硬墙。

GraphRAG 和 LightRAG 就是为了绕过这几堵墙而出现的。

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    A["📚 传统 RAG<br/>找语义相似的文本块"] --> P{"遇到三大痛点"}
    P --> P1["❌ 多跳推理干不了"]
    P --> P2["❌ 全局性问题答不了"]
    P --> P3["❌ 切块导致语义断裂"]
    P1 & P2 & P3 --> S{"解决思路：引入知识图谱"}
    S --> G["🕸️ GraphRAG<br/>微软 · 2024.4<br/>重型 · 深度分析"]
    S --> L["🪶 LightRAG<br/>港大 · 2024.10<br/>轻量 · 增量友好"]

    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style G fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style L fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## 第一部分：传统 RAG 的三大痛点

### 痛点 ① 多跳推理干不了

有些问题需要跨多个文档做链式推理：A → B → C。

比如："哪些欧洲供应商在过去一年的安全审计中没通过，而且还在处理个人隐私数据？"

这个问题需要同时关联三份文档：供应商档案、审计报告、合同文档。传统 RAG 只能分别检索到一些相关的 chunk，但没有任何机制把三个集合做"交集"。

### 痛点 ② 全局性问题答不了

"这本小说的主题思想是什么？""这份年报里公司战略发生了哪些变化？"

这类问题的答案不在某一个 chunk 里，需要通读所有文档归纳总结。但传统 RAG 只会找 Top-K 个最像的 chunk，哪个 chunk 会写"本书主题思想是 XXX"？几乎没有。

### 痛点 ③ 切块导致语义断裂

文档切块会把完整的因果关系拦腰斩断。比如原文写"卡托普利属于 ACEI 类药物，但严重肾功能不全者禁用"，切块后可能分散在两个 chunk 里，大模型拿到碎片化事实就容易搞混因果关系。

### 根本原因

传统 RAG 做的是"找相似文本"，但企业真正需要的是"理解实体关系"。

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph LR
    A["传统 RAG<br/>找语义相似的文本块"] -->|"缺乏"| B["关系理解能力"]
    B --> C["多跳推理 ❌"]
    B --> D["全局归纳 ❌"]
    B --> E["因果追溯 ❌"]
    F["GraphRAG / LightRAG<br/>引入知识图谱"] -->|"具备"| B2["关系理解能力"]
    B2 --> C2["多跳推理 ✅"]
    B2 --> D2["全局归纳 ✅"]
    B2 --> E2["因果追溯 ✅"]

    style A fill:#ffcdd2,stroke:#c62828,stroke-width:2px
    style F fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
```

---

## 第二部分：什么是 GraphRAG？

### 2.1 一句话理解

GraphRAG = 用 LLM 把文档"读成一张知识图谱"，然后基于这张图谱来做检索和回答。

传统 RAG 存的是一堆独立的文本块，GraphRAG 存的是一张实体和关系组成的网络：

```
传统 RAG：
  chunk 1: "张三在2025年创立了A公司..."  （独立文本块）
  chunk 2: "李四是A公司的CTO..."         （独立文本块）
  → 块和块之间没有任何联系

GraphRAG：
  [张三] --创立--> [A公司] --CTO是--> [李四]
  [A公司] --主营--> [自动驾驶]
  → 实体和关系形成一张网络
```

### 2.2 索引阶段：5 步把文档变成知识图谱

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    A["📄 ① 文档切块<br/>跟传统 RAG 一样"] 
    --> B["🔍 ② LLM 抽取实体和关系<br/>每个 chunk 单独发给 LLM"]
    --> C["📝 ③ 生成实体/关系摘要<br/>同一实体的多次描述合并"]
    --> D["🏘️ ④ 社区检测 Leiden 算法<br/>把图谱划分成层次化社区"]
    --> E["📋 ⑤ 生成社区摘要<br/>每个社区一份总结报告"]

    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style B fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style C fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style D fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style E fill:#fff9c4,stroke:#f57f17,stroke-width:2px
```

**社区是什么？** 图里那些互相抱团、关系特别密切的一群节点。比如在《三国演义》里，刘关张是一个小社区，曹操集团是一个社区，诸葛亮周边又是一个社区。

社区还有层次：最粗的一层可能是"蜀汉集团""曹魏集团""东吴集团"，再往下细分出"刘关张小团体""五虎上将"等。用户问得越宏观，用越高层的社区；问得越具体，用越低层的社区。

### 2.3 查询阶段：两种搜索模式

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    Q["🙋 用户提问"] --> R{"问题类型？"}
    R -->|"具体实体问题"| L["🔍 Local Search<br/>定位入口实体 → 沿图遍历扩展<br/>→ 收集邻居和关系 → LLM 生成"]
    R -->|"全局宏观问题"| G["🌐 Global Search<br/>Map: 每个社区摘要生成中间答案<br/>Reduce: 汇总中间答案 → LLM 生成"]

    style L fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style G fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

### 2.4 GraphRAG 的四大难点

| 难点 | 问题 | 影响 |
|:---|:---|:---|
| 💸 索引成本高 | Token 消耗是传统 RAG 的几十倍 | 100 万 token 约 $20-50 |
| 🔀 实体消歧难 | 同一实体被抽成多个节点 | 关系碎片化，检索召回不全 |
| 🐢 查询延迟高 | Global Search 遍历大量社区 | 端到端 10s~1min |
| 🔄 增量更新难 | 牵一发而动全身 | 社区重划 → 摘要重建 → 索引重算 |

> 🍳 比喻：GraphRAG 像一个很勤快的图书管理员，在你进图书馆之前已经把每个主题区都写好了概览海报。查得快，但维护贵。

---

## 第三部分：什么是 LightRAG？

### 3.1 一句话理解

LightRAG = GraphRAG 的轻量化替代方案。保留知识图谱的关系理解能力，但去掉社区检测和社区摘要，用双层检索替代。

### 3.2 三大核心创新

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    subgraph I1 ["🪶 创新一：去社区化"]
        A1["不做 Leiden 社区检测"]
        A2["不做社区摘要"]
        A3["省掉 80%+ 的 LLM 调用"]
    end
    subgraph I2 ["🔍 创新二：双层检索"]
        B1["低层关键词 → 检索实体（找具体的点）"]
        B2["高层关键词 → 检索关系（找抽象的线）"]
        B3["两路并行，秒级返回"]
    end
    subgraph I3 ["🔄 创新三：增量友好"]
        C1["新文档只需追加节点和边"]
        C2["不触发社区重划和摘要重建"]
        C3["几乎零额外成本"]
    end

    style I1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style I2 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style I3 fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

### 3.3 索引阶段：只有 3 步（对比 GraphRAG 的 5 步）

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    A["📄 ① 文档切块"] 
    --> B["🔍 ② LLM 抽取实体和关系<br/>+ 生成关系关键词"]
    --> C["🔑 ③ 键值对生成 + 去重<br/>实体名/关系关键词 → 详细描述<br/>同名实体合并"]

    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style B fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style C fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

对比 GraphRAG 省掉了什么：

| 步骤 | GraphRAG | LightRAG |
|:---|:---:|:---:|
| 实体抽取 | ✅ | ✅ |
| 关系抽取 | ✅ | ✅ |
| 实体/关系摘要 | ✅（重） | ✅（轻） |
| 社区检测 Leiden | ✅ | ❌ |
| 社区摘要 | ✅ | ❌ |

### 3.4 查询阶段：双层检索

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    Q["🙋 用户提问"] --> KW["🤖 LLM 抽取双层关键词"]
    KW --> LOW["低层关键词<br/>具体实体和术语"]
    KW --> HIGH["高层关键词<br/>抽象主题和概念"]
    LOW --> LS["🔍 检索实体节点<br/>+ 扩展一跳邻居"]
    HIGH --> HS["🔍 检索关系边<br/>+ 聚合涉及实体"]
    LS & HS --> CTX["📋 合并上下文"]
    CTX --> ANS["🤖 LLM 生成答案"]

    style LOW fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style HIGH fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style ANS fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

举个例子：

```
查询："国际贸易如何影响全球经济稳定？"

高层关键词：国际贸易、全球经济稳定、经济影响
  → 去向量库检索相关的"关系边"

低层关键词：贸易协定、关税、货币汇率、进口、出口
  → 去向量库检索相关的"实体节点"

两路结果合并 → 组装上下文 → LLM 生成答案
```

LightRAG 提供四种查询模式：

| 模式 | 做什么 | 适合场景 |
|:---|:---|:---|
| Naive | 纯向量检索（传统 RAG） | 简单事实问答 |
| Local | 只用低层检索（找实体） | 具体实体问题 |
| Global | 只用高层检索（找关系） | 主题归纳问题 |
| Hybrid | 双层同时用（默认推荐） | 复杂综合问题 |

### 3.5 增量更新：真正的轻量友好

```
新文档来了
  ↓ 切块 + 抽实体和关系（只处理新文档）
  ↓ 同名实体？合并描述。新实体？加新节点。
  ↓ 新关系追加到图上
  ↓ 新描述向量化，追加到向量库
  ↓ 完成。不用重算社区，不用重建摘要。
```

> 🪶 比喻：LightRAG 像一个聪明的助手，你告诉他你想了解什么，他现场帮你把相关的书和书之间的关系梳理出来。查得灵活，维护便宜。

---

## 第四部分：GraphRAG vs LightRAG 全面对比

### 4.1 核心对比表

| 维度 | 🕸️ GraphRAG | 🪶 LightRAG |
|:---|:---|:---|
| 发布时间 | 2024.4（微软） | 2024.10（港大） |
| 核心创新 | 社区检测 + 社区摘要 | 双层检索 + 增量友好 |
| 索引步骤 | 5 步（含社区） | 3 步（无社区） |
| 索引成本（100 万 token） | $20-50 | ~$0.15 |
| 单次查询延迟 | 10s~1min（Global） | 1~3s |
| 增量更新 | 难（级联重建） | 易（直接追加） |
| 全局深度洞察 | 强（社区摘要） | 一般（临场组装） |
| 实体消歧 | 多层策略，精度高 | 朴素名称匹配 |
| 适合场景 | 深度分析、静态知识库 | 实时更新、成本敏感 |

### 4.2 成本对比（真刀真枪的数字）

| 指标 | GraphRAG | LightRAG |
|:---|:---|:---|
| 索引 100 万 token | $20-50 | ~$0.15 |
| 单次查询 Token 消耗 | ~13,000 | 100~1,000 |
| 单次查询 API 调用 | 几百次 | 几次 |
| 增量一次更新 | 可能触发社区重建 | 几乎零额外成本 |

Token 消耗降低 99%，这是 LightRAG 最吸引人的数字。

---

## 第五部分：怎么选？

### 5.1 选型决策树

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    START(("🤔 我该选哪个？"))
    START --> Q1{"问题是否涉及<br/>跨文档关系推理？"}
    Q1 -->|"否，简单事实问答"| R0["📚 传统 RAG 就够了"]
    Q1 -->|"是"| Q2{"数据是否频繁更新？"}
    Q2 -->|"是，每天/每周都变"| R1["🪶 LightRAG"]
    Q2 -->|"否，相对静态"| Q3{"需要极深度的<br/>全局归纳分析？"}
    Q3 -->|"是"| R2["🕸️ GraphRAG"]
    Q3 -->|"不需要"| Q4{"预算充足吗？"}
    Q4 -->|"预算有限"| R3["🪶 LightRAG"]
    Q4 -->|"预算充足"| Q5{"对实时性<br/>要求高吗？"}
    Q5 -->|"是，C 端交互"| R4["🪶 LightRAG"]
    Q5 -->|"否，离线分析"| R5["🕸️ GraphRAG"]

    style R0 fill:#e3f2fd,color:#000,stroke-width:0px
    style R1 fill:#4caf50,color:#fff,stroke-width:0px
    style R3 fill:#4caf50,color:#fff,stroke-width:0px
    style R4 fill:#4caf50,color:#fff,stroke-width:0px
    style R2 fill:#ff9800,color:#fff,stroke-width:0px
    style R5 fill:#ff9800,color:#fff,stroke-width:0px
```

### 5.2 按数据规模选

| 数据规模 | 推荐方案 |
|:---|:---|
| < 10 万 token | 传统 RAG（没必要上图） |
| 10 万 ~ 500 万 token | LightRAG 最佳 |
| 500 万 ~ 5000 万 token | 看业务侧重，两者都可以 |
| > 5000 万 token | GraphRAG 更合适（规模越大，社区摘要价值越大） |

### 5.3 务实建议

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph LR
    A["① 先用传统 RAG<br/>搭 MVP"] --> B["② 发现需要关系推理<br/>升级 LightRAG"]
    B --> C["③ LightRAG 搞不定的<br/>深度分析再上 GraphRAG"]

    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style B fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style C fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

不要一上来就选重型方案。先跑 MVP 看用户到底在问什么类型的问题，再按需升级。

---

## 第六部分：RAG 技术演进全景

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    A["📚 传统 RAG<br/>向量检索 + LLM 生成<br/>简单、便宜、够用"] 
    -->|"痛点：不理解关系"| B["🕸️ GraphRAG<br/>知识图谱 + 社区摘要<br/>深度强、成本高"]
    B -->|"痛点：太贵太慢"| C["🪶 LightRAG<br/>知识图谱 + 双层检索<br/>轻量、增量友好"]
    
    A -.->|"80% 场景够用"| D["大多数企业"]
    C -.->|"15% 需要关系推理"| D
    B -.->|"5% 需要深度分析"| D

    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style B fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style C fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style D fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
```

RAG 技术的发展，本质上是在"精度""成本""速度""维护"之间不断做权衡。没有银弹，只有最适合你业务场景的方案。

---

## 📚 参考来源

- [字节二面：别光吹RAG，说说GraphRAG的多跳推理](https://mp.weixin.qq.com/s/f8M0MWpdcH4AeauZrdxgqw) — 小林coding，2026-04-22
- [鹅厂面试官：什么是 RAG？工作流程是怎样的？](https://mp.weixin.qq.com/s/KnNx_ewIeJ_CZhfs6HtnTA) — 小林coding，2026-04-20
- 微软 GraphRAG 论文：*From Local to Global: A Graph RAG Approach to Query-Focused Summarization*（2024.4）
- LightRAG 论文：*LightRAG: Simple and Fast Retrieval-Augmented Generation*（2024.10，EMNLP 2025）

> 内容基于以上文章整理，已重新组织结构并用通俗语言改写，适合初学者阅读。
