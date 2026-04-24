# 🧠 RAG & Fine-tuning 小白学习指南

**作者**: RJ.Wang
**邮箱**: wangrenjun@gmail.com
**创建时间**: 2026-04-23
**内容来源**: 小林coding 公众号文章整理

---

## 📖 前言：大模型为什么需要"补课"？

想象一下，你高考结束后再也不看新闻、不学新知识。三年后有人问你"今天股价多少"，你肯定答不上来。

**大模型（LLM）也是一样。** 它训练完之后，知识就"冻住"了：

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
mindmap
  root((🧊 LLM 知识冻结<br/>训练完 = 知识定格))
    🕐 时效性缺失
      训练数据有截止日期
      之后发生的事一无所知
      无法回答最新动态
    🔒 私有数据盲区
      公司内部文档看不到
      行业专有知识不了解
      用户业务数据无法覆盖
    🤥 幻觉问题
      不知道的也会瞎编
      无法区分已知与未知
      回答缺乏可靠依据
```

那怎么给大模型"补课"呢？业界有两条路：

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TB
    subgraph PROBLEM ["😰 问题：LLM 知识冻结，怎么办？"]
        P["🧊 模型训练完成后<br/>知识就固定了"]
    end

    P --> CHOICE{"💡 两条补课路径"}

    subgraph FT_PATH ["🔧 路径一：微调 Fine-tuning"]
        direction TB
        FT1["📝 准备标注数据"]
        FT2["🖥️ GPU 继续训练"]
        FT3["💾 知识写入模型参数"]
        FT1 --> FT2 --> FT3
    end

    subgraph RAG_PATH ["📚 路径二：RAG 检索增强生成"]
        direction TB
        RAG1["📂 建立外部知识库"]
        RAG2["🔍 用户提问时实时检索"]
        RAG3["📋 检索结果注入 Prompt"]
        RAG1 --> RAG2 --> RAG3
    end

    CHOICE -->|"改参数"| FT_PATH
    CHOICE -->|"不改参数"| RAG_PATH

    FT3 --> FT_RESULT["🎓 相当于重新上学<br/>知识刻进脑子"]
    RAG3 --> RAG_RESULT["📖 相当于开卷考试<br/>现场翻书回答"]

    style PROBLEM fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style FT_PATH fill:#fff8e1,stroke:#ff8f00,stroke-width:2px
    style RAG_PATH fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style CHOICE fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style FT_RESULT fill:#ff9800,color:#fff,stroke-width:0px
    style RAG_RESULT fill:#2196f3,color:#fff,stroke-width:0px
```

下面我们一个一个来学。

---

## 第一部分：什么是 RAG？

### 1.1 一句话理解

**RAG = Retrieval-Augmented Generation = 检索增强生成**

> 用户提问 → 先去知识库里搜相关资料 → 把资料和问题一起交给大模型 → 大模型基于资料回答

就像你考试时可以翻书一样，大模型不用死记硬背，现场查资料就行。

### 1.2 RAG 的完整工作流程

RAG 分为两个阶段：**离线建库** 和 **在线问答**。

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    subgraph OFFLINE ["🏗️ 离线阶段（建库，只做一次）"]
        O1["📄 文档加载"] --> O2["✂️ 文档切割"] --> O3["🔢 向量化"] --> O4["💾 入库"]
    end
    subgraph ONLINE ["🔍 在线阶段（每次提问都跑）"]
        N1["✏️ Query 改写"] --> N2["🔍 向量检索"] --> N3["🎯 精排 Rerank"] --> N4["🤖 LLM 生成"]
    end
    O4 -.->|"向量数据库"| N2

    style OFFLINE fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style ONLINE fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

---

#### 🏗️ 离线阶段：提前把知识库建好（只做一次）

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    A["📄 ① 文档加载<br/>PDF / Word / Markdown / 网页 / 数据库"]
    B["✂️ ② 文档切割 Chunking<br/>每段 500~1000 token，前后重叠 ~100 token"]
    C["🔢 ③ 向量化 Embedding<br/>Embedding 模型（OpenAI / BGE / E5）→ 高维向量"]
    D["💾 ④ 存入向量库<br/>向量 + 原文一起存储（Chroma / Milvus / Qdrant）"]

    A ==> B ==> C ==> D

    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style B fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style C fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style D fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
```

**① 文档加载**
把各种格式的原始数据读进来。PDF、Word、Markdown、网页都行。

**② 文档切割（Chunking）— 为什么要切？**

两个原因：
- 向量模型有长度限制，整篇文档塞不进去
- 整篇文档压缩成一个向量，细节会被"平均掉"

> 🍜 比喻：你问"这道菜怎么样"，对方回答"中国菜整体偏咸"
> → 具体哪道菜咸、咸到什么程度，全丢失了

实践建议：
- 每个 chunk 大小：500～1000 token
- 前后重叠 100 token，避免把完整语义从中间切断

**③ Embedding（向量化）— 最核心的一步**

把文字转成一串数字（高维向量），比如 1536 维的浮点数列表。

> 🗺️ 比喻：想象一个"语义坐标系"
> - "苹果手机怎么截图" 和 "iPhone 如何截屏"
> - 用词完全不同，但意思一样 → 向量位置非常接近
> - 这就是语义检索的基础：不匹配关键词，而是比较"意思相不相近"

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph LR
    subgraph SEMANTIC ["🗺️ 语义坐标系示意"]
        direction TB
        A["'苹果手机怎么截图'<br/>向量: [0.82, 0.15, ...]"]
        B["'iPhone 如何截屏'<br/>向量: [0.81, 0.16, ...]"]
        C["'今天天气怎么样'<br/>向量: [-0.3, 0.72, ...]"]
        A <-->|"✅ 距离很近<br/>语义相似！"| B
        A <-->|"❌ 距离很远<br/>语义无关"| C
    end

    style A fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style B fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style C fill:#ffcdd2,stroke:#c62828,stroke-width:2px
```

**④ 入库**
向量 + 原始文本一起存进向量数据库（Chroma、Milvus、Qdrant 等）。

---

<details>
<summary>🔻 学习批注 | 真实疑问，逐层递进（Q1~Q6）💬</summary>
<div style="color: #2e7d32; font-size: 0.85em;">

**Q1：为什么整篇文档不能直接做 Embedding？**

Embedding 模型不管你输入 100 字还是 10000 字，最终都只输出一个固定长度的向量。内容越多越杂，这个向量就越"模糊"——就像把很多种颜料混在一起，最后变成一团灰色。

```
一篇公司手册（请假 + 报销 + 绩效 + 出差）
        ↓ 整篇做 Embedding
  [0.3, 0.25, 0.1, ...]  ← 一个"啥都像又啥都不像"的模糊向量

用户问 "出差补贴标准是多少"
        ↓ 跟模糊向量比较
  相似度不高 ❌ → 因为"出差"只占四分之一，被其他话题稀释了
```

切成小段后，每段的向量精准代表那段内容，检索才能命中：

```
chunk 1: 请假制度 → 精准代表"请假"
chunk 2: 报销流程 → 精准代表"报销"
chunk 3: 绩效考核 → 精准代表"绩效"
chunk 4: 出差规定 → 精准代表"出差" ← 用户问出差，直接命中！
```

**Q2：Chunking 和 Embedding 是什么关系？**

完全不同的两件事，但是流水线上的前后两道工序：

- **Chunking** — 解决"怎么分"。纯文本处理，不涉及任何模型，本质就是分段。
- **Embedding** — 解决"怎么表示"。用模型把文本转成数字向量，让计算机能用数学方式比较语义。

```
长文档 → [Chunking 切成小段] → 一堆 chunk → [Embedding 逐个转向量] → 一堆向量 → 存入向量库
```

🥩 比喻：Chunking 是把一整头牛分切成部位（牛腩、牛腱、牛排），Embedding 是给每个部位贴上标签（适合炖、适合卤、适合煎）。切是切，贴标签是贴标签，但不切就没法精准贴标签。

**Q3："1536 维浮点数列表"到底是什么？**

拆开来看：

- **列表** — 就是一个一维数组
- **1536 维** — 数组里有 1536 个元素（格子）
- **浮点数** — 每个格子里是一个小数，如 `0.1234567`

```
[0.12, -0.34, 0.56, 0.01, ..., -0.78]
 格子1   格子2   格子3  格子4  ...  格子1536
```

为什么要这么多格子？因为语言的含义太丰富，少量数字描述不清楚。就像用 [身高, 体重] 两个数字描述一个人太粗糙，用 1536 个维度才能精细地刻画一段文字的语义。

不同模型维度不同，1536 不是物理定律，是设计者选的平衡点：

| 模型 | 维度 |
|:---|:---:|
| OpenAI text-embedding-ada-002 | 1536 |
| OpenAI text-embedding-3-large | 3072 |
| BGE-large | 1024 |
| BGE-small | 512 |

**Q4：浮点数的精度是无穷的吗？**

不是。计算机用固定位数的二进制存储浮点数，精度有上限：

| 类型 | 位数 | 有效精度 | 可表示的不同值 |
|:---|:---:|:---:|:---:|
| float32（单精度） | 32 bit | 约 7 位小数 | ≈ 43 亿个 |
| float16（半精度） | 16 bit | 约 3~4 位小数 | ≈ 65536 个 |

大多数 Embedding 模型输出 float32。单个格子有约 43 亿种取值，1536 个格子的组合数是 43亿^1536 —— 大到完全不用担心"不够用"，但严格来说确实是有限的、离散的。

**Q5：每个 chunk 最终对应什么数据结构？**

一个长度为 1536 的一维数组。向量数据库里每条记录存两样东西：

```
chunk 1: "请假需提前3天申请..."  →  [0.12, -0.34, 0.56, ..., -0.78]  (1536个float32)
chunk 2: "报销需提供发票..."    →  [0.45, 0.23, -0.11, ..., 0.33]   (1536个float32)
chunk 3: "出差补贴标准..."      →  [-0.08, 0.67, 0.29, ..., 0.15]   (1536个float32)
```

检索时，用户问题也过同一个 Embedding 模型得到同样长度的数组，然后逐个算距离（余弦相似度），距离最近的就是最相关的 chunk。

**Q6：数组里元素的位置是任意的还是固定的？**

**固定的，不能换。** 同一个 Embedding 模型输出的向量，第 N 个位置永远捕捉同一种语义特征。比较两个向量时，是同位置对同位置比的：

```
chunk 向量:  [0.12,  -0.34,  0.56,  ...,  -0.78]
问题向量:    [-0.07,  0.65,  0.31,  ...,   0.14]
               ↕       ↕      ↕             ↕
             位置1   位置2   位置3   ...   位置1536
             逐个对比，算整体距离
```

如果打乱某个向量的元素顺序，它就变成一个完全不同的"语义肖像"，检索结果会完全错乱。

但具体每个位置代表什么语义特征？人类说不清楚——这些特征是模型训练时自己学出来的，不是人为定义的。类比 RGB 颜色 `[红, 绿, 蓝]` 每个位置含义明确，而 Embedding 的 1536 个位置，模型自己懂，人类只知道"位置固定、不能乱换"。

💡 一句话总结：每个 token 在模型内部确实各有一个向量（模型在"逐词阅读"），但最终会通过 pooling 汇总成一个向量输出，这个向量才是存入向量库、用来做检索的东西。文档太长时，汇总向量会变模糊，所以要先 Chunking 再 Embedding。

</div>
</details>

<details>
<summary>🔻 学习批注 | Chunking 实操和数据最终形态（Q11~Q12）💬</summary>
<div style="color: #2e7d32; font-size: 0.85em;">

**Q11：Chunking 具体怎么做？是把一个长 PDF 切成若干短 PDF 吗？**

不是切文件，是切文字。PDF 只是原始数据的容器，加载器把文字提取出来之后，容器就没用了，后面全是对纯文本字符串的操作：

```
原始 PDF 文件
    ↓ 文档加载（提取文字，PDF 格式/排版/图片全丢掉，只留文字）
一大段纯文本字符串
    ↓ Chunking（按语义断点切割文本）
若干段短文本字符串（在内存里，不是新文件）
    ↓ Embedding → 入库
```

切割时会按优先级依次尝试在这些位置断开：段落分隔符 `\n\n` → 换行符 `\n` → 句号等句末标点 → 逗号等句中标点 → 空格 → 硬切。目标是尽量在语义完整的地方断开，而不是把一句话从中间劈开。

**Q12：数据的最终形态是什么？**

两样东西：原始 PDF（备份用，不影响系统运行）和向量数据库。向量数据库里每条记录同时存了**向量 + 原文 chunk**：

```
向量数据库里每条记录：
┌──────────────────────────────────────────────────┐
│  向量: [0.12, -0.34, 0.56, ..., -0.78]           │ ← 用来检索（算相似度）
│  原文: "出差补贴标准为每天200元，需提供..."         │ ← 用来给 LLM 读
└──────────────────────────────────────────────────┘
```

向量只是一串数字，LLM 读不懂。检索命中后，真正塞进 prompt 给 LLM 看的是原文文本，不是向量。

</div>
</details>

---

#### 🔍 在线阶段：用户提问时实时检索（每次都跑）

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    Q["🙋 用户提问<br/>'上次说的那个方案怎么样？'"]

    subgraph STEP_A ["✏️ ① Query 改写"]
        A1["用户原始问题可能模糊、口语化"]
        A2["🤖 LLM 改写为明确的检索 Query"]
        A3["改写后: '关于 XX 项目的技术方案评估'"]
        A1 --> A2 --> A3
    end

    subgraph STEP_B ["🔍 ② 向量检索 · 粗排"]
        B1["Query 转成向量"]
        B2["在向量库中计算余弦相似度"]
        B3["返回 Top-K 个最相似 chunk"]
        B4["⚡ 速度快：几十毫秒<br/>⚠️ 精度有限：可能混入噪声"]
        B1 --> B2 --> B3
    end

    subgraph STEP_C ["🎯 ③ Rerank · 精排"]
        C1["Cross-Encoder 模型"]
        C2["问题 + 每个 chunk 拼在一起<br/>深度理解相关性"]
        C3["重新打分排序<br/>Top-20 → Top-3~5"]
        C4["🐢 速度慢但更准确"]
        C1 --> C2 --> C3
    end

    subgraph STEP_D ["🤖 ④ LLM 生成答案"]
        D1["拼接 Prompt:<br/>系统指令 + 精排 chunk + 用户问题"]
        D2["指令: 只根据资料回答<br/>资料里没有就说不知道"]
        D3["✅ 生成最终答案<br/>附带来源引用"]
        D1 --> D2 --> D3
    end

    Q ==> STEP_A ==> STEP_B ==> STEP_C ==> STEP_D

    style Q fill:#ffeb3b,stroke:#f57f17,stroke-width:2px
    style STEP_A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style STEP_B fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style STEP_C fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style STEP_D fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
```

**① Query 改写**
用户说"上次说的那个方案怎么样"，检索系统根本不知道"上次"是什么。
→ 让 LLM 先把问题改写成更明确的形式

<details>
<summary>🔻 学习批注 | Query 改写的常见疑问（Q7~Q8）💬</summary>
<div style="color: #2e7d32; font-size: 0.85em;">

**Q7：LLM 是怎么知道"上次"指的是哪个项目？**

LLM 不是凭空猜的，它靠的是**对话历史（conversation memory）**。RAG 系统会把最近几轮对话一起传给 LLM，让它结合上下文来改写：

```
对话历史（系统维护的）:
├─ 用户: "帮我看看 XX 项目的技术方案"
├─ AI:   "XX 项目的技术方案包含三个模块..."
├─ 用户: "第二个模块的风险评估呢"
├─ AI:   "第二个模块主要风险有..."
└─ 用户: "上次说的那个方案怎么样"  ← 当前提问（模糊）

LLM 结合对话历史，改写为:
→ "XX 项目技术方案的评估和进展情况"  ← 指代消解完成
```

如果没有对话历史，LLM 确实不知道"上次"是什么。这时好的系统会让 AI 反问用户"你说的是哪个方案？"，而不是硬猜。

💡 所以 Query 改写的前提是系统必须维护对话记忆。它不只是改措辞，更重要的是**把对话上下文中的指代关系解析清楚**，让后面的向量检索能拿到一个语义完整的查询。

**Q8：Query 改写算 RAG 的一部分吗？**

严格来说，不算。RAG 论文的核心定义就三步：**检索 → 拼接 → 生成**。只要在生成之前从外部检索了知识注入 prompt，就算 RAG。

但在实际工程中，Query 改写几乎是标配。两者的关系是：

| 层面 | 包含什么 |
|:---|:---|
| RAG 的学术定义 | 检索 + 增强 + 生成 |
| RAG 的工程实践 | Query 改写、对话记忆、Chunking 策略、Embedding、向量检索、Rerank、Prompt 模板、生成、引用溯源、效果评估… |

🍳 比喻："做菜"的定义是把食材加热变熟，但实际做菜你还得洗菜、切菜、调味、摆盘。Query 改写就像"洗菜切菜"——不是做菜的定义，但你不做就不好吃。

</div>
</details>

---

**② 向量检索（粗排）**
问题也转成向量，在向量库里找距离最近的 Top-K 个 chunk。
- 速度快：百万级向量库，几十毫秒返回
- 但不够精：可能混入"看着近但其实不相关"的内容

<details>
<summary>🔻 学习批注 | 向量检索与 LLM 生成参数（Q9~Q10）💬</summary>
<div style="color: #2e7d32; font-size: 0.85em;">

**Q9：Top-K 是什么意思？**

K 就是你自己定的一个数字，Top-K = 取相似度排名最靠前的 K 个结果。假设向量库里有 10000 个 chunk，系统算出每个 chunk 跟问题的相似度后从高到低排序，K=20 就取前 20 个。

```
排名1: chunk_892   相似度 0.95  ← 最像
排名2: chunk_3401  相似度 0.91
排名3: chunk_127   相似度 0.88
...
排名20: chunk_44   相似度 0.72  ← Top-20 的末位
------- 以下丢弃 -------
排名21~10000: 不够相关，不要了
```

K 设多少合适？没有标准答案：粗排阶段一般 K 设大一点（如 20），多捞候选，宁可多不可漏；精排之后再缩到 3~5 个，只留真正相关的交给 LLM。K 太小容易漏掉相关内容，K 太大会给 LLM 塞入太多噪声还浪费 token，实际项目里需要根据效果调。

**Q9.1：LLM 生成时也有 Top-K，跟 RAG 检索的 Top-K 是一回事吗？**

"取前 K 个"这个操作是一样的，底层也都涉及向量运算，但任务性质不同：

```
RAG Top-K：
  "这 10000 个 chunk 里，哪 K 个跟我的问题最像？"
  → 在做搜索，用余弦相似度找语义最接近的

LLM Top-K：
  "下一个词可能是词表里的任何一个，哪 K 个最合理？"
  → 在做预测，选最合理的续写
```

一个是"找像的"，一个是"选对的"。具体来说，LLM 生成下一个词时不是拿向量去比距离，而是经过几十层 Transformer 计算后，用矩阵乘法算出每个词的"合理程度分数"（logits），再转成概率。这个分数不是余弦相似度，而是模型学出来的"这个上下文之后，每个词出现的合理程度"。

RAG 的 Top-K 是从向量库里选最相似的 K 个 chunk；LLM 的 Top-K 是从概率最高的 K 个候选词里选，其他词直接排除：

```
LLM 算出下一个词的概率：
  巴黎 85% | 里昂 5% | 马赛 3% | 北京 0.1% | 苹果 0.001% | ...

Top-K = 3 → 候选池：巴黎、里昂、马赛（其他全部排除）
Top-K = 1 → 只能选"巴黎"，没有随机性
Top-K = 50 → 前 50 个词都有机会，随机性更大
```

**Q9.2：温度（Temperature）和 Top-P 又是什么？跟 Top-K 什么关系？**

三个参数都在控制同一件事：LLM 生成下一个词时，从候选词里怎么选。它们依次过滤：

```
温度  → 先调整概率分布的形状（尖锐 or 平坦）
Top-K → 再砍掉排名靠后的词（固定数量）
Top-P → 再砍掉累计概率之外的词（动态数量）
        ↓
  从剩下的候选词里，按概率随机抽一个
```

**温度**控制概率分布的"尖锐程度"：温度低（如 0.1）→ 概率差距拉大，几乎只选最高的那个词，回答确定但死板；温度高（如 1.5）→ 概率被拉平，什么词都可能蹦出来，有创意但容易胡说。

**Top-P**（核采样）不是固定取前 K 个，而是取概率累加刚好达到 P 的那些词。比如 Top-P=0.9，巴黎 85% + 里昂 5% = 90%，候选池就只有这 2 个词。Top-P 比 Top-K 更聪明：模型很确定时候选自动变少，模型不确定时候选自动变多。

实际使用建议：

| 场景 | 温度 | Top-P | 效果 |
|:---|:---:|:---:|:---|
| 事实问答、代码生成 | 0~0.3 | 0.9 | 准确、稳定 |
| 日常对话 | 0.7 | 0.9 | 自然、有变化 |
| 创意写作、头脑风暴 | 1.0~1.5 | 0.95 | 发散、有惊喜 |

**Q10：余弦相似度是怎么比较两个向量的？**

核心直觉：比的是两组数字代表的"方向"像不像，跟数字大小无关。输出一个分数：1 = 完全一致，0 = 毫无关系，-1 = 完全相反。

计算分三步，用一个 3 维的小例子演示：

```
向量A = [1, 2, 3]
向量B = [2, 2, 1]

① 逐位相乘:  1×2, 2×2, 3×1  →  2, 4, 3
② 求和:      2 + 4 + 3      →  9（这叫"点积"）
③ 归一化:    9 ÷ (向量A的长度 × 向量B的长度) → 0.80
```

前两步好理解，关键是第三步：**为什么要除以长度？**

类比考试成绩——满分不同的两门课，不能直接比分数：

```
小明 数学 80 分（满分 100）→ 80/100 = 80%
小红 英语 120 分（满分 150）→ 120/150 = 80%
```

直接比分数 120 > 80，小红赢？不对，换算成百分比才公平。

余弦相似度里除以长度，道理一样：
- **点积 = 原始分数**（受向量数字大小影响，没法直接比）
- **除以长度 = 换算成百分比**（消除大小差异，只比方向）
- **结果 -1 到 1 = 方向相似度的"百分比"**

放到 1536 维也是一模一样的三步，只不过要乘 1536 次、加 1536 个数。计算机算这些很快，几十毫秒就能把上万个 chunk 都比完。

</div>
</details>

**③ Rerank（精排）**
用 Cross-Encoder 模型把问题和每个候选 chunk 拼在一起深度理解。

> 📚 比喻：
> - 粗排 = 用肉眼在书架上快速扫一遍，把可能相关的书都抽出来
> - 精排 = 一本一本翻开读目录，确认哪些真正有用

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph LR
    subgraph COARSE ["👀 粗排：快速扫书架"]
        CR1["📚📕📗📘📙📓"]
        CR2["速度快，但可能抽错"]
        CR3["返回 Top-20"]
    end
    subgraph FINE ["🔬 精排：逐本翻阅"]
        FR1["📖 翻开读目录"]
        FR2["确认真正有用的"]
        FR3["留下 Top-3~5"]
    end
    COARSE ==>|"候选集"| FINE

    style COARSE fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style FINE fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

通常对 Top-20 做精排，最终留 Top-3 到 Top-5。

**④ LLM 生成**
问题 + 精排后的 chunk 拼成 prompt，交给 LLM。
Prompt 里会明确说："只根据提供的资料回答，资料里没有就说不知道"。

<details>
<summary>🔻 学习批注 | RAG vs LLM 向量化的区别（Q13~Q15）💬</summary>
<div style="color: #2e7d32; font-size: 0.85em;">

**Q13：LLM 内部也有向量化和入库吗？**

LLM 内部有向量化，但没有入库。两者机制完全不同：

| | RAG 的向量化 | LLM 内部的向量化 |
|:---|:---|:---|
| 目的 | 存起来，以后检索用 | 当场计算用，用完就丢 |
| 存储 | 存入向量数据库，持久化 | 不存，只在内存中临时存在 |
| 模型 | 专门的 Embedding 模型 | LLM 自带的 Embedding 层 |
| 输出 | 一段文字 → 一个向量 | 每个 token → 一个向量 |
| 用途 | 语义检索（找相似内容） | 语义理解（生成下一个词） |

LLM 的知识不是存在某个数据库里，而是分散在模型的几十亿个权重参数中。你下载一个模型文件（如 Qwen3.5-9B），下载的就是这些权重参数（`.safetensors` 文件）+ 一个分词字典（`tokenizer.json`）。训练数据在训练完成后就不需要了——知识已经以数学参数的形式"烧"进了权重里，但不是原样存储，而是被压缩成了统计规律。

**Q14：检索到 chunk 后，LLM 是不是再做一次"比对"？**

不是。整个 RAG 流程里，"比对"只发生一次——向量检索那一步（用户问题向量 vs chunk 向量算余弦相似度）。

检索命中后，LLM 做的是**阅读理解 + 生成回答**，不是比对：

```
1. 向量检索命中 → 拿到原文 chunk（直接从向量数据库取，不用回 PDF 找）
2. 拼成 prompt：系统指令 + 检索到的原文 + 用户问题
3. LLM 内部：tokenizer → 向量化 → Transformer 计算（这是在"阅读"prompt）
4. LLM 输出答案（这是在"回答"，不是在"比对"）
```

就像你读一段材料然后回答问题，你的大脑确实在处理文字，但你不是在"比对"，你是在理解和组织答案。

**Q15：训练完模型后，原始训练数据还需要吗？**

从"使用模型"的角度，不需要了。模型是独立的，跑推理只需要权重文件。但知识是被"压缩"存储的，不是原样保留——模型学到了规律和模式，但无法逐字还原训练数据。保留训练数据通常是为了重新训练下一代模型或合规审计，跟使用模型无关。

</div>
</details>

---

### 1.3 RAG 的核心价值

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    subgraph VALUE ["🌟 RAG 三大核心价值"]
        direction TB
        subgraph V1 ["🔄 知识可热更新"]
            V1A["往知识库加新文档即可"]
            V1B["不需要重新训练模型"]
            V1C["成本极低，实时生效"]
        end
        subgraph V2 ["🔗 答案可溯源"]
            V2A["每条回答追溯到具体 chunk"]
            V2B["出错能快速定位原因"]
            V2C["可解释性远超纯 LLM"]
        end
        subgraph V3 ["💰 实现成本低"]
            V3A["OpenAI Embedding + Chroma"]
            V3B["小团队也能快速搭建"]
            V3C["无需 GPU 训练"]
        end
    end

    style VALUE fill:#fafafa,stroke:#bdbdbd,stroke-width:2px
    style V1 fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style V2 fill:#bbdefb,stroke:#1565c0,stroke-width:2px
    style V3 fill:#fff9c4,stroke:#f57f17,stroke-width:2px
```

---

## 第二部分：什么是微调（Fine-tuning）？

### 2.1 一句话理解

**微调 = 用你自己的数据继续训练模型，让它把新知识"记"进参数里**

> 相当于让模型重新上了一遍学，把你的业务知识刻进它的"大脑"

### 2.2 微调的工作原理

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph LR
    subgraph INPUT ["📝 准备阶段"]
        I1["收集业务数据"]
        I2["标注训练样本<br/>问答对 / 指令对"]
        I3["清洗 & 格式化"]
        I1 --> I2 --> I3
    end

    subgraph TRAIN ["🖥️ 训练阶段"]
        T1["加载基础模型<br/>如 LLaMA / Qwen"]
        T2["用业务数据继续训练<br/>LoRA / 全量微调"]
        T3["模型参数被修改<br/>新知识写入权重"]
        T1 --> T2 --> T3
    end

    subgraph OUTPUT ["🎯 使用阶段"]
        O1["用户直接提问"]
        O2["模型从参数中<br/>回忆知识并回答"]
        O3["无需额外检索"]
        O1 --> O2 --> O3
    end

    INPUT ==> TRAIN ==> OUTPUT

    style INPUT fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style TRAIN fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style OUTPUT fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

### 2.3 微调的优势与代价

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    subgraph PROS ["✅ 微调的优势"]
        direction TB
        P1["🎨 效果彻底<br/>专业词汇、输出格式<br/>回答风格深度定制"]
        P2["⚡ 推理延迟低<br/>不需要额外检索步骤<br/>直接从参数中回答"]
        P3["🧠 深度专业能力<br/>对特定领域有<br/>更深的推理能力"]
    end

    subgraph CONS ["❌ 微调的代价"]
        direction TB
        C1["💸 成本高<br/>标注数据 + GPU 训练<br/>时间和金钱都不低"]
        C2["🔄 更新困难<br/>业务数据每周变<br/>不可能每周重新微调"]
        C3["🔮 不透明<br/>不知道回答来自哪条知识<br/>出错难以定位"]
    end

    style PROS fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style CONS fill:#ffebee,stroke:#c62828,stroke-width:2px
    style P1 fill:#c8e6c9,stroke:#2e7d32
    style P2 fill:#c8e6c9,stroke:#2e7d32
    style P3 fill:#c8e6c9,stroke:#2e7d32
    style C1 fill:#ffcdd2,stroke:#c62828
    style C2 fill:#ffcdd2,stroke:#c62828
    style C3 fill:#ffcdd2,stroke:#c62828
```

> ⚠️ 常见误区：以为微调后模型就"懂"了你的业务
> 实际上模型只是记住了训练数据里的模式，它到底记住了什么、记错了什么，你无从得知

---

## 第三部分：RAG vs 微调 — 怎么选？

### 3.1 核心对比表

| 维度 | 🔧 微调 Fine-tuning | 📚 RAG |
|:---:|:---:|:---:|
| 本质 | 改模型参数 | 不改参数，推理时现查 |
| 知识更新 | 需重新训练，成本高 | 更新知识库即可，实时生效 |
| 推理延迟 | 低 | 较高（多一次检索） |
| 实现成本 | 高（GPU + 标注数据） | 低（向量库 + Embedding） |
| 答案溯源 | ❌ 不支持 | ✅ 可追溯到 chunk |
| 适合场景 | 定制风格 / 深度专业 | 知识问答 / 动态更新 |
| 知识上限 | 受限于训练数据 | 受限于检索质量 |

### 3.2 选型决策树

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    START(("🤔 我该选哪个？"))

    START --> Q1{"知识是否需要<br/>频繁更新？"}
    Q1 -->|"是，每周/每天都变"| R1["📚 首选 RAG<br/>更新知识库即可"]
    Q1 -->|"否，相对稳定"| Q2{"需要定制输出格式<br/>语气或行业术语？"}

    Q2 -->|"是"| R2["🔧 首选微调<br/>深度定制模型行为"]
    Q2 -->|"否"| Q3{"需要答案<br/>可溯源？"}

    Q3 -->|"是，必须追溯来源"| R3["📚 首选 RAG<br/>可追溯到具体 chunk"]
    Q3 -->|"不需要"| Q4{"预算充足吗？<br/>有 GPU 资源？"}

    Q4 -->|"预算有限"| R4["📚 首选 RAG<br/>成本低得多"]
    Q4 -->|"预算充足"| Q5{"需要深度专业<br/>推理能力？"}

    Q5 -->|"是"| R5["🔧 首选微调<br/>培养深度专业能力"]
    Q5 -->|"否"| R6["📚 首选 RAG<br/>够用且灵活"]

    R2 --> COMBO{"还需要<br/>动态知识？"}
    COMBO -->|"是"| BEST["🏆 组合使用<br/>先微调再套 RAG"]
    COMBO -->|"否"| DONE1["🔧 单独微调即可"]

    style START fill:#fce4ec,stroke:#c62828,stroke-width:3px
    style R1 fill:#2196f3,color:#fff,stroke-width:0px
    style R3 fill:#2196f3,color:#fff,stroke-width:0px
    style R4 fill:#2196f3,color:#fff,stroke-width:0px
    style R6 fill:#2196f3,color:#fff,stroke-width:0px
    style R2 fill:#ff9800,color:#fff,stroke-width:0px
    style R5 fill:#ff9800,color:#fff,stroke-width:0px
    style DONE1 fill:#ff9800,color:#fff,stroke-width:0px
    style BEST fill:#4caf50,color:#fff,stroke-width:2px
```

### 3.3 最佳实践：组合使用

实际工程中最常见的做法是**两个都用**：

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    subgraph COMBINED ["🏆 最佳实践：微调 + RAG 组合"]
        direction TB

        subgraph PHASE1 ["第一步：微调基础模型"]
            FT_IN["通用基础模型<br/>如 LLaMA / Qwen"]
            FT_DATA["业务标注数据<br/>格式 · 语气 · 术语"]
            FT_OUT["定制化模型<br/>会说行业话"]
            FT_IN --> FT_DATA --> FT_OUT
        end

        subgraph PHASE2 ["第二步：套上 RAG 系统"]
            RAG_KB["📂 外部知识库<br/>最新文档 · 业务数据"]
            RAG_SEARCH["🔍 实时检索<br/>找到相关资料"]
            RAG_INJECT["📋 注入 Prompt<br/>提供具体知识"]
            RAG_KB --> RAG_SEARCH --> RAG_INJECT
        end

        FT_OUT ==>|"定制化模型<br/>作为 RAG 的 LLM"| RAG_INJECT

        RESULT["🎯 最终效果"]
        RAG_INJECT ==> RESULT

        R1["✅ 会说行业话（微调的功劳）"]
        R2["✅ 知识最新最全（RAG 的功劳）"]
        R3["✅ 答案可溯源（RAG 的功劳）"]
        RESULT --- R1 & R2 & R3
    end

    style COMBINED fill:#fafafa,stroke:#bdbdbd,stroke-width:2px
    style PHASE1 fill:#fff3e0,stroke:#ff8f00,stroke-width:2px
    style PHASE2 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style RESULT fill:#4caf50,color:#fff,stroke-width:2px
    style R1 fill:#e8f5e9,stroke:#2e7d32
    style R2 fill:#e8f5e9,stroke:#2e7d32
    style R3 fill:#e8f5e9,stroke:#2e7d32
```

---

## 第四部分：一个关键洞察 💡

> **RAG 系统调优的主战场永远是检索层，不是换更强的 LLM。**

为什么？因为 LLM 只是在"复述和整理"检索到的内容：

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    subgraph GOOD ["✅ 正确做法：优化检索层"]
        G1["🔍 好的检索<br/>Chunking 策略优化<br/>Embedding 模型升级<br/>Rerank 精排调优"]
        G2["🤖 普通 LLM<br/>GPT-3.5 / Qwen"]
        G3["🎉 好答案！<br/>准确、有依据"]
        G1 --> G2 --> G3
    end

    subgraph BAD ["❌ 常见误区：只换更强的 LLM"]
        B1["🔍 差的检索<br/>chunk 切得不好<br/>相关内容没召回"]
        B2["🤖 最强 LLM<br/>GPT-4 / Claude"]
        B3["😢 还是答不好<br/>巧妇难为无米之炊"]
        B1 --> B2 --> B3
    end

    style GOOD fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style BAD fill:#ffebee,stroke:#c62828,stroke-width:2px
    style G3 fill:#4caf50,color:#fff,stroke-width:0px
    style B3 fill:#f44336,color:#fff,stroke-width:0px
```

很多人一上来就想用 GPT-4 换掉 GPT-3.5 来提升效果，这往往是南辕北辙。

真正该优化的三个方向：

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph LR
    CENTER["🎯 RAG 调优<br/>三大主战场"]
    CENTER --> A["✂️ Chunking 策略<br/>chunk 大小 · 重叠度<br/>按语义切 vs 按长度切"]
    CENTER --> B["🔢 Embedding 模型<br/>选择合适的向量模型<br/>BGE / E5 / OpenAI"]
    CENTER --> C["🎯 Rerank 精排<br/>Cross-Encoder 模型<br/>过滤噪声，提升精度"]

    style CENTER fill:#ff9800,color:#fff,stroke-width:2px
    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style B fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style C fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
```

---

## 第五部分：用生活比喻总结

```mermaid
%%{init: {"theme":"default","themeVariables":{"fontSize":"12px"},"flowchart":{"nodeSpacing":15,"rankSpacing":30}}}%%
graph TD
    subgraph SUMMARY ["📝 一张图总结 RAG & 微调"]
        direction TB

        subgraph FT_BOX ["🎓 微调 = 让学生重新上学"]
            FT1["📚 把新知识刻进脑子里"]
            FT2["💪 以后随时能用，不用翻书"]
            FT3["💸 但上学要花钱花时间"]
            FT4["🔄 知识过时了还得再上一遍"]
        end

        subgraph RAG_BOX ["📖 RAG = 给学生开卷考试"]
            RAG1["📂 不用死记硬背"]
            RAG2["🔍 考试时翻书就行"]
            RAG3["🔄 书可以随时换新的"]
            RAG4["📌 还能标注答案出自哪一页"]
        end

        subgraph BEST_BOX ["🏆 组合 = 上完学 + 开卷考试"]
            BEST1["🎓 有扎实的基础功底"]
            BEST2["📖 又能查阅最新资料"]
            BEST3["🎯 这才是工程中的最优解"]
        end

        FT_BOX -->|"+"| BEST_BOX
        RAG_BOX -->|"+"| BEST_BOX
    end

    style SUMMARY fill:#fafafa,stroke:#bdbdbd,stroke-width:2px
    style FT_BOX fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style RAG_BOX fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style BEST_BOX fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
```

---

## 📚 参考来源

- [鹅厂面试官：什么是 RAG？工作流程是怎样的？](https://mp.weixin.qq.com/s/KnNx_ewIeJ_CZhfs6HtnTA) — 小林coding，2026-04-20
- [阿里二面：你知道 RAG 和微调有什么区别吗？](https://mp.weixin.qq.com/s/pKpIXVuFGTebA7MvmqmXxA) — 小林coding，2026-04-21

> 内容基于以上文章整理，已重新组织结构并用通俗语言改写，适合初学者阅读。
