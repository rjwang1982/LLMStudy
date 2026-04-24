# AI 大模型学习 - 阶段二：学核心

**作者**: RJ.Wang
**邮箱**: wangrenjun@gmail.com
**创建时间**: 2026-04-22

---

## 前置条件

开始本阶段学习前，请确认你已具备以下条件：

| 类别 | 要求 | 如何验证 |
|------|------|----------|
| ✅ 阶段一 | 已完成阶段一全部练习 | 能用 PyTorch 训练 MNIST，理解 Transformer |
| 🐍 Python | 熟练使用 NumPy、函数、类 | 能独立写一个矩阵乘法并理解 shape |
| 🧠 神经网络 | 理解前向/反向传播 | 能解释 `loss.backward()` 做了什么 |
| 🤖 Transformer | 理解 Self-Attention | 能说出 Q/K/V 的含义和计算过程 |
| 💻 硬件 | Apple Silicon Mac（推荐） | oLMX 需要 M1/M2/M3/M4 芯片 |
| 🌐 网络 | 可访问 HuggingFace | 首次运行需下载预训练模型（约 100-500MB） |
| 🤖 oLMX | 已安装并运行 oLMX | 访问 `http://127.0.0.1:8019/admin` 确认 |

> ⚠️ 如果你跳过了阶段一，至少确保你理解：矩阵乘法、梯度下降、Softmax、注意力机制。否则本阶段的内容会很难跟上。

---

## 学习路线总览

本阶段在整个学习路线中的位置：

```mermaid
graph LR
    A["📌 阶段一<br/>打基础"] --> B["阶段二<br/>学核心"]
    B --> C["阶段三<br/>做应用"]

    style A fill:#86efac,stroke:#22c55e,color:#000
    style B fill:#fbbf24,stroke:#d97706,color:#000,stroke-width:2px
    style C fill:#e5e7eb,stroke:#9ca3af,color:#000
```

---

## 阶段二内部学习流程

```mermaid
graph TD
    subgraph M1["📘 模块四：Embeddings"]
        A1["01 词向量基础<br/>手动构造 + 余弦相似度"] --> A2["02 句子嵌入<br/>sentence-transformers"]
        A2 --> A3["03 向量检索<br/>FAISS 实战"]
    end

    subgraph M2["📗 模块五：RAG"]
        B1["01 文本分块<br/>分块策略与参数调优"] --> B2["02 向量数据库<br/>ChromaDB 增删改查"]
        B2 --> B3["03 RAG 管道<br/>检索 + Prompt + 生成"]
    end

    subgraph M3["📙 模块六：模型压缩"]
        C1["01 量化<br/>FP32→INT8 原理与实战"] --> C2["02 知识蒸馏<br/>大模型教小模型"]
    end

    subgraph M4["📕 模块七：多模态"]
        D1["01 CLIP 原理<br/>对比学习 + 图文对齐"] --> D2["02 图文搜索<br/>跨模态检索实战"]
    end

    M1 --> M2 --> M3 --> M4

    style M1 fill:#dbeafe,stroke:#3b82f6
    style M2 fill:#dcfce7,stroke:#22c55e
    style M3 fill:#fef3c7,stroke:#f59e0b
    style M4 fill:#fce7f3,stroke:#ec4899
```

---

## 本地 LLM 环境：oLMX

本阶段的 RAG 管道练习使用 [oLMX](https://github.com/jundot/omlx) 作为本地 LLM 推理后端，无需 OpenAI API Key。

```mermaid
graph TB
    subgraph oLMX["🖥️ oLMX 本地推理服务 (http://127.0.0.1:8019)"]
        direction LR
        M1["Qwen3.5-4B-MLX-4bit<br/>轻量快速 - 2.97GB"]
        M2["Qwen3.5-9B-MLX-4bit<br/>更强能力 - 5.82GB"]
    end

    subgraph usage["📚 本阶段使用场景"]
        U1["05_rag/03_rag_pipeline.py<br/>RAG 问答生成"]
    end

    oLMX -->|"OpenAI 兼容 API<br/>/v1/chat/completions"| usage

    style oLMX fill:#1e1b4b,stroke:#6366f1,color:#fff
    style usage fill:#dcfce7,stroke:#22c55e
```

**确认 oLMX 正在运行：**
```bash
# 检查服务状态
curl http://127.0.0.1:8019/v1/models -H "Authorization: Bearer your-api-key"

# 或打开管理后台
open http://127.0.0.1:8019/admin/dashboard
```

**推荐模型选择：**
| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| RAG 问答 | Qwen3.5-9B | 理解能力更强，回答质量更高 |
| 快速测试 | Qwen3.5-4B | 速度快，适合调试代码 |

---

## RAG 架构全景图

这是本阶段最核心的内容 —— RAG 的完整架构：

```mermaid
graph TB
    subgraph offline["📦 离线阶段：索引构建"]
        direction LR
        D["📄 原始文档"] --> S["✂️ 文本分块<br/>RecursiveTextSplitter"]
        S --> E["🧮 Embedding<br/>sentence-transformers"]
        E --> V["🗄️ 向量数据库<br/>ChromaDB / FAISS"]
    end

    subgraph online["🔍 在线阶段：问答"]
        direction LR
        Q["❓ 用户提问"] --> QE["🧮 Query Embedding"]
        QE --> R["🔎 向量检索<br/>Top-K 相似文档"]
        R --> P["📝 构造 Prompt<br/>检索结果 + 问题"]
        P --> LLM["🤖 oLMX 本地 LLM<br/>Qwen3.5-9B"]
        LLM --> A["💬 最终回答"]
    end

    V -.->|"检索"| R

    style offline fill:#eff6ff,stroke:#3b82f6
    style online fill:#f0fdf4,stroke:#22c55e
    style LLM fill:#6366f1,stroke:#4f46e5,color:#fff
```

---

## Embedding 的核心思想

```mermaid
graph LR
    subgraph 文本空间["文本（离散）"]
        T1["猫"]
        T2["狗"]
        T3["汽车"]
    end

    subgraph 向量空间["向量空间（连续）"]
        V1["[0.8, 0.2, 0.9]"]
        V2["[0.7, 0.3, 0.8]"]
        V3["[0.1, 0.7, 0.0]"]
    end

    T1 -->|"Embedding"| V1
    T2 -->|"Embedding"| V2
    T3 -->|"Embedding"| V3

    V1 ---|"相似 ✅"| V2
    V1 ---|"不相似 ❌"| V3

    style 文本空间 fill:#fee2e2,stroke:#ef4444
    style 向量空间 fill:#dbeafe,stroke:#3b82f6
```

---

## 模型压缩方法对比

```mermaid
graph TD
    BIG["🐘 大模型<br/>7B 参数 / 28GB"] --> Q["量化 Quantization<br/>降低精度：FP32→INT8→INT4"]
    BIG --> D["蒸馏 Distillation<br/>大模型教小模型"]
    BIG --> P["剪枝 Pruning<br/>删除不重要的连接"]

    Q --> SQ["同一个模型<br/>更小更快"]
    D --> SD["换一个小模型<br/>保留大部分能力"]
    P --> SP["同一个模型<br/>更稀疏"]

    SQ --> DEPLOY["📱 部署到<br/>手机/边缘设备"]
    SD --> DEPLOY
    SP --> DEPLOY

    style BIG fill:#f87171,stroke:#dc2626,color:#fff
    style DEPLOY fill:#34d399,stroke:#059669,color:#000
```

---

## 多模态：CLIP 对比学习

```mermaid
graph TD
    subgraph training["训练阶段"]
        IMG["🖼️ 图片"] --> IE["图像编码器<br/>ViT / ResNet"]
        TXT["📝 文字描述"] --> TE["文本编码器<br/>Transformer"]
        IE --> IV["图像向量"]
        TE --> TV["文本向量"]
        IV ---|"匹配对：拉近 ✅"| TV
        IV ---|"不匹配：推远 ❌"| TV
    end

    subgraph usage["应用阶段"]
        direction LR
        U1["🔍 文字搜图片"]
        U2["🏷️ 零样本分类"]
        U3["📸 图片搜文字"]
    end

    training --> usage

    style training fill:#eff6ff,stroke:#3b82f6
    style usage fill:#fef3c7,stroke:#f59e0b
```

---

## 项目结构

```
ai-learning-phase2/
├── 04_embeddings/                  # Embeddings 文本向量化
│   ├── 01_word_vectors.py              # 词向量基础：手动构造 + 余弦相似度
│   ├── 02_sentence_embeddings.py       # 句子嵌入：预训练模型编码文本
│   └── 03_similarity_search.py         # 向量检索：FAISS 实战
├── 05_rag/                         # RAG 检索增强生成
│   ├── 01_text_splitting.py            # 文本分块策略（含公司手册实战案例）
│   ├── 02_vector_store.py              # 向量数据库：ChromaDB 实战
│   ├── 03_rag_pipeline.py              # 完整 RAG 管道（接 oLMX）
│   └── 📖 RAG_and_Fine-tuning_学习指南.md  # 补充阅读：RAG & 微调原理详解
├── 06_model_compression/           # 模型蒸馏与压缩
│   ├── 01_quantization.py              # 量化：FP32→INT8
│   └── 02_distillation.py              # 知识蒸馏：大模型教小模型
├── 07_multimodal/                  # 多模态模型
│   ├── 01_clip_concept.py              # CLIP 原理：对比学习
│   └── 02_image_text_search.py         # 图文搜索实战
└── README.md
```

---

## 运行方式

```bash
cd ai-learning-phase2

# 运行某个练习
uv run python 04_embeddings/01_word_vectors.py
```

> 💡 首次运行 `02_sentence_embeddings.py` 和 `02_image_text_search.py` 时会自动下载预训练模型，请确保网络通畅。

---

## 每个练习的学习目标

```mermaid
graph LR
    subgraph 输入["你现在的状态"]
        I1["知道 Transformer"]
        I2["会用 PyTorch"]
        I3["不了解 RAG"]
    end

    subgraph 输出["完成后你能做到"]
        O1["用向量表示任意文本"]
        O2["构建完整 RAG 系统"]
        O3["理解模型量化和蒸馏"]
        O4["实现跨模态图文搜索"]
    end

    输入 --> 输出

    style 输入 fill:#fee2e2,stroke:#ef4444
    style 输出 fill:#dcfce7,stroke:#22c55e
```

| 模块 | 练习 | 你将学到 | 预计时间 |
|------|------|----------|----------|
| Embeddings | 01 词向量 | 为什么需要 Embedding、余弦相似度、类比推理 | 2h |
| Embeddings | 02 句子嵌入 | sentence-transformers 使用、语义搜索 | 2h |
| Embeddings | 03 向量检索 | FAISS 索引构建、暴力搜索 vs 近似搜索 | 2h |
| RAG | 01 文本分块 | chunk_size/overlap 调优、分块策略选择 | 2h |
| RAG | 02 向量数据库 | ChromaDB 增删改查、元数据过滤 | 2h |
| RAG | 03 RAG 管道 | 完整流程：分块→编码→存储→检索→生成 | 4h |
| 模型压缩 | 01 量化 | INT8 量化原理、PyTorch 动态量化 | 2h |
| 模型压缩 | 02 蒸馏 | 软标签、温度参数、对比实验 | 3h |
| 多模态 | 01 CLIP | 对比学习、零样本分类、图文对齐 | 2h |
| 多模态 | 02 图文搜索 | CLIP 模型使用、跨模态检索 | 2h |

---

## 环境变量（可选）

RAG 管道练习默认使用本地 oLMX 服务（`http://127.0.0.1:8019/v1`），无需任何 API Key 配置。

如果你想切换到 OpenAI 云端 API：
```bash
export OPENAI_API_KEY="your-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

---

## 📖 补充阅读

学完 RAG 模块后，强烈建议阅读：

**[RAG & Fine-tuning 小白学习指南](05_rag/RAG_and_Fine-tuning_学习指南.md)**

这篇指南用大量 Mermaid 图和生活比喻，深入讲解了：
- RAG 离线建库和在线问答的每一步细节
- 为什么必须 Chunking（公司手册的例子）
- Embedding 向量到底是什么（1536 维浮点数的直觉理解）
- 向量检索和 Rerank 的工作原理
- 微调 vs RAG 的选型决策树
- 两者组合使用的最佳实践

其中的 Q&A 部分来自真实学习过程中的疑问，逐层递进，非常适合巩固理解。

---

## 完成标志

当你能回答以下问题时，说明阶段二已经掌握：

- [ ] Embedding 和 One-Hot 编码有什么区别？
- [ ] 余弦相似度和欧氏距离有什么不同？
- [ ] RAG 的离线阶段和在线阶段分别做什么？
- [ ] chunk_size 设太大或太小会有什么问题？
- [ ] ChromaDB 的元数据过滤有什么用？
- [ ] INT8 量化能把模型压缩多少倍？
- [ ] 知识蒸馏中温度参数的作用是什么？
- [ ] CLIP 是如何实现图文对齐的？

全部打勾后，进入 **阶段三：做应用（LangChain + Agent）** 🚀
