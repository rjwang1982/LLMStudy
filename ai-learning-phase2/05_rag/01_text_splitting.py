"""
RAG - 文本分块策略

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 理解为什么需要文本分块
  - 掌握不同的分块策略
  - 理解 chunk_size 和 overlap 的影响
  - 为构建 RAG 系统做准备
"""

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ============================================================
# 1. 为什么需要文本分块？
# ============================================================
print("=" * 50)
print("1. 为什么需要文本分块？")
print("=" * 50)

print("""
问题: 一篇 10 万字的文档，不能直接塞给大模型。
  1. 大模型有上下文长度限制（如 4K/8K/128K tokens）
  2. 太长的文本会稀释关键信息
  3. Embedding 模型对长文本效果差

解决: 把长文档切成小块（chunks），分别编码和检索。

关键参数:
  - chunk_size: 每块的大小（字符数）
  - chunk_overlap: 相邻块的重叠部分（防止信息断裂）
""")

# 准备一段示例文档
sample_document = """
人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。AI 的发展历程可以追溯到 1950 年代，当时 Alan Turing 提出了著名的图灵测试。

机器学习是 AI 的核心方法之一。它让计算机能够从数据中学习，而不需要显式编程。常见的机器学习方法包括监督学习、无监督学习和强化学习。监督学习使用标注数据训练模型，无监督学习从未标注数据中发现模式，强化学习通过与环境交互来学习最优策略。

深度学习是机器学习的一个子领域，使用多层神经网络来处理复杂的数据。卷积神经网络（CNN）擅长处理图像数据，循环神经网络（RNN）适合处理序列数据，而 Transformer 架构则在自然语言处理领域取得了突破性进展。

大语言模型（LLM）是基于 Transformer 架构的超大规模语言模型。GPT 系列、LLaMA、Claude 等都是典型的大语言模型。这些模型通过在海量文本数据上进行预训练，学会了丰富的语言知识和推理能力。

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。它先从知识库中检索相关信息，然后将检索结果作为上下文提供给大模型，让模型基于这些信息生成回答。RAG 可以有效减少大模型的幻觉问题，并让模型能够访问最新的知识。
""".strip()

print(f"\n示例文档长度: {len(sample_document)} 字符")
print(f"段落数: {len(sample_document.split(chr(10) + chr(10)))}")

# ============================================================
# 2. 简单字符分块
# ============================================================
print("\n" + "=" * 50)
print("2. 简单字符分块 (CharacterTextSplitter)")
print("=" * 50)

splitter_simple = CharacterTextSplitter(
    separator="\n\n",     # 按段落分割
    chunk_size=200,       # 每块最多 200 字符
    chunk_overlap=20,     # 重叠 20 字符
)

chunks_simple = splitter_simple.split_text(sample_document)

print(f"分块数: {len(chunks_simple)}")
for i, chunk in enumerate(chunks_simple):
    print(f"\n--- Chunk {i+1} ({len(chunk)} 字符) ---")
    # 只显示前 80 字符
    preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
    print(f"  {preview}")

# ============================================================
# 3. 递归字符分块（推荐）
# ============================================================
print("\n" + "=" * 50)
print("3. 递归字符分块 (RecursiveCharacterTextSplitter)")
print("=" * 50)

print("""
递归分块的策略:
  依次尝试用这些分隔符切分: ["\\n\\n", "\\n", " ", ""]
  先尝试按段落分，段落太长就按行分，行太长就按空格分...
  这样能尽量保持语义完整性。
""")

splitter_recursive = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)

chunks_recursive = splitter_recursive.split_text(sample_document)

print(f"分块数: {len(chunks_recursive)}")
for i, chunk in enumerate(chunks_recursive):
    print(f"\n--- Chunk {i+1} ({len(chunk)} 字符) ---")
    preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
    print(f"  {preview}")

# ============================================================
# 4. chunk_size 的影响
# ============================================================
print("\n" + "=" * 50)
print("4. chunk_size 对分块的影响")
print("=" * 50)

for size in [100, 200, 400, 800]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=int(size * 0.1),
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_text(sample_document)
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    print(f"  chunk_size={size:4d} → {len(chunks):2d} 块, 平均长度 {avg_len:.0f} 字符")

print("""
选择建议:
  - chunk_size 太小: 块太多，上下文碎片化，检索噪声大
  - chunk_size 太大: 块太少，信息稀释，不够精确
  - 一般推荐: 500-1000 字符（中文），配合 10-20% 的 overlap
  - 需要根据实际数据和任务调优
""")

# ============================================================
# 5. overlap 的作用
# ============================================================
print("=" * 50)
print("5. overlap 的作用")
print("=" * 50)

# 无重叠
splitter_no_overlap = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=0,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
# 有重叠
splitter_with_overlap = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)

chunks_no = splitter_no_overlap.split_text(sample_document)
chunks_yes = splitter_with_overlap.split_text(sample_document)

print(f"无重叠: {len(chunks_no)} 块")
print(f"有重叠 (50字符): {len(chunks_yes)} 块")

# 展示重叠效果
if len(chunks_yes) >= 2:
    c1 = chunks_yes[0]
    c2 = chunks_yes[1]
    # 找重叠部分
    overlap_text = ""
    for length in range(min(len(c1), len(c2)), 0, -1):
        if c1.endswith(c2[:length]):
            overlap_text = c2[:length]
            break

    if overlap_text:
        print(f"\nChunk 1 结尾: ...{c1[-60:]}")
        print(f"Chunk 2 开头: {c2[:60]}...")
        print(f"重叠部分 ({len(overlap_text)} 字符): '{overlap_text[:50]}...'")
    print("\n→ 重叠确保了跨块的信息不会丢失！")

# ============================================================
# 6. 实战案例：公司手册 — 为什么必须分块？
# ============================================================
print("\n" + "=" * 50)
print("6. 实战案例：公司手册")
print("=" * 50)

print("""
场景: 你有一份公司员工手册，包含请假、报销、绩效、出差四个章节。
用户问: "出差补贴标准是多少？"

如果整篇文档压成一个向量会怎样？
""")

company_handbook = """
第一章：请假制度

员工请假需提前 3 个工作日提交申请。年假按工龄计算：1-5 年 5 天，5-10 年 10 天，10 年以上 15 天。病假需提供医院证明，每年累计不超过 30 天。事假每月不超过 3 天，需部门主管审批。婚假 3 天，产假按国家规定执行。

第二章：报销流程

报销需在费用发生后 30 天内提交。差旅费、办公用品、培训费用均可报销。报销金额 500 元以下由部门主管审批，500-5000 元需总监审批，5000 元以上需 VP 审批。报销时需提供正规发票和费用明细。电子发票需打印后粘贴在报销单上。

第三章：绩效考核

绩效考核每季度进行一次，采用 OKR + 360 度评估相结合的方式。考核结果分为 S/A/B/C/D 五个等级。S 级占比不超过 10%，D 级需进入绩效改进计划（PIP）。年终奖金与全年绩效挂钩，S 级 4 个月，A 级 3 个月，B 级 2 个月。

第四章：出差规定

出差需提前 5 个工作日提交出差申请。出差补贴标准：一线城市每天 200 元，二线城市每天 150 元，三线及以下城市每天 100 元。住宿标准：一线城市不超过 500 元/晚，二线城市不超过 350 元/晚。机票经济舱，高铁二等座。出差期间的餐费包含在补贴中，不再单独报销。出差超过 7 天需总监审批。
""".strip()

print(f"公司手册总长度: {len(company_handbook)} 字符")
print(f"包含 4 个章节: 请假、报销、绩效、出差\n")

# 方案 A: 不分块，整篇做 Embedding
print("--- 方案 A: 不分块（整篇文档 → 1 个向量）---")
print("""
  整篇手册 → [0.3, 0.25, 0.1, ...]
  → 一个"啥都像又啥都不像"的模糊向量
  → 用户问"出差补贴"，但"出差"只占四分之一，被其他话题稀释
  → 相似度不高，检索效果差 ❌
""")

# 方案 B: 按章节分块
print("--- 方案 B: 按章节分块（4 个 chunk → 4 个向量）---")

splitter_handbook = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    separators=["\n\n", "\n", "。", " ", ""],
)

handbook_chunks = splitter_handbook.split_text(company_handbook)
print(f"  分成 {len(handbook_chunks)} 个块:\n")

for i, chunk in enumerate(handbook_chunks):
    # 提取章节标题
    first_line = chunk.strip().split("\n")[0]
    print(f"  Chunk {i+1}: [{first_line}] ({len(chunk)} 字符)")

print("""
  → 每个 chunk 精准代表一个主题
  → 用户问"出差补贴"，直接命中"出差规定"那个 chunk ✅
  → 这就是为什么 Chunking 是 RAG 的第一步！
""")

# 用 Embedding 实际验证
print("--- 用 Embedding 实际验证检索效果 ---")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = "出差补贴标准是多少？"
    query_vec = model.encode([query])[0]

    # 方案 A: 整篇文档的向量
    whole_doc_vec = model.encode([company_handbook])[0]
    sim_whole = np.dot(query_vec, whole_doc_vec) / (
        np.linalg.norm(query_vec) * np.linalg.norm(whole_doc_vec)
    )

    # 方案 B: 分块后每个 chunk 的向量
    chunk_vecs = model.encode(handbook_chunks)
    sims_chunks = [
        np.dot(query_vec, cv) / (np.linalg.norm(query_vec) * np.linalg.norm(cv))
        for cv in chunk_vecs
    ]

    print(f"\n  查询: '{query}'\n")
    print(f"  方案 A（整篇）相似度: {sim_whole:.4f}")
    print(f"  方案 B（分块）相似度:")
    for i, (chunk, sim) in enumerate(zip(handbook_chunks, sims_chunks)):
        first_line = chunk.strip().split("\n")[0]
        marker = " ← 命中！" if sim == max(sims_chunks) else ""
        bar = "█" * int(sim * 30)
        print(f"    Chunk {i+1} [{first_line}]: {sim:.4f} {bar}{marker}")

    print(f"\n  结论: 分块后最高相似度 {max(sims_chunks):.4f} > 整篇 {sim_whole:.4f}")
    print(f"  → 分块让检索精度提升了 {((max(sims_chunks) - sim_whole) / sim_whole * 100):.1f}%！")

except ImportError:
    print("  (需要 sentence-transformers，跳过实际验证)")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 50)
print("💡 文本分块最佳实践")
print("=" * 50)
print("""
1. 优先使用 RecursiveCharacterTextSplitter
2. 中文文档建议 chunk_size=500-1000
3. overlap 设为 chunk_size 的 10-20%
4. 根据文档类型调整分隔符
5. 分块后检查质量：每块是否语义完整

分块质量直接影响 RAG 的检索效果！

📖 延伸阅读: 同目录下的 "RAG_and_Fine-tuning_学习指南.md"
   包含 RAG 和微调的完整原理讲解、常见疑问解答和选型决策树。
""")

print("✅ 文本分块策略练习完成！")
