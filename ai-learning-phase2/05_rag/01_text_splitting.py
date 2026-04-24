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
""")

print("✅ 文本分块策略练习完成！")
