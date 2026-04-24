"""
多模态模型 - 图文搜索实战

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 使用 sentence-transformers 的 CLIP 模型
  - 实现文字搜图片（text-to-image）
  - 实现图片搜文字（image-to-text）
  - 理解跨模态检索的实际应用
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 生成测试图片
# ============================================================
print("=" * 50)
print("1. 生成测试图片")
print("=" * 50)

# 用 matplotlib 生成一些简单的测试图片
output_dir = "07_multimodal/test_images"
os.makedirs(output_dir, exist_ok=True)


def create_test_images():
    """生成简单的测试图片"""
    images_info = []

    # 图片1: 红色圆形
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    circle = plt.Circle((0.5, 0.5), 0.4, color="red")
    ax.add_patch(circle)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    path = f"{output_dir}/red_circle.png"
    plt.savefig(path, bbox_inches="tight", dpi=72)
    plt.close()
    images_info.append(("red_circle.png", "红色圆形"))

    # 图片2: 蓝色方形
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, color="blue")
    ax.add_patch(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    path = f"{output_dir}/blue_square.png"
    plt.savefig(path, bbox_inches="tight", dpi=72)
    plt.close()
    images_info.append(("blue_square.png", "蓝色方形"))

    # 图片3: 绿色三角形
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    triangle = plt.Polygon([[0.5, 0.9], [0.1, 0.1], [0.9, 0.1]], color="green")
    ax.add_patch(triangle)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    path = f"{output_dir}/green_triangle.png"
    plt.savefig(path, bbox_inches="tight", dpi=72)
    plt.close()
    images_info.append(("green_triangle.png", "绿色三角形"))

    # 图片4: 折线图
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    x = np.linspace(0, 10, 50)
    ax.plot(x, np.sin(x), "b-", linewidth=2)
    ax.set_title("sin(x)", fontsize=10)
    ax.grid(True, alpha=0.3)
    path = f"{output_dir}/line_chart.png"
    plt.savefig(path, bbox_inches="tight", dpi=72)
    plt.close()
    images_info.append(("line_chart.png", "正弦波折线图"))

    # 图片5: 散点图
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    np.random.seed(42)
    ax.scatter(np.random.randn(30), np.random.randn(30), c="orange", alpha=0.7)
    ax.set_title("scatter", fontsize=10)
    ax.grid(True, alpha=0.3)
    path = f"{output_dir}/scatter_plot.png"
    plt.savefig(path, bbox_inches="tight", dpi=72)
    plt.close()
    images_info.append(("scatter_plot.png", "橙色散点图"))

    return images_info


images_info = create_test_images()
print(f"✅ 生成了 {len(images_info)} 张测试图片:")
for name, desc in images_info:
    print(f"  📷 {name}: {desc}")

# ============================================================
# 2. 使用 CLIP 模型编码图文
# ============================================================
print("\n" + "=" * 50)
print("2. 使用 CLIP 模型")
print("=" * 50)

print("加载 CLIP 模型（首次会下载）...")

from sentence_transformers import SentenceTransformer

# 加载支持图文的 CLIP 模型
model = SentenceTransformer("clip-ViT-B-32")
print("✅ CLIP 模型加载完成！")

# 编码图片
print("\n编码图片...")
image_paths = [f"{output_dir}/{info[0]}" for info in images_info]
images = [Image.open(p) for p in image_paths]
image_embeddings = model.encode(images)
print(f"图片嵌入 shape: {image_embeddings.shape}")

# 编码文本
texts = [
    "a red circle",
    "a blue square",
    "a green triangle",
    "a line chart showing a sine wave",
    "a scatter plot with orange dots",
    "a photo of a cat",
    "a bar chart",
    "a purple star",
]

print(f"\n编码 {len(texts)} 段文本...")
text_embeddings = model.encode(texts)
print(f"文本嵌入 shape: {text_embeddings.shape}")

# ============================================================
# 3. 文字搜图片 (Text → Image)
# ============================================================
print("\n" + "=" * 50)
print("3. 文字搜图片")
print("=" * 50)


def text_to_image_search(query_text, image_embs, image_names, model, top_k=3):
    """用文字搜索最匹配的图片"""
    query_emb = model.encode([query_text])

    # 计算余弦相似度
    similarities = np.dot(image_embs, query_emb.T).flatten()
    similarities = similarities / (
        np.linalg.norm(image_embs, axis=1) * np.linalg.norm(query_emb)
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]

    print(f"\n🔍 查询: '{query_text}'")
    for rank, idx in enumerate(top_indices, 1):
        print(f"   Top{rank} [{similarities[idx]:.4f}]: {image_names[idx]}")

    return top_indices


image_names = [info[1] for info in images_info]

# 测试搜索
search_queries = [
    "a red shape",
    "a chart or graph",
    "a blue geometric shape",
    "something green",
    "data visualization",
]

for query in search_queries:
    text_to_image_search(query, image_embeddings, image_names, model)

# ============================================================
# 4. 图片搜文字 (Image → Text)
# ============================================================
print("\n" + "=" * 50)
print("4. 图片搜文字")
print("=" * 50)


def image_to_text_search(image_idx, image_embs, text_embs, texts, top_k=3):
    """用图片搜索最匹配的文字描述"""
    query_emb = image_embs[image_idx:image_idx+1]

    similarities = np.dot(text_embs, query_emb.T).flatten()
    similarities = similarities / (
        np.linalg.norm(text_embs, axis=1) * np.linalg.norm(query_emb)
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]

    print(f"\n📷 图片: '{images_info[image_idx][1]}'")
    for rank, idx in enumerate(top_indices, 1):
        print(f"   Top{rank} [{similarities[idx]:.4f}]: {texts[idx]}")


for i in range(len(images_info)):
    image_to_text_search(i, image_embeddings, text_embeddings, texts)

# ============================================================
# 5. 图文相似度矩阵
# ============================================================
print("\n" + "=" * 50)
print("5. 图文相似度矩阵")
print("=" * 50)

# 计算完整的图文相似度矩阵
sim_matrix = np.dot(image_embeddings, text_embeddings.T)
# 归一化
norms_img = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
norms_txt = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
sim_matrix = sim_matrix / (norms_img @ norms_txt.T)

# 可视化
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(sim_matrix, cmap="YlOrRd", aspect="auto")

ax.set_yticks(range(len(images_info)))
ax.set_yticklabels([info[1] for info in images_info], fontsize=10)
ax.set_xticks(range(len(texts)))
ax.set_xticklabels(texts, fontsize=8, rotation=45, ha="right")
ax.set_title("CLIP 图文相似度矩阵", fontsize=14)

for i in range(len(images_info)):
    for j in range(len(texts)):
        ax.text(j, i, f"{sim_matrix[i,j]:.2f}",
                ha="center", va="center", fontsize=7)

plt.colorbar(im)
plt.tight_layout()
plt.savefig("07_multimodal/clip_similarity.png", dpi=150)
plt.close()
print("✅ 相似度矩阵已保存: clip_similarity.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 50)
print("💡 图文搜索总结")
print("=" * 50)
print("""
CLIP 的实际应用:
  1. 图片搜索引擎: 用文字描述搜索图片
  2. 图片自动标注: 给图片生成文字描述
  3. 内容审核: 检测图片是否包含特定内容
  4. 推荐系统: 根据用户文字偏好推荐图片

代码核心就三步:
  1. model = SentenceTransformer("clip-ViT-B-32")
  2. image_emb = model.encode(images)
  3. text_emb = model.encode(texts)
  然后计算相似度就行了！

进阶方向:
  - 使用更大的 CLIP 模型（ViT-L-14）
  - 结合向量数据库做大规模图片检索
  - 用 CLIP 做零样本图片分类
  - 学习 LLaVA 等视觉语言模型
""")

print("✅ 图文搜索实战完成！")
print("🎉 恭喜！阶段二全部完成！")
print("   你已经掌握了 Embedding、RAG、模型压缩和多模态的核心概念。")
print("   下一步: 阶段三 - LangChain 和 Agent 应用开发！")
