"""
Python 基础 - 数据结构练习

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 掌握列表、字典、集合的常用操作
  - 理解列表推导式
  - 学会函数定义和使用
"""

# ============================================================
# 1. 列表 (List) - 最常用的数据结构
# ============================================================
print("=" * 50)
print("1. 列表操作")
print("=" * 50)

# 创建一个成绩列表
scores = [85, 92, 78, 95, 88, 76, 90]
print(f"原始成绩: {scores}")
print(f"最高分: {max(scores)}")
print(f"最低分: {min(scores)}")
print(f"平均分: {sum(scores) / len(scores):.1f}")

# 列表推导式 - AI 编程中非常常用
# 筛选出 >= 90 分的成绩
high_scores = [s for s in scores if s >= 90]
print(f"90分以上: {high_scores}")

# 对每个成绩加 5 分（加分操作）
boosted = [s + 5 for s in scores]
print(f"加5分后: {boosted}")

# ============================================================
# 2. 字典 (Dict) - 键值对存储
# ============================================================
print("\n" + "=" * 50)
print("2. 字典操作")
print("=" * 50)

# 模拟一个模型的配置参数（AI 项目中很常见）
model_config = {
    "model_name": "my_first_model",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "hidden_layers": [128, 64, 32],
}

print(f"模型名称: {model_config['model_name']}")
print(f"学习率: {model_config['learning_rate']}")

# 遍历配置
for key, value in model_config.items():
    print(f"  {key}: {value}")

# 字典推导式
# 把成绩列表变成 {学生编号: 成绩} 的字典
student_scores = {f"学生{i+1}": score for i, score in enumerate(scores)}
print(f"\n学生成绩字典: {student_scores}")

# ============================================================
# 3. 函数 - 代码复用的基础
# ============================================================
print("\n" + "=" * 50)
print("3. 函数定义")
print("=" * 50)


def calculate_stats(data: list[float]) -> dict:
    """计算一组数据的统计信息

    Args:
        data: 数字列表

    Returns:
        包含统计信息的字典
    """
    return {
        "count": len(data),
        "mean": sum(data) / len(data),
        "max": max(data),
        "min": min(data),
        "range": max(data) - min(data),
    }


stats = calculate_stats(scores)
print(f"成绩统计: {stats}")


# 带默认参数的函数
def normalize(data: list[float], min_val: float = 0, max_val: float = 1) -> list[float]:
    """将数据归一化到指定范围（这在 AI 中非常重要！）

    Args:
        data: 原始数据
        min_val: 目标最小值
        max_val: 目标最大值

    Returns:
        归一化后的数据
    """
    data_min = min(data)
    data_max = max(data)
    return [
        min_val + (x - data_min) / (data_max - data_min) * (max_val - min_val)
        for x in data
    ]


normalized = normalize(scores)
print(f"\n归一化到 [0,1]: {[f'{x:.2f}' for x in normalized]}")

normalized_neg = normalize(scores, -1, 1)
print(f"归一化到 [-1,1]: {[f'{x:.2f}' for x in normalized_neg]}")

# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 50)
print("💡 练习题（取消注释后运行）")
print("=" * 50)

# 练习1: 写一个函数，接收一个列表，返回去重后的排序列表
# def unique_sorted(data):
#     pass  # 你的代码

# 练习2: 写一个函数，统计字符串中每个字符出现的次数，返回字典
# def char_count(text):
#     pass  # 你的代码

# 练习3: 用列表推导式生成一个 5x5 的乘法表
# multiplication_table = ...

print("\n✅ Python 基础 - 数据结构练习完成！")
