"""
模型压缩 - 知识蒸馏：大模型教小模型

作者: RJ.Wang
时间: 2026-04-20

学习目标:
  - 理解知识蒸馏的核心思想
  - 实现一个简单的蒸馏训练过程
  - 对比蒸馏前后小模型的性能
  - 理解温度参数的作用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# 1. 知识蒸馏的直觉
# ============================================================
print("=" * 50)
print("1. 知识蒸馏 - 直觉理解")
print("=" * 50)

print("""
类比: 老师教学生

  大模型（老师）: 参数多、能力强、但太大太慢
  小模型（学生）: 参数少、速度快、但能力弱

  知识蒸馏: 让大模型"教"小模型
    - 不是直接复制权重（大小不一样）
    - 而是让小模型学习大模型的"思考方式"

  具体怎么教？
    - 普通训练: 小模型学习 [1, 0, 0]（硬标签）
    - 蒸馏训练: 小模型学习 [0.7, 0.2, 0.1]（软标签）

  软标签包含更多信息:
    "这张图 70% 像猫，20% 像狗，10% 像兔子"
    比 "这是猫" 包含了更多关于类别关系的知识！
""")

# ============================================================
# 2. 温度参数 (Temperature)
# ============================================================
print("=" * 50)
print("2. 温度参数的作用")
print("=" * 50)

# 模拟大模型的输出 logits
logits = torch.tensor([5.0, 2.0, 0.5, -1.0])
labels = ["猫", "狗", "兔子", "鱼"]

print("大模型输出的 logits:", logits.tolist())
print()

for T in [1.0, 2.0, 5.0, 10.0]:
    probs = F.softmax(logits / T, dim=0)
    print(f"温度 T={T:4.1f}: ", end="")
    for label, p in zip(labels, probs):
        bar = "█" * int(p * 30)
        print(f"{label}={p:.3f} {bar}  ", end="")
    print()

print("""
观察:
  T=1.0: 概率集中在"猫"上（接近硬标签）
  T=10:  概率更均匀，保留了类别间的关系信息

蒸馏时用较高的温度（T=3~10），让软标签更"软"，
这样小模型能学到更多关于类别关系的知识。
""")

# ============================================================
# 3. 实现知识蒸馏
# ============================================================
print("=" * 50)
print("3. 知识蒸馏实战")
print("=" * 50)

# 生成模拟数据（二维分类问题）
np.random.seed(42)
torch.manual_seed(42)

n_samples = 1000
n_classes = 5
input_dim = 20

# 生成训练数据
X_train = torch.randn(n_samples, input_dim)
y_train = torch.randint(0, n_classes, (n_samples,))

X_test = torch.randn(200, input_dim)
y_test = torch.randint(0, n_classes, (200,))


# 大模型（老师）
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# 小模型（学生）
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


teacher_params = sum(p.numel() for p in TeacherModel().parameters())
student_params = sum(p.numel() for p in StudentModel().parameters())
print(f"老师模型参数量: {teacher_params:,}")
print(f"学生模型参数量: {student_params:,}")
print(f"压缩比: {teacher_params / student_params:.1f}x")


# ============================================================
# 训练函数
# ============================================================

def train_model(model, X, y, epochs=100, lr=0.01):
    """普通训练"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def evaluate(model, X, y):
    """评估准确率"""
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        acc = (pred == y).float().mean().item()
    return acc


def distill_train(
    student, teacher, X, y,
    temperature=5.0, alpha=0.7,
    epochs=100, lr=0.01,
):
    """蒸馏训练

    损失 = α * 蒸馏损失 + (1-α) * 硬标签损失

    Args:
        student: 学生模型
        teacher: 老师模型（已训练好）
        temperature: 温度参数
        alpha: 蒸馏损失的权重
    """
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion_hard = nn.CrossEntropyLoss()

    teacher.eval()

    for epoch in range(epochs):
        student.train()

        # 学生输出
        student_logits = student(X)

        # 老师输出（不需要梯度）
        with torch.no_grad():
            teacher_logits = teacher(X)

        # 蒸馏损失: 学生的软概率 vs 老师的软概率
        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        distill_loss = distill_loss * (temperature ** 2)  # 缩放

        # 硬标签损失
        hard_loss = criterion_hard(student_logits, y)

        # 总损失
        loss = alpha * distill_loss + (1 - alpha) * hard_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return student


# ============================================================
# 4. 对比实验
# ============================================================
print("\n" + "=" * 50)
print("4. 对比实验")
print("=" * 50)

# Step 1: 训练老师模型
print("\n训练老师模型...")
teacher = TeacherModel()
teacher = train_model(teacher, X_train, y_train, epochs=200, lr=0.005)
teacher_acc = evaluate(teacher, X_test, y_test)
print(f"  老师模型测试准确率: {teacher_acc:.2%}")

# Step 2: 直接训练学生模型（不蒸馏）
print("\n直接训练学生模型（无蒸馏）...")
student_no_distill = StudentModel()
student_no_distill = train_model(student_no_distill, X_train, y_train, epochs=200, lr=0.005)
student_no_distill_acc = evaluate(student_no_distill, X_test, y_test)
print(f"  学生模型（无蒸馏）测试准确率: {student_no_distill_acc:.2%}")

# Step 3: 蒸馏训练学生模型
print("\n蒸馏训练学生模型...")
student_distilled = StudentModel()
student_distilled = distill_train(
    student_distilled, teacher, X_train, y_train,
    temperature=5.0, alpha=0.7, epochs=200, lr=0.005,
)
student_distilled_acc = evaluate(student_distilled, X_test, y_test)
print(f"  学生模型（蒸馏后）测试准确率: {student_distilled_acc:.2%}")

# 结果对比
print("\n" + "=" * 50)
print("📊 结果对比")
print("=" * 50)
print(f"  老师模型 ({teacher_params:,} 参数):  {teacher_acc:.2%}")
print(f"  学生-无蒸馏 ({student_params:,} 参数): {student_no_distill_acc:.2%}")
print(f"  学生-蒸馏后 ({student_params:,} 参数): {student_distilled_acc:.2%}")

improvement = student_distilled_acc - student_no_distill_acc
print(f"\n  蒸馏提升: {improvement:+.2%}")

# ============================================================
# 5. 温度参数的影响
# ============================================================
print("\n" + "=" * 50)
print("5. 不同温度的蒸馏效果")
print("=" * 50)

for T in [1.0, 3.0, 5.0, 10.0, 20.0]:
    student_t = StudentModel()
    student_t = distill_train(
        student_t, teacher, X_train, y_train,
        temperature=T, alpha=0.7, epochs=200, lr=0.005,
    )
    acc = evaluate(student_t, X_test, y_test)
    bar = "█" * int(acc * 40)
    print(f"  T={T:5.1f}: {acc:.2%} {bar}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 50)
print("💡 知识蒸馏总结")
print("=" * 50)
print("""
核心思想:
  大模型的"软标签"包含了类别间关系的知识，
  小模型通过学习这些软标签，能获得超越自身能力的表现。

关键参数:
  - Temperature (T): 控制软标签的"软度"，通常 3-10
  - Alpha (α): 蒸馏损失的权重，通常 0.5-0.9

实际应用:
  - DistilBERT: BERT 的蒸馏版，速度快 60%，保留 97% 性能
  - TinyLLaMA: LLaMA 的小型版本
  - 移动端部署: 大模型蒸馏成小模型，在手机上运行

蒸馏 vs 量化:
  - 量化: 同一个模型，降低精度 → 更小更快
  - 蒸馏: 换一个小模型，学习大模型的知识 → 更小更快
  - 可以组合使用: 先蒸馏再量化！
""")

print("✅ 知识蒸馏练习完成！")
