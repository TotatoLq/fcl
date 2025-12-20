import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 需要你配置的部分
# =========================

# 存放 csv 文件的目录
csv_dir = "../acc_records"

# 要画的 csv 文件名（按顺序画）
csv_files = [
    "acc_Cora_10_Louvain_GCN.csv",
    "acc_Cora_10_Metis_GCN.csv",
]

# 对应每条曲线的名字（用于图例）
labels = [
    "Metis Split",
    "Louvain Community",
]

# 保存图片路径（None 表示不保存，只显示）
save_path = "../acc_records/Cora_Metis_Louvain.png"

# =========================
# 主逻辑
# =========================

plt.figure(figsize=(8, 5))

for csv_file, label in zip(csv_files, labels):
    csv_path = os.path.join(csv_dir, csv_file)

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skip.")
        continue

    df = pd.read_csv(csv_path)

    if "test_acc" not in df.columns:
        print(f"Warning: test_acc not in {csv_file}, skip.")
        continue

    test_acc = df["test_acc"].values
    rounds = range(1, len(test_acc) + 1)

    plt.plot(rounds, test_acc, label=label)

# =========================
# 画图美化
# =========================

plt.xlabel("Round")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()

if save_path is not None:
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")

plt.show()
