import os
import csv


def save_evaluating_indicator (
        data_name,
        num_clients,
        partition,
        gmodel_name,
        num_rounds,
        NMI,
        ARI,
        Purity,
        Modularity,
        save_dir="./acc_records"
    ):
    """
    创建或追加写入 acc_xxx.csv 文件
    文件格式：
        num_rounds, val_acc, test_acc
    """

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 文件名
    file_name = f"evaluate_{data_name}_{num_clients}_{partition}_{gmodel_name}.csv"
    file_path = os.path.join(save_dir, file_name)

    # 判断文件是否存在
    file_exists = os.path.isfile(file_path)

    # 如果不存在，新建并写入表头
    if not file_exists:
        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["num_rounds", "NMI", "ARI", "Purity", "Modularity"])
        print(f"[INFO] 创建新文件：{file_path}")

    # 追加写入一行数据
    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([num_rounds, NMI, ARI,Purity,Modularity])
