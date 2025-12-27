import argparse
parser = argparse.ArgumentParser()

# -------------------- 数据划分与随机性相关参数 --------------------

# Dirichlet 分布参数，用于控制客户端之间的数据非 IID 程度
# α 越小 → 数据越不均匀（更极端的 non-IID）
# α 越大 → 越接近均匀分布
parser.add_argument('--dirichlet_alpha', help='Dirichlet 分布参数（控制非IID程度）',
                    type=float, default=0.5)

# 选择 GPU 的编号（0 表示使用第一块 GPU；如果无 GPU，则自动回退到 CPU）
parser.add_argument('--gpu_id', help='GPU 编号', type=int, default=0)

# -------------------- AdaFGL 特有的超参数 --------------------

# 图传播阈值，用于控制个性化阶段的邻域信息融合强度
parser.add_argument('--threshold', help='个性化阶段的传播阈值',
                    type=float, default=0.9)

# 平滑项系数 α：用于调节本地消息的加权（针对异构图结构）
parser.add_argument('--alpha', help='异构传播中的平滑权重系数 alpha',
                    type=float, default=0.05)

# 平衡项系数 β：用于调节全局与本地表示之间的损失平衡
parser.add_argument('--beta', help='本地-全局一致性损失的平衡系数 beta',
                    type=float, default=0.2)

# -------------------- 随机性控制 --------------------

# 随机种子，确保实验可复现
parser.add_argument('--seed', help='随机种子', type=int, default=2023)

# -------------------- 数据集与划分方式 --------------------

# 图数据集名称：Cora / CiteSeer / PubMed / 自定义数据集
parser.add_argument('--data_name', help='数据集名称', type=str, default="Cora")

# 客户端数量（图将被划分为 num_clients 个子图）
parser.add_argument('--num_clients', help='客户端数量', type=int, default=10)

# 图划分方法：Metis / Louvain
parser.add_argument('--partition', help='图划分方式',
                    type=str, default="Metis")

# -------------------- 全局模型（Federated GNN）参数 --------------------

# 全局模型结构：ChebNet / GCN / GAT 等（AdaFGL 默认 ChebNet）
parser.add_argument('--gmodel_name', help='联邦全局模型类型',
                    type=str, default="GCN")

# 联邦训练的全局通信轮数（FedAvg rounds）
parser.add_argument('--num_rounds', help='联邦训练的全局迭代轮数',
                    type=int, default=100)

# 每个客户端本地训练的 epoch 次数（全局模型阶段）
parser.add_argument('--num_epochs', help='全局模型的本地训练 epoch 数',
                    type=int, default=3)

# 全局模型的学习率（local SGD）
parser.add_argument('--lr', help='全局模型学习率',
                    type=float, default=1e-2)

# 全局模型权重衰减（L2 正则）
parser.add_argument('--weight_decay', help='全局模型权重衰减',
                    type=float, default=5e-4)

# 全局模型 dropout 比例
parser.add_argument('--drop', help='全局模型 dropout 概率',
                    type=float, default=0.5)

# -------------------- 个性化阶段（AdaFGL Step2） --------------------

# 个性化训练重复次数
# 每个 subgraph 会运行 normalize_train 次，取平均值
parser.add_argument('--normalize_train', help='个性化阶段的重复训练次数',
                    type=int, default=1)

# 个性化模型（MyModel）隐藏维度大小
parser.add_argument('--hidden_dim', help='个性化模型的隐藏层维度',
                    type=int, default=64)

# 个性化传播 epoch 数（越大传播越充分）
parser.add_argument('--epochs', help='个性化阶段的传播训练 epoch 数',
                    type=int, default=200)

# -------------------- 聚类相关参数 --------------------
parser.add_argument('--emb_dim', type=int, default=64)        # embedding 维度
parser.add_argument('--k_clusters', type=int, default=7)      # 本地聚类簇数(可选)
parser.add_argument('--neg_ratio', type=int, default=1)       # 负采样比例
parser.add_argument('--max_pos_edges', type=int, default=20000)
parser.add_argument('--log_every', type=int, default=5)



parser.add_argument('--eval_every', type=int, default=1)              # 全局评估间隔：每隔多少轮联邦通信评估一次全局 encoder
parser.add_argument('--probe_epochs', type=int, default=100)          # probe 评估时线性分类器/探针模型的训练轮数
parser.add_argument('--probe_lr', type=float, default=1e-2)           # probe 评估阶段使用的学习率
parser.add_argument('--probe_weight_decay', type=float, default=0.0)  # probe 评估阶段的权重衰减系数（L2 正则）

parser.add_argument('--visualize_clusters', action='store_true',default=False)  # 是否开启聚类结果可视化
parser.add_argument('--viz_every', type=int, default=10)               # 聚类可视化的轮次间隔：每隔多少轮进行一次可视化
parser.add_argument('--viz_out_dir', type=str, default='./cluster_viz')# 聚类可视化结果的输出目录
parser.add_argument('--viz_max_points', type=int, default=2000)        # 可视化时最多绘制的节点/样本数量（防止点太多）
parser.add_argument('--viz_kmeans_iters', type=int, default=50)        # 可视化中 K-means 聚类的最大迭代次数

# -------------------- 模型导出相关参数 --------------------
parser.add_argument('--best_model_dir', type=str, default="./checkpoints",  # 最佳模型导出目录
                    help='保存全局 encoder 最佳权重的路径')

args = parser.parse_args()

# 数据集名称
args.data_name="Cora"

# 划分方式  # 图划分方法：Metis / Louvain
args.partition="Louvain"

# 全局模型结构
args.gmodel_name="GCN"

# 客户端数量
args.num_clients=10

# 全局迭代轮数
args.num_rounds=100

# 本地训练 epoch 数
args.num_epochs=3


# print(args)
