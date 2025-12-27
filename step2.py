import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import args
from data.graph_fl_datasets import GraphFLDataset
from models.gcn_encoder import GCNEncoder
from utils.utils import set_seed

warnings.filterwarnings("ignore")


def best_encoder_checkpoint_path():
    """返回全局阶段最佳编码器的检查点路径。"""
    best_dir = getattr(args, "best_model_dir", "./checkpoints")
    file_name = (
        f"best_encoder_{args.data_name}_{args.partition}_"
        f"{args.gmodel_name}_{args.num_clients}c_seed{args.seed}.pt"
    )
    return Path(best_dir) / file_name

def _load_best_encoder(dataset, device):
    """加载全局阶段保存的最佳编码器权重。"""
    best_path = best_encoder_checkpoint_path()
    if not best_path.exists():
        raise FileNotFoundError(
            f"Best encoder checkpoint not found: {best_path}. "
            "Please run the federated pretraining stage first."
        )

    state = torch.load(best_path, map_location=device)
    model = GCNEncoder(
        feat_dim=dataset.input_dim,
        hidden_dim=args.hidden_dim,
        emb_dim=getattr(args, "emb_dim", args.hidden_dim),
        dropout=args.drop,
    )
    model.load_state_dict(state["model_state_dict"])
    return model


def _to_dense_adj(adj, device):
    """
    尽量把 subgraph.adj 转成 torch 稠密矩阵（小图可行）。
    你的 Cora/CiteSeer/PubMed 这种通常能撑住；如果你是大图，请把这里改成稀疏乘法版本。
    """
    if torch.is_tensor(adj):
        if adj.is_sparse:
            return adj.to(device).to_dense()
        return adj.to(device)
    # scipy sparse
    try:
        import scipy.sparse as sp  # noqa: F401

        if hasattr(adj, "tocoo"):
            coo = adj.tocoo()
            indices = torch.tensor(
                np.vstack([coo.row, coo.col]), dtype=torch.long, device=device
            )
            values = torch.tensor(coo.data, dtype=torch.float32, device=device)
            A = torch.sparse_coo_tensor(
                indices, values, size=coo.shape, device=device
            ).to_dense()
            return A
    except Exception:
        pass
    return torch.tensor(adj, dtype=torch.float32, device=device)


def _sym_norm_adj(A, eps=1e-12):
    """A 是稠密邻接矩阵，做对称归一化：D^{-1/2} (A + I) D^{-1/2}"""
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    A_hat = A + I
    deg = A_hat.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def _propagate(Z, A, steps=2):
    """简单的图平滑：反复做 Z <- A_norm Z"""
    A_norm = _sym_norm_adj(A)
    H = Z
    for _ in range(max(0, int(steps))):
        H = A_norm @ H
    return H


def _cluster_kmeans(X, k, seed):
    """KMeans 聚类（优先 sklearn，没有就用一个简单 torch 版兜底）。"""
    X_np = X.astype(np.float32, copy=False)
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        pred = km.fit_predict(X_np)
        return pred
    except Exception:
        # 简单 torch kmeans 兜底（不如 sklearn 稳定，但能跑）
        Xt = torch.tensor(X_np)
        n, d = Xt.shape
        g = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(n, generator=g)[:k]
        C = Xt[idx].clone()  # k x d
        for _ in range(30):
            dist = torch.cdist(Xt, C)  # n x k
            assign = dist.argmin(dim=1)  # n
            newC = []
            for j in range(k):
                mask = assign == j
                if mask.any():
                    newC.append(Xt[mask].mean(dim=0))
                else:
                    newC.append(C[j])
            newC = torch.stack(newC, dim=0)
            if torch.norm(newC - C) < 1e-6:
                break
            C = newC
        return assign.cpu().numpy()


def _purity_score(y_true, y_pred):
    """Purity = (1/N) * sum_k max_j |C_k ∩ L_j|"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = y_true.shape[0]
    if N == 0:
        return 0.0
    purity = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        if idx.size == 0:
            continue
        labels, counts = np.unique(y_true[idx], return_counts=True)
        purity += counts.max()
    return float(purity) / float(N)


def _clustering_metrics(y_true, y_pred):
    """NMI/ARI/Purity（y_true 只用于评估，不参与训练）。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    try:
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

        nmi = float(normalized_mutual_info_score(y_true, y_pred))
        ari = float(adjusted_rand_score(y_true, y_pred))
    except Exception:
        nmi, ari = 0.0, 0.0
    purity = _purity_score(y_true, y_pred)
    return nmi, ari, purity


def realtrain(datasets=None):
    """改成：冻结全局 encoder ->（可选）图平滑 -> 本地 KMeans 聚类 -> 聚类指标评估"""
    main_normalize_train = getattr(args, "normalize_train", 1)
    gpu_id = args.gpu_id
    num_clients = args.num_clients
    k = getattr(args, "k_clusters", None)

    if k is None:
        raise ValueError("args.k_clusters 未设置：聚类任务需要指定聚类数 k_clusters")

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if datasets is None:
        datasets = GraphFLDataset(
            root="./datasets",
            name=args.data_name,
            sampling=args.partition,
            num_clients=num_clients,
            analysis_local_subgraph=True, #对每一个客户端本地子图做结构分析
            analysis_global_graph=False,  #对全局原始大图做结构分析
        )

    prop_steps = int(getattr(args, "prop_steps", 2))  # 你也可以在 config 里加这个参数
    use_l2norm = int(getattr(args, "use_l2norm", 1))  # 是否对 embedding 做 L2 归一化
    use_smoothing = int(getattr(args, "use_smoothing", 1))  # 是否做图平滑

    print("\n| ★  Start Local Client Clustering (Personalized)")

    global_record = {
        "nmi_mean": 0.0,
        "ari_mean": 0.0,
        "purity_mean": 0.0,
        "nmi_std": 0.0,
        "ari_std": 0.0,
        "purity_std": 0.0,
    }

    t_total = time.time()
    N_global = datasets.global_data.num_nodes

    for i, subgraph in enumerate(datasets.subgraphs):
        subgraph.y = subgraph.y.to(device)

        local_runs = {"nmi": [], "ari": [], "purity": []}

        for rep in range(main_normalize_train):
            # 1) 冻结使用全局 encoder 提取表示
            gmodel = _load_best_encoder(datasets, device)
            gmodel.preprocess(subgraph.adj, subgraph.x)
            gmodel = gmodel.to(device)
            gmodel.eval()

            with torch.no_grad():
                Z = gmodel.model_forward(range(subgraph.num_nodes), device)  # [N, d]
                # 不要 softmax：聚类用连续表示更合适
                if use_l2norm:
                    Z = F.normalize(Z, p=2, dim=1)

            # 2) 可选：用图结构做简单平滑/传播，得到更适合聚类的表示
            if use_smoothing:
                A = _to_dense_adj(subgraph.adj, device=device)
                Z = _propagate(Z, A, steps=prop_steps)
                if use_l2norm:
                    Z = F.normalize(Z, p=2, dim=1)

            # 3) 本地聚类（在本客户端子图所有节点上聚类）
            X = Z.detach().cpu().numpy()
            pred = _cluster_kmeans(X, k=int(k), seed=int(args.seed) + int(i) * 97 + int(rep))

            # 4) 聚类评估（只用 y 做评估）
            y_true = subgraph.y.detach().cpu().numpy()
            nmi, ari, purity = _clustering_metrics(y_true, pred)

            local_runs["nmi"].append(nmi)
            local_runs["ari"].append(ari)
            local_runs["purity"].append(purity)

        # 5) 按客户端节点数加权汇总到全局
        weight = float(subgraph.num_nodes) / float(N_global)
        global_record["nmi_mean"] += float(np.mean(local_runs["nmi"])) * weight
        global_record["ari_mean"] += float(np.mean(local_runs["ari"])) * weight
        global_record["purity_mean"] += float(np.mean(local_runs["purity"])) * weight

        # 标准差也给一个加权版本（与你原本代码风格一致）
        if len(local_runs["nmi"]) > 1:
            global_record["nmi_std"] += float(np.std(local_runs["nmi"], ddof=1)) * weight
            global_record["ari_std"] += float(np.std(local_runs["ari"], ddof=1)) * weight
            global_record["purity_std"] += float(np.std(local_runs["purity"], ddof=1)) * weight
        else:
            global_record["nmi_std"] += 0.0
            global_record["ari_std"] += 0.0
            global_record["purity_std"] += 0.0

    print("| ★  Clustering Completed")
    print(
        "| Normalize Train: {}, Total Time Elapsed: {:.4f}s".format(
            main_normalize_train,
            time.time() - t_total,
        )
    )
    print(
        "| Mean NMI ± Std: {}±{}, Mean ARI ± Std: {}±{}, Mean Purity ± Std: {}±{}".format(
            round(global_record["nmi_mean"], 4),
            round(global_record["nmi_std"], 4),
            round(global_record["ari_mean"], 4),
            round(global_record["ari_std"], 4),
            round(global_record["purity_mean"], 4),
            round(global_record["purity_std"], 4),
        )
    )

    # 你也可以改成返回三项，这里先返回主指标 NMI
    return round(global_record["nmi_mean"], 4)


if __name__ == "__main__":
    realtrain()
