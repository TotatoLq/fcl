import numpy as np
import torch
import torch.nn.functional as F


# ======================================================
# Utilities
# ======================================================

def _to_dense_adj(adj, device):
    """
    将 subgraph.adj 转成 torch 稠密邻接矩阵
    兼容 torch sparse / scipy sparse / numpy
    """
    if torch.is_tensor(adj):
        if adj.is_sparse:
            return adj.to(device).to_dense()
        return adj.to(device)

    try:
        if hasattr(adj, "tocoo"):
            coo = adj.tocoo()
            indices = torch.tensor(
                np.vstack([coo.row, coo.col]),
                dtype=torch.long,
                device=device,
            )
            values = torch.tensor(
                coo.data,
                dtype=torch.float32,
                device=device,
            )
            return torch.sparse_coo_tensor(
                indices, values, size=coo.shape, device=device
            ).to_dense()
    except Exception:
        pass

    return torch.tensor(adj, dtype=torch.float32, device=device)


def _cluster_kmeans(X: np.ndarray, k: int, seed: int):
    """
    使用工程中已有的 torch kmeans
    """
    from utils.cluster_metrics import kmeans_torch

    X_t = torch.tensor(X, dtype=torch.float32)
    return kmeans_torch(
        X_t,
        k,
        num_iters=50,
        seed=int(seed),
    ).cpu().numpy()


# ======================================================
# AdaFGL Clustering Client (Step 2)
# ======================================================

class AdaFGLClusteringClient:
    """
    AdaFGL-style Step 2 for Federated Graph Clustering (per-client)

    核心思想：
    - 冻结全局 encoder（federated knowledge extractor）
    - 构造知识引导的结构校正矩阵
    - homo / hete 两个表示分支
    - 在表示空间进行 gamma 加权融合
    - 只在融合表示上做一次聚类
    """

    def __init__(
        self,
        client_id: int,
        k: int,
        device: torch.device,
        *,
        alpha_struct: float = 0.5,
        tau: float = 0.2,
        use_l2norm: bool = True,
        prop_steps_homo: int = 2,
        prop_steps_hete: int = 0,
        use_topk: int = 20,
        gamma_mode: str = "auto",
        gamma_fixed: float = 0.5,
        seed: int = 0,
    ):
        self.client_id = int(client_id)
        self.k = int(k)
        self.device = device

        self.alpha_struct = float(alpha_struct)
        self.tau = float(tau)
        self.use_l2norm = bool(use_l2norm)

        self.prop_steps_homo = int(prop_steps_homo)
        self.prop_steps_hete = int(prop_steps_hete)

        self.use_topk = int(use_topk)
        self.gamma_mode = str(gamma_mode)
        self.gamma_fixed = float(gamma_fixed)

        self.seed = int(seed)

    # --------------------------------------------------
    # Step 2.1  冻结 encoder 提取表示
    # --------------------------------------------------

    @torch.no_grad()
    def encode(self, encoder, subgraph):
        encoder.eval()
        encoder = encoder.to(self.device)
        encoder.preprocess(subgraph.adj, subgraph.x)
        Z = encoder.model_forward(range(subgraph.num_nodes), self.device)
        if self.use_l2norm:
            Z = F.normalize(Z, p=2, dim=1)
        return Z  # [N, d]

    # --------------------------------------------------
    # Step 2.2  知识引导相似性
    # --------------------------------------------------

    @torch.no_grad()
    def build_similarity(self, Z: torch.Tensor) -> torch.Tensor:
        sim = (Z @ Z.t()) / max(self.tau, 1e-6)

        if self.use_topk and 0 < self.use_topk < sim.size(0):
            vals, idx = torch.topk(sim, k=self.use_topk, dim=1)
            S = torch.zeros_like(sim)
            S.scatter_(1, idx, vals)
            sim = S

        return torch.softmax(sim, dim=1)

    @torch.no_grad()
    def correct_structure(self, A: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        P = self.alpha_struct * A + (1.0 - self.alpha_struct) * S
        P = 0.5 * (P + P.t())
        P = P / P.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return P

    # --------------------------------------------------
    # Step 2.3  表示传播
    # --------------------------------------------------

    @torch.no_grad()
    def propagate_with_P(self, Z: torch.Tensor, P: torch.Tensor, steps: int) -> torch.Tensor:
        H = Z
        for _ in range(max(0, int(steps))):
            H = P @ H
        if self.use_l2norm:
            H = F.normalize(H, p=2, dim=1)
        return H

    # --------------------------------------------------
    # Step 2.4  聚类版 Homophily Confidence
    # --------------------------------------------------

    @torch.no_grad()
    def homophily_confidence(self, A: torch.Tensor, Z: torch.Tensor) -> float:
        edge_idx = (A > 0).nonzero(as_tuple=False)
        edge_idx = edge_idx[edge_idx[:, 0] != edge_idx[:, 1]]
        if edge_idx.numel() == 0:
            return 0.5

        sim_edge = (Z[edge_idx[:, 0]] * Z[edge_idx[:, 1]]).sum(dim=1)
        m = sim_edge.numel()

        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.seed + 97 * (self.client_id + 1))
        u = torch.randint(0, Z.size(0), (m,), generator=rng, device=self.device)
        v = torch.randint(0, Z.size(0), (m,), generator=rng, device=self.device)
        sim_rand = (Z[u] * Z[v]).sum(dim=1)

        gap = (sim_edge.mean() - sim_rand.mean()).item()
        gamma = 1.0 / (1.0 + np.exp(-5.0 * gap))
        return float(gamma)

    # --------------------------------------------------
    # Step 2.5  主流程（关键修改点）
    # --------------------------------------------------

    def run(self, encoder, subgraph):
        """
        返回：
        - pred_final: 最终聚类标签
        - info: 记录 gamma
        """

        # 1) 邻接 & 表示
        A = _to_dense_adj(subgraph.adj, device=self.device)
        Z = self.encode(encoder, subgraph)

        # 2) 结构校正
        S = self.build_similarity(Z)
        P = self.correct_structure(A, S)

        # 3) homo / hete 表示分支
        Z_homo = self.propagate_with_P(Z, P, self.prop_steps_homo)
        Z_hete = self.propagate_with_P(Z, P, self.prop_steps_hete)

        # 4) gamma
        if self.gamma_mode == "fixed":
            gamma = float(np.clip(self.gamma_fixed, 0.0, 1.0))
        else:
            gamma = self.homophily_confidence(A, Z)

        # ==================================================
        # ★ 关键改动：表示级融合 + 再聚一次
        # ==================================================

        Z_fused = gamma * Z_homo + (1.0 - gamma) * Z_hete
        if self.use_l2norm:
            Z_fused = F.normalize(Z_fused, p=2, dim=1)

        pred_final = _cluster_kmeans(
            Z_fused.detach().cpu().numpy(),
            k=self.k,
            seed=self.seed + 1000 + self.client_id,
        )

        info = {
            "gamma": gamma,
        }
        return pred_final, info
