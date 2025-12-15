import numpy as np
import torch
import torch.nn.functional as F


class LinkReconstructionTask:
    """
    无监督本地训练：用正边/负边的 BCE 训练 encoder
    只依赖 subgraph.adj 与 subgraph.x
    """
    def __init__(self, subgraph, model, device,
                 lr=1e-3, weight_decay=1e-5, epochs=3,
                 neg_ratio=1, max_pos_edges=20000, seed=0):
        self.subgraph = subgraph
        self.model = model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.neg_ratio = max(1, int(neg_ratio))
        self.max_pos_edges = int(max_pos_edges)
        self.seed = int(seed)

        adj = subgraph.adj.tocoo()
        row = adj.row
        col = adj.col
        mask = row != col
        self.pos_edges = np.vstack([row[mask], col[mask]]).T.astype(np.int64)
        self.num_nodes = int(subgraph.num_nodes)

        self._pos_set = set((int(u), int(v)) for u, v in self.pos_edges)

    def _sample_pos(self, rng):
        if self.pos_edges.shape[0] <= self.max_pos_edges:
            return self.pos_edges
        idx = rng.choice(self.pos_edges.shape[0], size=self.max_pos_edges, replace=False)
        return self.pos_edges[idx]

    def _sample_neg(self, rng, num_samples):
        neg = []
        tries = 0
        max_tries = num_samples * 50 + 1000
        while len(neg) < num_samples and tries < max_tries:
            u = int(rng.integers(0, self.num_nodes))
            v = int(rng.integers(0, self.num_nodes))
            if u == v:
                tries += 1
                continue
            if (u, v) in self._pos_set or (v, u) in self._pos_set:
                tries += 1
                continue
            neg.append((u, v))
            tries += 1
        if len(neg) < num_samples:
            while len(neg) < num_samples:
                u = int(rng.integers(0, self.num_nodes))
                v = int(rng.integers(0, self.num_nodes))
                if u != v:
                    neg.append((u, v))
        return np.asarray(neg, dtype=np.int64)

    def execute(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.train()

        rng = np.random.default_rng(self.seed)

        for _ in range(self.epochs):
            z = self.model(self.device)

            pos = self._sample_pos(rng)
            neg = self._sample_neg(rng, num_samples=pos.shape[0] * self.neg_ratio)

            pos_u = torch.from_numpy(pos[:, 0]).long().to(self.device)
            pos_v = torch.from_numpy(pos[:, 1]).long().to(self.device)
            neg_u = torch.from_numpy(neg[:, 0]).long().to(self.device)
            neg_v = torch.from_numpy(neg[:, 1]).long().to(self.device)

            pos_logit = (z[pos_u] * z[pos_v]).sum(dim=1)
            neg_logit = (z[neg_u] * z[neg_v]).sum(dim=1)

            pos_y = torch.ones_like(pos_logit)
            neg_y = torch.zeros_like(neg_logit)

            loss = F.binary_cross_entropy_with_logits(pos_logit, pos_y) + \
                   F.binary_cross_entropy_with_logits(neg_logit, neg_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        return self.model
