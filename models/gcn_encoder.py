# models/gcn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import sparse_mx_to_torch_sparse_tensor, adj_to_symmetric_norm


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.sparse.mm(adj, x)
        return x


class GCNEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, emb_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.processed_feature = None
        self.adj = None

    def preprocess(self, adj, feature):
        self.processed_feature = feature
        adj = adj_to_symmetric_norm(adj, r=0.5)
        self.adj = sparse_mx_to_torch_sparse_tensor(adj)

    def forward(self, device):
        x = self.processed_feature.to(device)
        adj = self.adj.to(device)

        z = self.conv1(x, adj)
        z = self.relu(z)
        z = self.dropout(z)
        z = self.conv2(z, adj)

        return F.normalize(z, dim=1)

    def model_forward(self, idx, device):
        """Forward helper that mirrors AdaFGL's interface.

        Args:
            idx: Iterable of node indices to fetch embeddings for.
            device: Target device.

        Returns:
            Tensor of node embeddings for the provided indices.
        """
        if isinstance(idx, range):
            idx = list(idx)
        index_tensor = torch.as_tensor(idx, dtype=torch.long, device=device)
        all_embeddings = self.forward(device)
        return all_embeddings[index_tensor]
