import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch import Tensor


class GraphOp:
    def __init__(self, prop_steps: int):
        self._prop_steps = prop_steps
        self._adj = None

    def construct_adj(self, adj: sp.spmatrix) -> sp.csr_matrix:
        raise NotImplementedError

    def _validate_inputs(self, adj: sp.spmatrix, feature: np.ndarray):
        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        if not isinstance(feature, np.ndarray):
            raise TypeError("The feature matrix must be a numpy.ndarray!")
        if adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

    def propagate(self, adj: sp.spmatrix, feature: np.ndarray):
        self._adj = self.construct_adj(adj)
        if not isinstance(feature, np.ndarray):
            feature = feature.numpy()
        self._validate_inputs(self._adj, feature)

        prop_feat_list = [feature]
        for _ in range(self._prop_steps):
            feat_temp = self._adj.dot(prop_feat_list[-1])
            prop_feat_list.append(feat_temp)
        return [torch.FloatTensor(feat) for feat in prop_feat_list]

    def init_lp_propagate(self, adj: sp.spmatrix, feature: torch.Tensor, init_label: torch.Tensor, alpha: float):
        self._adj = self.construct_adj(adj)
        feat_np = feature.cpu().numpy()
        self._validate_inputs(self._adj, feat_np)

        prop_feat_list = [feat_np]
        for _ in range(self._prop_steps):
            feat_temp = self._adj.dot(prop_feat_list[-1])
            feat_temp = alpha * feat_temp + (1 - alpha) * feat_np
            feat_temp[init_label] += feat_np[init_label]
            prop_feat_list.append(feat_temp)

        return [torch.FloatTensor(feat) for feat in prop_feat_list]

    def res_lp_propagate(self, adj: sp.spmatrix, feature: np.ndarray, alpha: float):
        self._adj = self.construct_adj(adj)
        self._validate_inputs(self._adj, feature)

        prop_feat_list = [feature]
        for _ in range(self._prop_steps):
            feat_temp = self._adj.dot(prop_feat_list[-1])
            feat_temp = alpha * feat_temp + (1 - alpha) * feature
            prop_feat_list.append(feat_temp)

        return [torch.FloatTensor(feat) for feat in prop_feat_list]


class MessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super().__init__()
        self._aggr_type = None
        self._start, self._end = start, end

    @property
    def aggr_type(self):
        return self._aggr_type

    def combine(self, feat_list):
        raise NotImplementedError

    def aggregate(self, feat_list):
        if not isinstance(feat_list, list):
            raise TypeError("The input must be a list consists of feature matrices!")
        for feat in feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The feature matrices must be tensors!")

        return self.combine(feat_list)