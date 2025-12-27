import scipy.sparse as sp

from personalization.base_op import GraphOp
from utils.utils import adj_to_symmetric_norm


class LaplacianGraphOp(GraphOp):
    def __init__(self, prop_steps: int, r: float = 0.5):
        super().__init__(prop_steps)
        self.r = r

    def construct_adj(self, adj: sp.spmatrix) -> sp.csr_matrix:
        if isinstance(adj, sp.csr_matrix):
            adj = adj.tocoo()
        elif not isinstance(adj, sp.coo_matrix):
            raise TypeError("The adjacency matrix must be a scipy.sparse.coo_matrix/csr_matrix!")

        adj_normalized = adj_to_symmetric_norm(adj, self.r)
        return adj_normalized.tocsr()