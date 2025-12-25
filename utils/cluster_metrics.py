import numpy as np
import torch

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    _HAS_SK_METRICS = True
except Exception:
    _HAS_SK_METRICS = False

try:
    import networkx as nx
    from networkx.algorithms.community.quality import modularity as nx_modularity
    _HAS_NX = True
except Exception:
    _HAS_NX = False


def purity_score(true_labels, pred_labels):
    true_labels = np.asarray(true_labels).astype(int)
    pred_labels = np.asarray(pred_labels).astype(int)
    if true_labels.size == 0:
        return 0.0
    clusters = np.unique(pred_labels)
    correct = 0
    for cluster_id in clusters:
        mask = pred_labels == cluster_id
        if not np.any(mask):
            continue
        labels_in_cluster = true_labels[mask]
        majority = np.bincount(labels_in_cluster).argmax()
        correct += np.sum(labels_in_cluster == majority)
    return correct / true_labels.size


def build_nx_graph(adj):
    if not _HAS_NX:
        return None
    return nx.from_scipy_sparse_array(adj)


def evaluate_clustering(labels_true, labels_pred, graph=None):
    metrics = {}
    if _HAS_SK_METRICS:
        metrics["nmi"] = normalized_mutual_info_score(labels_true, labels_pred)
        metrics["ari"] = adjusted_rand_score(labels_true, labels_pred)
    else:
        metrics["nmi"] = None
        metrics["ari"] = None

    metrics["purity"] = purity_score(labels_true, labels_pred)

    if _HAS_NX and graph is not None:
        communities = []
        for cluster_id in np.unique(labels_pred):
            nodes = np.where(labels_pred == cluster_id)[0]
            communities.append(set(nodes.tolist()))
        metrics["modularity"] = nx_modularity(graph, communities)
    else:
        metrics["modularity"] = None

    return metrics


def kmeans_torch(z, k, num_iters=50, seed=0):
    if k <= 0:
        return torch.zeros(z.size(0), dtype=torch.long, device=z.device)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    perm = torch.randperm(z.size(0), generator=rng)
    perm = perm.to(z.device)

    centers = z[perm[:k]].clone()
    labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

    for _ in range(num_iters):
        dist = torch.cdist(z, centers)
        labels = dist.argmin(dim=1)

        for idx in range(k):
            mask = labels == idx
            if mask.any():
                centers[idx] = z[mask].mean(dim=0)
            else:
                ridx = torch.randint(0, z.size(0), (1,), generator=rng).item()
                centers[idx] = z[ridx]

    return labels