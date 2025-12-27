import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from config import args
from data.graph_fl_datasets import GraphFLDataset
from models.gcn_encoder import GCNEncoder
from utils.cluster_metrics import build_nx_graph, evaluate_clustering
from utils.utils import set_seed

# ★ 新增：AdaFGL-style 聚类客户端
from personalization.adafgl_clustering_client import AdaFGLClusteringClient

warnings.filterwarnings("ignore")


# =========================
#   Encoder Checkpoint
# =========================

def best_encoder_checkpoint_path():
    best_dir = getattr(args, "best_model_dir", "./checkpoints")
    file_name = (
        f"best_encoder_{args.data_name}_{args.partition}_"
        f"{args.gmodel_name}_{args.num_clients}c_seed{args.seed}.pt"
    )
    return Path(best_dir) / file_name


def _load_best_encoder(dataset, device):
    best_path = best_encoder_checkpoint_path()
    if not best_path.exists():
        raise FileNotFoundError(f"Best encoder checkpoint not found: {best_path}")

    state = torch.load(best_path, map_location=device)
    model = GCNEncoder(
        feat_dim=dataset.input_dim,
        hidden_dim=args.hidden_dim,
        emb_dim=getattr(args, "emb_dim", args.hidden_dim),
        dropout=args.drop,
    )
    model.load_state_dict(state["model_state_dict"])
    return model


# =========================
#   Linear Probe (Global)
# =========================

def _mask_to_index(mask):
    if mask is None:
        return torch.tensor([], dtype=torch.long)
    if torch.is_tensor(mask):
        if mask.dtype == torch.bool:
            return mask.nonzero(as_tuple=False).view(-1)
        return (mask > 0).nonzero(as_tuple=False).view(-1)
    return torch.tensor(mask, dtype=torch.long)


def _accuracy(logits, labels, idx):
    if idx.numel() == 0:
        return 0.0
    pred = logits[idx].argmax(dim=1)
    return (pred == labels[idx]).float().mean().item()


def _f1_score(logits, labels, idx, average="macro"):
    if idx.numel() == 0:
        return 0.0
    pred = logits[idx].argmax(dim=1)
    return f1_score(labels[idx].cpu(), pred.cpu(), average=average)


def _evaluate_encoder_on_global_graph(dataset, device):
    model = _load_best_encoder(dataset, device)
    model = model.to(device)
    model.eval()
    model.preprocess(dataset.global_data.adj, dataset.global_data.x)

    with torch.no_grad():
        z = model(device).detach()

    labels = dataset.global_data.y.to(device)
    train_idx = _mask_to_index(dataset.global_data.train_idx).to(device)
    val_idx = _mask_to_index(dataset.global_data.val_idx).to(device)
    test_idx = _mask_to_index(dataset.global_data.test_idx).to(device)

    num_classes = int(labels.max().item() + 1)
    clf = nn.Linear(z.size(1), num_classes).to(device)
    opt = torch.optim.Adam(
        clf.parameters(),
        lr=getattr(args, "linear_eval_lr", 0.01),
        weight_decay=getattr(args, "linear_eval_weight_decay", 0.0),
    )
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(int(getattr(args, "linear_eval_epochs", 100))):
        clf.train()
        opt.zero_grad()
        loss = loss_fn(clf(z[train_idx]), labels[train_idx])
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        logits = clf(z)
        val_acc = _accuracy(logits, labels, val_idx)
        test_acc = _accuracy(logits, labels, test_idx)
        val_f1 = _f1_score(logits, labels, val_idx)
        test_f1 = _f1_score(logits, labels, test_idx)

    return val_acc, test_acc, val_f1, test_f1


# =========================
#   Global Clustering Eval
# =========================

def _evaluate_clustering_on_global_graph(dataset, device, k):
    model = _load_best_encoder(dataset, device)
    model = model.to(device)
    model.eval()

    model.preprocess(dataset.global_data.adj, dataset.global_data.x)
    with torch.no_grad():
        Z = model.model_forward(range(dataset.global_data.num_nodes), device)

    X = Z.detach().cpu().numpy()
    from utils.cluster_metrics import kmeans_torch
    pred = kmeans_torch(torch.tensor(X), k, num_iters=50, seed=int(args.seed)).cpu().numpy()

    labels_true = dataset.global_data.y.detach().cpu().numpy()
    graph = build_nx_graph(dataset.global_data.adj)
    metrics = evaluate_clustering(labels_true, pred, graph=graph)
    return metrics


# =========================
#   Step 2: AdaFGL Clustering
# =========================

def realtrain(datasets=None):
    gpu_id = args.gpu_id
    num_clients = args.num_clients
    k = getattr(args, "k_clusters", None)
    if k is None:
        raise ValueError("k_clusters must be set for clustering")

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if datasets is None:
        datasets = GraphFLDataset(
            root="./datasets",
            name=args.data_name,
            sampling=args.partition,
            num_clients=num_clients,
            analysis_local_subgraph=True,
            analysis_global_graph=False,
        )

    print("\n| ★  Start AdaFGL-style Federated Graph Clustering")

    global_record = {
        "nmi": 0.0,
        "ari": 0.0,
        "purity": 0.0,
    }

    t_total = time.time()
    N_global = datasets.global_data.num_nodes

    # 冻结一次 encoder（避免重复 load）
    encoder = _load_best_encoder(datasets, device)
    encoder = encoder.to(device)
    encoder.eval()

    for cid, subgraph in enumerate(datasets.subgraphs):
        subgraph.y = subgraph.y.to(device)

        client = AdaFGLClusteringClient(
            client_id=cid,
            k=int(k),
            device=device,
            alpha_struct=float(getattr(args, "alpha_struct", 0.5)),
            tau=float(getattr(args, "tau", 0.2)),
            use_l2norm=bool(int(getattr(args, "use_l2norm", 1))),
            prop_steps_homo=int(getattr(args, "prop_steps_homo", 2)),
            prop_steps_hete=int(getattr(args, "prop_steps_hete", 0)),
            use_topk=int(getattr(args, "sim_topk", 20)),
            gamma_mode=str(getattr(args, "gamma_mode", "auto")),
            gamma_fixed=float(getattr(args, "gamma_fixed", 0.5)),
            seed=int(args.seed),
        )

        pred, info = client.run(encoder, subgraph)

        y_true = subgraph.y.detach().cpu().numpy()
        from utils.cluster_metrics import normalized_mutual_info_score, adjusted_rand_score
        from utils.cluster_metrics import purity_score

        nmi = normalized_mutual_info_score(y_true, pred)
        ari = adjusted_rand_score(y_true, pred)
        purity = purity_score(y_true, pred)

        weight = float(subgraph.num_nodes) / float(N_global)
        global_record["nmi"] += nmi * weight
        global_record["ari"] += ari * weight
        global_record["purity"] += purity * weight

        print(
            f"Client {cid:02d} | gamma={info['gamma']:.3f} | "
            f"NMI={nmi:.4f} ARI={ari:.4f} Purity={purity:.4f}"
        )

    print(
        "| ★  Local Clustering Finished | Time {:.2f}s".format(time.time() - t_total)
    )
    print(
        "| Mean NMI={:.4f} | Mean ARI={:.4f} | Mean Purity={:.4f}".format(
            global_record["nmi"],
            global_record["ari"],
            global_record["purity"],
        )
    )

    # ===== Global Evaluation =====
    val_acc, test_acc, val_f1, test_f1 = _evaluate_encoder_on_global_graph(
        datasets, device
    )
    global_metrics = _evaluate_clustering_on_global_graph(
        datasets, device, k
    )

    print(
        f"| Global Eval | val_acc={val_acc:.4f} | test_acc={test_acc:.4f} || "
        f"NMI={global_metrics.get('nmi'):.4f} | "
        f"ARI={global_metrics.get('ari'):.4f} | "
        f"Purity={global_metrics.get('purity'):.4f} | "
        f"Modularity={global_metrics.get('modularity'):.4f} | "
        f"val_f1={val_f1:.4f} | test_f1={test_f1:.4f}"
    )


if __name__ == "__main__":
    realtrain()
