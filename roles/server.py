import os
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
from utils.visualize_utils import DrawCluster
from config import args
from models.gcn_encoder import GCNEncoder
from utils.utils import save_accuracy_record
from utils.cluster_metrics import build_nx_graph,evaluate_clustering,kmeans_torch
from sklearn.metrics import f1_score

class ServerManager:
    """
    联邦图聚类的 Server：
    只做 encoder 的下发、聚合，不再做 val/test acc，也不保存 best acc。
    """
    def __init__(self, model_name, datasets, num_clients, device, num_round, client_sample_ratio):
        self.model_name = model_name
        self.datasets = datasets
        self.input_dim = datasets.input_dim
        self.global_data = datasets.global_data
        self.subgraphs = datasets.subgraphs
        self.cluster_drawer = DrawCluster(self)
        self.num_clients = num_clients
        self.device = device
        self.client_sample_ratio = client_sample_ratio

        self.num_rounds = num_round  # 你 manager 里传的是 num_round
        self.hidden_dim = args.hidden_dim
        self.emb_dim = getattr(args, "emb_dim", args.hidden_dim)

        self.init_model()

    def init_model(self):
        self.model = GCNEncoder(
            feat_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            emb_dim=self.emb_dim,
            dropout=args.drop
        ).to(self.device)

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict=state_dict)

    def model_aggregation(self, models, mixing_coefficients):
        aggregated = OrderedDict()
        sd_list = [m.state_dict() for m in models]
        for it, sd in enumerate(sd_list):
            for k in sd.keys():
                if it == 0:
                    aggregated[k] = mixing_coefficients[it] * sd[k]
                else:
                    aggregated[k] = aggregated[k] + mixing_coefficients[it] * sd[k]
        return aggregated

    @staticmethod
    def _mask_to_index(mask):
        if mask is None:
            return torch.tensor([], dtype=torch.long)
        if mask.dtype == torch.bool:
            return mask.nonzero(as_tuple=False).view(-1)
        return (mask > 0).nonzero(as_tuple=False).view(-1)

    @staticmethod
    def _accuracy(logits, labels, idx):
        if idx.numel() == 0:
            return 0.0
        pred = logits[idx].argmax(dim=1)
        return (pred == labels[idx]).float().mean().item()

    @staticmethod
    def _f1_score(logits, labels, idx, average="macro"):
        if idx.numel() == 0:
            return 0.0
        pred = logits[idx].argmax(dim=1)
        return f1_score(labels[idx].cpu(), pred.cpu(), average=average)

    @staticmethod
    def _pca_2d(z):
        z = z - z.mean(dim=0, keepdim=True)
        _, _, v_t = torch.linalg.svd(z, full_matrices=False)
        components = v_t[:2].T
        return z @ components

    @staticmethod
    def _kmeans(z, k, num_iters=50, seed=0):
        return kmeans_torch(z, k, num_iters=num_iters, seed=seed)

    def _get_nx_graph(self):
        if getattr(self, "_nx_graph", None) is not None:
            return self._nx_graph
        graph = build_nx_graph(self.global_data.adj)
        self._nx_graph = graph
        return graph

    def evaluate_clustering(self, k, seed=0):
        if k <= 0:
            return None
        self.model.eval()
        self.model.preprocess(self.global_data.adj, self.global_data.x)
        with torch.no_grad():
            z = self.model(self.device).detach()
        labels_true = self.global_data.y.cpu().numpy()
        labels_pred = self._kmeans(
            z,
            k,
            num_iters=getattr(args, "viz_kmeans_iters", 50),
            seed=seed
        ).cpu().numpy()

        graph = self._get_nx_graph()
        return evaluate_clustering(labels_true, labels_pred, graph=graph)
         # if k <= 0:
        #     return torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        #
        # # 统一使用 CPU generator
        # rng = torch.Generator(device="cpu")
        # rng.manual_seed(seed)
        #
        # # 初始中心随机打乱索引（CPU）
        # perm = torch.randperm(z.size(0), generator=rng)
        # perm = perm.to(z.device)

        # centers = z[perm[:k]].clone()
        # labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        # for _ in range(num_iters):
        #     dist = torch.cdist(z, centers)
        #     labels = dist.argmin(dim=1)

            # for idx in range(k):
            #     mask = labels == idx
            #     if mask.any():
            #         centers[idx] = z[mask].mean(dim=0)
            #     else:
            #         # 随机选一个点作为空簇中心（CPU 采样）
            #         ridx = torch.randint(
            #             0, z.size(0), (1,), generator=rng
            #         ).item()
            #         centers[idx] = z[ridx]

        # return labels



    def evaluate_global_encoder(self, epochs=100, lr=0.01, weight_decay=0.0):
        self.model.eval()
        self.model.preprocess(self.global_data.adj, self.global_data.x)
        with torch.no_grad():
            z = self.model(self.device).detach()

        labels = self.global_data.y.to(self.device)
        train_idx = self._mask_to_index(self.global_data.train_idx).to(self.device)
        val_idx = self._mask_to_index(self.global_data.val_idx).to(self.device)
        test_idx = self._mask_to_index(self.global_data.test_idx).to(self.device)

        num_classes = int(labels.max().item() + 1)
        classifier = torch.nn.Linear(z.size(1), num_classes).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            classifier.train()
            optimizer.zero_grad()
            logits = classifier(z[train_idx])
            loss = loss_fn(logits, labels[train_idx])
            loss.backward()
            optimizer.step()

        classifier.eval()
        with torch.no_grad():
            logits = classifier(z)
            val_acc = self._accuracy(logits, labels, val_idx)
            test_acc = self._accuracy(logits, labels, test_idx)

            val_f1 = self._f1_score(logits, labels, val_idx)
            test_f1 = self._f1_score(logits, labels, test_idx)

        return val_acc, test_acc, val_f1, test_f1


    def collaborative_training_encoder(self, clients):
        print("| ★ Start Federated Graph Clustering: train global encoder (unsupervised)")

        for rnd in range(self.num_rounds):
            all_ids = list(range(self.num_clients))
            random.shuffle(all_ids)

            sample_num = max(1, int(len(all_ids) * self.client_sample_ratio))
            sample_ids = sorted(all_ids[:sample_num])

            mix = [clients[i].num_nodes for i in sample_ids]
            s = float(sum(mix))
            mix = [v / s for v in mix]

            local_models = []

            for cid in sample_ids:
                clients[cid].set_state_dict(self.model)

                local_model = clients[cid].local_train_encoder(
                    device=self.device,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    epochs=args.num_epochs,
                    neg_ratio=getattr(args, "neg_ratio", 1),
                    max_pos_edges=getattr(args, "max_pos_edges", 20000),
                    seed=args.seed + rnd * 1000 + cid
                )
                local_models.append(local_model)

            agg = self.model_aggregation(local_models, mix)
            self.set_state_dict(agg)

            log_every = max(1, getattr(args, "log_every", 1))
            eval_every = getattr(args, "eval_every", log_every)
            if eval_every > 0 and (rnd % eval_every == 0 or rnd == self.num_rounds - 1):
                val_acc, test_acc, val_f1, test_f1 = self.evaluate_global_encoder(
                    epochs=getattr(args, "probe_epochs", 100),
                    lr=getattr(args, "probe_lr", 0.01),
                    weight_decay=getattr(args, "probe_weight_decay", 0.0)
                )

                save_accuracy_record(
                    data_name=args.data_name,
                    num_clients=args.num_clients,
                    partition=args.partition,
                    gmodel_name=args.gmodel_name,
                    num_rounds=rnd,
                    val_acc=val_acc,
                    test_acc=test_acc,
                    val_f1=val_f1,
                    test_f1=test_f1
                )


                k = getattr(args, "k_clusters", 0)
                if k and k > 0:
                    metrics = self.evaluate_clustering(k, seed=args.seed + rnd)
                    if metrics is not None:
                        nmi = metrics.get("nmi")
                        ari = metrics.get("ari")
                        purity = metrics.get("purity")
                        modularity = metrics.get("modularity")
                        nmi_str = f"{nmi:.4f}" if nmi is not None else "N/A"
                        ari_str = f"{ari:.4f}" if ari is not None else "N/A"
                        purity_str = f"{purity:.4f}" if purity is not None else "N/A"
                        modularity_str = f"{modularity:.4f}" if modularity is not None else "N/A"

                        from utils.save_evaluate import save_evaluating_indicator
                        save_evaluating_indicator(
                            data_name = args.data_name,
                            num_clients = args.num_clients,
                            partition = args.partition,
                            gmodel_name = args.gmodel_name,
                            num_rounds = rnd,
                            NMI=nmi_str,
                            ARI=ari_str,
                            Purity=purity_str,
                            Modularity=modularity_str,
                        )

                        print(
                            f"Round {rnd:03d} | "
                            f"val_acc={val_acc:.4f} | test_acc={test_acc:.4f} || "
                            f"NMI={nmi_str} | ARI={ari_str} | Purity={purity_str} | Modularity={modularity_str} | "
                            f"val_f1={val_f1} | test_f1={test_f1}"
                        )
            if getattr(args, "visualize_clusters", False):
                viz_every = max(1, getattr(args, "viz_every", log_every))
                if rnd % viz_every == 0 or rnd == self.num_rounds - 1:
                    self.cluster_drawer.visualize_clusters(rnd)

