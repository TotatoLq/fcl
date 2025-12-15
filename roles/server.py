import random
from collections import OrderedDict

import torch

from config import args
from models.gcn_encoder import GCNEncoder


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

            if rnd % max(1, getattr(args, "log_every", 5)) == 0 or rnd == self.num_rounds - 1:
                print(f"Round {rnd:03d} | sampled={len(sample_ids)}/{self.num_clients} | encoder updated")
