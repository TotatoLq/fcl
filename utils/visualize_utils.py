import os

import matplotlib.pyplot as plt
import torch
from config import args

class DrawCluster:
    def __init__(self, server):
        self.server = server

    def visualize_clusters(self, round_idx):
        self.server.model.eval()
        self.server.model.preprocess(
            self.server.global_data.adj,
            self.server.global_data.x
        )
        with torch.no_grad():
            z = self.server.model(self.server.device).detach()

        max_points = getattr(args, "viz_max_points", 2000)
        if max_points > 0 and z.size(0) > max_points:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(args.seed + round_idx)
            perm = torch.randperm(z.size(0), generator=rng)[:max_points]
            perm = perm.to(z.device)
            z = z[perm]
        else:
            perm = None

        z_2d = self.server._pca_2d(z).cpu().numpy()

        k = getattr(args, "k_clusters", 0)
        if k and k > 0:
            labels = self.server._kmeans(
                z,
                k,
                num_iters=getattr(args, "viz_kmeans_iters", 50),
                seed=args.seed + round_idx
            ).cpu().numpy()
        else:
            labels = self.server.global_data.y
            labels = labels[perm].cpu().numpy() if perm is not None else labels.cpu().numpy()

        base_dir = getattr(args, "viz_out_dir", "./cluster_viz")
        dataset_name = getattr(args, "data_name", "unknown")
        num_clients = getattr(args, "num_clients", "N")
        partition = getattr(args, "partition", "unknown")
        gmodel = getattr(args, "gmodel_name", "model")

        out_dir = os.path.join(base_dir, dataset_name)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir,f"{dataset_name}_{num_clients}_{partition}_{gmodel}_{round_idx:03d}.png")

        plt.figure(figsize=(7, 6))
        plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab20", s=8, alpha=0.8)
        plt.title(f"Cluster visualization (round {round_idx:03d})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[INFO] Cluster visualization saved: {out_path}")