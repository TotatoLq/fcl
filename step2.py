import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import args
from data.graph_fl_datasets import GraphFLDataset
from models.gcn_encoder import GCNEncoder
from personalization.my_model import MyModel
from utils.utils import set_seed


warnings.filterwarnings("ignore")


def accuracy(output, labels):
    preds = output.max(1)[1]
    correct = preds.eq(labels).double()
    return (correct.sum() / labels.shape[0]).item()


def _load_best_encoder(dataset, device):
    """加载全局阶段保存的最佳编码器权重。"""
    best_dir = getattr(args, "best_model_dir", "./checkpoints")
    file_name = (
        f"best_encoder_{args.data_name}_{args.partition}_"
        f"{args.gmodel_name}_{args.num_clients}c_seed{args.seed}.pt"
    )
    best_path = Path(best_dir) / file_name
    if not best_path.exists():
        raise FileNotFoundError(
            f"Best encoder checkpoint not found: {best_path}. "
            "Please run the federated pretraining stage first."
        )

    state = torch.load(best_path, map_location=device)
    model = GCNEncoder(
        feat_dim=dataset.input_dim,
        hidden_dim=args.hidden_dim,
        emb_dim=getattr(args, "emb_dim", args.hidden_dim),
        dropout=args.drop,
    )
    model.load_state_dict(state["model_state_dict"])
    return model


def step2_main(datasets=None):
    """个性化阶段主入口，逻辑与 AdaFGL 的 step2_main 保持一致。"""
    main_normalize_train = getattr(args, "normalize_train", 1)
    gpu_id = args.gpu_id
    num_clients = args.num_clients

    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    )
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

    print("\n| ★  Start Local Client Personalized Training")

    global_normalize_record = {
        "acc_val_mean": 0,
        "acc_val_std": 0,
        "acc_test_mean": 0,
        "acc_test_std": 0,
    }

    t_total = time.time()

    for i, subgraph in enumerate(datasets.subgraphs):
        subgraph.y = subgraph.y.to(device)

        local_normalize_record = {"acc_val": [], "acc_test": []}

        for _ in range(main_normalize_train):
            gmodel = _load_best_encoder(datasets, device)
            gmodel.preprocess(subgraph.adj, subgraph.x)
            gmodel = gmodel.to(device)

            nodes_embedding = gmodel.model_forward(
                range(subgraph.num_nodes), device
            ).detach().cpu()
            nodes_embedding = nn.Softmax(dim=1)(nodes_embedding)

            acc_val = accuracy(
                nodes_embedding[subgraph.val_idx],
                subgraph.y[subgraph.val_idx],
            )
            acc_test = accuracy(
                nodes_embedding[subgraph.test_idx],
                subgraph.y[subgraph.test_idx],
            )

            model = MyModel(
                prop_steps=3,
                feat_dim=datasets.input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=datasets.output_dim,
                threshold=args.threshold,
            )

            model.non_para_lp(
                subgraph=subgraph,
                nodes_embedding=nodes_embedding,
                x=subgraph.x,
                device=device,
            )

            model.preprocess(adj=subgraph.adj)
            model = model.to(device)

            loss_ce_fn = nn.CrossEntropyLoss()
            loss_mse_fn = nn.MSELoss()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            epochs = args.epochs

            best_val = 0.0
            best_test = 0.0

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                if model.homo:
                    local_smooth_emb, global_emb = model.homo_forward(device)

                    loss_train1 = loss_ce_fn(
                        local_smooth_emb[subgraph.train_idx],
                        subgraph.y[subgraph.train_idx],
                    )
                    loss_train2 = loss_mse_fn(
                        local_smooth_emb,
                        global_emb,
                    )
                    loss_train = loss_train1 + loss_train2
                    loss_train.backward()
                    optimizer.step()

                    model.eval()
                    local_smooth_emb, global_emb = model.homo_forward(device)
                    output = (
                        F.softmax(local_smooth_emb.data, 1)
                        + F.softmax(global_emb.data, 1)
                    ) / 2

                    acc_val = accuracy(
                        output[subgraph.val_idx],
                        subgraph.y[subgraph.val_idx],
                    )
                    acc_test = accuracy(
                        output[subgraph.test_idx],
                        subgraph.y[subgraph.test_idx],
                    )
                else:
                    (
                        local_ori_emb,
                        local_smooth_emb,
                        local_message_propagation,
                    ) = model.hete_forward(device)

                    loss_train1 = loss_ce_fn(
                        local_ori_emb[subgraph.train_idx],
                        subgraph.y[subgraph.train_idx],
                    )
                    loss_train2 = loss_ce_fn(
                        local_smooth_emb[subgraph.train_idx],
                        subgraph.y[subgraph.train_idx],
                    )
                    loss_train3 = loss_ce_fn(
                        local_message_propagation[subgraph.train_idx],
                        subgraph.y[subgraph.train_idx],
                    )

                    loss_train = loss_train1 + loss_train2 + loss_train3
                    loss_train.backward()
                    optimizer.step()

                    model.eval()
                    (
                        local_ori_emb,
                        local_smooth_emb,
                        local_message_propagation,
                    ) = model.hete_forward(device)

                    output = (
                        F.softmax(local_ori_emb.data, 1)
                        + F.softmax(local_smooth_emb.data, 1)
                        + F.softmax(local_message_propagation.data, 1)
                    ) / 3

                    acc_val = accuracy(
                        output[subgraph.val_idx],
                        subgraph.y[subgraph.val_idx],
                    )
                    acc_test = accuracy(
                        output[subgraph.test_idx],
                        subgraph.y[subgraph.test_idx],
                    )

                if acc_val > best_val:
                    best_val = acc_val
                    best_test = acc_test

            local_normalize_record["acc_val"].append(best_val)
            local_normalize_record["acc_test"].append(best_test)

        global_normalize_record["acc_val_mean"] += (
            np.mean(local_normalize_record["acc_val"])
            * subgraph.num_nodes
            / datasets.global_data.num_nodes
        )
        global_normalize_record["acc_val_std"] += (
            np.std(local_normalize_record["acc_val"], ddof=1)
            * subgraph.num_nodes
            / datasets.global_data.num_nodes
        )
        global_normalize_record["acc_test_mean"] += (
            np.mean(local_normalize_record["acc_test"])
            * subgraph.num_nodes
            / datasets.global_data.num_nodes
        )
        global_normalize_record["acc_test_std"] += (
            np.std(local_normalize_record["acc_test"], ddof=1)
            * subgraph.num_nodes
            / datasets.global_data.num_nodes
        )

    print("| ★  Normalize Train Completed")
    print(
        "| Normalize Train: {}, Total Time Elapsed: {:.4f}s".format(
            args.normalize_train,
            time.time() - t_total,
        )
    )
    print(
        "| Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(
            round(global_normalize_record["acc_val_mean"], 4),
            round(global_normalize_record["acc_val_std"], 4),
            round(global_normalize_record["acc_test_mean"], 4),
            round(global_normalize_record["acc_test_std"], 4),
        )
    )

    return round(global_normalize_record["acc_test_mean"], 4)


if __name__ == "__main__":
    step2_main()