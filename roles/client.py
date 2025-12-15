import copy
import torch

from config import args
from models.gcn_encoder import GCNEncoder
from tasks.fed_graph_clustering import LinkReconstructionTask


try:
    from sklearn.cluster import KMeans
    _HAS_SK = True
except Exception:
    _HAS_SK = False


class ClientsManager:
    """
    仍然负责：根据 datasets.subgraphs 初始化 Client 列表
    你原来的 ClientsManager 结构可以直接复用，只是模型换成 Encoder。
    """
    def __init__(self, model_name, datasets, num_clients, device, eval_single_client=False):
        self.model_name = model_name
        self.global_data = datasets.global_data
        self.input_dim = datasets.input_dim
        self.output_dim = getattr(datasets, "output_dim", None)  # 聚类不使用
        self.subgraphs = datasets.subgraphs
        self.device = device
        self.num_clients = num_clients

        self.hidden_dim = args.hidden_dim
        self.emb_dim = getattr(args, "emb_dim", args.hidden_dim)

        self.clients = []
        self.tot_nodes = 0
        self.initClient()

    def initClient(self):
        for client_id in range(self.num_clients):
            c = Client(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                emb_dim=self.emb_dim,
                client_id=client_id,
                local_subgraph=self.subgraphs[client_id],
            )
            self.clients.append(c)
            self.tot_nodes += c.num_nodes


class Client(object):
    def __init__(self, input_dim, hidden_dim, emb_dim, client_id, local_subgraph):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.client_id = client_id
        self.local_subgraph = local_subgraph
        self.num_nodes = int(self.local_subgraph.num_nodes)

        self.model = None
        self.init_model()

    def init_model(self):
        self.model = GCNEncoder(
            feat_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            emb_dim=self.emb_dim,
            dropout=args.drop
        )
        self.model.preprocess(self.local_subgraph.adj, self.local_subgraph.x)

    def set_state_dict(self, global_model):
        self.model.load_state_dict(state_dict=copy.deepcopy(global_model.state_dict()))
        self.model.preprocess(self.local_subgraph.adj, self.local_subgraph.x)

    def local_train_encoder(self, device, lr, weight_decay, epochs, neg_ratio=1, max_pos_edges=20000, seed=0):
        task = LinkReconstructionTask(
            subgraph=self.local_subgraph,
            model=self.model.to(device),
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            neg_ratio=neg_ratio,
            max_pos_edges=max_pos_edges,
            seed=seed
        )
        self.model = task.execute()
        return self.model

    def local_cluster(self, device, k):
        if not _HAS_SK:
            return None
        self.model.eval()
        with torch.no_grad():
            z = self.model(device).detach().cpu().numpy()
        labels = KMeans(n_clusters=k, n_init=10).fit_predict(z)
        return labels
