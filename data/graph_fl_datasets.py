from torch_geometric.datasets import Planetoid
from torch_geometric.data import Dataset
import torch
from torch import Tensor

import time
import torch
import os
import os.path as osp
import scipy.sparse as sp
from torch import Tensor
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Planetoid
from data.data_sampling import data_partitioning
from data.geom_data import load_geom_data
from torch_geometric.data import Dataset
from data.utils import analysis_graph_structure_statis_info, analysis_graph_structure_homo_hete_info


class GraphFLDataset(Dataset):
    def __init__(self,
                 root,                     #数据集根目录
                 name,                #数据集名称
                 sampling,                #采样策略/异质性模拟方法
                 num_clients,              #客户端数量
                 analysis_local_subgraph,  #是否分析局部子图属性
                 analysis_global_graph,    #是否分析全局大图属性
                 ratio_train=0.2,          #训练集比例
                 ratio_val=0.4,            #验证集比例
                 ratio_test=0.4,           #测试集比例
                 ratio_iso=0.5,            #孤立节点比例 (通常用于噪声或异常检测)
                 ratio_homo=0.001,         #同质性比率 (Homophily Ratio) 相关参数
                 ratio_hete=0.001,         #异质性比率 (Heterophily Ratio) 相关参数
                 transform=None,           #动态数据变换 (TorchGeometric 标准参数)
                 pre_transform=None,       #预处理变换 (TorchGeometric 标准参数)
                 pre_filter=None):         #预过滤条件 (TorchGeometric 标准参数):
        self.name = name
        self.sampling = sampling
        self.num_clients = num_clients
        self.ratio_train = ratio_train
        self.ratio_val = ratio_val
        self.ratio_test = ratio_test
        self.ratio_homo = ratio_homo
        self.ratio_hete = ratio_hete
        self.ratio_iso = ratio_iso
        self.analysis_local_subgraph = analysis_local_subgraph
        self.analysis_global_graph = analysis_global_graph
        super(GraphFLDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.load_data()

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "datasets") if self.name in ["Cora", "CiteSeer", "PubMed"] else self.root

    @property
    def processed_dir(self) -> str:
        return osp.join(self.raw_dir, self.name, "Client{}".format(self.num_clients), self.sampling)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self) -> str:
        files_names = ['data{}.pt'.format(i) for i in range(self.num_clients)]
        return files_names

    def download(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, f"data{idx}.pt"),
            weights_only=False
        )

        return data

    def load_global_graph(self, process=False):
        print("| ★  Load Global Data: {}".format(self.name))
        if self.name in ["Cora"]:
            self.global_dataset = Planetoid(root=self.raw_dir, name=self.name)
            self.input_dim = self.global_dataset.num_features
            self.output_dim = self.global_dataset.num_classes
            self.global_data = self.global_dataset._data
            self.global_data.adj = sp.coo_matrix((torch.ones([len(self.global_data.edge_index[0])]),
                                                  (self.global_data.edge_index[0], self.global_data.edge_index[1])),
                                                 shape=(self.global_data.num_nodes, self.global_data.num_nodes))
            self.global_data.row, self.global_data.col, self.global_data.edge_weight = self.global_data.adj.row, self.global_data.adj.col, self.global_data.adj.data
            if isinstance(self.global_data.row, Tensor) or isinstance(self.global_data.col, Tensor):
                self.global_data.adj = csr_matrix((self.global_data.edge_weight.numpy(),
                                                   (self.global_data.row.numpy(), self.global_data.col.numpy())),
                                                  shape=(self.global_data.num_nodes, self.global_data.num_nodes))
            else:
                self.global_data.adj = csr_matrix(
                    (self.global_data.edge_weight, (self.global_data.row, self.global_data.col)),
                    shape=(self.global_data.num_nodes, self.global_data.num_nodes))
        else:
            raise ValueError("Not supported for this dataset, please check root file path and dataset name")

    def process(self):
        self.load_global_graph(process=True)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        subgraph_list = data_partitioning(
            G=self.global_data,
            sampling=self.sampling,
            num_clients=self.num_clients,
            ratio_train=self.ratio_train,
            ratio_val=self.ratio_val,
            ratio_test=self.ratio_test,
            ratio_iso=self.ratio_iso,
            ratio_homo=self.ratio_homo,
            ratio_hete=self.ratio_hete
        )
        for i in range(self.num_clients):
            torch.save(subgraph_list[i], self.processed_paths[i])

    def load_data(self):
        self.load_global_graph()
        self.subgraphs = [self.get(i) for i in range(self.num_clients)]
        for i in range(len(self.subgraphs)):
            if i == 0:
                self.global_data.train_idx = self.subgraphs[i].global_train_idx
                self.global_data.val_idx = self.subgraphs[i].global_val_idx
                self.global_data.test_idx = self.subgraphs[i].global_test_idx
            else:
                self.global_data.train_idx += self.subgraphs[i].global_train_idx
                self.global_data.val_idx += self.subgraphs[i].global_val_idx
                self.global_data.test_idx += self.subgraphs[i].global_test_idx