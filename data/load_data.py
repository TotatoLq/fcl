from data.graph_fl_datasets import GraphFLDataset


def load_dataset(args):
    dataset=GraphFLDataset(
        root="./datasets",
        name=args.data_name,
        sampling=args.partition,
        num_clients=args.num_clients,
        analysis_local_subgraph=False,
        analysis_global_graph=False,
    )
    return dataset