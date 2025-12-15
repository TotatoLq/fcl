from config import args
from roles.server import ServerManager
from roles.client import ClientsManager


def load_client_server(datasets):
    server = ServerManager(
        model_name=args.gmodel_name,
        datasets=datasets,
        num_clients=args.num_clients,
        device=args.gpu_id,
        num_round=getattr(args, "num_rounds", args.num_rounds),
        client_sample_ratio=1.0
    )
    client_manager = ClientsManager(
        model_name=args.gmodel_name,
        datasets=datasets,
        num_clients=args.num_clients,
        device=args.gpu_id,
        eval_single_client=False,
    )
    return server, client_manager
