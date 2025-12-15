from config import args
from data.load_data import load_dataset
from roles.manager import load_client_server


def main():
    print(args)
    datasets = load_dataset(args)
    server, client_manager = load_client_server(datasets)

    server.collaborative_training_encoder(client_manager.clients)

    k = getattr(args, "k_clusters", None)
    if k is not None:
        for c in client_manager.clients:
            labels = c.local_cluster(args.gpu_id, k)
            if labels is not None:
                print(f"Client {c.client_id} clustering done, K={k}")


if __name__ == "__main__":
    main()
