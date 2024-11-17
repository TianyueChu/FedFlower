import torch
from flwr.simulation import run_simulation
from flwr.server import ServerApp
from flwr.client import ClientApp
from client import client_fn
from server import server_fn

DEVICE = torch.device("cpu")

client = ClientApp(client_fn=client_fn)
server = ServerApp(server_fn=server_fn)

backend_config = {"client_resources": None}
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1}}

NUM_PARTITIONS = 50

if __name__ == '__main__':
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_PARTITIONS,
        backend_config=backend_config,
    )


