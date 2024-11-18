from flwr.simulation import run_simulation
from flwr.server import ServerApp
from flwr.client import ClientApp
from client import client_fn
from server import server_fn
import configs.config as cfg


client = ClientApp(client_fn=client_fn)
server = ServerApp(server_fn=server_fn)

NUM_PARTITIONS = cfg.NUM_PARTITIONS

if __name__ == '__main__':
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_PARTITIONS,
        backend_config=cfg.backend_config,
    )


