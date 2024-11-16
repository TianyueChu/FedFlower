import flwr as fl
import torch
from flwr.common import Context,ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from torch.backends.opt_einsum import strategy
from task import test_fn
from stratergy.FedAvg import FedCustomAvg
from models.mobilenetv2 import CelebAMobileNet
from data.data_loader import load_datasets
from task import get_weights

Device = "cuda" if torch.cuda.is_available() else "cpu"

params = get_weights(CelebAMobileNet(num_classes=4))

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training

    config = ServerConfig(num_rounds=3)

    return ServerAppComponents(strategy=strategy, config=config)


_,_,test_data = load_datasets(0,10)


# Create FedAvg strategy
strategy = FedCustomAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=50,  # Sample 100% of available clients for evaluation
    min_fit_clients=40,  # Never sample less than 5 clients for training
    min_evaluate_clients=40,  # Never sample less than 5 clients for evaluation
    min_available_clients=40,  # Wait until all 5 clients are available
    initial_parameters=ndarrays_to_parameters(params),
    evaluate_fn=test_fn,
    server_testset=test_data,
    net=CelebAMobileNet(num_classes=4),
    device=Device,
)

# Run the server
fl.server.start_server(server_address="0.0.0.0:8080", config=ServerConfig(num_rounds=10), strategy=strategy)