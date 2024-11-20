import flwr.client
import torch
from ray.tune.examples.pbt_dcgan_mnist.common import batch_size

from models.mobilenetv2 import CelebAMobileNet
import os
from filelock import FileLock

from configs.client_id import generate_client_id

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from task import train_fn, test_fn
from data.data_loader import load_datasets
from task import set_weights, get_weights

import configs.config as cfg
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
client_id = generate_client_id()


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs,learning_rate,partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        self.learning_rate = learning_rate
        self.partition_id = partition_id

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train_fn(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.learning_rate,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        print("client parameters: ", parameters)
        set_weights(self.net, parameters)
        folder_path = f"./results/client_{self.partition_id}"
        os.makedirs(folder_path, exist_ok=True)
        loss, metrics = test_fn(self.net, self.valloader, self.device, results_dir=folder_path)
        return loss, len(self.valloader.dataset), metrics

# Flower.ai lose the ability to track the client id when running in a real distributed environment.
def initialize_partition_file(num_clients, config_dir="./configs"):
    """
    Initialize the partition_id.txt file with a list of partition IDs if it does not already exist.
    If the file exists, return its current content.
    """
    os.makedirs(config_dir, exist_ok=True)
    partition_file = os.path.join(config_dir, "partition_id.txt")

    if os.path.exists(partition_file):
        return

    # If the file does not exist, initialize it with the partition IDs
    with open(partition_file, "w") as f:
        partition_ids = " ".join(map(str, range(num_clients)))
        f.write(partition_ids + "\n")
    print(f"Initialized {partition_file} with {num_clients} partition IDs.")
    return


def get_or_create_partition_id(client_id, config_dir="./configs"):
    """
    Safely get or reuse a partition ID for the current client.
    """
    os.makedirs(config_dir, exist_ok=True)
    assigned_file = os.path.join(config_dir, f"client_{client_id}_partitionid.txt")
    partition_file = os.path.join(config_dir, "partition_id.txt")
    lock_file = partition_file + ".lock"

    # Check if the client already has an assigned partition ID
    if os.path.exists(assigned_file):
        with open(assigned_file, "r") as f:
            return int(f.readline().strip())

    # If no assigned ID exists, get the first ID from partition_id.txt
    with FileLock(lock_file):  # Ensure only one client accesses partition_id.txt at a time
        with open(partition_file, "r") as f:
            partition_ids = list(map(int, f.readline().strip().split()))

        if not partition_ids:
            raise ValueError("No partition IDs left to assign!")

        assigned_id = partition_ids.pop(0)  # Assign the first available ID

        # Write the remaining IDs back to partition_id.txt
        with open(partition_file, "w") as f:
            f.write(" ".join(map(str, partition_ids)) + "\n")

    # Save the assigned ID for this client
    with open(assigned_file, "w") as f:
        f.write(str(assigned_id))

    return assigned_id


def client_fn(context: Context):
    net = CelebAMobileNet(num_classes=4).to(DEVICE)
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, _ = load_datasets(partition_id, num_partitions, batch_size=cfg.BATCH_SIZE, non_iid=False)
    local_epochs = cfg.LOCAL_EPOCHS
    learning_rate = cfg.LearningRate
    return FlowerClient(net, trainloader, valloader, local_epochs, learning_rate, partition_id).to_client()


# Run the client in the real setting
# def client_fn(context: Context):
#     # Load model and data
#     net = CelebAMobileNet(num_classes=4).to(DEVICE)
#     # print(f"context: {context}")
#     # context: Context(node_id=-1, node_config={}, state=RecordSet(parameters_records={}, metrics_records={}, configs_records={}), run_config={})
#     # partition_id = context.node_config.get("partition-id", None)
#     # if partition_id is None:
#     #     raise KeyError("partition-id is missing in node_config")
#     partition_id = get_or_create_partition_id(client_id)
#     # num_partitions = context.node_config["num-partitions"]
#     num_partitions = 50
#     # local_epochs = context.run_config["local-epochs"]
#     local_epochs = 1
#     learning_rate = 0.001
#
#     # Return Client instance
#     return FlowerClient(net, trainloader, valloader, local_epochs, learning_rate, partition_id).to_client()
# number_clients = 50
# initialize_partition_file(number_clients)
# flwr.client.start_client(server_address="127.0.0.1:8080", client_fn=client_fn)
