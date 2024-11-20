import torch

BATCH_SIZE = 8
NUM_ROUNDS = 10
LearningRate = 0.001
LOCAL_EPOCHS = 1
NUM_PARTITIONS = 50

DEVICE = torch.device("cpu")
# Define the backend configuration
backend_config = {"client_resources": None}
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1}}