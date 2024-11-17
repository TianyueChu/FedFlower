import datasets
import torchvision.transforms as transforms
from data.fed_dataset import FedDataset
from data.partitioner.visual_dirichlet_partitioner import VisualDirichletPartitioner
from data.partitioner.iid_partitioner import IidPartitioner
from torch.utils.data import DataLoader

BATCH_SIZE = 32

def load_datasets(partition_id: int, num_partitions: int, distribution: str = "iid") -> tuple:
    """
    Load and partition the CelebA dataset for federated learning.

    Args:
        partition_id (int): The ID of the partition to load for the specific client.
        num_partitions (int): The total number of partitions to divide the dataset into.

    Returns:
        tuple: A tuple containing three DataLoader objects:
            - trainloader: DataLoader for the training data of the specified partition.
            - valloader: DataLoader for the validation data (20% of the partition's data).
            - testloader: DataLoader for the global test set.

    Steps:
    1. Load the CelebA dataset using the `datasets` library.
    2. Add demographic labels to the dataset using the `add_demographic_labels` function.
    3. Partition the dataset into `num_partitions` using a Visual Dirichlet Partitioner based on demographic labels.
    4. Load the specified partition (identified by `partition_id`) and split it into 80% training and 20% validation data.
    5. Apply PyTorch transforms (e.g., normalization) to the images in the dataset.
    6. Create PyTorch DataLoaders for the training, validation, and global test sets.

    :param num_partitions:
    :param partition_id:
    :param distribution:
    """
    # Step 1: Load the CelebA dataset
    dataset = datasets.load_dataset(path="flwrlabs/celeba")

    # Step 2: Add demographic labels
    labeled_dataset = add_demographic_labels(dataset)

    # Step 3: Select partitioning strategy
    if distribution == "iid":
        partitioner = IidPartitioner(num_partitions=num_partitions, partition_by="Demographic_Label")
    else:
        partitioner = VisualDirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="Demographic_Label",
            alpha=0.5,
            min_partition_size=10
        )

    # Step 4: Partition the dataset and load the specified partition
    fds = FedDataset(dataset=labeled_dataset, partitioners={"train": partitioner})
    partition = fds.load_partition(partition_id)

    # Split partition into 80% training and 20% validation
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Define PyTorch transformations
    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Apply transformations to the dataset
    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # Step 5: Create DataLoaders for training, validation, and testing
    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)

    # Load global test set and apply transformations
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    return trainloader, valloader, testloader


# Function to calculate demographic label
def assign_demographic_label(row):
    """
    Assign a demographic label based on the 'Male' and 'Young' attributes in the dataset.

    Args:
        row (dict): A dictionary representing a single data entry with attributes 'Male' and 'Young'.
                    These attributes are expected to be boolean values.

    Returns:
        int: An integer representing the demographic label, where:
             0 -> "Not Male & Not Young"
             1 -> "Not Male & Young"
             2 -> "Male & Not Young"
             3 -> "Male & Young"
    """
    # Ensure 'Male' and 'Young' values are integers (0 or 1)
    male = int(row['Male'])
    young = int(row['Young'])
    return (male << 1) | young

# Apply the function to assign demographic labels
def add_demographic_labels(dataset):
    """
    Add a 'Demographic_Label' field to the dataset.

    Args:
    dataset (list of dict): The dataset where each row is a dictionary.

    Returns:
    list of dict: The dataset with added demographic labels.
    """
    return dataset.map(lambda x: {"Demographic_Label": assign_demographic_label(x)})



