from flwr_datasets.partitioner.partitioner import Partitioner
from collections import defaultdict
from typing import List, Optional
from datasets import Dataset
import numpy as np

class IidPartitioner(Partitioner):
    """
    IID Partitioner with stratification to ensure each shard has the same size
    and the same class distribution as the original dataset.

    Parameters
    ----------
    num_partitions : int
        The number of partitions to create.
    partition_by : str
        The name of the column containing class labels.
    """

    def __init__(self, num_partitions: int, partition_by: str) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
        self._num_partitions = num_partitions
        self.partition_by = partition_by
        self.partitions: Optional[List[List[int]]] = None  # Will hold partition indices

    def _create_partitions(self) -> List[List[int]]:
        """Create stratified partitions based on class labels."""
        if self._dataset is None:
            raise AttributeError("The dataset must be assigned before creating partitions.")

        # Group indices by class
        label_to_indices = defaultdict(list)
        for idx, example in enumerate(self._dataset):
            label_to_indices[example[self.partition_by]].append(idx)

        # Shuffle indices within each class
        for label, indices in label_to_indices.items():
            np.random.shuffle(indices)

        # Distribute indices evenly across partitions
        partitions = [[] for _ in range(self._num_partitions)]
        for label, indices in label_to_indices.items():
            num_samples = len(indices)
            shard_size = num_samples // self._num_partitions
            remainder = num_samples % self._num_partitions

            start_idx = 0
            for i in range(self._num_partitions):
                end_idx = start_idx + shard_size + (1 if i < remainder else 0)
                partitions[i].extend(indices[start_idx:end_idx])
                start_idx = end_idx

        # Shuffle partitions to mix classes randomly
        for partition in partitions:
            np.random.shuffle(partition)

        return partitions

    def load_partition(self, partition_id: int) -> Dataset:
        """
        Load a specific partition by ID.

        Parameters
        ----------
        partition_id : int
            The partition ID to load.

        Returns
        -------
        datasets.Dataset
            The dataset partition corresponding to the given partition ID.
        """
        if not self.is_dataset_assigned():
            raise AttributeError("The dataset must be assigned before loading partitions.")
        if not (0 <= partition_id < self._num_partitions):
            raise ValueError(f"Partition ID must be between 0 and {self._num_partitions - 1}.")

        # Create partitions if not already created
        if self.partitions is None:
            self.partitions = self._create_partitions()

        indices = self.partitions[partition_id]
        return self._dataset.select(indices)

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        return self._num_partitions