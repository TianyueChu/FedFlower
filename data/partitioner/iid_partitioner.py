import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class IidPartitioner(Partitioner):
    """Partitioner creates each partition sampled randomly from the dataset.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    """

    def __init__(self, num_partitions: int) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
        self._num_partitions = num_partitions

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a single IID partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """
        return self.dataset.shard(
            num_shards=self._num_partitions, index=partition_id, contiguous=True
        )

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        return self._num_partitions