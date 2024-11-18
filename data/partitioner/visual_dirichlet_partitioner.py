import numpy as np
from typing import Union, Optional, List
import datasets
from flwr_datasets.common.typing import NDArrayFloat
from flwr_datasets.partitioner.partitioner import Partitioner


class VisualDirichletPartitioner(Partitioner):
    """Partitioner based on Dirichlet distribution for visual classifcation

    Implementation based on Measuring the effects of non-identical data distribution
    for federated visual classification.arXiv preprint arXiv:1909.06335 (2019).

    The class distribution for each client is represented by a vector q, where qi denotes
    the probability of class i and the sum of all qi equals 1. To generate the client-specific
    class distribution vectors q, the authors sample from a Dirichlet distribution parameterized
    by αlpha * p, where: p is a prior distribution over classes, reflecting the overall class
    distribution in the dataset. alpha is a concentration parameter that controls the similarity
    between the client distributions and the prior distribution p.


    A higher alpha value results in client distributions q that are more similar to the prior distribution p,
    indicating more homogeneous data across clients. A lower alpha value leads to client distributions that
    are more skewed, with clients having data concentrated in fewer classes, thus increasing heterogeneity.


    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    partition_by : str
        Column name of the labels (targets) based on which Dirichlet sampling works.
    alpha : Union[int, float, List[float]]
        Concentration parameter to the Dirichlet distribution
    min_partition_size : int
        The minimum number of samples that each partitions will have (the sampling
        process is repeated if any partition is too small).
    shuffle: bool
        Whether to randomize the order of samples. Shuffling applied after the
        samples assignment to partitions.
    seed: int
        Seed used for dataset shuffling. It has no effect if `shuffle` is False.
    """

    def __init__(
        self,
        num_partitions: int,
        partition_by: str,
        alpha: Union[int, float, list[float]],
        min_partition_size: int = 10,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        # Attributes based on the constructor
        self._num_partitions = num_partitions
        self._check_num_partitions_greater_than_zero()
        self._alpha = self._initialize_alpha(alpha)
        self._partition_by = partition_by
        self._min_partition_size:int = min_partition_size
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

        # Utility attributes
        self._avg_num_of_samples_per_partition: Optional[float] = None
        self._unique_classes: Optional[Union[list[int], list[str]]] = None
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_indices_determined = False

        # Attributes to store partitions
        self.partitions = None
        self.partition_determined = False


    def _initialize_alpha(
        self, alpha: Union[int, float, list[float], NDArrayFloat]
    ) -> NDArrayFloat:
        """Convert alpha to the used format in the code a NDArrayFloat.

        The alpha can be provided in constructor can be in different format for user
        convenience. The format into which it's transformed here is used throughout the
        code for computation.

        Parameters
        ----------
            alpha : Union[int, float, List[float], NDArrayFloat]
                Concentration parameter to the Dirichlet distribution

        Returns
        -------
        alpha : NDArrayFloat
            Concentration parameter in a format ready to used in computation.
        """
        if isinstance(alpha, int):
            alpha = np.array([float(alpha)], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, float):
            alpha = np.array([alpha], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, list):
            if len(alpha) != self._num_partitions:
                raise ValueError(
                    "If passing alpha as a List, it needs to be of length of equal to "
                    "num_partitions."
                )
            alpha = np.asarray(alpha)
        elif isinstance(alpha, np.ndarray):
            if alpha.ndim == 1 and alpha.shape[0] != self._num_partitions:
                raise ValueError(
                    "If passing alpha as an NDArray, its length needs to be of length "
                    "equal to num_partitions."
                )
            elif alpha.ndim == 2:
                alpha = alpha.flatten()
                if alpha.shape[0] != self._num_partitions:
                    raise ValueError(
                        "If passing alpha as an NDArray, its size needs to be of length"
                        " equal to num_partitions."
                    )
        else:
            raise ValueError("The given alpha format is not supported.")
        if not (alpha > 0).all():
            raise ValueError(
                f"Alpha values should be strictly greater than zero. "
                f"Instead it'd be converted to {alpha}"
            )
        return alpha

    def _determine_partition_id_to_indices_if_needed(
        self,
    ) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return

        # Generate information needed for Dirichlet partitioning
        self._unique_classes = self.dataset.unique(self._partition_by)
        assert self._unique_classes is not None
        # This is needed only if self._self_balancing is True (the default option)
        self._avg_num_of_samples_per_partition = (
            self.dataset.num_rows / self._num_partitions
        )

        # Change targets list data type to numpy
        targets = np.array(self.dataset[self._partition_by])

        # Repeat the sampling procedure based on the Dirichlet distribution until the
        # min_partition_size is reached.
        sampling_try = 0
        while True:
            # Prepare data structure to store indices assigned to partition ids
            partition_id_to_indices: dict[int, list[int]] = {nid: [] for nid in range(self._num_partitions)}
            # Perform Dirichlet-based partitioning
            for class_label in self._unique_classes:
                # Access all the indices associated with class_label
                class_indices = np.nonzero(targets == class_label)[0]
                class_count = len(class_indices)

                # Determine proportions using Dirichlet distribution
                dirichlet_proportions = self._rng.dirichlet(self._alpha)
                # Normalize proportions to ensure minimum partition size
                normalized_proportions = np.clip(
                    dirichlet_proportions * class_count,
                    self._min_partition_size,
                    None
                ).astype(int)

                # Adjust proportions to maintain total sample count
                total_assigned = normalized_proportions.sum()
                if total_assigned < class_count:
                    normalized_proportions[np.argmax(dirichlet_proportions)] += class_count - total_assigned

                # Split class indices based on proportions
                cumulative_splits = np.cumsum(normalized_proportions)[:-1]
                split_indices = np.split(class_indices, cumulative_splits)

                # Assign indices to partitions
                for nid, indices in enumerate(split_indices):
                    partition_id_to_indices[nid].extend(indices)

            # Verify minimum partition size constraint
            min_sample_size_on_client = min(
                len(indices) for indices in partition_id_to_indices.values()
            )
            if min_sample_size_on_client >= self._min_partition_size:
                break

            sampling_try += 1
            if sampling_try >= 10:
                raise ValueError(
                    "Failed to generate partitions meeting the minimum size requirement "
                    "after 10 attempts. Adjust alpha or min_partition_size."
                )

        # Shuffle indices within each partition if required
        if self._shuffle:
            for indices in partition_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)

        # Store the partition indices
        self._partition_id_to_indices = partition_id_to_indices
        self._partition_id_to_indices_determined = True


    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a single partition based on the partition index."""
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions


    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test num_partitions when the dataset is given (in load_partition)."""
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of "
                    "samples in the dataset."
                )

    def _check_num_partitions_greater_than_zero(self) -> None:
        """Test num_partition left sides correctness."""
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")