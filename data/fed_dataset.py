from typing import Any, Dict, Optional, Tuple, Union

from datasets import Dataset, DatasetDict
from flwr_datasets.common import EventType, event
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.preprocessor import Preprocessor
from flwr_datasets.utils import _instantiate_merger_if_needed, _instantiate_partitioners


class FedDataset:
    """Representation of a dataset for federated learning/evaluation/analytics.

    Directly uses a pre-loaded DatasetDict object for federated operations.

    Parameters
    ----------
    dataset : DatasetDict
        The dataset already loaded in the form of a Hugging Face DatasetDict.
    preprocessor : Optional[Union[Preprocessor, Dict[str, Tuple[str, ...]]]]
        `Callable` that transforms `DatasetDict` by resplitting, removing
        features, creating new features, performing any other preprocessing operation,
        or configuration dict for `Merger`. Applied after shuffling. If None,
        no operation is applied.
    partitioners : Dict[str, Union[Partitioner, int]]
        A dictionary mapping the Dataset split (a `str`) to a `Partitioner` or an `int`
        (representing the number of IID partitions that this split should be
        partitioned into, i.e., using the default partitioner).
    shuffle : bool
        Whether to randomize the order of samples. Applied prior to preprocessing
        operations, separately to each of the present splits in the dataset.
        Defaults to True.
    seed : Optional[int]
        Seed used for dataset shuffling. It has no effect if `shuffle` is False.
        Defaults to 42.
    """

    def __init__(
        self,
        *,
        dataset: DatasetDict,
        preprocessor: Optional[Union[Preprocessor, Dict[str, Tuple[str, ...]]]] = None,
        partitioners: Dict[str, Union[Partitioner, int]],
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        self._dataset: DatasetDict = dataset
        self._preprocessor: Optional[Preprocessor] = _instantiate_merger_if_needed(
            preprocessor
        )
        # If the dataset is not already a dictionary with splits as keys,
        # we assume it represents a single split, and we name it "train"
        if not hasattr(dataset, "keys"):  # Check if dataset has keys() method
            self._dataset = {"train": dataset}  # Wrap in a dictionary with "train" key
        else:
            self._dataset = dataset

        self._partitioners: Dict[str, Partitioner] = _instantiate_partitioners(
            partitioners
        )
        self._shuffle = shuffle
        self._seed = seed
        self._dataset_prepared: bool = False
        self._event = {
            "load_partition": {split: False for split in self._partitioners},
        }

    def load_partition(
        self,
        partition_id: int,
        split: Optional[str] = None,
    ) -> Dataset:
        """Load the partition specified by the idx in the selected split."""
        if not self._dataset_prepared:
            self._prepare_dataset()
        if split is None:
            self._check_if_no_split_keyword_possible()
            split = list(self._partitioners.keys())[0]
        self._check_if_split_present(split)
        self._check_if_split_possible_to_federate(split)
        partitioner: Partitioner = self._partitioners[split]
        self._assign_dataset_to_partitioner(split)
        partition = partitioner.load_partition(partition_id)
        if not self._event["load_partition"][split]:
            event(
                EventType.LOAD_PARTITION_CALLED,
                {
                    "federated_dataset_id": id(self),
                    "split": split,
                    "partitioner": partitioner.__class__.__name__,
                    "num_partitions": partitioner.num_partitions,
                },
            )
            self._event["load_partition"][split] = True
        return partition

    def load_split(self, split: str) -> Dataset:
        """Load the full split of the dataset."""
        if not self._dataset_prepared:
            self._prepare_dataset()
        self._check_if_split_present(split)
        dataset_split = self._dataset[split]
        return dataset_split

    def _check_if_split_present(self, split: str) -> None:
        """Check if the split is in the dataset."""
        available_splits = list(self._dataset.keys())
        if split not in available_splits:
            raise ValueError(
                f"The given split: '{split}' is not present in the dataset's splits: "
                f"'{available_splits}'."
            )

    def _check_if_split_possible_to_federate(self, split: str) -> None:
        """Check if the split has a corresponding partitioner."""
        partitioners_keys = list(self._partitioners.keys())
        if split not in partitioners_keys:
            raise ValueError(
                f"The given split: '{split}' does not have a partitioner to perform "
                f"partitioning. Partitioners were specified for the following splits:"
                f"'{partitioners_keys}'."
            )

    def _assign_dataset_to_partitioner(self, split: str) -> None:
        """Assign the corresponding split of the dataset to the partitioner."""
        if not self._partitioners[split].is_dataset_assigned():
            self._partitioners[split].dataset = self._dataset[split]

    def _prepare_dataset(self) -> None:
        """Prepare the dataset (prior to partitioning) by shuffle and preprocess."""
        if self._shuffle:
            self._dataset = self._dataset.shuffle(seed=self._seed)
        if self._preprocessor:
            self._dataset = self._preprocessor(self._dataset)
        self._dataset_prepared = True

    def _check_if_no_split_keyword_possible(self) -> None:
        if len(self._partitioners) != 1:
            raise ValueError(
                "Please set the `split` argument. You can only omit the split keyword "
                "if there is exactly one partitioner specified."
            )