"""
Contains the various function linked to dataset. 
Specificity : training on sevral dataset at the same time. 
    - define a dataset that aggregate sevral other dataset
    - define a sampler that aggregate sevral other sampler
Config :

"""

from typing import List, Iterator, Iterable, Tuple, Union, Optional, Any, Dict
import torch
from torch.utils.data import Dataset, IterableDataset


class AlternateBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Wrap sevral batch samplers.
    The batch will be sampled from the first dataset, then the second, then the third, etc.
    """

    def __init__(self, samplers: List[torch.utils.data.Sampler]):
        self.samplers = samplers
        self.number_it = min(len(sampler) for sampler in self.samplers)
        self.number_dataset = len(self.samplers)

    def __iter__(self) -> Iterator[List[int]]:
        samplers = [iter(sampler) for sampler in self.samplers]

        for _ in range(self.number_it):
            for dataset_index in range(self.number_dataset):
                sampler_index = next(samplers[dataset_index])

                index_datasets = [dataset_index] * len(sampler_index)
                # tuple of batch -> batch of tuple
                yield zip(index_datasets, sampler_index)

    def __len__(self):
        return len(self.samplers) * self.number_it


class MultiSourceDataset(Dataset):
    """
    Create a dataset by wrapping sevral datasets.
    In order to keep track of the source dataset, the dataset also returns the dataset name.
    Adapted from :
    https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset
    """

    def __init__(self, datasets: Iterable[Dataset], list_name: List[str]):
        self.datasets = list(datasets)
        self.list_name = list_name
        assert len(self.datasets) == len(self.list_name), "datasets should not be empty"

        # check that all dataset are not iterable
        for dataset in self.datasets:
            assert not isinstance(
                dataset, IterableDataset
            ), "MultiSourceDataset does not support IterableDataset"
        self.size = sum(len(dataset) for dataset in self.datasets)

    def __len__(self):
        return self.size

    def __getitem__(self, index: tuple[int, int]):
        dataset_index, index_data = index
        return (self.list_name[dataset_index], self.datasets[dataset_index][index_data])

