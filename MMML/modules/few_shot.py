import torch
from torch.utils.data import Dataset
import learn2learn.data.transforms as l2ltransforms
from learn2learn.data import MetaDataset
from torch import nn


class MeanRegister(nn.Module):
    def __init__(self, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        self.running_mean = None

    def forward(self, input):
        # input : bs, dim

        new_mean_estimate = torch.mean(input.detach(), axis=(0, -1, -2))
        if self.running_mean is None:
            self.running_mean = new_mean_estimate
        else:
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * new_mean_estimate


class FeatureNormalizer(nn.Module):
    def __init__(self, mean_register: MeanRegister):
        super().__init__()
        self.mean_register = mean_register

    def forward(self, feature):
        # feature : bs, n_exemple, n_class, n_dim

        normalized_vector = feature - self.mean_register.running_mean[
            None, None, None, :
        ].to(feature.device)
        return normalized_vector / torch.norm(feature, dim=-1, p=2)[..., None]


class InductiveNCM:
    """
    Adapted from EASY.
    """

    def __call__(self, shots: torch.Tensor, queries: torch.Tensor):
        assert len(shots.shape) == 4
        assert len(queries.shape) == 4
        bs_s, n_ways_s, n_shots, dim_s = shots.shape
        bs_q, n_ways_q, n_queries, dim_q = queries.shape
        assert (bs_s == bs_q) and (n_ways_s == n_ways_q) and (dim_s == dim_q)

        bs, n_ways, dim = bs_s, n_ways_s, dim_s

        means = torch.mean(shots, dim=2)

        # make a new axis for class prediction and take the minimum
        distances = torch.norm(
            queries.reshape(bs, n_ways, 1, -1, dim)
            - means.reshape(bs, 1, n_ways, 1, dim),
            dim=4,
            p=2,
        )
        winners = torch.min(distances, dim=2)
        # winners : bs x n_ways x n_queries
        return winners.indices


class FeatureDataset(Dataset):
    def __init__(self, n_class):
        self.n_class = n_class
        self.reset_data()

    def reset_data(self):
        self.all_features = None
        self.all_targets = None
        self.is_compiled = False

    def add_features(self, features, targets):
        if self.all_features is None:
            self.all_features = features
            self.all_targets = targets
        else:
            self.all_features = torch.concatenate([self.all_features, features], axis=0)
            self.all_targets = torch.concatenate([self.all_targets, targets], axis=0)

    def __getitem__(self, idx):
        return self.all_features[idx], self.all_targets[idx]

    def __len__(self):
        return len(self.images)


def get_dataset_to_transform(n_ways, n_shots, n_queries):
    global get_task_transform

    # global needed for pickling
    def get_task_transform(dataset):
        return [
            l2ltransforms.NWays(dataset, n=n_ways),
            l2ltransforms.KShots(dataset, k=n_shots + n_queries),
            l2ltransforms.LoadData(dataset),
            # l2ltransforms.RemapLabels(dataset),
            # l2ltransforms.ConsecutiveLabels(dataset),
        ]

    return get_task_transform


def get_number_class_few_shot_dataset(dataset):
    meta_dataset = MetaDataset(dataset)
    return len(meta_dataset.labels)


class Transforml2lTask:
    """
    Transform the tensor from l2l task into tensor of shape batch_size, n_example, feature, num_class

    """

    def __init__(self, n_ways, n_shots, n_queries):
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries

    def __call__(self, few_shot_tasks):
        features, labels = few_shot_tasks
        bs, n_exemples, dim = features.shape
        assert n_exemples == (self.n_shots + self.n_queries) * self.n_ways

        features = features.reshape(bs, self.n_ways, self.n_shots + self.n_queries, dim)
        labels = labels.reshape(bs, self.n_ways, self.n_shots + self.n_queries)
        # all labels should be the same along one way
        assert torch.all(labels == labels[..., 0][..., None])
        # note that this is also possible to use l2ltransforms to remap index.
        label_queries = torch.tile(
            torch.arange(self.n_ways), (bs, self.n_queries, 1)
        ).permute(0, 2, 1)
        label_queries = label_queries.to(features.device)
        features_shots = features[:, :, : self.n_shots, :]
        feature_queries = features[:, :, self.n_shots :, :]
        return features_shots, feature_queries, label_queries
