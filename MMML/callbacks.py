
from tqdm import tqdm
from torch import save
import os
import torch
from learn2learn.data import MetaDataset, TaskDataset
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.functional import avg_pool2d


from MMML.utils.utils import transfer_to, TrainingConfig
from MMML.modules.few_shot import FeatureDataset, get_dataset_to_transform
import warnings

class SaveStateCallback:
    def __init__(self, path_output, state, each_n_epoch):
        self.path_output = path_output
        self.state = state
        self.each_n_epoch = each_n_epoch
    
    def __call__(self, writer, training_config : TrainingConfig):
        if training_config.epochs % self.each_n_epoch == 0:
            save_path = os.path.join(self.path_output, "checkpoint.pt")
            save(self.state, save_path)

class WriteLogsCallback:
    def __init__(self, add_lr = False):

        self.add_lr = add_lr
        
    def __call__(self, writer, training_config : TrainingConfig):
        if writer is None:
            return 0
        module = training_config.training_module
        epoch = training_config.epochs
        prefix = training_config.name_split
        dict_logs = module.accumulate_and_get_logs()
        
        write_logs(dict_logs, writer, prefix, epoch)
        
        if self.add_lr:
            writer.add_scalar("lr", training_config.scheduler.get_last_lr()[0], epoch)

from torch.utils.data import Dataset
from dataclasses import dataclass, InitVar, field
from typing import Any

@dataclass
class ValidationConfig:
    validation_module : Any    
    dataset : Dataset
    dataloader_kwargs : dict
    device: str = "cuda:0"
    prefix: str = "validation-"
    @property
    def dataloader(self):
        return DataLoader(self.dataset, **self.dataloader_kwargs)

@dataclass
class FewShotValidationConfig:
    n_ways: int
    n_shots: int
    n_queries: int
    backbone : Any
    fs_module: Any
    dataset : Dataset
    dataloader_kwargs : dict
    dataloader_few_shot_kwargs: dict
    device : str = "cuda:0"
    device_few_shot : str = "cpu"
    prefix: str = "validation-fs-"
    num_tasks : int = 20000
    dataset_to_transform : callable= field(init = False)
    labels_to_indices: list = field(init= False)
    indices_to_labels : list = field(init=False)
    number_class : int = field(init = False)
    
    def __post_init__(self):
        if self.dataloader_kwargs["shuffle"] is True:
            warnings.warn("Setting shuffle to False for dataloader")
        self.dataloader_kwargs["shuffle"] = False
        meta_dataset = MetaDataset(self.dataset)
        self.labels_to_indices = meta_dataset.labels_to_indices
        self.indices_to_labels = meta_dataset.indices_to_labels
        self.number_class = len(meta_dataset.labels)
        self.dataset_to_transform = get_dataset_to_transform(self.n_ways, self.n_shots, self.n_queries)

    @property
    def dataloader(self):
        return DataLoader(self.dataset, **self.dataloader_kwargs)

class SaveValidationMetricsCallback:
    def __init__(self, config : ValidationConfig):
        self.config = config


    def __call__(self, writer, training_config: TrainingConfig):

        epoch = training_config.epochs
        self.config.validation_module.to(self.config.device)
        self.config.validation_module.eval()

        for input_data in tqdm(self.config.dataloader):
            # ----- INPUT -------
            input_data = transfer_to(input_data, self.config.device)
            # ----- FORWARD  ------
            self.config.validation_module(input_data)

        dict_logs = self.config.validation_module.accumulate_and_get_logs()
        self.config.validation_module.to("cpu")
        write_logs(dict_logs, writer, self.config.prefix, epoch)



def write_logs(dict_logs, writer, prefix: str, epoch: int):
    for name_metric, value in dict_logs.items():
        writer.add_scalar(prefix + "-" + name_metric, value, epoch)




class SaveFewShotValidationMetricCallback:
    def __init__(
        self,
        fs_config : FewShotValidationConfig,
    ):
        self.fs_config = fs_config

    def __call__(self, writer, training_config : TrainingConfig):
        if writer is None:
            raise Warning("Write not setup")
        epoch = training_config.epochs
        feature_dataset = FeatureDataset(self.fs_config.number_class)
        self.fs_config.backbone.eval()
        self.fs_config.backbone.to(self.fs_config.device)
        with torch.no_grad():
            for batch in tqdm(self.fs_config.dataloader, desc = "few_shot"):
                batch_image, targets = batch
                batch_image = batch_image.to(self.fs_config.device)
                features = self.fs_config.backbone(batch_image)
                features = avg_pool2d(features, kernel_size = features.shape[-1])
                features = torch.flatten(features, start_dim=1, end_dim=-1)
                feature_dataset.add_features(features.cpu(), targets.cpu())

            meta_dataset = MetaDataset(
                feature_dataset,
                labels_to_indices=self.fs_config.labels_to_indices,
                indices_to_labels=self.fs_config.indices_to_labels,
            )
            
            taskset = TaskDataset(
                meta_dataset, self.fs_config.dataset_to_transform(meta_dataset), num_tasks=self.fs_config.num_tasks
            )

            transfer_to(self.fs_config.fs_module, self.fs_config.device_few_shot)
            dataloader = DataLoader(taskset, **self.fs_config.dataloader_few_shot_kwargs)
            for few_shot_tasks in tqdm(dataloader, desc = "few_shot_classification"):
                transfer_to(few_shot_tasks, self.fs_config.device_few_shot)
                self.fs_config.fs_module(few_shot_tasks)
            transfer_to(self.fs_config.fs_module, "cpu")

        dict_logs = self.fs_config.fs_module.accumulate_and_get_logs()
        write_logs(dict_logs, writer, self.fs_config.prefix, epoch)
         