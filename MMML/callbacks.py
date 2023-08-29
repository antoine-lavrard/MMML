import os
import warnings
from tqdm import tqdm
import torch
from torch import save
from torch.utils.data import Dataset

from learn2learn.data import MetaDataset, TaskDataset
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.functional import avg_pool2d
from dataclasses import dataclass, InitVar, field
from typing import Any
from copy import deepcopy
import numpy as np

# from MMML.utils.utils import transfer_to, TrainingConfig
from MMML.modules.few_shot import FeatureDataset#, get_dataset_to_transform
from MMML.train.configs import ClassicalTraining, MultiStepTraining

class SaveIfImprovement:
    def __init__(self, path_output, to_save, name_metric, save_name:str, goal="increase"):
        self.path_output = path_output
        self.to_save= to_save
        self.name_metric=name_metric
        self.save_name = save_name
        self.goal = goal
        if goal == "increase":
            self.best = -np.inf
        elif goal == "decrease":
            self.best = np.inf
    def __call__(self, epoch, writer, training_config, dict_evals):
        if writer is None or dict_evals is None:
            return None
        
        current_value = dict_evals[self.name_metric]
        is_more = current_value > self.best
        is_better = (is_more and self.goal == "increase") or (not(is_more) and self.goal =="decrease")

        if is_better:
            self.best = current_value
            save_path = os.path.join(self.path_output, self.save_name)
            save(self.to_save, save_path)


class SaveBackbone:
    def __init__(self, path_output, to_save, save_name:str):
        self.path_output = path_output
        self.to_save= to_save

        self.save_name = save_name

    def __call__(self, epoch, writer, training_config, dict_evals):
        if writer is None or dict_evals is None:
            return None
        save_path = os.path.join(self.path_output, self.save_name)
        save(self.to_save, save_path)


class SaveStateCallback:
    def __init__(self, path_output, state, each_n_epoch):
        self.path_output = path_output
        self.state = state
        self.each_n_epoch = each_n_epoch

    def __call__(self, epoch, writer, training_config, dict_evals):
        if writer is None:
            return None
        if (epoch+1) % self.each_n_epoch == 0:
            save_path = os.path.join(self.path_output, "checkpoint.pt")
            save(self.state, save_path)


class WriteLogsCallback:
    def __call__(self, epoch,  writer, training_config: ClassicalTraining, dict_evals):
        module = training_config.training_methode.module
        prefix = training_config.dataloader_config.name_split

        dict_logs = module.accumulate_and_get_logs()
        print(dict_logs)
        if writer is None:
            return None

        write_logs(dict_logs, writer, prefix, epoch)
        scheduler = training_config.optim_config.scheduler
        if scheduler is not None:
            print("lr : ", scheduler.get_last_lr())
            writer.add_scalar(
                prefix + "-lr",
                scheduler.get_last_lr()[0],
                epoch,
            )


class WriteLogsMultistepCallback:
    def __call__(self, epoch, writer, training_config: MultiStepTraining, dict_evals):
        modules = training_config.training_methode
        prefix = training_config.dataloader_config.name_split

        for i, module_config in enumerate(modules):
            module=  module_config.module
            dict_logs= module.accumulate_and_get_logs()
            print("step i : ", i)
            print(dict_logs)
            if writer is None:
                return None

            write_logs(dict_logs, writer, f"step-{i}-"+prefix, epoch)
        scheduler = training_config.optim_config.scheduler
        if scheduler is not None:
            print("lr : ", scheduler.get_last_lr())
            writer.add_scalar(
                prefix + "-lr",
                scheduler.get_last_lr()[0],
                epoch,
            )


def write_logs(dict_logs, writer, prefix: str, epoch: int):
    for name_metric, value in dict_logs.items():
        writer.add_scalar(prefix + "-" + name_metric, value, epoch)

