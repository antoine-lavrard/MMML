import warnings

from dataclasses import dataclass, field
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset

from learn2learn.data import MetaDataset, TaskDataset


from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.functional import avg_pool2d
from dataclasses import dataclass, InitVar, field
from typing import Union
from copy import deepcopy

from MMML.modules.few_shot import get_dataset_to_transform
from MMML.train.lookahead import Lookahead


@dataclass
class OptimizerConfig:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler


@dataclass
class QuantisationConfig:
    enable_amd: bool = False
    dtype_amd: Any = torch.float16
    scaler: torch.cuda.amp.GradScaler = field(init=None)

    def __post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amd)


from abc import ABC, abstractmethod


@dataclass
class Prepare(ABC):
    @abstractmethod
    def prepare_training_module(self):
        pass

    @abstractmethod
    def prepare_model(self):
        pass


@dataclass
class PrepareLookahead(Prepare):
    optimizer_config: OptimizerConfig
    model: torch.nn.Module

    def __post_init__(self):
        assert type(self.optimizer) is Lookahead, "type of optimizer is not lookahead"
        self.state = "training"

    def prepare_training_module(self):
        assert self.state == "evaluation"
        self.optimizer_config.optimizer._clear_and_load_backup()
        self.state = "training"

    def prepare_model(self):
        assert self.state == "training", f"{self.state}"

        self.optimizer_config.optimizer._backup_and_load_cache()
        self.state = "evaluation"


from MMML.utils import update_bn


@dataclass
class DefaultPreparer(Prepare):
    def prepare_training_module(self):
        pass

    def prepare_model(self):
        pass


# MethodeConfig : how the network should be runed


@dataclass
class MethodeConfig:
    module: torch.nn.Module

    DEVICE: str = "cuda:0"
    def get_evaluation_module(self):
        return self.module
   


# avoid local fn in order to be able to pickle functino
def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
    return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (
        num_averaged + 1
    )


class StepCounter:
    def __init__(self):
        self.reset()
    def step(self):
        self.number_step +=1
    def reset(self):
        self.number_step =0

@dataclass
class SWAMethodeConfig:
    module: torch.nn.Module
    swa_model: torch.optim.swa_utils.AveragedModel = None
    DEVICE: str = "cuda:0"
    update_every_n: int = 1
    step_counter =  StepCounter()

    def __post_init__(self):
        self.number_batch_since_average = -self.skip_n_step * self.update_every_n
        if self.swa_model is None:
            self.swa_model = torch.optim.swa_utils.AveragedModel(
                self.module, avg_fn=avg_fn
            )
    def get_evaluation_module(self):
        return self.swa_model
    
    def update_swa(self, module):
        self.number_batch_since_average += 1
        self.step_counter.step()
        if self.step_counter.number_step == self.update_every_n:
            self.swa_model.update_parameters(module)
            self.number_batch_since_average = 0
            self.step_counter.reset()



@dataclass
class SharedSWAMethodeConfig:
    module: torch.nn.Module
    swa_model: torch.optim.swa_utils.AveragedModel
    current_module: int
    DEVICE: str = "cuda:0"
    update_every_n: int = 1
    step_counter: StepCounter = None

    def get_evaluation_module(self):
        return self.swa_model.module[self.current_module]
    
    def update_swa(self, module):
        self.step_counter.step()
        if self.step_counter.number_step == self.update_every_n:
            self.swa_model.update_parameters(module)
            self.number_batch_since_average = 0
            self.step_counter.reset()


def get_multi_swa_methode_config(list_module, **kwargs) -> list[SharedSWAMethodeConfig]:

    list_module =  torch.nn.ModuleList(list_module)
   
    swa_model = torch.optim.swa_utils.AveragedModel(
                list_module, avg_fn=avg_fn
    )
    shared_step_counter = StepCounter()
    return [
        SharedSWAMethodeConfig(list_module[i], swa_model, i, step_counter = shared_step_counter, **kwargs) for i in range(len(list_module))
    ]

@dataclass
class DataLoaderConfig:
    dataset: Dataset
    dataloader_kwargs: dict
    name_split: str = ""

    def __post_init__(self):
        self.dataloader = DataLoader(self.dataset, **self.dataloader_kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["dataloader"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        self.dataloader = DataLoader(self.dataset, **self.dataloader_kwargs)


@dataclass
class PrepareSWA(Prepare):
    dataloader_config: DataLoaderConfig
    swa_config: SWAMethodeConfig


    def prepare_training_module(self):
        pass

    def prepare_model(self):
        update_bn(
            self.dataloader_config.dataloader,
            self.swa_config.get_evaluation_module().to(self.swa_config.DEVICE),
            device=self.swa_config.DEVICE,
        )


@dataclass
class UnshuffleDataloaderConfig:
    dataset: Dataset
    dataloader_kwargs: dict
    name_split: str = ""
    dataset_to_transform: callable = field(init=False)

    def __post_init__(self):
        if self.dataloader_kwargs["shuffle"] is True:
            warnings.warn("Setting shuffle to False for dataloader")
            self.dataloader_kwargs = deepcopy(self.dataloader_kwargs)
            self.dataloader_kwargs["shuffle"] = False

        self.dataloader = DataLoader(self.dataset, **self.dataloader_kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["dataloader"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        self.dataloader = DataLoader(self.dataset, **self.dataloader_kwargs)


@dataclass
class FeatureDatasetConfig:
    n_ways: int
    n_shots: int
    n_queries: int
    dataset: Dataset
    dataloader_kwargs: dict
    labels_to_indices: list = field(init=False)
    indices_to_labels: list = field(init=False)
    number_class_novel: int = field(init=False)
    num_tasks: int = 10000

    def __post_init__(self):
        meta_dataset = MetaDataset(self.dataset)
        self.labels_to_indices = meta_dataset.labels_to_indices
        self.indices_to_labels = meta_dataset.indices_to_labels
        self.number_class_novel = len(meta_dataset.labels)
        self.dataset_to_transform = get_dataset_to_transform(
            self.n_ways, self.n_shots, self.n_queries
        )


@dataclass
class EvaluationConfig:
    methode_config: MethodeConfig
    dataloader_config: DataLoaderConfig
    evaluation_each_n_epoch: int = 1


@dataclass
class FewShotEvaluationConfig:
    backbone_methode_config: MethodeConfig
    fs_module: MethodeConfig
    dataloader_config: UnshuffleDataloaderConfig
    dataloader_feature_config: FeatureDatasetConfig
    intermediate_device: str = None
    evaluation_each_n_epoch: int = 1

    def __post_init__(self):
        if self.intermediate_device is None:
            self.intermediate_device = self.fs_module.DEVICE
        using_multiprocess = (
            self.dataloader_config.dataloader_kwargs.get("dataloader_kwargs", 1) > 1
        )
        using_multiprocess = using_multiprocess or (
            self.dataloader_feature_config.dataloader_kwargs.get("dataloader_kwargs", 1)
            > 1
        )
        if using_multiprocess and "cuda" in (self.intermediate_device.DEVICE):
            warnings.warn(
                "Sharing CUDA tensor between sevral core, not tested. Either use intermediate_device=cpu or only one core"
            )

@dataclass
class TrainingConfig(ABC):
    training_methode : Any
    quantisation_config: QuantisationConfig
    optim_config: OptimizerConfig
    dataloader_config: Union[DataLoaderConfig, UnshuffleDataloaderConfig]
    final_epoch: int
    module_preparer: Prepare = DefaultPreparer()
    current_epoch: int = 0
    callbacks: list = field(default_factory=list)
    validation_config: Union[EvaluationConfig, FewShotEvaluationConfig] = None


@dataclass
class ClassicalTraining:
    training_methode: MethodeConfig
    quantisation_config: QuantisationConfig
    optim_config: OptimizerConfig
    dataloader_config: Union[DataLoaderConfig, UnshuffleDataloaderConfig]
    final_epoch: int
    module_preparer: Prepare = DefaultPreparer()
    current_epoch: int = 0
    callbacks: list = field(default_factory=list)
    validation_config: Union[EvaluationConfig, FewShotEvaluationConfig] = None


@dataclass
class MultiStepTraining:
    training_methode: list[MethodeConfig]
    quantisation_config: QuantisationConfig
    optim_config: OptimizerConfig
    dataloader_config: Union[DataLoaderConfig, UnshuffleDataloaderConfig]
    final_epoch: int
    module_preparer: Prepare = DefaultPreparer()
    current_epoch: int = 0
    callbacks: list = field(default_factory=list)
    validation_config: Union[EvaluationConfig, FewShotEvaluationConfig] = None
