import torch
import os
from typing import Union
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# from MMML.utils import transfer_to, TrainingConfig, SWAMethodeConfig, update_bn
from MMML.callbacks import SaveStateCallback, write_logs
from MMML.train.configs import *
from MMML.utils import transfer_to, update_bn
from MMML.modules.few_shot import FeatureDataset

torch.backends.cudnn.benchmark = (
    True  # should accelerate training, delete if it does not work
)


def train_one_epoch_classic(
    dataloader,
    training_config: MethodeConfig,
    optimizer_config: OptimizerConfig,
    quantisation_config: QuantisationConfig,
):
    # training_methode = config.training_methode
    # optimizer_config = config.optim_config
    # quantisation_config = config.quantisation_config

    optimizer = optimizer_config.optimizer
    scheduler = optimizer_config.scheduler
    scaler = quantisation_config.scaler
    training_module = training_config.module

    training_module.train()
    training_module.to(training_config.DEVICE)  # data can be moved by callbacks

    
    if hasattr(training_config, "swa_model"): 
        training_config.swa_model.to(training_config.DEVICE)

    for input_data in tqdm(dataloader):
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=quantisation_config.dtype_amd,
            enabled=quantisation_config.enable_amd,
        ):
            input_data = transfer_to(input_data, training_config.DEVICE)
            loss = training_module(input_data)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
       
        if hasattr(training_config, "update_swa"):
            print("updating swa")
            training_config.update_swa(training_module)
        scaler.update()

    if scheduler is not None:
        scheduler.step()


def train_one_epoch_multistep(
    dataloader,
    list_training_config: list[MethodeConfig],
    optimizer_config: OptimizerConfig,
    quantisation_config: QuantisationConfig,
):
    # training_methode = config.training_methode
    # optimizer_config = config.optim_config
    # quantisation_config = config.quantisation_config

    optimizer = optimizer_config.optimizer
    scheduler = optimizer_config.scheduler
    scaler = quantisation_config.scaler

    for training_config in list_training_config:
        training_module = training_config.module

        training_module.train()
        training_module.to(training_config.DEVICE)  # data can be moved by callbacks
        
        if hasattr(training_config, "swa_model"):
            training_config.swa_model.to(training_config.DEVICE)
           
    for input_data in tqdm(dataloader):
        for training_config in list_training_config:
            training_module = training_config.module
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=quantisation_config.dtype_amd,
                enabled=quantisation_config.enable_amd,
            ):
                input_data = transfer_to(input_data, training_config.DEVICE)
                loss = training_module(input_data)

            scaler.scale(loss).backward()
            if hasattr(training_config, "update_swa"):
               training_config.update_swa(training_module)
            scaler.step(optimizer)
            scaler.update()

    if scheduler is not None:
        scheduler.step()

# TODO : use polymorphism ?
def train_one_epoch(training_config):
    if type(training_config) is ClassicalTraining:
        train_one_epoch_classic(
            training_config.dataloader_config.dataloader,
            training_config.training_methode,
            training_config.optim_config,
            training_config.quantisation_config,
        )
    elif type(training_config) is MultiStepTraining:
        train_one_epoch_multistep(
            training_config.dataloader_config.dataloader,
            training_config.training_methode,
            training_config.optim_config,
            training_config.quantisation_config,
        )


def compute_validation_classic(validation_config: EvaluationConfig):
    dataloader_validation = validation_config.dataloader_config.dataloader
    validation_module = validation_config.methode_config.get_evaluation_module()
    device = validation_config.methode_config.DEVICE

    validation_module.to(device)
    validation_module.eval()

    for input_data in tqdm(dataloader_validation):
        # ----- INPUT -------
        input_data = transfer_to(input_data, device)
        # ----- FORWARD  ------
        validation_module(input_data)

    dict_logs = validation_module.accumulate_and_get_logs()
    return dict_logs


def compute_validation_fs(fs_config: FewShotEvaluationConfig):
    feature_dataset = FeatureDataset(
        fs_config.dataloader_feature_config.number_class_novel
    )
    backbone = fs_config.backbone_methode_config.get_evaluation_module()
    device_backbone = fs_config.backbone_methode_config.DEVICE
    fs_module = fs_config.fs_module.module
    device_fs = fs_config.fs_module.DEVICE
    dataloader = fs_config.dataloader_config.dataloader
    intermediate_device = fs_config.intermediate_device
    backbone.eval()
    backbone.to(device_backbone)

    fs_module.to(device_fs)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="few-shot"):
            batch_image, targets = batch
            batch_image = batch_image.to(device_backbone)
            features = backbone(batch_image)
            features = avg_pool2d(
                features, kernel_size=(features.shape[-2], features.shape[-1])
            )
            features = torch.flatten(features, start_dim=1, end_dim=-1)
            feature_dataset.add_features(
                features.to(intermediate_device), targets.to(intermediate_device)
            )

        # modify the element of the dataset without modifying index
        meta_dataset = MetaDataset(
            feature_dataset,
            labels_to_indices=fs_config.dataloader_feature_config.labels_to_indices,
            indices_to_labels=fs_config.dataloader_feature_config.indices_to_labels,
        )
        fs_config.dataloader_feature_config.update_dataset(meta_dataset)
        taskset = fs_config.dataloader_feature_config.taskset
        # only need to update the LoadData dataset 
        # taskset = TaskDataset(
        #     meta_dataset,
        #     fs_config.dataloader_feature_config.dataset_to_transform(meta_dataset),
        #     num_tasks=fs_config.dataloader_feature_config.num_tasks,
        # )
        
        dataloader = DataLoader(
            taskset, **fs_config.dataloader_feature_config.dataloader_kwargs
        )
        
        for few_shot_tasks in tqdm(dataloader, desc="few-shot-classification"):
            transfer_to(few_shot_tasks, device_fs)
            fs_module(few_shot_tasks)
        # transfer_to(fs_module, "cpu")

    dict_logs = fs_module.accumulate_and_get_logs()
    return dict_logs


def launch_evaluation(
    writer,
    epoch,
    preparer: Prepare,
    validation_config: Union[EvaluationConfig, FewShotEvaluationConfig],
):
    if (epoch+1) % validation_config.evaluation_each_n_epoch != 0:
        return None
    # TODO : move the preparer to the validation config and use polymorphism 
    preparer.prepare_model()
    if type(validation_config) is EvaluationConfig:
        dict_loss = compute_validation_classic(validation_config)
    elif type(validation_config) is FewShotEvaluationConfig:
        dict_loss = compute_validation_fs(validation_config)
    else:
        raise NotImplementedError(f"non valid type : {validation_config}")
    preparer.prepare_training_module()
    
    write_logs(dict_loss, writer, validation_config.dataloader_config.name_split, epoch)
    return dict_loss

def train(writer, training_config):
    for epoch in range(training_config.current_epoch, training_config.final_epoch):
        train_one_epoch(training_config)

        if training_config.validation_config is not None:
            dict_eval = launch_evaluation(
                writer,
                epoch,
                training_config.module_preparer,
                training_config.validation_config,
            )
        else:
            dict_eval = None
        for callback in training_config.callbacks:
            callback(epoch, writer, training_config, dict_eval)

def launch_training(training_configs: list, path_output=None):
    if path_output is not None:

        for training_config in training_configs:
            
            writer = get_writer(path_output, training_config.dataloader_config.name_split)
            train(writer, training_config)
            

def get_writer(path_output, name_split):
    if path_output is not None:
        path_writer = os.path.join(
            path_output, name_split
        )
        writer = SummaryWriter(log_dir=path_writer)
    else:
        writer = None
    return writer