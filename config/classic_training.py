from torch import nn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, StepLR
from torchmetrics.classification import MulticlassAccuracy
from copy import copy

from MMML.data import get_normalized_cifar10
from MMML.meta_network import ForwardModule, SplitConfigurationBuilder, handle_classification_input
from MMML.modules.backbone import get_resnet12_easy
from MMML.modules.metrics import Projection, WrappedLossFunction
from MMML.train import launch_training
from MMML.callbacks import SaveValidationMetricsCallback, WriteLogsCallback, ValidationConfig
from MMML.utils import TrainingConfig


def launch(path_output, arg_parser):
    # ---------- CONSTANTS -----------------
    # path_output = None # No saving
    DATALOADER_KWARGS = dict(
        shuffle = True,
        batch_size = 64,
        num_workers = 3
    )

    NUM_CLASSES = 10
    EPOCHS = 1000

    # ---------- MODEL DEFINITION -----------------
    
    backbone = get_resnet12_easy("tiny", dropout = 0)
    feature_maps_backbone = 320
    head = Projection(feature_maps_backbone, NUM_CLASSES)
    metric = MulticlassAccuracy(NUM_CLASSES)
    loss = WrappedLossFunction(nn.CrossEntropyLoss())

    train_split = (
        SplitConfigurationBuilder()
        .connect_node(handle_classification_input, ["input"], ["images", "targets"])
        .connect_node(backbone, ["images"], ["feature"])
        .connect_node(head, ["feature"], ["pred"])
        .connect_metric(metric, ["pred", "targets"])
        .connect_loss(loss, ["pred", "targets"], metric_name="cross entropy")
    )

    train_module = ForwardModule(train_split)
    validation_module = ForwardModule(train_split) # same config for train / test

    # ---------- DATA DEFINITION -----------------

    train_dataset = get_normalized_cifar10(root="./datasets",train=True, transform = transforms.RandomHorizontalFlip())
    validation_dataset = get_normalized_cifar10(root="./datasets", train=False, transform = transforms.RandomHorizontalFlip())

    #validation_loader = DataLoader(validation_dataset, **DATALOADER_ARGS)

    validation_config = ValidationConfig(
        validation_module = validation_module,
        dataset = validation_dataset,
        dataloader_kwargs= DATALOADER_KWARGS,
    )
    validation_callback = SaveValidationMetricsCallback(validation_config)
    save_log_callback = WriteLogsCallback(add_lr = True)

    # ------------ TRAINING DEFINITION -----------

    warmup_epochs = 5

    optimizer_warmup = SGD(train_module.parameters(), lr=0.1)
    scheduler_warmup = LinearLR(optimizer_warmup, start_factor = 1/(warmup_epochs+1), end_factor = 1, total_iters = warmup_epochs)

    optimizer = SGD(train_module.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.3)
    
    # ---------------- Config training  ------------------
    
    training_config = TrainingConfig(
        training_methode = "classical_training",
        training_module = train_module,
        dataset = train_dataset,
        dataloader_kwargs= DATALOADER_KWARGS,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        callbacks = [validation_callback, save_log_callback]
    )
    
    # shallow copy mandatory to keep weight sharing
    config_warmup = copy(training_config)
    config_warmup.optimizer = optimizer_warmup
    config_warmup.scheduler = scheduler_warmup


    train_configs = {
        "warmup" : config_warmup,
        "train" : training_config
    }
    
    launch_training(
        train_configs, path_output, save_each_n_epoch = 1
    )
    