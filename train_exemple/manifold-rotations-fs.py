import os
from datetime import datetime
from torch import nn
from torchvision import datasets, transforms
from learn2learn.vision.datasets.cifarfs import CIFARFS
import learn2learn as l2l
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD, lr_scheduler
from torchmetrics.classification import MulticlassAccuracy
from models.backbone import SequentialManifoldMix


from models.heads_loss import WrapperLossMixup
from train import launch_training
from models.pretext_task import PretextRotation
from models.meta_network import (
    ForwardModule,
    SplitConfigurationBuilder,
    handle_classification_input,
)
from models.heads_loss import Projection, WrappedLossFunction
from models.vgg import VGG_MODELS


def launch(path_output):

    # ---------- CONSTANTS -----------------
    BATCH_SIZE = 128
    NUM_CLASSES = 10
    EPOCHS = 200


    # ---------- MODEL DEFINITION -----------------

    rotation_pretext = PretextRotation()
    sequential_backbone = VGG_MODELS["vgg11"]
    backbone = SequentialManifoldMix(sequential_backbone)
    

    head = Projection(512, NUM_CLASSES)
    head_pred_rotation = Projection(512, 4)
    metric = MulticlassAccuracy(NUM_CLASSES)
    loss = WrapperLossMixup(nn.CrossEntropyLoss(reduction = "none"))
    loss_rotation = WrappedLossFunction(nn.CrossEntropyLoss())

    train_split = (
        SplitConfigurationBuilder()
        .connect_node(handle_classification_input, ["input"], ["images", "targets"])
        .connect_node(rotation_pretext, ["images"], ["rotated_images", "target_rotation"])
        .connect_node(
            backbone,
            ["rotated_images", "targets"],
            ["feature", "modified_target"],
        )
        .connect_node(
            head_pred_rotation,
            ["feature"],
            ["pred_rotation"]
        )
        .connect_node(
            head,
            ["feature"],
            ["pred_class"],
        )
        .connect_loss(loss, ["pred_class", "modified_target"], metric_name = "loss classification")
        .connect_loss(loss_rotation, ["pred_rotation", "target_rotation"], metric_name = "loss_rotation")
    )

    validation_config = (
        SplitConfigurationBuilder()
        .connect_node(handle_classification_input, ["input"], ["images", "targets"])
        .connect_node(
            backbone,
            ["images"],
            ["feature"],
    
        )
        .connect_node(
            head,
            ["feature"],
            ["pred"],
        )
        .connect_metric(metric, ["pred", "targets"])
    )

    
    train_module = ForwardModule(train_split)
    validation_module = ForwardModule(validation_config)


    # ---------- DATA DEFINITION -----------------
    train_dataset = CIFARFS(
        root="./datasets",
        mode="train",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
                ),
            ]
        ),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=1, persistent_workers=True
    )

    validation_dataset = CIFARFS(
        root="./datasets",
        mode="validation",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
                ),
            ]
        ),
    )
    
    BATCH_SIZE_FS = 32
    
    dataset = l2l.data.MetaDataset(dataset)
    transforms = [
        l2l.data.transforms.NWays(dataset, n=5),
        l2l.data.transforms.KShots(dataset, k=1),
        l2l.data.transforms.LoadData(dataset),
        ]
        
    taskset = l2l.data.TaskDataset(dataset, transforms, num_tasks=1)

    validation_loader = DataLoader(
        taskset, batch_size=BATCH_SIZE_FS, num_workers=1, persistent_workers=True
    )

    # ------------ TRAINING DEFINITION ---------------------
    optimizer = SGD(
        train_module.parameters(),
        lr=0.1,
    )

    scheduler = lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.3)

    train_config = dict(
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=0,
        epochs=EPOCHS,
        name_split="train",
    )
    
    launch_training(
        train_config,
        train_module,
        train_loader,
        path_output,
        validation_module=validation_module,
        validation_dataloader=validation_loader,
    )