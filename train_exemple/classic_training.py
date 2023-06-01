import os
from datetime import datetime
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD, lr_scheduler
from torchmetrics.classification import MulticlassAccuracy

from train import launch_training
from models.meta_network import (
    ForwardModule,
    SplitConfigurationBuilder,
    handle_classification_input,
)
from models.heads_loss import Projection, WrappedLossFunction
from models.vgg import VGG_MODELS


def launch(path_output):

    # ---------- CONSTANTS -----------------
    BATCH_SIZE = 256
    NUM_CLASSES = 10
    EPOCHS = 1000


    # ---------- MODEL DEFINITION -----------------

    backbone = VGG_MODELS["vgg11"]
    head = Projection(512, NUM_CLASSES)
    metric = MulticlassAccuracy(NUM_CLASSES)
    loss = WrappedLossFunction(nn.CrossEntropyLoss())

    train_split = (
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
        .connect_loss(loss, ["pred", "targets"], metric_name = "cross entropy")
    )

    test_split = SplitConfigurationBuilder.copy_config(train_split)

    train_module = ForwardModule(train_split)
    validation_module = ForwardModule(test_split)


    # ---------- DATA DEFINITION -----------------

    train_dataset = datasets.CIFAR10(
        root="./datasets",
        train=True,
        download=False,
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
        train_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True
    )

    validation_dataset = datasets.CIFAR10(
        root="./datasets",
        train=False,
        download=False,
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

    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True
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
