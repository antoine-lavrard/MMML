"""
python main.py --file_name manifold --save_name resnet12_cutmixmixup_02_tals_manifold --save --command --backbone resnet12 --dtype_amd float16 --scheduler quick_eval --mix_aug manifold-mixup-cutmix --alpha_mixup 0.2 --data_augmentation trivial_aug --label_smoothing 0.1 --epochs 300

"""

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD
from torch.optim import lr_scheduler
from torchmetrics.classification import MulticlassAccuracy
from copy import copy
from argparse import ArgumentParser
from torchinfo import summary
import os
from torch.optim.swa_utils import AveragedModel


from MMML.data import get_normalized_cifar10, get_normalized_cifar100
from MMML.meta_network import (
    ForwardModule,
    SplitConfigurationBuilder,
    handle_classification_input,
)
from MMML.modules.pretext_task import PretextRotation
from MMML.modules.backbone import (
    get_resnet12_easy,
    get_resnet9_easy,
    SequentialManifoldMix,
    BasicBlockRN,
)
from MMML.modules.backbone.manifold_mixup import RandomMixup, RandomCutmix
from MMML.modules.backbone.resnet import (
    get_resnet_from_stage_number,
    get_resnet_doubled_from_stage_number,
)
from MMML.modules.metrics import Projection, WrappedLossFunction, MixCrossEntropyLoss
from MMML.train import launch_training, Lookahead
from MMML.train.schedulers import SGDR
from MMML.callbacks import (
    WriteLogsCallback,
)
from MMML.train.configs import (
    ClassicalTraining,
    SWAMethodeConfig,
    OptimizerConfig,
    QuantisationConfig,
    EvaluationConfig,
    DataLoaderConfig,
    MethodeConfig,
    PrepareLookahead,
    PrepareSWA,
)

from MMML.utils import save_summary


def get_cifar_dataloader(data_augmentation, dataset, dataloader_kwargs):
    transform_pil = None
    if data_augmentation == "simple":
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        )
    elif data_augmentation == "trivial_aug":
        # aug from https://github.com/vgripon/SOTA-routine
        transform_pil = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.TrivialAugmentWide(),
            ]
        )

        transform_train = transforms.Compose(
            [transforms.Resize(32, antialias=True), transforms.RandomErasing(0.1)]
        )

    if dataset == "cifar10":
        NUM_CLASSES = 10
        train_dataset = get_normalized_cifar10(
            root="./datasets",
            train=True,
            transform_pil=transform_pil,
            transform=transform_train,
            download=False,
        )
        validation_dataset = get_normalized_cifar10(
            root="./datasets", train=False, download=False
        )
    elif dataset == "cifar100":
        NUM_CLASSES = 100
        train_dataset = get_normalized_cifar100(
            root="./datasets",
            train=True,
            transform_pil=transform_pil,
            transform=transform_train,
            download=True,
        )
        validation_dataset = get_normalized_cifar100(
            root="./datasets", train=False, download=True
        )

    validation_dataloader_config = DataLoaderConfig(
        validation_dataset, dataloader_kwargs=dataloader_kwargs, name_split="validation"
    )
    train_dataloader_config = DataLoaderConfig(
        train_dataset, dataloader_kwargs, name_split="train"
    )

    return train_dataloader_config, validation_dataloader_config, NUM_CLASSES


def get_backbone_resnet(args):
    if args.backbone == "resnet9":
        feature_maps_backbone = 320
        backbone = get_resnet9_easy(
            "classic",
            post_activate=args.post_activate,
            stochastic_depth=args.stochastic_depth,
        )
    elif args.backbone == "resnet12":
        feature_maps_backbone = 640
        backbone = get_resnet12_easy(
            "classic",
            post_activate=args.post_activate,
            stochastic_depth=args.stochastic_depth,
        )
    elif args.backbone == "resnet9-strided":
        feature_maps_backbone = 320
        feature_maps = 64
        backbone = nn.Sequential(
            BasicBlockRN(
                3,
                feature_maps,
                stride=2,
                post_activate=args.post_activate,
                stochastic_depth=args.stochastic_depth,
            ),
            BasicBlockRN(
                feature_maps,
                int(2.5 * feature_maps),
                stride=2,
                post_activate=args.post_activate,
                stochastic_depth=args.stochastic_depth,
            ),
            BasicBlockRN(
                int(2.5 * feature_maps),
                5 * feature_maps,
                stride=2,
                post_activate=args.post_activate,
                stochastic_depth=args.stochastic_depth,
            ),
        )
    elif "resnet-easy-" in args.backbone:
        numbers = args.backbone.split("-")
        print(numbers)
        numbers = [int(number) for number in numbers[2:]]
        feature_maps_backbone = 320 if len(numbers) == 3 else 640

        backbone = get_resnet_from_stage_number(
            numbers,
            64,
            number_layer=3,
            post_activate=args.post_activate,
            stochastic_depth=args.stochastic_depth,
        )
    elif "resnet-doubled-" in args.backbone:
        numbers = args.backbone.split("-")
        print(numbers)
        numbers = [int(number) for number in numbers[2:]]
        feature_maps_backbone = 320 if len(numbers) == 3 else 640
        backbone = get_resnet_doubled_from_stage_number(
            numbers,
            64,
            post_activate=args.post_activate,
            stochastic_depth=args.stochastic_depth,
        )
    elif "resnet-" in args.backbone:
        numbers = args.backbone.split("-")
        print(numbers)
        numbers = [int(number) for number in numbers[1:]]
        feature_maps_backbone = 320 if len(numbers) == 3 else 640

        backbone = get_resnet_from_stage_number(
            numbers,
            64,
            post_activate=args.post_activate,
            stochastic_depth=args.stochastic_depth,
        )
    return backbone, feature_maps_backbone


def define_network(args, backbone, feature_maps_backbone, num_classes):
    head = Projection(feature_maps_backbone, num_classes)

    if args.mix_aug == "cutmix_mixup":
        mixup = transforms.RandomChoice(
            [
                RandomMixup(p=1.0, alpha=args.alpha_mixup),
                RandomCutmix(p=1.0, alpha=args.alpha_cutmix),
            ]
        )
        loss = MixCrossEntropyLoss(label_smoothing=args.label_smoothing)

        train_split = (
            SplitConfigurationBuilder()
            .connect_node(handle_classification_input, ["input"], ["images", "targets"])
            .connect_node(mixup, ["images"], ["mixed_image", "lmbd", "index"])
            .connect_node(backbone, ["mixed_image"], ["feature"])
            .connect_node(head, ["feature"], ["pred"])
            .connect_loss(
                loss,
                ["pred", "targets", "lmbd", "index"],
                metric_name="loss-classification",
            )
        )
    elif args.mix_aug == "cutmix":
        # aug from https://github.com/vgripon/SOTA-routine
        mixup = RandomCutmix(p=1.0, alpha=args.alpha_mixup)


        loss = MixCrossEntropyLoss(label_smoothing=args.label_smoothing)

        train_split = (
            SplitConfigurationBuilder()
            .connect_node(handle_classification_input, ["input"], ["images", "targets"])
            .connect_node(mixup, ["images"], ["mixed_image", "lmbd", "index"])
            .connect_node(backbone, ["mixed_image"], ["feature"])
            .connect_node(head, ["feature"], ["pred"])
            .connect_loss(
                loss,
                ["pred", "targets", "lmbd", "index"],
                metric_name="loss-classification",
            )
        )
    elif args.mix_aug == "mixup":
        # aug from https://github.com/vgripon/SOTA-routine
        mixup = RandomMixup(p=1.0, alpha=args.alpha_mixup)

        loss = MixCrossEntropyLoss(
            label_smoothing=args.label_smoothing, type_loss=args.type_loss
        )

        train_split = (
            SplitConfigurationBuilder()
            .connect_node(handle_classification_input, ["input"], ["images", "targets"])
            .connect_node(mixup, ["images"], ["mixed_image", "lmbd", "index"])
            .connect_node(backbone, ["mixed_image"], ["feature"])
            .connect_node(head, ["feature"], ["pred"])
            .connect_loss(
                loss,
                ["pred", "targets", "lmbd", "index"],
                metric_name="loss-classification",
            )
        )
    elif args.mix_aug == "manifold-mixup-cutmix":
        mixup = transforms.RandomChoice(
            [
                RandomMixup(p=1.0, alpha=args.alpha_mixup),
                RandomCutmix(p=1.0, alpha=args.alpha_cutmix),
            ]
        )
        manifold_backbone = SequentialManifoldMix(backbone, augmentation=mixup)
        loss = MixCrossEntropyLoss(label_smoothing=args.label_smoothing)
        train_split = (
            SplitConfigurationBuilder()
            .connect_node(handle_classification_input, ["input"], ["images", "targets"])
            .connect_node(manifold_backbone, ["images"], ["feature", "lmbd", "index"])
            .connect_node(head, ["feature"], ["pred"])
            .connect_loss(
                loss,
                ["pred", "targets", "lmbd", "index"],
                metric_name="loss-classification",
            )
        )
    else:
        accuracy_train = MulticlassAccuracy(num_classes)
        loss = WrappedLossFunction(
            nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        )
        train_split = (
            SplitConfigurationBuilder()
            .connect_node(handle_classification_input, ["input"], ["images", "targets"])
            .connect_node(backbone, ["images"], ["feature"])
            .connect_node(head, ["feature"], ["pred"])
            .connect_loss(loss, ["pred", "targets"], metric_name="loss-classification")
            .connect_metric(accuracy_train, ["pred", "targets"], metric_name="accuracy")
        )

    train_module = ForwardModule(train_split)

    # ---------- MODEL DEFINITION Validation-----------------
    loss_validation = WrappedLossFunction(nn.CrossEntropyLoss())
    accuracy_validation = MulticlassAccuracy(num_classes)
    
    # TODO : add the averaged network in case of swa
    validation_network = (
        SplitConfigurationBuilder()
        .connect_node(handle_classification_input, ["input"], ["images", "targets"])
        .connect_node(backbone, ["images"], ["feature"])
        .connect_node(head, ["feature"], ["pred"])
        .connect_metric(accuracy_validation, ["pred", "targets"], "test accuracy")
        .connect_metric(loss_validation, ["pred", "targets"], "test validation")
    )

    validation_module = ForwardModule(validation_network)

    return train_module, validation_module


def get_optimisation_config(args, parameters, la_steps):
    if args.optimizer == "SGD":
        optimizer = SGD(
            parameters,
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )

    if args.use_lookheahead:
        optimizer = Lookahead(optimizer, la_steps=la_steps)

    if args.scheduler == "step_lr":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[(i + 1) * 100 for i in range(args.epochs // 100)],
            gamma=0.1,
        )
    elif args.scheduler == "SGDR":
        scheduler = SGDR(optimizer, T0=100, cycle_decay=0.9, number_restart=10)

    elif args.scheduler == "quick_eval":
        # original resnet evaluation (without validation split)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[81, 122], gamma=0.1)
    optimizer_config = OptimizerConfig(optimizer, scheduler)
    return optimizer_config


def launch(path_output, list_args: list[str]):
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--epochs", type=int, default=200)
    arg_parser.add_argument(
        "--mix_aug",
        nargs="?",
        choices=("manifold-mixup-cutmix", "mixup", "cutmix_mixup", "cutmix"),
    )
    arg_parser.add_argument("--stochastic_depth", type=float, default=0.0)  # 0.2
    arg_parser.add_argument("--scheduler", type=str, default="step_lr")
    arg_parser.add_argument("--dtype_amd", type=str, default="float16")
    arg_parser.add_argument("--label_smoothing", default=0.0, type=float)  # 0.1
    arg_parser.add_argument("--lr", type=float, default=0.1)
    arg_parser.add_argument("--backbone", type=str, default="resnet9")
    arg_parser.add_argument("--use_strides", action="store_true")
    arg_parser.add_argument("--type_loss", type=str, default="classic")
    arg_parser.add_argument("--no_post_activate", action="store_true")
    arg_parser.add_argument("--dataset", type=str, default="cifar10")
    arg_parser.add_argument("--data_augmentation", type=str, default="simple")
    arg_parser.add_argument("--alpha_mixup", type=float, default=2.0)
    arg_parser.add_argument("--alpha_cutmix", type=float, default=1.0)
    arg_parser.add_argument("--optimizer", type=str, default="SGD")
    arg_parser.add_argument("--use_swa", action="store_true")
    arg_parser.add_argument("--use_lookheahead", action="store_true")

    args = arg_parser.parse_args(args=list_args)
    args.post_activate = not (args.no_post_activate)

    # ---------- CONSTANTS -----------------

    DATALOADER_KWARGS = dict(
        shuffle=True, batch_size=128, num_workers=10, persistent_workers=True
    )

    la_steps = 5  # lookheahead_steps

    # ---------- Dataloader -----------------

    (
        train_dataloader_config,
        validation_dataloader_config,
        NUM_CLASSES,
    ) = get_cifar_dataloader(args.data_augmentation, args.dataset, DATALOADER_KWARGS)

    # ---------- Backbone Architecture -----------------
    backbone, feature_maps_backbone = get_backbone_resnet(args)

    save_summary(path_output, backbone)

    train_module, validation_module = define_network(
        args, backbone, feature_maps_backbone, NUM_CLASSES
    )

    # ------------ OPTIMIZER  -----------

    optimizer_config = get_optimisation_config(
        args, train_module.parameters(), la_steps
    )

    # --------------- Quantization ----------------------

    dtype = getattr(torch, args.dtype_amd)  # torch, "float16" -> torch.float16
    quantisation_config = QuantisationConfig(True, dtype_amd=dtype)

    # ---------------- Config training  ------------------

    save_log_callback = WriteLogsCallback()

    if args.use_swa:
        if args.use_lookheahead:
            # update swa only on slow weight
            # this way, only slow weights will be taken into acount
            methode_config = SWAMethodeConfig(train_module, update_every_n=la_steps)
        else:
            methode_config = SWAMethodeConfig(train_module)
    else:
        methode_config = MethodeConfig(train_module)

    validation_module_config = MethodeConfig(validation_module)

    validation_config = EvaluationConfig(
        validation_module_config, validation_dataloader_config
    )

    kwargs_training = dict(
        quantisation_config=quantisation_config,
        optim_config=optimizer_config,
        dataloader_config=train_dataloader_config,
        callbacks=[save_log_callback],
        final_epoch=args.epochs,
        validation_config=validation_config,
    )
    # if use_swa & lookahead : swa only use slow weights
    if args.use_swa:
        preparer = PrepareSWA(train_dataloader_config, methode_config)
        kwargs_training["preparer"]=preparer
        
    elif args.use_lookahead:
        preparer = PrepareLookahead(optimizer_config)
        kwargs_training["preparer"]=preparer

    training_config = ClassicalTraining(
            training_methode=methode_config, **kwargs_training
        )

    launch_training(
        [training_config],
        path_output
    )
