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
import os
from torch.optim.swa_utils import AveragedModel
import itertools
import warnings

from MMML.train.configs import (
    ClassicalTraining,
    SWAMethodeConfig,
    OptimizerConfig,
    QuantisationConfig,
    FewShotEvaluationConfig,
    DataLoaderConfig,
    UnshuffleDataloaderConfig,
    MethodeConfig,
    PrepareSWA,
    MultiStepTraining,
    PrepareLookahead,
    DefaultPreparer,
    FeatureDatasetConfig,
    get_multi_swa_methode_config
)


from MMML.callbacks import (
    WriteLogsCallback, WriteLogsMultistepCallback
)
from MMML.data import get_normalized_miniimagenet, get_normalized_cifarfs
from MMML.meta_network import (
    ForwardModule,
    SplitConfigurationBuilder,
    handle_classification_input,
)

from MMML.modules.few_shot import (
    get_number_class_few_shot_dataset,
    MeanRegister,
    FeatureNormalizer,
    InductiveNCM,
    Transforml2lTask,
)
from MMML.modules.pretext_task import PretextRotation
from MMML.modules.metrics import Projection, WrappedLossFunction, MixCrossEntropyLoss
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
from MMML.train import launch_training, Lookahead


from MMML.train.schedulers import SGDR
from MMML.train.methodes import (
    get_manifold_mixup_baseline,
    get_mixup_baseline,
    get_ss_rotation_methode,
)

from MMML.utils import save_summary


def get_dataloaders(args, input_size):
    transform_pil = None
    if args.data_augmentation == "simple":
        if args.use_rotation_pretext:
            transform_train = transforms.RandomCrop(input_size, padding=4)
        else:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(input_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
    elif args.data_augmentation == "trivial_aug":
        # aug from https://github.com/vgripon/SOTA-routine
        if args.use_rotation_pretext:
            transform_pil = transforms.Compose(
                [
                    transforms.RandomCrop(input_size, padding=4),
                    transforms.TrivialAugmentWide(),
                ]
            )
        else:
            transform_pil = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(input_size, padding=4),
                    transforms.TrivialAugmentWide(),
                ]
            )

        transform_train = transforms.Compose(
            [
                transforms.Resize(input_size, antialias=True),
                transforms.RandomErasing(0.1),
            ]
        )
    if args.dataset == "miniimagenet":
        train_dataset = get_normalized_miniimagenet(
            root="./datasets",
            mode="train",
            transform_pil=transform_pil,
            transform=transform_train,
            download=True,
        )
        validation_dataset = get_normalized_miniimagenet(
            root="./datasets",
            mode="validation",
            download=True,
        )
        test_dataset = get_normalized_miniimagenet(
            root="./datasets",
            mode="test",
            download=True,
        )
    elif args.dataset == "cifar-fs":
        train_dataset = get_normalized_cifarfs(
            root="./datasets",
            mode="train",
            transform_pil=transform_pil,
            transform=transform_train,
            download=True,
        )
        validation_dataset = get_normalized_cifarfs(
            root="./datasets",
            mode="validation",
            download=True,
        )
        test_dataset = get_normalized_cifarfs(
            root="./datasets",
            mode="validation",
            download=True,
        )
    return train_dataset, validation_dataset, test_dataset

def get_backbone_resnet(args):
    if args.backbone == "resnet9":
        feature_maps_backbone = 320
        backbone = get_resnet9_easy(
            "classic",
            post_activate=args.post_activate,
        )
    elif args.backbone == "resnet12":
        feature_maps_backbone = 640
        backbone = get_resnet12_easy(
            "classic",
            post_activate=args.post_activate,
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
            ),
            BasicBlockRN(
                feature_maps,
                int(2.5 * feature_maps),
                stride=2,
                post_activate=args.post_activate,
            ),
            BasicBlockRN(
                int(2.5 * feature_maps),
                5 * feature_maps,
                stride=2,
                post_activate=args.post_activate,
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
        )
    return backbone, feature_maps_backbone

def define_network(args, backbone, feature_maps_backbone, NUM_CLASSES):
    head = Projection(feature_maps_backbone, NUM_CLASSES)

    train_split = SplitConfigurationBuilder().connect_node(
        handle_classification_input, ["input"], ["images", "targets"]
    )

    if args.mix_aug == "cutmix_mixup":
        mixup = transforms.RandomChoice(
            [
                RandomMixup(p=1.0, alpha=args.alpha_mixup),
                RandomCutmix(p=1.0, alpha=args.alpha_cutmix),
            ]
        )

        train_split = get_mixup_baseline(
            mixup, backbone, feature_maps_backbone, NUM_CLASSES, args.label_smoothing
        )

    elif args.mix_aug == "cutmix":
        # aug from https://github.com/vgripon/SOTA-routine
        mixup = RandomCutmix(p=1.0, alpha=args.alpha_mixup)
        train_split = get_mixup_baseline(
            mixup, backbone, feature_maps_backbone, NUM_CLASSES, args.label_smoothing
        )

    elif args.mix_aug == "mixup":
        # aug from https://github.com/vgripon/SOTA-routine
        mixup = RandomMixup(p=1.0, alpha=args.alpha_mixup)
        train_split = get_mixup_baseline(
            mixup, backbone, feature_maps_backbone, NUM_CLASSES, args.label_smoothing
        )

    elif args.mix_aug == "manifold-mixup":
        mixup = transforms.RandomChoice([RandomMixup(p=1.0, alpha=args.alpha_mixup)])

        train_split = get_manifold_mixup_baseline(
            mixup, backbone, feature_maps_backbone, NUM_CLASSES, args.label_smoothing
        )

    elif args.mix_aug == "manifold-mixup-cutmix":
        mixup = transforms.RandomChoice(
            [
                RandomMixup(p=1.0, alpha=args.alpha_mixup),
                RandomCutmix(p=1.0, alpha=args.alpha_cutmix),
            ]
        )
        train_split = get_manifold_mixup_baseline(
            mixup, backbone, feature_maps_backbone, NUM_CLASSES, args.label_smoothing
        )

    else:
        accuracy_train = MulticlassAccuracy(NUM_CLASSES)
        loss = WrappedLossFunction(
            nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        )
        train_split = (
            train_split.connect_node(backbone, ["images"], ["feature"])
            .connect_node(head, ["feature"], ["pred"])
            .connect_loss(loss, ["pred", "targets"], metric_name="loss-classification")
            .connect_metric(accuracy_train, ["pred", "targets"], metric_name="accuracy")
        )

    mean_register = MeanRegister()
    train_split.connect_node(mean_register, ["feature"], [])

    train_module = ForwardModule(train_split)

    train_parameters = train_module.parameters()

    # --------- Second training step -------------------
    if args.use_rotation_pretext:
        accuracy_train = MulticlassAccuracy(NUM_CLASSES)
        loss = WrappedLossFunction(
            nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        )
        loss_rotation = WrappedLossFunction(
            nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        )
        head_projection = Projection(feature_maps_backbone, 4)
        pretext_rotation = PretextRotation()
        second_train_split = (
            SplitConfigurationBuilder()
            .connect_node(handle_classification_input, ["input"], ["images", "targets"])
            .connect_node(
                pretext_rotation, ["images"], ["rotated_image", "target_rotation"]
            )
            .connect_node(backbone, ["rotated_image"], ["feature"])
            .connect_node(head, ["feature"], ["pred"])
            .connect_node(head_projection, ["feature"], ["pred_rotation"])
            .connect_loss(loss, ["pred", "targets"], metric_name="loss-classification")
            .connect_loss(
                loss_rotation,
                ["pred_rotation", "target_rotation"],
                metric_name="loss-rotation",
            )
            .connect_metric(accuracy_train, ["pred", "targets"], metric_name="accuracy")
        )

        mean_register = MeanRegister()
        second_train_split.connect_node(mean_register, ["feature"], [])

        second_train_module = ForwardModule(second_train_split)
        train_parameters = itertools.chain(train_parameters, loss_rotation.parameters())
    else:
        second_train_module = None

    return train_parameters, train_module, second_train_module, mean_register
 
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
    arg_parser.add_argument("--epochs", type=int, default=500)
    arg_parser.add_argument(
        "--mix_aug",
        nargs="?",
        choices=(
            "manifold-mixup",
            "manifold-mixup-cutmix",
            "mixup",
            "cutmix_mixup",
            "cutmix",
        ),
    )
    arg_parser.add_argument("--scheduler", type=str, default="step_lr")
    arg_parser.add_argument("--dtype_amd", type=str, default="float16")
    arg_parser.add_argument("--label_smoothing", default=0.1, type=float)
    arg_parser.add_argument("--lr", type=float, default=0.1)
    arg_parser.add_argument("--backbone", type=str, default="resnet12")
    arg_parser.add_argument("--use_strides", action="store_true")
    arg_parser.add_argument("--no_post_activate", action="store_true")
    arg_parser.add_argument("--data_augmentation", type=str, default="trivial_aug")
    arg_parser.add_argument("--alpha_mixup", type=float, default=0.2)
    arg_parser.add_argument("--alpha_cutmix", type=float, default=1.0)
    arg_parser.add_argument("--optimizer", type=str, default="SGD")
    arg_parser.add_argument("--use_swa", action="store_true")
    arg_parser.add_argument("--use_lookheahead", action="store_true")

    arg_parser.add_argument("--use_rotation_pretext", action="store_true")
    arg_parser.add_argument("--dataset", type=str, default="cifar-fs")

    args = arg_parser.parse_args(args=list_args)
    args.post_activate = not (args.no_post_activate)
    
    # ---------- CONSTANTS -----------------
    intermediate_device = "cpu"
    DATALOADER_KWARGS = dict(
        shuffle=True, batch_size=128, num_workers=10, persistent_workers=True
    )
    n_ways, n_shot, n_queries = 5, 5, 20
    DATALOADERFS_KWARGS = dict(
        shuffle=False, batch_size=128, num_workers=5, persistent_workers=True
    )
    DATALOADERFS_feature_KWARGS = dict(shuffle=False, batch_size=1024, num_workers=5)
    la_steps = 5
    save_each_n_epoch=10 # save the whole training state (with optimiser  )
    evaluation_each_n_epoch = 1


    # ---------- DATAS -----------------

    input_size = 84 if args.dataset == "miniimagenet" else 32

    train_dataset, validation_dataset, test_dataset = get_dataloaders(args, input_size)
    NUM_CLASSES = get_number_class_few_shot_dataset(train_dataset)

    train_dataloader_config = DataLoaderConfig(
        train_dataset, DATALOADER_KWARGS, name_split="train"
    )

    # use unshuffled dataset in order to build a feature dataset
    dataloader_validation_config = UnshuffleDataloaderConfig(
        validation_dataset,
        dataloader_kwargs=DATALOADERFS_KWARGS,
        name_split="validation",
    )

    dataloader_test_config = UnshuffleDataloaderConfig(
        test_dataset,
        dataloader_kwargs=DATALOADERFS_KWARGS,
        name_split="validation",
    )

    # ---------- MODEL DEFINITION -----------------
    backbone, feature_maps_backbone = get_backbone_resnet(args)

    save_summary(path_output=path_output, backbone=backbone)

    train_parameters, train_module, second_train_module, mean_register = (
        define_network(
            args, backbone, feature_maps_backbone, NUM_CLASSES
        )
    )

    if args.use_swa:
        if args.use_lookheahead:
            update_every_n = la_steps
        else:
            update_every_n = 1
        backbone_config, methode_config_train, methode_config_second_train = get_multi_swa_methode_config([backbone, train_module, second_train_module], update_every_n=update_every_n)
        
    else:
        methode_config_train = MethodeConfig(train_module)
        if second_train_module is not None:
            methode_config_second_train = MethodeConfig(second_train_module)

    if args.use_swa:
        # for swa : prepare bn of the backbone ()
        preparer = PrepareSWA(train_dataloader_config, methode_config_train)
    elif args.use_lookahead:
        # lookahead : replace module weights (fast weights) with slow ones
        preparer = PrepareLookahead(optimizer_config)
        
    else:
        preparer = DefaultPreparer()
    # ------------ TRAINING DEFINITION -----------
    optimizer_config = get_optimisation_config(args, train_parameters, la_steps)
    

    # --------- Validation Definition -------------------

    few_shot_module = InductiveNCM()
    accuracyfs = MulticlassAccuracy(n_ways)
    l2l_input_manager = Transforml2lTask(n_ways, n_shot, n_queries)
    normalizer = FeatureNormalizer(mean_register)

    fs_config = (
        SplitConfigurationBuilder()
        .connect_node(
            l2l_input_manager,
            ["input"],
            ["shots-features", "queries-features", "targets"],
        )
        .connect_node(normalizer, ["shots-features"], ["normalized_shots"])
        .connect_node(normalizer, ["queries-features"], ["normalized_queries"])
        .connect_node(
            few_shot_module, ["normalized_shots", "normalized_queries"], ["prediction"]
        )
        .connect_metric(accuracyfs, ["prediction", "targets"], metric_name ="fs-accuracy")
    )

    fs_module = ForwardModule(fs_config)

  
    fs_module_config = MethodeConfig(fs_module)

    # --------------- Quantization ----------------------

    dtype = getattr(torch, args.dtype_amd)  # torch, "float16" -> torch.float16
    quantisation_config = QuantisationConfig(True, dtype_amd=dtype)

    # ---------------- Config training  ------------------
    dataloader_feature_config = FeatureDatasetConfig(
        n_ways, n_shot, n_queries, validation_dataset, DATALOADERFS_feature_KWARGS
    )

    validation_config = FewShotEvaluationConfig(
        backbone_config,
        fs_module_config,
        dataloader_validation_config,
        dataloader_feature_config,
        intermediate_device=intermediate_device,
        evaluation_each_n_epoch=evaluation_each_n_epoch,
    )
    if second_train_module is None:
        save_log_callback = WriteLogsCallback()
    else:
        save_log_callback = WriteLogsMultistepCallback()
    
    callbacks = [save_log_callback]
    if path_output is not None:
        from MMML.callbacks import SaveIfImprovement, SaveStateCallback
        
        callbacks.append(
            SaveIfImprovement(path_output, backbone_config, "fs-accuracy", "best_backbone.pt")
        )


    kwargs_training = dict(
        quantisation_config=quantisation_config,
        optim_config=optimizer_config,
        dataloader_config=train_dataloader_config,
        callbacks=callbacks,
        final_epoch=args.epochs,
        module_preparer =preparer,
        validation_config=validation_config,
    )
    # swa & lookahead require additional steps before validation / test
    # if use_swa & lookahead : swa only use slow weights

    
    
    if second_train_module is not None:
        training_config = MultiStepTraining(
            [methode_config_train, methode_config_second_train],
            **kwargs_training
        )
    else:
        training_config = ClassicalTraining(
            methode_config_train, **kwargs_training
        )
    
    if path_output is not None:
        
        training_config.callbacks.append( 
            SaveStateCallback(
                path_output, training_config, save_each_n_epoch
            )
        )

    launch_training(
        [training_config],
        path_output
    )

    # ---------- TEST ----------
    if path_output is not None:
        with open(path_output +  "/" + "best_backbone.pt", "rb") as f:
            best_backbone = torch.load(f)
    else:
        warnings.warn("output path not set Computing test loss using the last backbone")
        best_backbone = backbone_config
    
    test_validation =  FewShotEvaluationConfig(
        best_backbone,
        fs_module_config,
        dataloader_test_config,
        dataloader_feature_config,
        intermediate_device=intermediate_device
    )
    from MMML.train.train import compute_validation_fs, get_writer

    dict_logs = compute_validation_fs(test_validation)
    writer = get_writer(path_output, "test")
    writer.add_hparams(
        vars(args), dict_logs
    )