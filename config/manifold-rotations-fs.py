from torch import nn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, StepLR
from torchmetrics.classification import MulticlassAccuracy
from copy import copy

from MMML.data import get_normalized_cifarfs
from MMML.meta_network import ForwardModule, SplitConfigurationBuilder, handle_classification_input
from MMML.modules.pretext_task import PretextRotation
from MMML.modules.backbone import get_resnet12_easy, SequentialManifoldMix
from MMML.modules.metrics import Projection, WrappedLossFunction, WrapperLossMixup
from MMML.train import launch_training
from MMML.modules.few_shot import NCM, get_number_class_few_shot_dataset, Transforml2lTask
from MMML.callbacks import SaveFewShotValidationMetricCallback, WriteLogsCallback, FewShotValidationConfig
from MMML.utils import TrainingConfig

def launch(path_output, arg_parser):
    # ---------- CONSTANTS -----------------
    DATALOADER_KWARGS = dict(
        shuffle = True,
        batch_size = 64,
        num_workers = 3,
        persistent_workers = True
    )
    DATALOADERFS_KWARGS = dict(
        shuffle = False,
        batch_size = 64,
        num_workers = 3,
        persistent_workers = True
    )
    EPOCHS = 1000
    # ---------- DATA DEFINITION -----------------

    train_dataset = get_normalized_cifarfs(root="./datasets", mode="train", transform = transforms.RandomHorizontalFlip())
    validation_dataset = get_normalized_cifarfs(root="./datasets", mode="validation", transform = transforms.RandomHorizontalFlip())

    #train_loader = DataLoader(train_dataset, shuffle = True, **DATALOADER_ARGS)
    #validation_loader = DataLoader(validation_dataset, shuffle=False, **DATALOADER_ARGS)
    
    save_log_callback = WriteLogsCallback(add_lr = True)
    num_class_train = get_number_class_few_shot_dataset(train_dataset)
    
    
    # ---------- MODEL DEFINITION -----------------
    feature_maps_backbone = 320

    rotation_pretext = PretextRotation()
    backbone = SequentialManifoldMix(get_resnet12_easy("tiny", dropout = 0))

    head = Projection(feature_maps_backbone, num_class_train)

    head_pred_rotation = Projection(feature_maps_backbone, 4)
    
    loss = WrapperLossMixup(nn.CrossEntropyLoss(reduction="none"))
    loss_rotation = WrappedLossFunction(nn.CrossEntropyLoss())

    # --------- Train definition -------------------
    train_split = (
        SplitConfigurationBuilder()
        .connect_node(handle_classification_input, ["input"], ["images", "targets"])
        .connect_node(rotation_pretext, ["images"], ["rotated_images", "target_rotation"])
        .connect_node(backbone, ["rotated_images", "targets"], ["feature", "modified_target"])
        .connect_node(head_pred_rotation, ["feature"], ["pred_rotation"])
        .connect_node(head, ["feature"], ["pred_class"])
        .connect_loss(loss, ["pred_class", "modified_target"], metric_name="loss classification")
        .connect_loss(loss_rotation, ["pred_rotation", "target_rotation"], metric_name="loss_rotation")
    )

    train_module = ForwardModule(train_split)

    # --------- Validation Definition -------------------

    n_ways, n_shot, n_queries = 5, 1, 20
    few_shot_module = NCM()
    accuracyfs = MulticlassAccuracy(n_ways)
    l2l_input_manager = Transforml2lTask(n_ways, n_shot, n_queries)
    fs_config = (
        SplitConfigurationBuilder()
        .connect_node(l2l_input_manager, ["input"], ["shots-features", "queries-features", "targets"])
        .connect_node(few_shot_module, ["shots-features","queries-features"], ["prediction"])
        .connect_metric(accuracyfs, ["prediction", "targets"])
    )
    fs_module = ForwardModule(fs_config)
    few_shot_config = FewShotValidationConfig(n_ways, n_shot, n_queries, backbone, fs_module, validation_dataset, DATALOADER_KWARGS, DATALOADERFS_KWARGS)
    few_shot_validation = SaveFewShotValidationMetricCallback(fs_config=few_shot_config)

    # ------------ OPTIMIZATION TRAINING DEFINITION -----------
    warmup_epochs = 5
    optimizer_warmup = SGD(train_module.parameters(), lr=0.1)
    scheduler_warmup = LinearLR(optimizer_warmup, start_factor = 1/(warmup_epochs+1), end_factor = 1, total_iters = warmup_epochs)

    optimizer = SGD(train_module.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.3)
    
    # ---------------- Config *  ------------------
    
    training_config = TrainingConfig(
        training_methode = "classical_training",
        training_module = train_module,
        dataset = train_dataset,
        dataloader_kwargs=DATALOADER_KWARGS,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        callbacks = [few_shot_validation, save_log_callback]
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
