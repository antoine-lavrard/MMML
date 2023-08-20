from MMML.modules.metrics import Projection, WrappedLossFunction, MixCrossEntropyLoss
from MMML.meta_network import (
    ForwardModule,
    SplitConfigurationBuilder,
    handle_classification_input,
)
from MMML.modules.backbone import SequentialManifoldMix
from MMML.modules.pretext_task import PretextRotation
from torch import nn


def get_mixup_baseline(
    mixup, backbone, fmap_backbone_output, num_classes, label_smoothing
):
    head = Projection(fmap_backbone_output, num_classes)
    loss = MixCrossEntropyLoss(label_smoothing=label_smoothing)
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
    return train_split


def get_manifold_mixup_baseline(
    mixup, backbone, fmap_backbone_output, num_classes, label_smoothing
):
    head = Projection(fmap_backbone_output, num_classes)
    loss = MixCrossEntropyLoss(label_smoothing=label_smoothing)
    manifold_backbone = SequentialManifoldMix(backbone, augmentation=mixup)
    loss = MixCrossEntropyLoss(label_smoothing=label_smoothing)
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

    return train_split


def get_ss_rotation_methode(
    backbone, fmap_backbone_output, num_classes, label_smoothing
):
    head = Projection(fmap_backbone_output, num_classes)
    rotation_pretext = PretextRotation()
    head_pred_rotation = Projection(fmap_backbone_output, num_classes)
    loss = nn.CrossEntropyLoss()
    loss_rotation = MixCrossEntropyLoss(label_smoothing=label_smoothing)

    train_split = (
        SplitConfigurationBuilder()
        .connect_node(handle_classification_input, ["input"], ["images", "targets"])
        .connect_node(
            rotation_pretext, ["images"], ["rotated_images", "target_rotation"]
        )
        .connect_node(backbone, ["images"], ["feature", "lmbd", "index"])
        .connect_node(head_pred_rotation, ["feature"], ["rotation_prediction"])
        .connect_node(head, ["feature"], ["pred"])
        .connect_loss(
            loss,
            ["pred", "targets", "lmbd", "index"],
            metric_name="loss-classification",
        )
        .connect_loss(
            loss_rotation,
            ["rotation_prediction", "target_rotation", "lmbd", "index"],
            metric_name="loss-rotation",
        )
    )

    return train_split
