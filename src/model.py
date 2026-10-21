"""
model.py
--------
Builds the CNN used for freshness classification (transfer learning).

Architecture (default: MobileNetV2)
    input image 3 x 224 x 224
        -> MobileNetV2 convolutional backbone, pretrained on ImageNet
           (extracts visual features: edges -> textures -> shapes -> "rotten spot")
        -> global average pooling -> 1280 numbers per image
        -> Dropout(0.3)            (randomly zeroes features during training to reduce over-fitting)
        -> Linear(1280 -> N classes)   <- NEW layer, trained on our dataset
        -> softmax (applied in predict / loss) -> probability for each class
"""

import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHS = ("mobilenet_v2", "efficientnet_b0")


def build_model(num_classes: int, arch: str = "mobilenet_v2", pretrained: bool = True) -> nn.Module:
    """Create the backbone with ImageNet weights and replace its final layer."""
    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {SUPPORTED_ARCHS}")

    in_features = model.classifier[-1].in_features     # 1280 for both architectures
    # The original ImageNet head predicts 1000 ImageNet classes; we replace it
    # with a new head that predicts OUR classes (e.g. 28 fresh/rotten classes).
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1: keep all pretrained convolution layers fixed; train only the new head."""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, num_blocks: int) -> None:
    """Phase 2 (fine-tuning): also train the last `num_blocks` blocks of the backbone."""
    blocks = list(model.features.children())
    for block in blocks[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True


def keep_frozen_batchnorm_in_eval_mode(model: nn.Module) -> None:
    """
    BatchNorm layers keep running averages of their inputs. For layers we are
    NOT training, we keep the ImageNet averages instead of updating them.
    Call this after model.train().
    """
    for module in model.features.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
