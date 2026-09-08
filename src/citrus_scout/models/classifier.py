"""Model construction.

Backbones come from `timm`, pre-trained on ImageNet. Training from scratch on 5,000
images would badly overfit; the pre-trained features already encode edges, texture
and colour gradients, which is most of what separates a necrotic lesion from a
healthy lamina.

The default is EfficientNet-B0. It is small enough to train on a 4 GB laptop GPU at
a useful batch size, and on this problem the ceiling is data quality rather than
model capacity, so a larger backbone mostly buys longer epochs.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

DEFAULT_BACKBONE = "efficientnet_b0"


def build_model(
    *,
    backbone: str = DEFAULT_BACKBONE,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Create a classifier with a fresh head on a pre-trained backbone.

    Dropout is enabled by default because it is what MC Dropout later samples over
    to estimate uncertainty. Uncertainty is not an afterthought here: the product
    routes uncertain trees to human inspection, so the model must be able to say it
    does not know.
    """
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )


def freeze_backbone(model: nn.Module) -> None:
    """Freeze everything except the classifier head.

    Useful for a short warmup: a randomly initialised head produces large gradients
    that can wreck pre-trained features in the first few hundred steps.
    """
    classifier = model.get_classifier()
    classifier_params = set(id(p) for p in classifier.parameters())

    for param in model.parameters():
        param.requires_grad = id(param) in classifier_params


def unfreeze_all(model: nn.Module) -> None:
    """Make every parameter trainable again."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def select_device(preference: str = "auto") -> torch.device:
    """Pick the compute device.

    'auto' uses CUDA when present. Everything in this project runs on CPU too, just
    slowly, so the fallback is a real option for a quick sanity check rather than a
    broken path.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
