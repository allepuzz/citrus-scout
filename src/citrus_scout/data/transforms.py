"""Image augmentation and preprocessing.

Augmentation choices are constrained by what the deployed system will actually see.
A drone descending over a tree photographs leaves at arbitrary rotation, in sunlight
that swings between direct and shaded within one flight, and at varying distance.
Those are the transforms worth applying.

Transforms that would teach the model something false are deliberately absent:

- **No vertical flip beyond what rotation covers.** Leaf lesions have no consistent
  gravity-relative orientation, so this one is harmless, but strong perspective
  warps are not: they distort lesion shape, which is a diagnostic feature.
- **No aggressive hue shift.** Colour is the signal. Chlorosis is a yellow shift and
  sooty mould is a darkening; rotating hue teaches the model to ignore exactly what
  distinguishes them. Only mild jitter, for camera and white-balance variation.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics. The backbones are pre-trained with these, and normalising
# differently would discard part of what the pre-training learned.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 224


def train_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    """Augmentation for training.

    RandomResizedCrop is the workhorse: it varies scale and framing, which mimics a
    drone at different standoff distances and off-centre framing.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.85, 1.18),
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.5, border_mode=0),
            # Field lighting varies far more than colour balance does, so brightness
            # and contrast move more than hue and saturation.
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
            A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            # Motion blur and defocus are real failure modes for a moving drone.
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=(3, 5)),
                ],
                p=0.2,
            ),
            A.GaussNoise(p=0.15),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def eval_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    """Deterministic preprocessing for validation and test.

    Resizes the shorter side then centre-crops, the standard evaluation protocol.
    Any randomness here would make results irreproducible between runs.
    """
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=int(image_size * 1.14)),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
