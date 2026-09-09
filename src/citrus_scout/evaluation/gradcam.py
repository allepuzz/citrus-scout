"""Grad-CAM: where the model is looking.

Every metric in this project scores *what* the model answers. None of them check
*why*. On this dataset that gap is wide enough to matter, because the public leaf
images are close-range shots against controlled backgrounds, and a model can reach
a perfect score by reading the background rather than the leaf.

That is not a hypothetical. The baseline run hit PR-AUC 1.0000 on 762 test images.
Either the task is trivial, or the model found a shortcut; the metrics cannot tell
those apart and a heatmap can. If attention sits on the lamina and the lesions, the
features are likely to survive a change of viewpoint. If it sits on the corners, the
model learned which photographer shot which class, and none of it transfers to a
drone over a Murcian grove.

Grad-CAM weights the final convolutional feature maps by the gradient of the target
class score with respect to each map, then averages them. Channels whose activation
raises the score get more weight, so the result shows which spatial positions
supported the prediction.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization" (ICCV 2017).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from citrus_scout.data.transforms import IMAGENET_MEAN, IMAGENET_STD

# Last layer before global pooling, per architecture family. Grad-CAM needs a layer
# that still has spatial extent: anything after the pool has been reduced to 1x1 and
# carries no localisation left to read.
DEFAULT_TARGET_LAYERS: dict[str, str] = {
    "efficientnet": "bn2",
    "convnext": "norm_pre",
    "resnet": "layer4",
    "mobilenet": "bn2",
}


@dataclass(frozen=True)
class AttentionSummary:
    """Where a heatmap's mass sits, as numbers rather than a picture.

    A picture needs a human to read it. These let the check run over a whole split
    and flag the images worth looking at.
    """

    border_fraction: float
    """Share of total attention in the border ring.

    The shortcut detector. A leaf photographed against a uniform background puts no
    diagnostic information at the edges, so attention there is attention on the
    photograph rather than the plant.
    """

    border_area_fraction: float
    """Share of the frame's *area* the border ring occupies.

    Needed to read `border_fraction` at all. Grad-CAM maps are small, 7x7 for
    EfficientNet-B0, so a one-cell ring covers 49% of the area there against 26% at
    14x14. Comparing attention to a fixed constant would therefore flag a uniform
    map as suspicious at one resolution and clear it at another.
    """

    centre_fraction: float
    """Share of attention in the central half, where the leaf usually sits."""

    peak_value: float
    concentration: float
    """Share of attention in the top 10% of pixels.

    Near 1.0 means one sharp focus, which on a diffuse symptom like chlorosis is
    itself suspicious. Near 0.1 means the map is flat and explains nothing.
    """

    @property
    def border_excess(self) -> float:
        """How much more attention the border gets than its area alone would earn.

        1.0 means exactly proportional, the value a uniform map produces at any
        resolution. Above 1.0 means the model prefers the edges. This is the
        resolution-independent form of `border_fraction` and the one to threshold.
        """
        if self.border_area_fraction <= 0.0:
            return 0.0
        return self.border_fraction / self.border_area_fraction

    def looks_like_background_shortcut(self, *, excess_threshold: float = 1.15) -> bool:
        """Whether the border draws disproportionate attention.

        Thresholds `border_excess`, not the raw fraction, so the verdict does not
        change with feature-map size. The default allows 15% above proportional
        before complaining: these are centred leaf photographs, and a model reading
        the plant should put at most its area share at the edges, never clearly more.
        """
        return self.border_excess > excess_threshold

    def __str__(self) -> str:
        verdict = " [suspect]" if self.looks_like_background_shortcut() else ""
        return (
            f"border={self.border_fraction:.1%} of attention for "
            f"{self.border_area_fraction:.1%} of area (excess {self.border_excess:.2f}x) "
            f"centre={self.centre_fraction:.1%} concentration={self.concentration:.1%}{verdict}"
        )


def resolve_target_layer(model: nn.Module, backbone: str) -> str:
    """Pick the layer to hook, by architecture family.

    Raises rather than guessing when the family is unknown. A silently wrong layer
    produces a plausible-looking heatmap of nothing in particular, which is worse
    than an error because it invites a confident wrong conclusion.
    """
    available = {name for name, _ in model.named_modules()}

    for family, layer in DEFAULT_TARGET_LAYERS.items():
        if family in backbone.lower():
            if layer not in available:
                raise ValueError(
                    f"backbone {backbone!r} looks like {family} but has no layer "
                    f"{layer!r}. Pass target_layer explicitly."
                )
            return layer

    raise ValueError(
        f"no default Grad-CAM layer known for backbone {backbone!r}. Pass "
        f"target_layer explicitly: it must be the last layer with spatial extent, "
        f"before global pooling."
    )


class GradCAM:
    """Grad-CAM for one model and one target layer.

    Used as a context manager so the forward and backward hooks are always removed.
    A leaked hook keeps every activation tensor alive, which on a full split leaks
    memory steadily and is easy to mistake for something else.
    """

    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        modules = dict(model.named_modules())
        if target_layer not in modules:
            raise ValueError(
                f"layer {target_layer!r} not found. Available leaf modules include: "
                f"{sorted(n for n in modules if n)[-5:]}"
            )
        self._module = modules[target_layer]

    def __enter__(self) -> GradCAM:
        self._handles.append(self._module.register_forward_hook(self._save_activations))
        # full_backward_hook rather than backward_hook: the latter is documented as
        # unreliable for modules whose forward is not a single operation.
        self._handles.append(self._module.register_full_backward_hook(self._save_gradients))
        return self

    def __exit__(self, *args: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._activations = None
        self._gradients = None

    def _save_activations(self, _module, _inputs, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def heatmap(
        self,
        images: torch.Tensor,
        *,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Attention maps for a batch, each normalised to [0, 1].

        `target_class` defaults to whatever the model predicted, which answers "why
        this answer". Passing a class explicitly answers a different and sometimes
        more useful question: what evidence the model sees for a class it rejected.

        Returns shape (batch, height, width) at the feature map's resolution, not
        the input's. Upsampling is left to the caller so it can choose the
        interpolation rather than inherit one.
        """
        if not self._handles:
            raise RuntimeError("use GradCAM as a context manager so hooks are registered")

        was_training = self.model.training
        self.model.eval()

        # Gradients are required even though this is inference: Grad-CAM's weights
        # *are* gradients. A no_grad block here yields an empty map.
        images = images.clone().requires_grad_(False)
        logits = self.model(images)

        if target_class is None:
            targets = logits.argmax(dim=1)
        else:
            targets = torch.full((logits.shape[0],), target_class, device=logits.device)

        score = logits.gather(1, targets.unsqueeze(1)).sum()

        self.model.zero_grad(set_to_none=True)
        score.backward()

        # Clear the parameter gradients this backward pass just produced. Grad-CAM
        # reads them from the hooked activations, not from the parameters, and
        # leaving them behind would be added to the next optimiser step if this ran
        # mid-training.
        self.model.zero_grad(set_to_none=True)

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                f"no activations captured at {self.target_layer!r}. The layer is "
                "probably not on the forward path for this input."
            )

        # Channel weights: mean gradient per feature map. A channel whose activation
        # raises the class score gets a positive weight.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        maps = (weights * self._activations).sum(dim=1)

        # ReLU: only evidence *for* the class is wanted. Negative contributions are
        # evidence for some other class and would muddy the map.
        maps = torch.relu(maps)

        if was_training:
            self.model.train()

        return _normalise_per_image(maps.cpu().numpy())


def _normalise_per_image(maps: np.ndarray) -> np.ndarray:
    """Scale each map to [0, 1] independently.

    Per image, not per batch: the absolute gradient magnitude varies with how
    confident the prediction was, and normalising across a batch would make
    confident images look like they have more attention everywhere.
    """
    flat = maps.reshape(maps.shape[0], -1)
    minimum = flat.min(axis=1).reshape(-1, 1, 1)
    maximum = flat.max(axis=1).reshape(-1, 1, 1)
    span = np.maximum(maximum - minimum, 1e-12)
    return (maps - minimum) / span


def summarise_attention(heatmap: np.ndarray, *, border_width: float = 0.125) -> AttentionSummary:
    """Reduce one heatmap to the numbers that indicate a background shortcut."""
    if heatmap.ndim != 2:
        raise ValueError(f"expected a single 2D heatmap, got shape {heatmap.shape}")
    if not 0.0 < border_width < 0.5:
        raise ValueError(f"border_width must be in (0, 0.5), got {border_width}")

    height, width = heatmap.shape

    margin_y = max(int(height * border_width), 1)
    margin_x = max(int(width * border_width), 1)

    # Guard the degenerate case: on a map small enough that the margins meet, the
    # interior slice is empty and every cell counts as border.
    interior_height = max(height - 2 * margin_y, 0)
    interior_width = max(width - 2 * margin_x, 0)
    border_area_fraction = 1.0 - (interior_height * interior_width) / (height * width)

    total = float(heatmap.sum())
    if total <= 0.0:
        # A flat zero map carries no information; report zeros rather than dividing.
        return AttentionSummary(
            border_fraction=0.0,
            border_area_fraction=border_area_fraction,
            centre_fraction=0.0,
            peak_value=0.0,
            concentration=0.0,
        )

    interior_mass = (
        float(heatmap[margin_y : height - margin_y, margin_x : width - margin_x].sum())
        if interior_height and interior_width
        else 0.0
    )

    quarter_y, quarter_x = height // 4, width // 4
    centre_mass = float(
        heatmap[quarter_y : height - quarter_y, quarter_x : width - quarter_x].sum()
    )

    flat = np.sort(heatmap.ravel())[::-1]
    top_tenth = max(int(flat.size * 0.1), 1)

    return AttentionSummary(
        border_fraction=(total - interior_mass) / total,
        border_area_fraction=border_area_fraction,
        centre_fraction=centre_mass / total,
        peak_value=float(heatmap.max()),
        concentration=float(flat[:top_tenth].sum()) / total,
    )


def denormalise(image: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation, returning HWC uint8 for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    restored = (image.cpu() * std + mean).clamp(0, 1)
    return (restored.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay(image: torch.Tensor, heatmap: np.ndarray, *, alpha: float = 0.5) -> np.ndarray:
    """Blend a heatmap over its image as an RGB uint8 array.

    Uses a plain red channel rather than a perceptual colourmap so the module stays
    free of a matplotlib dependency. The point here is localisation, which a single
    channel conveys adequately.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    base = denormalise(image)
    height, width = base.shape[:2]

    # Nearest-neighbour upsample from the feature map's resolution, typically 7x7.
    # Deliberately blocky: a smooth interpolation implies spatial precision the
    # underlying map does not have.
    y_index = (np.arange(height) * heatmap.shape[0] // height).clip(0, heatmap.shape[0] - 1)
    x_index = (np.arange(width) * heatmap.shape[1] // width).clip(0, heatmap.shape[1] - 1)
    upsampled = heatmap[np.ix_(y_index, x_index)]

    tint = np.zeros_like(base, dtype=float)
    tint[..., 0] = upsampled * 255

    blended = base * (1 - alpha * upsampled[..., None]) + tint * alpha
    return blended.clip(0, 255).astype(np.uint8)
