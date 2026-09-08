"""Public dataset catalogue for Phase 0.

These are close-range leaf images with controlled backgrounds, so they are only a
proxy for the inspection pass of the pipeline (a drone descending to under 2 m over
a flagged tree). They are useless for the nadir screening pass, where a canopy is
tens of centimetres across and a leaf lesion sits below one pixel.

What they buy us is a working pipeline, a backbone pre-trained on citrus leaf
texture, and an honest sense of which architectures behave. The labels themselves
do not transfer: most of these datasets cover diseases that are absent from Spain.

Licences are recorded because this project is intended to become a commercial
product. Datasets with unknown or non-commercial licences are deliberately excluded,
even when their imagery looks useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSource:
    """A public dataset that can be fetched from Kaggle."""

    key: str
    """Short local identifier; also the folder name under data/raw/."""
    ref: str
    """Kaggle reference, as owner/dataset-slug."""
    licence: str
    approx_size_mb: int
    description: str
    relevant_classes: tuple[str, ...] = field(default_factory=tuple)
    """Classes that map onto pests or diseases actually present in Murcia."""
    caveats: str = ""

    @property
    def is_commercially_usable(self) -> bool:
        """Whether the licence permits commercial use.

        Conservative by design: anything not explicitly known to be permissive is
        treated as unusable, since this code is meant to ship in a product.
        """
        permissive = ("CC0", "CC BY", "Public Domain", "MIT", "Apache")
        return any(token.lower() in self.licence.lower() for token in permissive)


CITRUS_LEAF_PATHOLOGY = DatasetSource(
    key="citrus_leaf_pathology",
    ref="chayanmondalabir/citrus-leaf-pathology-multi-class-image-dataset",
    licence="CC0: Public Domain",
    approx_size_mb=3686,
    description=(
        "Multi-class citrus leaf pathology dataset, 14 classes, with train/test splits "
        "and an augmented variant."
    ),
    relevant_classes=("Citrus Mite", "Curl Leaf"),
    caveats=(
        "Most classes are diseases absent from Spain (canker, black spot, bacterial "
        "blight). Ships both a raw and a pre-augmented copy; only the raw copy should "
        "feed a split, or augmented duplicates of one leaf will straddle train and "
        "validation."
    ),
)

CITRUS_DISEASES = DatasetSource(
    key="citrus_diseases",
    ref="superlord/citrus-diseases",
    licence="Attribution 4.0 International (CC BY 4.0)",
    approx_size_mb=2076,
    description="Citrus disease and pest imagery grouped by class, including aphids.",
    relevant_classes=("aphids",),
    caveats="CC BY 4.0 requires attribution, which is recorded in docs/data_sources.md.",
)

CATALOGUE: tuple[DatasetSource, ...] = (
    CITRUS_LEAF_PATHOLOGY,
    CITRUS_DISEASES,
)


def get_source(key: str) -> DatasetSource:
    """Look up a catalogue entry by its short key."""
    for source in CATALOGUE:
        if source.key == key:
            return source
    available = ", ".join(s.key for s in CATALOGUE)
    raise KeyError(f"unknown dataset {key!r}; available: {available}")
