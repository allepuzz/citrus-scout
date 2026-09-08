"""Package the working set into a single archive for upload to Colab.

Uploading the raw download to Drive is impractical: 14,227 files totalling 5.5 GB,
where per-file latency dominates and the transfer takes hours. Only 5,951 of those
files survive filtering, and the model never sees them at full resolution anyway.

This module writes one archive holding only the images that make the splits,
downscaled to the size training actually uses. That turns hours of upload into
minutes, and Colab unpacks a single file rather than walking a deep tree on a
network filesystem.

The split assignment is baked into the archive so that training on Colab uses the
identical partition to a local run. Recomputing it remotely would risk a different
split if any input changed, quietly invalidating the comparison.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from citrus_scout.data.build import build_splits
from citrus_scout.data.splits import SplitResult

# Longest side of the exported images. Training crops to 224, so 512 leaves room
# for random-resized-crop augmentation without carrying full camera resolution.
DEFAULT_MAX_SIDE = 512

# JPEG quality for the export. 90 is visually near-lossless for this content and
# roughly a quarter the size of quality 100.
DEFAULT_QUALITY = 90


@dataclass(frozen=True)
class PackageStats:
    """Outcome of packaging, for reporting."""

    archive: Path
    images: int
    skipped: int
    bytes_written: int

    @property
    def megabytes(self) -> float:
        return self.bytes_written / 1e6

    def describe(self) -> str:
        return f"{self.archive.name}: {self.images} images, {self.megabytes:.0f} MB" + (
            f", {self.skipped} skipped" if self.skipped else ""
        )


def _resized(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest side is at most `max_side`, preserving aspect.

    Images already smaller are returned untouched rather than upscaled, which would
    add no information and cost space.
    """
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (round(image.width * scale), round(image.height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def package_splits(
    split: SplitResult,
    output: Path,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_QUALITY,
) -> PackageStats:
    """Write one zip holding every split image, downscaled, plus a manifest.

    Inside the archive images are laid out as `<split>/<label>/<index>.jpg`, and
    `manifest.json` records the class list and per-split counts so the remote side
    can verify it received what it expected.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    partitions = {"train": split.train, "val": split.val, "test": split.test}
    manifest: dict[str, object] = {
        "max_side": max_side,
        "quality": quality,
        "classes": sorted({s.label for group in partitions.values() for s in group}),
        "counts": {name: len(group) for name, group in partitions.items()},
        "duplicates_removed": split.duplicates_removed,
    }

    written = 0
    skipped = 0

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        # ZIP_STORED, not DEFLATE: JPEG is already compressed, so deflating it
        # burns CPU on both ends for a fraction of a percent of size.
        for split_name, samples in partitions.items():
            for index, sample in enumerate(samples):
                try:
                    with Image.open(sample.path) as source:
                        image = _resized(source.convert("RGB"), max_side)
                        # Encode in memory rather than writing a temporary file per image.
                        buffer = BytesIO()
                        image.save(buffer, format="JPEG", quality=quality, optimize=True)
                except (OSError, ValueError):
                    # Truncated or unreadable file: skip rather than abort a long export.
                    skipped += 1
                    continue

                name = f"{split_name}/{sample.label}/{index:06d}.jpg"
                archive.writestr(name, buffer.getvalue())
                written += 1

        manifest["images_written"] = written
        manifest["images_skipped"] = skipped
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))

    return PackageStats(
        archive=output,
        images=written,
        skipped=skipped,
        bytes_written=output.stat().st_size,
    )


def package_for_colab(
    output: Path,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_QUALITY,
    seed: int = 42,
) -> PackageStats:
    """Build the splits and package them in one step."""
    split, _ = build_splits(seed=seed)
    return package_splits(split, output, max_side=max_side, quality=quality)
