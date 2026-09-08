"""Download public datasets from Kaggle into data/raw/.

Authentication is handled by the Kaggle client itself, which reads credentials from
the user's Kaggle config directory. Recent accounts get an `access_token` file;
older ones a `kaggle.json`. The client tries the access token first, so either works
without anything special here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from citrus_scout.data.sources import CATALOGUE, DatasetSource
from citrus_scout.utils.paths import RAW_DATA_DIR

# Extensions we treat as usable imagery when counting what landed on disk.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


class KaggleAuthError(RuntimeError):
    """Raised when Kaggle credentials are missing or rejected."""


def _authenticated_api():
    """Return an authenticated Kaggle client.

    Imported lazily: the Kaggle package authenticates at import time in some
    versions, which would make the whole package unimportable without credentials
    and break unrelated tests.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise KaggleAuthError("the kaggle package is not installed") from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise KaggleAuthError(
            "Kaggle authentication failed. Create a token at "
            "https://www.kaggle.com/settings under API, and place the file it gives "
            "you in your Kaggle config directory (~/.kaggle on Linux and macOS, "
            "%USERPROFILE%\\.kaggle on Windows)."
        ) from exc
    return api


def count_images(directory: Path) -> int:
    """Count image files anywhere under `directory`."""
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def download_source(
    source: DatasetSource,
    *,
    destination: Path | None = None,
    force: bool = False,
) -> Path:
    """Download and unpack one dataset, returning the directory it landed in.

    Skips the download when the target already holds images, unless `force` is set.
    A partially extracted directory counts as present, so pass `force` after an
    interrupted run.
    """
    target = (destination or RAW_DATA_DIR) / source.key

    existing = count_images(target)
    if existing and not force:
        return target

    if force and target.exists():
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)

    api = _authenticated_api()
    api.dataset_download_files(source.ref, path=str(target), unzip=True, quiet=False)

    if not count_images(target):
        raise RuntimeError(
            f"{source.ref} downloaded but no images were found under {target}. "
            "The dataset layout may have changed, or it may ship a non-image format."
        )
    return target


def download_all(
    *,
    destination: Path | None = None,
    force: bool = False,
    commercial_only: bool = True,
) -> dict[str, Path]:
    """Download every catalogue entry, returning key to directory.

    By default this skips datasets whose licence does not permit commercial use,
    since the project is meant to ship as a product.
    """
    results: dict[str, Path] = {}
    for source in CATALOGUE:
        if commercial_only and not source.is_commercially_usable:
            continue
        results[source.key] = download_source(source, destination=destination, force=force)
    return results
