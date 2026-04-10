"""I/O helpers."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


def ensure_dir(path: Path) -> None:
    """Create a directory if it doesn't exist.

    @param path Directory path to create.
    @return None
    """
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk (BGR).

    @param path Image path to load.
    @return Loaded image in BGR format.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an image to disk, creating parent directories if needed.

    @param path Output image path.
    @param image Image array to save (BGR).
    @return None
    """
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image)


def timestamp_id() -> str:
    """Return a sortable timestamp string.

    @return UTC timestamp string (YYYYMMDD_HHMMSS_us).
    """
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB.

    @param image_bgr Input image in BGR.
    @return Image converted to RGB.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR.

    @param image_rgb Input image in RGB.
    @return Image converted to BGR.
    """
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
