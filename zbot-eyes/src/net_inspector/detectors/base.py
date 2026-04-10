"""Detector base classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class Detection:
    """A detection result."""

    label: str
    bbox: Tuple[int, int, int, int]
    score: float
    meta: Dict[str, Any] | None = None


class Detector:
    """Base detector interface."""

    def analyze(self, image: np.ndarray) -> List[Detection]:
        """Analyze an image and return detections.

        @param image Input image (BGR).
        @return List of detections.
        """
        raise NotImplementedError
