"""Annotation helpers."""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from net_inspector.detectors.base import Detection

COLOR_MAP = {
    "debris": (0, 140, 255),
    "tear": (0, 0, 255),
    "fire": (0, 165, 255),
    "smoke": (128, 128, 128),
    "sky": (200, 220, 255),
    "unknown": (160, 160, 200),
    "ml_fire": (0, 200, 255),
    "ml_object": (255, 128, 0),
}


def draw_detections(image: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """Return a copy of the image annotated with detections.

    @param image Input image (BGR).
    @param detections Iterable of Detection objects.
    @return Annotated image copy.
    """
    annotated = image.copy()
    for det in detections:
        x, y, w, h = det.bbox
        color = COLOR_MAP.get(det.label, (255, 255, 255))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"{det.label}:{det.score:.2f}"
        cv2.putText(
            annotated,
            label,
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated
