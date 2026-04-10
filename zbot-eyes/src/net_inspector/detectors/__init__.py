"""Detector subpackage."""

from .base import Detection, Detector
from .fast import FastDetector
from .deep import DeepDetector

__all__ = ["Detection", "Detector", "FastDetector", "DeepDetector"]
