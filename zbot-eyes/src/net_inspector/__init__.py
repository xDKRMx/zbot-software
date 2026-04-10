"""Net Inspector package."""

from .config import AppConfig
from .detectors.fast import FastDetector
from .detectors.deep import DeepDetector

__all__ = ["AppConfig", "FastDetector", "DeepDetector"]
