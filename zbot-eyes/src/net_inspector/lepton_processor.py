"""Lepton Thermal Processing Module

Provides fixed-scale normalization for FLIR Lepton thermal cameras.
Based on lepton_fixed_scale.py reference implementation.

This ensures:
1. Frame-to-frame color consistency (critical for panoramic stitching)
2. Absolute temperature perception (15°C always same color)
3. Better heat detection (consistent thresholds)

Usage:
    processor = LeptonThermalProcessor()
    thermal_8bit, thermal_color = processor.process_frame(raw_uint16_frame)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional


class LeptonThermalProcessor:
    """Process FLIR Lepton thermal frames with fixed-scale normalization."""
    
    # Fixed display range (raw counts)
    # Formula: T_celsius = (raw_value / 100) - 273.15
    # MINV = 28815 → ~15°C
    # MAXV = 33315 → ~60°C (tunable based on use case)
    DEFAULT_MIN_RAW = 28815  # ~15°C
    DEFAULT_MAX_RAW = 33315  # ~60°C
    
    # Lepton 2.5 resolution
    LEPTON_2_5_WIDTH = 160
    LEPTON_2_5_HEIGHT = 120
    
    # Lepton 2.0 resolution
    LEPTON_2_0_WIDTH = 80
    LEPTON_2_0_HEIGHT = 60
    
    def __init__(
        self,
        min_raw: int = DEFAULT_MIN_RAW,
        max_raw: int = DEFAULT_MAX_RAW,
        colormap: int = cv2.COLORMAP_JET,
        apply_histogram_eq: bool = False,
    ):
        """Initialize Lepton thermal processor.
        
        Args:
            min_raw: Minimum raw thermal value (default: 28815 = ~15°C)
            max_raw: Maximum raw thermal value (default: 33315 = ~60°C)
            colormap: OpenCV colormap (default: COLORMAP_JET)
            apply_histogram_eq: Apply histogram equalization (default: False for fixed scale)
        """
        self.min_raw = min_raw
        self.max_raw = max_raw
        self.colormap = colormap
        self.apply_histogram_eq = apply_histogram_eq
        
        # Precompute normalization factor
        self._range = max(self.max_raw - self.min_raw, 1)
        self._norm_factor = 255.0 / self._range
    
    def raw_to_celsius(self, raw_value: int | float) -> float:
        """Convert raw thermal value to Celsius.
        
        Formula: T_celsius = (raw_value / 100) - 273.15
        
        Args:
            raw_value: Raw thermal value from Lepton
            
        Returns:
            Temperature in Celsius
        """
        return (raw_value / 100.0) - 273.15
    
    def celsius_to_raw(self, celsius: float) -> int:
        """Convert Celsius to raw thermal value.
        
        Args:
            celsius: Temperature in Celsius
            
        Returns:
            Raw thermal value
        """
        return int((celsius + 273.15) * 100)
    
    def process_frame(
        self, 
        raw_frame: np.ndarray,
        return_colorized: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Process raw thermal frame with fixed-scale normalization.
        
        Args:
            raw_frame: Raw uint16 thermal frame (H x W)
            return_colorized: If True, return JET colormap version
            
        Returns:
            Tuple of (thermal_8bit_gray, thermal_color_bgr)
            - thermal_8bit_gray: Normalized 8-bit grayscale (0-255)
            - thermal_color_bgr: JET colormap BGR (only if return_colorized=True)
        """
        if raw_frame.dtype != np.uint16:
            raise ValueError(f"Expected uint16 frame, got {raw_frame.dtype}")
        
        # Clip to fixed range
        clipped = np.clip(raw_frame, self.min_raw, self.max_raw)
        
        # Normalize to 0-255
        thermal_8bit = ((clipped - self.min_raw) * self._norm_factor).astype(np.uint8)
        
        # Optional histogram equalization (usually NOT needed with fixed scale)
        if self.apply_histogram_eq:
            thermal_8bit = cv2.equalizeHist(thermal_8bit)
        
        # Apply colormap if requested
        thermal_color = None
        if return_colorized:
            thermal_color = cv2.applyColorMap(thermal_8bit, self.colormap)
        
        return thermal_8bit, thermal_color
    
    def get_temperature_stats(self, raw_frame: np.ndarray) -> dict:
        """Get temperature statistics from raw frame.
        
        Args:
            raw_frame: Raw uint16 thermal frame
            
        Returns:
            Dict with min/max/mean temperatures in Celsius and raw
        """
        raw_min = int(raw_frame.min())
        raw_max = int(raw_frame.max())
        raw_mean = float(raw_frame.mean())
        
        return {
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_mean": raw_mean,
            "celsius_min": self.raw_to_celsius(raw_min),
            "celsius_max": self.raw_to_celsius(raw_max),
            "celsius_mean": self.raw_to_celsius(raw_mean),
        }
    
    def is_valid_lepton_frame(self, frame: np.ndarray) -> bool:
        """Check if frame has valid Lepton resolution.
        
        Args:
            frame: Input frame
            
        Returns:
            True if frame matches Lepton 2.0 or 2.5 resolution
        """
        if len(frame.shape) != 2:
            return False
        
        h, w = frame.shape
        return (
            (h == self.LEPTON_2_5_HEIGHT and w == self.LEPTON_2_5_WIDTH) or
            (h == self.LEPTON_2_0_HEIGHT and w == self.LEPTON_2_0_WIDTH)
        )


# Global singleton instance (optional, for shared config)
_global_processor: Optional[LeptonThermalProcessor] = None


def get_global_processor() -> LeptonThermalProcessor:
    """Get or create global Lepton processor instance."""
    global _global_processor
    if _global_processor is None:
        _global_processor = LeptonThermalProcessor()
    return _global_processor


def set_global_processor(processor: LeptonThermalProcessor) -> None:
    """Set global Lepton processor instance."""
    global _global_processor
    _global_processor = processor
