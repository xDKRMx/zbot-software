"""Fast heuristic detector (CPU-friendly)."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from net_inspector.config import AppConfig
from net_inspector.detectors.base import Detection, Detector


class FastDetector(Detector):
    """Real-time detector using simple HSV and morphology heuristics."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self._prev_gray: np.ndarray | None = None

    def analyze(self, image: np.ndarray, enable_temporal: bool = False) -> List[Detection]:
        """Run FAST heuristics.

        @param image Input image (BGR).
        @param enable_temporal Enable flicker/motion cue for fire.
        @return List of detections.
        """
        detections: List[Detection] = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = self._green_mask(hsv)
        net_region = self._net_region(green_mask)
        sky_mask = self._sky_mask(hsv)

        detections.extend(self._detect_debris(image, green_mask, net_region))
        detections.extend(self._detect_tears(image, net_region, green_mask, hsv))
        detections.extend(self._detect_fire(hsv, net_region, enable_temporal))
        detections.extend(self._detect_sky(image, sky_mask))
        detections.extend(self._detect_unknown(image, green_mask, net_region, sky_mask))

        return detections

    def _green_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Create a mask for green net regions.

        @param hsv HSV image.
        @return Binary mask of green areas.
        """
        cfg = self.config.heuristic
        lower = (cfg.green_h_min, cfg.green_s_min, cfg.green_v_min)
        upper = (cfg.green_h_max, 255, 255)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _net_region(self, green_mask: np.ndarray) -> np.ndarray:
        """Estimate net region from green mask.

        @param green_mask Binary green mask.
        @return Dilated net region mask.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        region = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        region = cv2.dilate(region, kernel, iterations=1)
        return region

    def _detect_debris(
        self, image: np.ndarray, green_mask: np.ndarray, net_region: np.ndarray
    ) -> List[Detection]:
        """Detect non-green debris within net region.

        @param image Input image (BGR).
        @param green_mask Binary green mask.
        @param net_region Estimated net mask.
        @return List of debris detections.
        """
        cfg = self.config.heuristic
        non_green = cv2.bitwise_not(green_mask)
        debris_mask = cv2.bitwise_and(non_green, net_region)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        debris_mask = cv2.morphologyEx(debris_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(debris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        img_area = image.shape[0] * image.shape[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.debris_min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            score = min(1.0, area / (img_area * 0.05))
            detections.append(Detection("debris", (x, y, w, h), score))
        return detections

    def _detect_tears(
        self,
        image: np.ndarray,
        net_region: np.ndarray,
        green_mask: np.ndarray,
        hsv: np.ndarray,
    ) -> List[Detection]:
        """Detect tears/holes using dark regions and green suppression.

        @param image Input image (BGR).
        @param net_region Estimated net mask.
        @param green_mask Binary green mask.
        @param hsv HSV image.
        @return List of tear detections.
        """
        cfg = self.config.heuristic
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
        green_shadow = cv2.inRange(
            hsv,
            (cfg.green_h_min, cfg.green_s_min, 0),
            (cfg.green_h_max, 255, 80),
        )
        hole_mask = cv2.bitwise_and(dark, net_region)
        hole_mask = cv2.bitwise_and(hole_mask, cv2.bitwise_not(green_shadow))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        img_area = image.shape[0] * image.shape[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.tear_min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            roi = green_mask[y : y + h, x : x + w]
            roi_area = max(1, w * h)
            green_ratio = float(cv2.countNonZero(roi)) / float(roi_area)
            if green_ratio > cfg.tear_green_ratio_max:
                continue
            score = min(1.0, area / (img_area * 0.08))
            detections.append(Detection("tear", (x, y, w, h), score))
        return detections

    def _detect_fire(
        self, hsv: np.ndarray, net_region: np.ndarray, enable_temporal: bool
    ) -> List[Detection]:
        """Detect fire/smoke via HSV thresholds and optional temporal cue.

        @param hsv HSV image.
        @param net_region Estimated net mask.
        @param enable_temporal Enable flicker/motion cue.
        @return List of fire/smoke detections.
        """
        cfg = self.config.heuristic

        # Fire colors: reds/oranges/yellows
        fire_mask1 = cv2.inRange(hsv, (0, 120, 120), (25, 255, 255))
        fire_mask2 = cv2.inRange(hsv, (160, 120, 120), (179, 255, 255))
        fire_mask = cv2.bitwise_or(fire_mask1, fire_mask2)
        fire_mask = cv2.morphologyEx(
            fire_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )

        # Smoke heuristic: low saturation, high value
        smoke_mask = cv2.inRange(hsv, (0, 0, cfg.smoke_v_min), (179, cfg.smoke_s_max, 255))
        smoke_mask = cv2.morphologyEx(
            smoke_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )

        detections: List[Detection] = []
        img_area = hsv.shape[0] * hsv.shape[1]

        for label, mask, min_area in (
            ("fire", fire_mask, cfg.fire_min_area),
            ("smoke", smoke_mask, cfg.fire_min_area * 2),
        ):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                score = min(1.0, area / (img_area * 0.05))
                detections.append(Detection(label, (x, y, w, h), score))

        if enable_temporal:
            gray = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            if self._prev_gray is not None:
                diff = cv2.absdiff(gray, self._prev_gray)
                motion_ratio = float((diff > 20).mean())
                if motion_ratio > 0.02:
                    detections.append(
                        Detection(
                            "fire",
                            (0, 0, hsv.shape[1], hsv.shape[0]),
                            min(1.0, motion_ratio * 5.0),
                            meta={"note": "temporal flicker"},
                        )
                    )
            self._prev_gray = gray

        return detections

    def _sky_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Create a mask for sky-like regions.

        @param hsv HSV image.
        @return Binary sky mask.
        """
        cfg = self.config.heuristic
        lower = (cfg.sky_h_min, cfg.sky_s_min, cfg.sky_v_min)
        upper = (cfg.sky_h_max, 255, 255)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _detect_sky(self, image: np.ndarray, sky_mask: np.ndarray) -> List[Detection]:
        """Detect large sky regions.

        @param image Input image (BGR).
        @param sky_mask Binary sky mask.
        @return List of sky detections.
        """
        cfg = self.config.heuristic
        contours, _ = cv2.findContours(sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        img_area = image.shape[0] * image.shape[1]
        min_area = img_area * cfg.sky_min_area_ratio
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            score = min(1.0, area / (img_area * 0.6))
            detections.append(Detection("sky", (x, y, w, h), score))
        return detections

    def _detect_unknown(
        self,
        image: np.ndarray,
        green_mask: np.ndarray,
        net_region: np.ndarray,
        sky_mask: np.ndarray,
    ) -> List[Detection]:
        """Detect large non-net, non-sky regions.

        @param image Input image (BGR).
        @param green_mask Binary green mask.
        @param net_region Estimated net mask.
        @param sky_mask Binary sky mask.
        @return List of unknown detections.
        """
        cfg = self.config.heuristic
        non_green = cv2.bitwise_not(green_mask)
        unknown_mask = cv2.bitwise_and(non_green, cv2.bitwise_not(net_region))
        unknown_mask = cv2.bitwise_and(unknown_mask, cv2.bitwise_not(sky_mask))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        unknown_mask = cv2.morphologyEx(unknown_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(
            unknown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        detections: List[Detection] = []
        img_area = image.shape[0] * image.shape[1]
        min_area = img_area * cfg.unknown_min_area_ratio
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            score = min(1.0, area / (img_area * 0.5))
            detections.append(Detection("unknown", (x, y, w, h), score))
        return detections
