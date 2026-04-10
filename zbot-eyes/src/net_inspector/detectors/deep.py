"""Deeper (slower) detector with optional ML hooks."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from net_inspector.config import AppConfig, MODELS_DIR
from net_inspector.detectors.base import Detection, Detector
from net_inspector.detectors.fast import FastDetector


class DeepDetector(Detector):
    """On-demand detector that runs heavier heuristics and optional ML models."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self._fast = FastDetector(self.config)
        self._fire_interpreter = None
        self._object_interpreter = None

    def analyze(self, image: np.ndarray) -> List[Detection]:
        """Run DEEP heuristics and optional ML hooks.

        @param image Input image (BGR).
        @return List of detections.
        """
        detections: List[Detection] = []
        detections.extend(self._heuristic_multi_scale(image))

        fire_det = self._run_fire_classifier(image)
        if fire_det is not None:
            detections.append(fire_det)

        detections.extend(self._run_object_detector(image))
        return detections

    def _heuristic_multi_scale(self, image: np.ndarray) -> List[Detection]:
        """Run heuristics at multiple scales.

        @param image Input image (BGR).
        @return List of detections merged across scales.
        """
        detections: List[Detection] = []
        for scale in (1.0, 0.7):
            if scale == 1.0:
                resized = image
            else:
                resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            dets = self._slow_heuristics(resized)
            if scale != 1.0:
                detections.extend(self._scale_detections(dets, 1.0 / scale))
            else:
                detections.extend(dets)
        return detections

    def _slow_heuristics(self, image: np.ndarray) -> List[Detection]:
        """Run slower heuristic pipeline with edge enhancement.

        @param image Input image (BGR).
        @return List of detections.
        """
        # Use the fast detector but with extra smoothing and edge enhancement.
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 120)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        edge_overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(blurred, 0.9, edge_overlay, 0.1, 0)
        detections = self._fast.analyze(enhanced, enable_temporal=False)

        boosted: List[Detection] = []
        for det in detections:
            boosted.append(Detection(det.label, det.bbox, min(1.0, det.score * 1.15), det.meta))
        return boosted

    def _scale_detections(self, detections: List[Detection], scale: float) -> List[Detection]:
        """Scale detection boxes by a factor.

        @param detections List of detections.
        @param scale Scale factor.
        @return Scaled detections.
        """
        scaled: List[Detection] = []
        for det in detections:
            x, y, w, h = det.bbox
            bbox = (int(x * scale), int(y * scale), int(w * scale), int(h * scale))
            scaled.append(Detection(det.label, bbox, det.score, det.meta))
        return scaled

    def _run_fire_classifier(self, image: np.ndarray) -> Optional[Detection]:
        """Run optional fire classifier if available.

        @param image Input image (BGR).
        @return Detection if fire classifier fires, else None.
        """
        model_path = MODELS_DIR / "fire_classifier.tflite"
        if not model_path.exists():
            return None
        interpreter = self._get_interpreter(model_path, kind="fire")
        if interpreter is None:
            return None

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_shape = input_details["shape"]
        target_h, target_w = int(input_shape[1]), int(input_shape[2])

        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        input_data = resized.astype(np.float32)
        if self.config.fire_classifier.normalize_0_1:
            input_data = input_data / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        if input_details["dtype"] == np.uint8:
            input_data = (input_data * 255).astype(np.uint8)

        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"]).reshape(-1)

        score = self._extract_classifier_score(output, output_details)
        if score < self.config.fire_classifier.score_threshold:
            return None

        return Detection(
            "ml_fire",
            (0, 0, image.shape[1], image.shape[0]),
            float(score),
            meta={"model": str(model_path), "type": "fire_classifier"},
        )

    def _run_object_detector(self, image: np.ndarray) -> List[Detection]:
        """Run optional object detector if available.

        @param image Input image (BGR).
        @return List of ML detections.
        """
        model_path = MODELS_DIR / "object_detector.tflite"
        if not model_path.exists():
            return []
        interpreter = self._get_interpreter(model_path, kind="object")
        if interpreter is None:
            return []

        input_details = interpreter.get_input_details()[0]
        input_shape = input_details["shape"]
        target_h, target_w = int(input_shape[1]), int(input_shape[2])
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        input_data = resized.astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        if input_details["dtype"] == np.uint8:
            input_data = np.clip(input_data, 0, 255).astype(np.uint8)
        else:
            input_data = input_data / 255.0

        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()

        output_details = interpreter.get_output_details()
        outputs = [interpreter.get_tensor(d["index"]) for d in output_details]
        detections: List[Detection] = []

        if len(outputs) >= 4:
            boxes = outputs[0][0]
            classes = outputs[1][0]
            scores = outputs[2][0]
            count = int(outputs[3][0])
            labels = self._load_labels()

            for i in range(count):
                score = float(scores[i])
                if score < self.config.object_detector.score_threshold:
                    continue
                ymin, xmin, ymax, xmax = boxes[i]
                x = int(xmin * image.shape[1])
                y = int(ymin * image.shape[0])
                w = int((xmax - xmin) * image.shape[1])
                h = int((ymax - ymin) * image.shape[0])
                class_id = int(classes[i])
                label = labels.get(class_id, f"class_{class_id}")
                detections.append(
                    Detection(
                        "ml_object",
                        (x, y, w, h),
                        score,
                        meta={"class_id": class_id, "label": label, "model": str(model_path)},
                    )
                )
        return detections

    def _load_labels(self) -> dict[int, str]:
        """Load labels for object detector.

        @return Mapping of class id to label name.
        """
        labels: dict[int, str] = {}
        path = self.config.object_detector.labels_path
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    name = line.strip()
                    if not name:
                        continue
                    labels[idx] = name
        return labels

    def _extract_classifier_score(self, output: np.ndarray, output_details: dict) -> float:
        """Extract a scalar score from classifier output.

        @param output Raw model output.
        @param output_details Interpreter output details.
        @return Score as float.
        """
        # Handle quantized outputs if available.
        if output_details.get("dtype") == np.uint8:
            scale, zero = output_details.get("quantization", (1.0, 0))
            output = (output.astype(np.float32) - zero) * scale

        if output.size == 1:
            return float(output[0])
        idx = min(self.config.fire_classifier.positive_index, output.size - 1)
        return float(output[idx])

    def _get_interpreter(self, model_path: Path, kind: str):
        """Create or reuse a TFLite interpreter.

        @param model_path Path to model file.
        @param kind Cache key (fire/object).
        @return Interpreter or None if unavailable.
        """
        cache = self._fire_interpreter if kind == "fire" else self._object_interpreter
        if cache is not None:
            return cache

        interpreter = None
        try:
            from tflite_runtime.interpreter import Interpreter

            interpreter = Interpreter(model_path=str(model_path))
        except Exception:
            try:
                from tensorflow.lite.python.interpreter import Interpreter

                interpreter = Interpreter(model_path=str(model_path))
            except Exception:
                return None

        interpreter.allocate_tensors()
        if kind == "fire":
            self._fire_interpreter = interpreter
        else:
            self._object_interpreter = interpreter
        return interpreter
