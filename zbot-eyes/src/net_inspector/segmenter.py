"""ML segmentation helper (optional, PyTorch)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from net_inspector.config import MODELS_DIR


DEFAULT_WEIGHTS_PATH = MODELS_DIR / "deeplabv3_resnet50.pth"
DEFAULT_LABELS_PATH = MODELS_DIR / "segment_labels.txt"
DEFAULT_MAX_SIDE = 720
DEFAULT_LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class Segmenter:
    """PyTorch semantic segmentation runner (DeepLabV3)."""

    def __init__(
        self,
        weights_path: Path = DEFAULT_WEIGHTS_PATH,
        labels_path: Path = DEFAULT_LABELS_PATH,
        max_side: int = DEFAULT_MAX_SIDE,
    ) -> None:
        self.weights_path = weights_path
        self.labels_path = labels_path
        self.max_side = max_side
        self.labels = _load_labels(labels_path)
        self._model = self._load_model(weights_path)

    def available(self) -> bool:
        """Check whether the segmentation model is available.

        @return True if model is loaded.
        """
        return self._model is not None

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a class mask resized to the input image size.

        @param image_bgr Input image (BGR).
        @return Class id mask (H x W).
        """
        if self._model is None:
            raise RuntimeError("Segmentation model not available.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized, scale = _resize_max_side(image_rgb, self.max_side)
        tensor = _to_tensor(resized)

        with _torch_no_grad():
            output = self._model(tensor)["out"][0]
            mask = output.argmax(0).byte().cpu().numpy()

        if scale != 1.0:
            mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def _load_model(self, weights_path: Path):
        """Load PyTorch weights into a DeepLabV3 model.

        @param weights_path Path to .pth weights.
        @return Model or None if unavailable.
        """
        if not weights_path.exists():
            return None
        try:
            import torch
            from torchvision.models.segmentation import deeplabv3_resnet50
        except Exception:
            return None

        model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=21)
        state = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval()
        return model


def render_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    labels: List[str],
    alpha: float = 0.45,
    max_classes: int = 5,
    show_legend: bool = True,
) -> np.ndarray:
    """Overlay a colorized mask and add a top-k legend.

    @param image_bgr Input image (BGR).
    @param mask Class id mask.
    @param labels Label list for class ids.
    @param alpha Blend factor.
    @param max_classes Max classes to list in legend.
    @param show_legend Whether to draw legend text.
    @return Overlay image (BGR).
    """
    color_mask = colorize_mask(mask)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, color_mask, alpha, 0)

    if show_legend:
        class_counts = _top_classes(mask, max_classes)
        y = 24
        for class_id, ratio in class_counts:
            name = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
            text = f"{name}: {ratio*100:.1f}%"
            _draw_text(overlay, text, (10, y))
            y += 20

    return overlay


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map class ids to colors (BGR).

    @param mask Class id mask.
    @return Colorized mask (BGR).
    """
    h, w = mask.shape[:2]
    color = np.zeros((h, w, 3), dtype=np.uint8)
    unique = np.unique(mask)
    for idx in unique:
        bgr = _class_color(int(idx))
        color[mask == idx] = bgr
    return color


def _class_color(idx: int) -> Tuple[int, int, int]:
    """Deterministic color for a class id.

    @param idx Class id.
    @return BGR color tuple.
    """
    rng = (idx * 37) % 255
    return (
        int((rng * 3) % 255),
        int((rng * 7) % 255),
        int((rng * 11) % 255),
    )


def _resize_max_side(image_rgb: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Resize image to fit within max side length.

    @param image_rgb Input image (RGB).
    @param max_side Maximum side length.
    @return Tuple of resized image and scale.
    """
    h, w = image_rgb.shape[:2]
    if max(h, w) <= max_side:
        return image_rgb, 1.0
    scale = max_side / float(max(h, w))
    resized = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _to_tensor(image_rgb: np.ndarray):
    """Convert RGB image to normalized torch tensor.

    @param image_rgb Input image (RGB).
    @return Torch tensor of shape [1, 3, H, W].
    """
    import torch

    tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


class _torch_no_grad:
    def __enter__(self):
        """Enter torch.no_grad context.

        @return Context manager.
        """
        import torch

        self._ctx = torch.no_grad()
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        """Exit torch.no_grad context.

        @param exc_type Exception type.
        @param exc Exception instance.
        @param tb Traceback.
        @return Boolean for suppression.
        """
        return self._ctx.__exit__(exc_type, exc, tb)


def _load_labels(path: Path) -> List[str]:
    """Load label list from file.

    @param path Path to labels file.
    @return List of labels.
    """
    if not path.exists():
        return DEFAULT_LABELS
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels if labels else DEFAULT_LABELS


def _top_classes(mask: np.ndarray, max_classes: int) -> List[Tuple[int, float]]:
    """Compute top classes by pixel share.

    @param mask Class id mask.
    @param max_classes Max number of classes to return.
    @return List of (class_id, ratio).
    """
    total = mask.size
    if total == 0:
        return []
    values, counts = np.unique(mask, return_counts=True)
    pairs = sorted(zip(values, counts), key=lambda x: x[1], reverse=True)
    top = pairs[:max_classes]
    return [(int(v), float(c) / float(total)) for v, c in top]


def _draw_text(image: np.ndarray, text: str, origin: Tuple[int, int]) -> None:
    """Draw text with outline for readability.

    @param image Image to draw on.
    @param text Text string.
    @param origin (x, y) origin.
    @return None
    """
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
