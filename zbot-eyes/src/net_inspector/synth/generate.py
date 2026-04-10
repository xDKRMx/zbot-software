"""Synthetic demo image generator for net inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json
import random

import cv2
import numpy as np

from net_inspector.utils.io import ensure_dir, save_image, timestamp_id


@dataclass
class SynthObject:
    """Synthetic object metadata."""

    label: str
    bbox: Tuple[int, int, int, int]


@dataclass
class SynthResult:
    """Synthetic image output."""

    image: np.ndarray
    objects: List[SynthObject]


def generate_demo_image(
    width: int = 640,
    height: int = 480,
    add_debris: bool = True,
    add_tear: bool = True,
    add_fire: bool = True,
    add_sky: bool = False,
    seed: int | None = None,
) -> SynthResult:
    """Generate a synthetic net image with optional defects.

    @param width Output width.
    @param height Output height.
    @param add_debris Include debris blob.
    @param add_tear Include tear/hole.
    @param add_fire Include fire-like blob.
    @param add_sky Include sky background.
    @param seed Optional random seed.
    @return SynthResult with image and object list.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if add_sky:
        base, sky_bbox = _draw_sky_background(width, height)
    else:
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base[:] = (30, 120, 30)
        sky_bbox = None

    _draw_mesh(base, alpha=0.75 if add_sky else 1.0)
    objects: List[SynthObject] = []

    if sky_bbox is not None:
        objects.append(SynthObject("sky", sky_bbox))

    if add_debris:
        bbox = _draw_debris(base)
        objects.append(SynthObject("debris", bbox))

    if add_tear:
        bbox = _draw_tear(base)
        objects.append(SynthObject("tear", bbox))

    if add_fire:
        bbox = _draw_fire(base)
        objects.append(SynthObject("fire", bbox))

    return SynthResult(base, objects)


def generate_batch(output_dir: Path, n: int = 10) -> None:
    """Generate a batch of synthetic images and JSON labels.

    @param output_dir Output directory.
    @param n Number of images to generate.
    @return None
    """
    ensure_dir(output_dir)
    for _ in range(n):
        add_debris = random.random() > 0.3
        add_tear = random.random() > 0.3
        add_fire = random.random() > 0.5
        add_sky = random.random() > 0.6
        result = generate_demo_image(
            add_debris=add_debris,
            add_tear=add_tear,
            add_fire=add_fire,
            add_sky=add_sky,
        )
        stamp = timestamp_id()
        img_path = output_dir / f"demo_{stamp}.jpg"
        save_image(img_path, result.image)

        label_path = output_dir / f"demo_{stamp}.json"
        record = {
            "image": img_path.name,
            "width": result.image.shape[1],
            "height": result.image.shape[0],
            "objects": [
                {"type": obj.label, "bbox": list(obj.bbox)} for obj in result.objects
            ],
        }
        with label_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)


def _draw_mesh(image: np.ndarray, alpha: float = 1.0) -> None:
    """Draw a mesh-like grid on the image.

    @param image Image to draw on.
    @param alpha Blend strength.
    @return None
    """
    h, w = image.shape[:2]
    spacing = 40
    color_dark = (20, 90, 20)
    color_light = (50, 160, 50)

    overlay = image.copy()
    for x in range(0, w, spacing):
        cv2.line(overlay, (x, 0), (x, h), color_dark, 2)
    for y in range(0, h, spacing):
        cv2.line(overlay, (0, y), (w, y), color_light, 2)

    # Diagonal mesh lines
    for offset in range(-h, w, spacing):
        cv2.line(overlay, (offset, 0), (offset + h, h), color_light, 1)

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    else:
        image[:] = overlay

    # Add subtle noise
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    image[:] = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _draw_debris(image: np.ndarray) -> Tuple[int, int, int, int]:
    """Draw a debris blob.

    @param image Image to draw on.
    @return Bounding box (x, y, w, h).
    """
    h, w = image.shape[:2]
    x = random.randint(40, w - 120)
    y = random.randint(40, h - 120)
    size = random.randint(40, 90)
    color = random.choice([(60, 60, 60), (90, 70, 50), (110, 90, 70)])
    center = (x + size // 2, y + size // 2)
    angle = random.randint(0, 180)
    axes = (size // 2, int(size * 0.35))
    cv2.ellipse(image, center, axes, angle, 0, 360, color, -1)
    cv2.circle(
        image,
        (x + size // 3, y + size // 3),
        size // 4,
        (min(color[0] + 30, 255), min(color[1] + 20, 255), min(color[2] + 10, 255)),
        -1,
    )
    return (x, y, size, size)


def _draw_tear(image: np.ndarray) -> Tuple[int, int, int, int]:
    """Draw a tear/hole ellipse.

    @param image Image to draw on.
    @return Bounding box (x, y, w, h).
    """
    h, w = image.shape[:2]
    x = random.randint(60, w - 160)
    y = random.randint(60, h - 160)
    w_box = random.randint(80, 140)
    h_box = random.randint(60, 120)
    center = (x + w_box // 2, y + h_box // 2)
    axes = (w_box // 2, h_box // 2)
    cv2.ellipse(image, center, axes, random.randint(0, 45), 0, 360, (10, 10, 10), -1)
    cv2.ellipse(image, center, (axes[0] + 6, axes[1] + 6), 0, 0, 360, (30, 30, 30), 2)
    return (x, y, w_box, h_box)


def _draw_fire(image: np.ndarray) -> Tuple[int, int, int, int]:
    """Draw a fire-like blob with smoke.

    @param image Image to draw on.
    @return Bounding box (x, y, w, h).
    """
    h, w = image.shape[:2]
    x = random.randint(80, w - 160)
    y = random.randint(80, h - 200)
    w_box = random.randint(60, 110)
    h_box = random.randint(80, 140)
    base_center = (x + w_box // 2, y + h_box)

    for i in range(6):
        radius = int(w_box * (0.3 + 0.15 * i))
        color = (0, 80 + i * 20, 200 + i * 6)
        cv2.circle(image, base_center, radius, color, -1)

    # Add a smoke-like haze above
    smoke_top = max(0, y - h_box // 2)
    smoke = np.zeros((h_box, w_box, 3), dtype=np.uint8)
    smoke[:] = (160, 160, 160)
    alpha = 0.35
    roi = image[smoke_top : smoke_top + h_box, x : x + w_box]
    if roi.shape == smoke.shape:
        blended = cv2.addWeighted(roi, 1 - alpha, smoke, alpha, 0)
        image[smoke_top : smoke_top + h_box, x : x + w_box] = blended

    return (x, y, w_box, h_box)


def _draw_sky_background(width: int, height: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Create a sky-like background with a mild gradient and cloud hints.

    @param width Output width.
    @param height Output height.
    @return Tuple of image and sky bounding box.
    """
    top = np.array([255, 200, 120], dtype=np.float32)
    bottom = np.array([180, 150, 120], dtype=np.float32)
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / max(1, height - 1)
        color = (1 - t) * top + t * bottom
        gradient[y, :] = color.astype(np.uint8)

    # Light cloud hints
    cloud_layer = gradient.copy()
    for _ in range(10):
        cx = random.randint(0, width - 1)
        cy = random.randint(0, int(height * 0.5))
        radius = random.randint(40, 90)
        cv2.circle(cloud_layer, (cx, cy), radius, (255, 255, 255), -1)
    gradient = cv2.addWeighted(cloud_layer, 0.2, gradient, 0.8, 0)

    sky_h = int(height * 0.45)
    return gradient, (0, 0, width, sky_h)
