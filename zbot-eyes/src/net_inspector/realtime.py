"""Realtime camera loop (FAST mode with DEEP on-demand)."""

from __future__ import annotations

import time
from typing import Optional

import cv2

from net_inspector.config import AppConfig
from net_inspector.detectors.fast import FastDetector
from net_inspector.detectors.deep import DeepDetector
from net_inspector.utils.annotate import draw_detections
from net_inspector.utils.io import ensure_dir, save_image, timestamp_id
from net_inspector.utils.logging_utils import detections_to_dicts, iso_timestamp, log_jsonl


def run_realtime(
    camera: int = 0,
    mode: str = "fast",
    width: int = 640,
    height: int = 480,
    save_annotated: bool = False,
    save_raw: bool = False,
    log_all: bool = False,
    no_display: bool = False,
    log_every_n: int = 5,
    segment: bool = False,
    segment_weights=None,
    segment_every_n: int = 8,
    segment_alpha: float = 0.45,
) -> None:
    """Run the realtime FAST detector and allow DEEP analysis via keypress.

    @param camera Camera index.
    @param mode Realtime mode (FAST only).
    @param width Frame width.
    @param height Frame height.
    @param save_annotated Save annotated frames on events.
    @param save_raw Save raw frames on events.
    @param log_all Log every frame event.
    @param no_display Disable preview window.
    @param log_every_n Log every N frames in FAST mode.
    @param segment Enable segmentation overlay.
    @param segment_weights Path to segmentation weights.
    @param segment_every_n Run segmentation every N frames.
    @param segment_alpha Segmentation overlay alpha.
    @return None
    """
    config = AppConfig()
    fast = FastDetector(config)
    deep = DeepDetector(config)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera {camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    ensure_dir(config.outputs_annotated)
    ensure_dir(config.outputs_raw)

    help_on = False
    frame_idx = 0
    last_fps_time = time.time()
    fps = 0.0
    deep_display_until = 0.0
    last_deep_annotated: Optional[object] = None

    if mode != "fast":
        print("Realtime mode only supports FAST inference; running FAST.")

    seg_enabled = False
    segmenter = None
    seg_mask = None
    seg_labels = []
    if segment:
        try:
            from net_inspector.segmenter import Segmenter, render_overlay

            segmenter = Segmenter(weights_path=segment_weights)
            if segmenter.available():
                seg_enabled = True
                seg_labels = segmenter.labels
                print("[SEG] segmentation enabled")
            else:
                print("[SEG] model not available, skipping")
        except Exception as exc:
            print(f"[SEG] unavailable: {exc}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            now = time.time()
            if frame_idx % 10 == 0:
                elapsed = now - last_fps_time
                if elapsed > 0:
                    fps = 10.0 / elapsed
                last_fps_time = now

            detections = fast.analyze(frame, enable_temporal=True)
            annotated = draw_detections(frame, detections)

            if seg_enabled and segmenter is not None:
                if segment_every_n <= 1 or frame_idx % segment_every_n == 0:
                    seg_mask = segmenter.segment(frame)
                if seg_mask is not None:
                    annotated = render_overlay(
                        annotated, seg_mask, seg_labels, alpha=segment_alpha
                    )

            status = _format_status(detections)
            _overlay_text(annotated, f"FPS: {fps:.1f}", (10, 20))
            _overlay_text(annotated, status, (10, 42))

            if help_on:
                _draw_help(annotated)

            display_frame = annotated
            if now < deep_display_until and last_deep_annotated is not None:
                display_frame = last_deep_annotated or annotated
                _overlay_text(display_frame, "DEEP done", (10, 64))

            key = None
            if not no_display:
                cv2.imshow("Net Inspector - Realtime", display_frame)
                key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("h"), ord("H")):
                help_on = not help_on

            if key in (ord("m"), ord("M")) and segmenter is not None:
                seg_enabled = not seg_enabled
                state = "on" if seg_enabled else "off"
                print(f"[SEG] toggled {state}")

            if key in (ord("s"), ord("S")):
                annotated_path = _save_snapshot(
                    config, "fast", annotated, frame if save_raw else None
                )
                _log_event(
                    config,
                    mode="fast",
                    camera=camera,
                    frame=frame,
                    detections=detections,
                    annotated_path=annotated_path,
                )
                print(f"[FAST] snapshot saved: {annotated_path}")

            if key in (ord("d"), ord("D")):
                deep_detections = deep.analyze(frame)
                deep_annotated = draw_detections(frame, deep_detections)
                annotated_path = _save_snapshot(
                    config, "deep", deep_annotated, frame if save_raw else None
                )
                _log_event(
                    config,
                    mode="deep",
                    camera=camera,
                    frame=frame,
                    detections=deep_detections,
                    annotated_path=annotated_path,
                )
                deep_display_until = time.time() + 1.0
                last_deep_annotated = deep_annotated
                print(f"[DEEP] detections={len(deep_detections)} saved={annotated_path}")

            if log_all or (log_every_n > 0 and frame_idx % log_every_n == 0):
                annotated_path = None
                if save_annotated:
                    annotated_path = _save_snapshot(config, "fast", annotated, None)
                _log_event(
                    config,
                    mode="fast",
                    camera=camera,
                    frame=frame,
                    detections=detections,
                    annotated_path=annotated_path,
                )
                print(f"[FAST] {status}")
    finally:
        cap.release()
        if not no_display:
            cv2.destroyAllWindows()


def _format_status(detections) -> str:
    """Build a short status line for FAST detections.

    @param detections List of detections.
    @return Status string.
    """
    debris = sum(1 for d in detections if d.label == "debris")
    hole = sum(1 for d in detections if d.label == "tear")
    fire = sum(1 for d in detections if d.label == "fire")
    sky = sum(1 for d in detections if d.label == "sky")
    unknown = sum(1 for d in detections if d.label == "unknown")
    return f"FAST: debris={debris} hole={hole} fire={fire} sky={sky} unknown={unknown}"


def _overlay_text(image, text: str, origin: tuple[int, int]) -> None:
    """Draw outlined text on an image.

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


def _draw_help(image) -> None:
    """Draw help overlay.

    @param image Image to draw on.
    @return None
    """
    lines = [
        "Keys: q=quit  d=deep  s=save  h=help  m=seg",
    ]
    y = 86
    for line in lines:
        _overlay_text(image, line, (10, y))
        y += 18


def _save_snapshot(config: AppConfig, mode: str, annotated, raw) -> Optional[str]:
    """Save annotated (and optionally raw) snapshot.

    @param config AppConfig with output paths.
    @param mode Label prefix for filename.
    @param annotated Annotated frame.
    @param raw Raw frame (optional).
    @return Saved annotated path or None.
    """
    stamp = timestamp_id()
    annotated_path = config.outputs_annotated / f"{mode}_{stamp}.jpg"
    save_image(annotated_path, annotated)
    if raw is not None:
        raw_path = config.outputs_raw / f"{mode}_{stamp}.jpg"
        save_image(raw_path, raw)
    return str(annotated_path)


def _log_event(
    config: AppConfig,
    mode: str,
    camera: int,
    frame,
    detections,
    annotated_path: Optional[str],
) -> None:
    """Append a realtime event to events.jsonl.

    @param config AppConfig with log path.
    @param mode Event mode (fast/deep).
    @param camera Camera index.
    @param frame Input frame.
    @param detections List of detections.
    @param annotated_path Saved annotated path if any.
    @return None
    """
    height, width = frame.shape[:2]
    record = {
        "timestamp": iso_timestamp(),
        "mode": mode,
        "camera": camera,
        "resolution": {"width": int(width), "height": int(height)},
        "detections": detections_to_dicts(detections),
        "image": {"annotated": annotated_path},
    }
    log_jsonl(config.events_log_path, record)
