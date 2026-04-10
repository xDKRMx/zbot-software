"""Command-line interface for net inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from net_inspector.config import AppConfig
from net_inspector.detectors.fast import FastDetector
from net_inspector.detectors.deep import DeepDetector
from net_inspector.gui import launch_gui
from net_inspector.realtime import run_realtime
from net_inspector.synth.generate import generate_batch
from net_inspector.utils.annotate import draw_detections
from net_inspector.utils.io import ensure_dir, load_image, save_image, timestamp_id
from net_inspector.utils.logging_utils import detections_to_dicts, iso_timestamp, log_jsonl


def main() -> None:
    """CLI entrypoint.

    @return None
    """
    parser = argparse.ArgumentParser(prog="net-inspector", description="Net inspection CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen-demo", help="Generate synthetic demo images")
    gen.add_argument("--out", type=Path, default=Path("data/demo"))
    gen.add_argument("--n", type=int, default=20)

    sub.add_parser("gui", help="Launch the GUI")

    analyze = sub.add_parser("analyze", help="Analyze a single image")
    analyze.add_argument("--image", type=Path, required=True)
    analyze.add_argument("--mode", choices=["fast", "deep"], default="fast")
    analyze.add_argument("--save-annotated", action="store_true")
    analyze.add_argument("--save-raw", action="store_true")

    rt = sub.add_parser("realtime", help="Run realtime detector")
    rt.add_argument("--camera", type=int, default=0)
    rt.add_argument("--mode", choices=["fast"], default="fast")
    rt.add_argument("--width", type=int, default=640)
    rt.add_argument("--height", type=int, default=480)
    rt.add_argument("--save-annotated", action="store_true")
    rt.add_argument("--save-raw", action="store_true")
    rt.add_argument("--log-all", action="store_true")
    rt.add_argument("--no-display", action="store_true")
    rt.add_argument("--log-every-n", type=int, default=5)
    rt.add_argument("--segment", action="store_true")
    rt.add_argument("--segment-weights", type=Path, default=Path("models/deeplabv3_resnet50.pth"))
    rt.add_argument("--segment-every-n", type=int, default=8)
    rt.add_argument("--segment-alpha", type=float, default=0.45)

    sub.add_parser("print-download-commands", help="Print model download commands")

    seg = sub.add_parser("segment", help="Run ML segmentation on images")
    seg.add_argument("--image", type=Path)
    seg.add_argument("--input-dir", type=Path)
    seg.add_argument("--out", type=Path, default=Path("outputs/segmentation"))
    seg.add_argument("--weights", type=Path, default=Path("models/deeplabv3_resnet50.pth"))
    seg.add_argument("--max-classes", type=int, default=5)

    args = parser.parse_args()

    if args.command == "gen-demo":
        generate_batch(args.out, n=args.n)
        print(f"Generated {args.n} images in {args.out}")
        return

    if args.command == "gui":
        launch_gui()
        return

    if args.command == "analyze":
        run_analyze(args)
        return

    if args.command == "realtime":
        run_realtime(
            camera=args.camera,
            mode=args.mode,
            width=args.width,
            height=args.height,
            save_annotated=args.save_annotated,
            save_raw=args.save_raw,
            log_all=args.log_all,
            no_display=args.no_display,
            log_every_n=args.log_every_n,
            segment=args.segment,
            segment_weights=args.segment_weights,
            segment_every_n=args.segment_every_n,
            segment_alpha=args.segment_alpha,
        )
        return

    if args.command == "print-download-commands":
        print_download_commands()
        return

    if args.command == "segment":
        run_segment(args)
        return


def run_analyze(args: argparse.Namespace) -> None:
    """Run single-image analysis with FAST or DEEP.

    @param args Parsed CLI args.
    @return None
    """
    config = AppConfig()
    image = load_image(args.image)
    if args.mode == "fast":
        detector = FastDetector(config)
    else:
        detector = DeepDetector(config)

    detections = detector.analyze(image)
    annotated = draw_detections(image, detections)

    annotated_path = None
    raw_path = None

    if args.save_annotated:
        ensure_dir(config.outputs_annotated)
        annotated_path = config.outputs_annotated / f"{args.mode}_{timestamp_id()}.jpg"
        save_image(annotated_path, annotated)

    if args.save_raw:
        ensure_dir(config.outputs_raw)
        raw_path = config.outputs_raw / f"{args.mode}_{timestamp_id()}.jpg"
        save_image(raw_path, image)

    record = {
        "timestamp": iso_timestamp(),
        "mode": args.mode,
        "detections": detections_to_dicts(detections),
        "image": {
            "input": str(args.image),
            "annotated": str(annotated_path) if annotated_path else None,
            "raw": str(raw_path) if raw_path else None,
        },
    }
    log_jsonl(config.log_path, record)

    print(f"Detections: {len(detections)}")
    for det in detections:
        print(f"- {det.label} score={det.score:.2f} bbox={det.bbox}")

    if args.save_annotated:
        cv2.imshow("Annotated", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def print_download_commands() -> None:
    """Print optional model download commands.

    @return None
    """
    commands = [
        "# Fire classifier (TFLite) - place at models/fire_classifier.tflite",
        "curl -L -o models/fire_classifier.tflite https://example.com/path/to/fire_classifier.tflite",
        "",
        "# Generic object detector (TFLite) - place at models/object_detector.tflite",
        "curl -L -o models/object_detector.tflite https://example.com/path/to/object_detector.tflite",
        "",
        "# Optional label map for object detector",
        "curl -L -o models/object_labels.txt https://example.com/path/to/object_labels.txt",
        "",
        "# Semantic segmentation weights (PyTorch) - place at models/deeplabv3_resnet50.pth",
        "curl -L -o models/deeplabv3_resnet50.pth https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
    ]
    print("\n".join(commands))


def run_segment(args: argparse.Namespace) -> None:
    """Run segmentation on one image or a folder.

    @param args Parsed CLI args.
    @return None
    """
    from net_inspector.segmenter import Segmenter, render_overlay
    from net_inspector.utils.io import ensure_dir, load_image, save_image

    if not args.image and not args.input_dir:
        raise SystemExit("Provide --image or --input-dir")

    segmenter = Segmenter(weights_path=args.weights)
    if not segmenter.available():
        raise SystemExit(
            "Segmentation model not available. Place a model at models/segmenter.tflite"
        )

    ensure_dir(args.out)

    paths = []
    if args.image:
        paths.append(args.image)
    if args.input_dir:
        for p in sorted(args.input_dir.glob("*.png")):
            paths.append(p)
        for p in sorted(args.input_dir.glob("*.jpg")):
            paths.append(p)
        for p in sorted(args.input_dir.glob("*.jpeg")):
            paths.append(p)

    for path in paths:
        image = load_image(path)
        mask = segmenter.segment(image)
        overlay = render_overlay(image, mask, segmenter.labels, max_classes=args.max_classes)
        out_path = args.out / f"seg_{path.stem}.png"
        save_image(out_path, overlay)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
