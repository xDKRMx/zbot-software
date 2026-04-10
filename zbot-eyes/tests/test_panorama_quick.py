"""Quick functional test for panorama stitcher pipeline."""
import sys
import tempfile
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import cv2
from net_inspector.panorama_stitcher import PanoramaStitcher
from net_inspector.config import PanoramaConfig


def make_checkerboard(h=240, w=320, cell=20):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if (y // cell + x // cell) % 2 == 0:
                img[y:y+cell, x:x+cell] = [180, 180, 180]
            else:
                img[y:y+cell, x:x+cell] = [40, 40, 40]
    noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def main():
    print("=== Full Pipeline Test: 5 overlapping frames ===")

    cfg = PanoramaConfig(enabled=True, capture_interval_s=0.1, max_frames=50)
    stitcher = PanoramaStitcher(config=cfg)
    stitcher.start()

    base_rgb = make_checkerboard()
    base_thermal = np.tile(np.linspace(30, 170, 320, dtype=np.uint8), (240, 1))

    for i in range(5):
        dx = i * 40
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        rgb_shifted = cv2.warpAffine(base_rgb, M, (320, 240))
        thermal_shifted = cv2.warpAffine(base_thermal, M, (320, 240))
        stitcher.feed_frame(rgb_shifted, thermal_shifted, time.time())
        time.sleep(0.3)

    time.sleep(2.0)

    tmp = Path(tempfile.mkdtemp())
    result = stitcher.export(tmp)
    stitcher.stop()

    if result and result.exists():
        img = cv2.imread(str(result))
        print(f"  Exported: {result.name}")
        print(f"  Image shape: {img.shape}")

        meta_files = list((tmp / "panorama").glob("*_metadata.json"))
        if meta_files:
            meta = json.loads(meta_files[0].read_text())
            print(f"  Stitched: {meta['frames_stitched']}, Skipped: {meta['frames_skipped']}")
            print(f"  Canvas: {meta['canvas_width']}x{meta['canvas_height']}")

        preview = stitcher.get_preview()
        if preview is not None:
            print(f"  Preview shape: {preview.shape}")

        print("=== FULL PIPELINE TEST PASSED ===")
    else:
        print("  EXPORT FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
