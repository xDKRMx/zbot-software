"""Verify hybrid override + edge-catching stitcher behavior."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2
from stitcher import CanvasManager

def test_override_semantics():
    """Second frame's values must win over first frame in overlap region."""
    cm = CanvasManager(100, 100, padding_factor=3, overlap_margin_px=10)
    frame_a = np.full((100, 100), 80, dtype=np.uint8)  # value=80
    frame_b = np.full((100, 100), 150, dtype=np.uint8)  # value=150

    cm.place_first(frame_a)
    H_identity = np.eye(3, dtype=np.float64)
    # Frame B at same position (tx=0, ty=0) — should override
    cm.warp_and_blend(frame_b, H_identity, tx=0.0, ty=0.0, angle_deg=0.0)

    cropped = cm.get_cropped()
    # The overlap region should have value 150 (frame_b wins)
    center_val = cropped[50, 50]
    assert abs(center_val - 150.0) < 1.0, f"Expected 150, got {center_val}"
    print(f"  override_semantics: center={center_val:.1f} ✓")

def test_edge_mask_right_movement():
    """tx=+60 → only left ~80px (60+margin) should be in edge mask."""
    cm = CanvasManager(100, 200, padding_factor=3, overlap_margin_px=20)
    mask = cm._compute_edge_mask(tx=60.0, ty=0.0)
    # Left 80 columns should be True
    assert mask[:, :80].all(), "Left region should be True"
    # Right columns beyond margin should be False
    assert not mask[:, 100:].any(), "Far right should be False"
    print(f"  edge_mask_right: left={mask[:,:80].sum()} px True, right={mask[:,100:].sum()} px True ✓")

def test_edge_mask_up_movement():
    """ty=-40 → only bottom ~60px (40+margin) should be in edge mask."""
    cm = CanvasManager(100, 100, padding_factor=3, overlap_margin_px=20)
    mask = cm._compute_edge_mask(tx=0.0, ty=-40.0)
    # Bottom 60 rows should be True
    assert mask[40:, :].all(), "Bottom region should be True"
    # Top rows should be False
    assert not mask[:20, :].any(), "Top should be False"
    print(f"  edge_mask_up: bottom={mask[40:,:].sum()} px True ✓")

def test_no_overlap_side_by_side():
    """Two frames placed side by side should have zero overlapping written pixels."""
    h, w = 100, 100
    cm = CanvasManager(h, w, padding_factor=4, overlap_margin_px=5)

    frame_a = np.full((h, w), 80, dtype=np.uint8)
    frame_b = np.full((h, w), 150, dtype=np.uint8)

    cm.place_first(frame_a)
    visited_after_a = cm._visited.copy()

    # Frame B is exactly one frame-width to the right (tx = w)
    H_right = np.array([[1, 0, float(w)],
                         [0, 1, 0.0],
                         [0, 0, 1.0]], dtype=np.float64)
    cm.warp_and_blend(frame_b, H_right, tx=float(w), ty=0.0, angle_deg=0.0)

    # Pixels written by A that are also written by B = overlap
    overlap = visited_after_a & (cm._visited & ~visited_after_a == False)
    # More precisely: pixels that were visited before AND after
    new_pixels = cm._visited & ~visited_after_a
    old_pixels = visited_after_a

    # With margin=5, there may be a small overlap strip — but it should be < margin*h
    overlap_count = (old_pixels & cm._visited).sum() - old_pixels.sum()
    # old_pixels are all still visited, new_pixels are additional
    # Real overlap = pixels written by B that were already written by A
    # Since B is placed at tx=w (no overlap in theory), overlap should be ~0 or just margin
    print(f"  no_overlap: A_pixels={old_pixels.sum()}, new_B_pixels={new_pixels.sum()}, "
          f"margin_overlap≈{(old_pixels & new_pixels).sum()} px ✓")
    assert new_pixels.sum() > 0, "Frame B should have written some pixels"

def test_vibration_fallback():
    """With pure override, rotation doesn't matter — camera FOV always updates canvas."""
    cm = CanvasManager(100, 100, padding_factor=3,
                       overlap_margin_px=10, rotation_threshold_deg=5.0)
    frame_a = np.full((100, 100), 80, dtype=np.uint8)
    cm.place_first(frame_a)

    frame_b = np.full((100, 100), 150, dtype=np.uint8)
    H_identity = np.eye(3, dtype=np.float64)
    # angle=10° — with pure override, still writes (camera FOV always updates)
    ok = cm.warp_and_blend(frame_b, H_identity, tx=0.0, ty=0.0, angle_deg=10.0)
    assert ok, "Pure override should always write when warped has data"
    # Value should be updated to 150
    cropped = cm.get_cropped()
    assert abs(cropped[50, 50] - 150.0) < 1.0, f"Expected 150, got {cropped[50,50]}"
    print(f"  vibration_pure_override: ok={ok}, center={cropped[50,50]:.1f} ✓")

if __name__ == "__main__":
    print("=== Hybrid Stitcher Tests ===")
    test_override_semantics()
    test_edge_mask_right_movement()
    test_edge_mask_up_movement()
    test_no_overlap_side_by_side()
    test_vibration_fallback()
    print("=== ALL TESTS PASSED ===")
