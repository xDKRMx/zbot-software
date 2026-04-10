"""Manual image stitching tester — Z-BOT Panoramic Heat Extraction

Upload images in order (left→right or top→bottom as robot moves),
stitch them, and see the result. No camera needed.

Usage:
    python stitch_test.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk
except ImportError:
    print("[ERROR] pip install Pillow")
    sys.exit(1)

from stitcher import PanoramaConfig, FeatureAligner, CanvasManager, HeatMapRenderer

# ── Layout ───────────────────────────────────────────────────────────────────
THUMB_W, THUMB_H = 120, 90
RESULT_W, RESULT_H = 900, 500


# ── Core batch stitcher (no threading needed for manual test) ─────────────────

class BatchStitcher:
    """Stitch a list of images in order using the same pipeline as live mode."""

    def __init__(self, minv: int = 40, maxv: int = 160,
                 min_inliers: int = 6, min_move_px: float = 5.0,
                 max_move_px: float = 300.0) -> None:
        cfg = PanoramaConfig()
        self._aligner = FeatureAligner(
            akaze_threshold=cfg.akaze_threshold,
            ransac_threshold=cfg.ransac_threshold,
            min_inliers=min_inliers,
        )
        self._renderer = HeatMapRenderer(minv, maxv)
        self._min_move = min_move_px
        self._max_move = max_move_px

    def stitch(self, images: list[np.ndarray],
               progress_cb=None) -> tuple[np.ndarray, dict]:
        """
        Stitch images in order. Returns (colorized_result, stats).
        images: list of BGR numpy arrays (can be RGB thermal or regular).
        progress_cb: optional callable(step, total, message).
        """
        n = len(images)
        if n == 0:
            raise ValueError("No images provided")

        # Convert each image to grayscale for thermal canvas
        grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

        h, w = grays[0].shape[:2]
        canvas_mgr = CanvasManager(h, w, padding_factor=4)
        canvas_mgr.place_first(grays[0])

        H_acc = np.eye(3, dtype=np.float64)
        stitched = [images[0]]  # keep originals for fallback matching
        H_list = [H_acc.copy()]

        stats = {
            "total": n,
            "stitched": 1,
            "skipped": 0,
            "low_confidence": 0,
            "skip_reasons": [],
        }

        if progress_cb:
            progress_cb(1, n, f"Placed frame 1/{n}")

        for i in range(1, n):
            if progress_cb:
                progress_cb(i + 1, n, f"Matching frame {i+1}/{n}…")

            H, inliers, reason = self._match(images[i - 1], images[i], i - 1, stitched)

            if H is None:
                stats["skipped"] += 1
                stats["skip_reasons"].append(f"Frame {i+1}: {reason}")
                H_list.append(None)
                if progress_cb:
                    progress_cb(i + 1, n, f"Frame {i+1} skipped — {reason}")
                continue

            # Sanity check: translation magnitude
            tx, ty = H[0, 2], H[1, 2]
            move = (tx**2 + ty**2) ** 0.5
            if move < self._min_move:
                stats["skipped"] += 1
                stats["skip_reasons"].append(
                    f"Frame {i+1}: too little movement ({move:.1f}px)")
                H_list.append(None)
                continue
            if move > self._max_move:
                stats["skipped"] += 1
                stats["skip_reasons"].append(
                    f"Frame {i+1}: too much movement ({move:.1f}px) — bad match?")
                H_list.append(None)
                continue

            if inliers < 10:
                stats["low_confidence"] += 1

            # Accumulate: H maps prev→curr, so curr_on_canvas = H_acc @ inv(H)
            H_acc = H_acc @ np.linalg.inv(H)
            H_list.append(H_acc.copy())

            canvas_mgr.expand_if_needed()
            ok = canvas_mgr.warp_and_blend(grays[i], H_acc)
            if ok:
                stats["stitched"] += 1
                stitched.append(images[i])
            else:
                stats["skipped"] += 1
                stats["skip_reasons"].append(f"Frame {i+1}: empty warp")

        cropped = canvas_mgr.get_cropped()
        colored = self._renderer.render(cropped)
        with_bar = self._renderer.add_scale_bar(colored)
        return with_bar, stats

    def _match(self, prev: np.ndarray, curr: np.ndarray,
               prev_idx: int, all_prev: list) -> tuple:
        """Try matching curr against prev, then fallback to earlier frames."""
        H, inliers = self._aligner.compute_homography(prev, curr)
        if H is not None:
            return H, inliers, None

        # Fallback: try 2 frames back
        if prev_idx >= 1:
            H, inliers = self._aligner.compute_homography(all_prev[-2], curr)
            if H is not None:
                return H, inliers, "used fallback frame"

        return None, 0, f"no match (inliers too low)"


# ── GUI ───────────────────────────────────────────────────────────────────────

class StitchTestGUI:
    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title("Z-BOT — Manual Stitch Tester")
        self._root.resizable(True, True)
        self._root.configure(bg="#1e1e2e")

        self._images: list[np.ndarray] = []   # BGR originals
        self._paths: list[str] = []
        self._result: Optional[np.ndarray] = None

        self._minv_var = tk.IntVar(value=40)
        self._maxv_var = tk.IntVar(value=160)
        self._min_move_var = tk.DoubleVar(value=5.0)
        self._max_move_var = tk.DoubleVar(value=300.0)
        self._min_inliers_var = tk.IntVar(value=6)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self._root, bg="#313244", pady=6)
        top.pack(fill="x")

        tk.Label(top, text="Z-BOT  Manual Stitch Tester",
                 font=("Helvetica", 12, "bold"),
                 bg="#313244", fg="#cdd6f4").pack(side="left", padx=12)

        self._btn_stitch = tk.Button(
            top, text="⚙  Stitch", width=10,
            bg="#a6e3a1", fg="#1e1e2e", font=("Helvetica", 9, "bold"),
            relief="flat", cursor="hand2", state="disabled",
            command=self._run_stitch)
        self._btn_stitch.pack(side="right", padx=4)

        self._btn_save = tk.Button(
            top, text="💾  Save", width=10,
            bg="#89b4fa", fg="#1e1e2e", font=("Helvetica", 9, "bold"),
            relief="flat", cursor="hand2", state="disabled",
            command=self._save_result)
        self._btn_save.pack(side="right", padx=4)

        self._btn_clear = tk.Button(
            top, text="🗑  Clear", width=10,
            bg="#f38ba8", fg="#1e1e2e", font=("Helvetica", 9, "bold"),
            relief="flat", cursor="hand2", command=self._clear)
        self._btn_clear.pack(side="right", padx=4)

        self._btn_add = tk.Button(
            top, text="➕  Add Images", width=12,
            bg="#cba6f7", fg="#1e1e2e", font=("Helvetica", 9, "bold"),
            relief="flat", cursor="hand2", command=self._add_images)
        self._btn_add.pack(side="right", padx=4)

        # ── Main area ─────────────────────────────────────────────────────────
        main = tk.Frame(self._root, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=8, pady=8)

        # Left: image list
        left = tk.Frame(main, bg="#1e1e2e", width=160)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        tk.Label(left, text="Images (in order)",
                 bg="#1e1e2e", fg="#89b4fa",
                 font=("Helvetica", 9, "bold")).pack(anchor="w")

        list_frame = tk.Frame(left, bg="#11111b")
        list_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self._listbox = tk.Listbox(
            list_frame, bg="#11111b", fg="#cdd6f4",
            selectbackground="#45475a", font=("Helvetica", 8),
            yscrollcommand=scrollbar.set, activestyle="none")
        self._listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self._listbox.yview)

        # Reorder buttons
        reorder = tk.Frame(left, bg="#1e1e2e")
        reorder.pack(fill="x", pady=2)
        tk.Button(reorder, text="↑", width=4, bg="#45475a", fg="#cdd6f4",
                  relief="flat", command=self._move_up).pack(side="left", padx=2)
        tk.Button(reorder, text="↓", width=4, bg="#45475a", fg="#cdd6f4",
                  relief="flat", command=self._move_down).pack(side="left", padx=2)
        tk.Button(reorder, text="✕", width=4, bg="#f38ba8", fg="#1e1e2e",
                  relief="flat", command=self._remove_selected).pack(side="left", padx=2)

        # Middle: params
        mid = tk.Frame(main, bg="#1e1e2e", width=180)
        mid.pack(side="left", fill="y", padx=(0, 8))
        mid.pack_propagate(False)

        tk.Label(mid, text="Parameters",
                 bg="#1e1e2e", fg="#89b4fa",
                 font=("Helvetica", 9, "bold")).pack(anchor="w")

        params = [
            ("Thermal min (cold)", self._minv_var, 0, 255),
            ("Thermal max (hot)", self._maxv_var, 0, 255),
            ("Min move (px)", self._min_move_var, 0, 50),
            ("Max move (px)", self._max_move_var, 50, 500),
            ("Min inliers", self._min_inliers_var, 3, 50),
        ]
        for label, var, lo, hi in params:
            f = tk.Frame(mid, bg="#1e1e2e")
            f.pack(fill="x", pady=3)
            tk.Label(f, text=label, bg="#1e1e2e", fg="#a6adc8",
                     font=("Helvetica", 8), anchor="w").pack(fill="x")
            tk.Scale(f, variable=var, from_=lo, to=hi,
                     orient="horizontal", bg="#1e1e2e", fg="#cdd6f4",
                     highlightthickness=0, troughcolor="#313244",
                     length=160).pack(fill="x")

        # Right: result
        right = tk.Frame(main, bg="#1e1e2e")
        right.pack(side="left", fill="both", expand=True)

        tk.Label(right, text="Stitched Result",
                 bg="#1e1e2e", fg="#f38ba8",
                 font=("Helvetica", 9, "bold")).pack(anchor="w")

        self._result_label = tk.Label(right, bg="#11111b",
                                      width=RESULT_W, height=RESULT_H)
        self._result_label.pack(fill="both", expand=True)
        self._show_placeholder()

        # ── Status bar ────────────────────────────────────────────────────────
        sb = tk.Frame(self._root, bg="#313244", pady=3)
        sb.pack(fill="x")

        self._status_var = tk.StringVar(value="Add images to begin")
        tk.Label(sb, textvariable=self._status_var,
                 bg="#313244", fg="#a6adc8",
                 font=("Helvetica", 9), anchor="w").pack(side="left", padx=10)

        self._progress_var = tk.StringVar(value="")
        tk.Label(sb, textvariable=self._progress_var,
                 bg="#313244", fg="#a6e3a1",
                 font=("Helvetica", 9), anchor="e").pack(side="right", padx=10)

    def _show_placeholder(self) -> None:
        img = np.zeros((RESULT_H, RESULT_W, 3), dtype=np.uint8)
        img[:] = (17, 17, 27)
        cv2.putText(img, "Add images and press Stitch",
                    (RESULT_W // 2 - 160, RESULT_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (88, 91, 112), 1)
        self._set_result_image(img)

    # ── Image management ──────────────────────────────────────────────────────

    def _add_images(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select images (in traversal order)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                       ("All files", "*.*")])
        if not paths:
            return

        added = 0
        for p in sorted(paths):  # sort by filename for natural order
            img = cv2.imread(p)
            if img is None:
                continue
            self._images.append(img)
            self._paths.append(p)
            self._listbox.insert("end", Path(p).name)
            added += 1

        self._status_var.set(f"{len(self._images)} images loaded")
        if len(self._images) >= 2:
            self._btn_stitch.config(state="normal")

    def _move_up(self) -> None:
        sel = self._listbox.curselection()
        if not sel or sel[0] == 0:
            return
        i = sel[0]
        self._images[i-1], self._images[i] = self._images[i], self._images[i-1]
        self._paths[i-1], self._paths[i] = self._paths[i], self._paths[i-1]
        name = self._listbox.get(i)
        self._listbox.delete(i)
        self._listbox.insert(i-1, name)
        self._listbox.selection_set(i-1)

    def _move_down(self) -> None:
        sel = self._listbox.curselection()
        if not sel or sel[0] >= len(self._images) - 1:
            return
        i = sel[0]
        self._images[i], self._images[i+1] = self._images[i+1], self._images[i]
        self._paths[i], self._paths[i+1] = self._paths[i+1], self._paths[i]
        name = self._listbox.get(i)
        self._listbox.delete(i)
        self._listbox.insert(i+1, name)
        self._listbox.selection_set(i+1)

    def _remove_selected(self) -> None:
        sel = self._listbox.curselection()
        if not sel:
            return
        i = sel[0]
        self._images.pop(i)
        self._paths.pop(i)
        self._listbox.delete(i)
        if len(self._images) < 2:
            self._btn_stitch.config(state="disabled")
        self._status_var.set(f"{len(self._images)} images loaded")

    def _clear(self) -> None:
        self._images.clear()
        self._paths.clear()
        self._listbox.delete(0, "end")
        self._result = None
        self._btn_stitch.config(state="disabled")
        self._btn_save.config(state="disabled")
        self._status_var.set("Cleared")
        self._progress_var.set("")
        self._show_placeholder()

    # ── Stitching ─────────────────────────────────────────────────────────────

    def _run_stitch(self) -> None:
        if len(self._images) < 2:
            messagebox.showwarning("Not enough images", "Add at least 2 images.")
            return

        self._btn_stitch.config(state="disabled")
        self._btn_save.config(state="disabled")
        self._status_var.set("Stitching…")

        def _worker():
            stitcher = BatchStitcher(
                minv=self._minv_var.get(),
                maxv=self._maxv_var.get(),
                min_inliers=self._min_inliers_var.get(),
                min_move_px=self._min_move_var.get(),
                max_move_px=self._max_move_var.get(),
            )

            def _progress(step, total, msg):
                self._root.after(0, lambda: self._progress_var.set(
                    f"{step}/{total}  {msg}"))

            try:
                result, stats = stitcher.stitch(self._images, progress_cb=_progress)
                self._result = result
                self._root.after(0, lambda: self._on_stitch_done(result, stats))
            except Exception as e:
                self._root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self._root.after(0, lambda: self._btn_stitch.config(state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_stitch_done(self, result: np.ndarray, stats: dict) -> None:
        self._set_result_image(result)
        self._btn_stitch.config(state="normal")
        self._btn_save.config(state="normal")

        skips = "\n".join(stats["skip_reasons"]) if stats["skip_reasons"] else "none"
        summary = (f"Done — stitched {stats['stitched']}/{stats['total']}  |  "
                   f"skipped {stats['skipped']}  |  "
                   f"low-confidence {stats['low_confidence']}")
        self._status_var.set(summary)
        self._progress_var.set("")

        if stats["skip_reasons"]:
            messagebox.showinfo("Stitch complete",
                                f"{summary}\n\nSkipped frames:\n{skips}")

    def _set_result_image(self, img_bgr: np.ndarray) -> None:
        h, w = img_bgr.shape[:2]
        # Fit into result panel
        scale = min(RESULT_W / max(w, 1), RESULT_H / max(h, 1))
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((RESULT_H, RESULT_W, 3), dtype=np.uint8)
        canvas[:] = (17, 17, 27)
        y0 = (RESULT_H - nh) // 2
        x0 = (RESULT_W - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized

        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        tk_img = ImageTk.PhotoImage(pil)
        self._result_label.config(image=tk_img)
        self._result_label.image = tk_img

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_result(self) -> None:
        if self._result is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            initialfile="stitched_result.png")
        if path:
            cv2.imwrite(path, self._result)
            self._status_var.set(f"Saved → {Path(path).name}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    StitchTestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
