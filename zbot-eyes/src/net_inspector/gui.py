"""Tkinter GUI focused on segmentation and live camera preview."""

from __future__ import annotations

import threading
import time
from typing import Optional
import colorsys
import re

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, ttk, messagebox
import subprocess
import sys
import io
import os
import shutil

from net_inspector.config import AppConfig
from net_inspector.llm_glm import ChatGLMVisionClient
from net_inspector.reporting import IncidentStore
from net_inspector.segmenter import Segmenter, render_overlay
from net_inspector.synth.generate import generate_demo_image
from net_inspector.utils.io import ensure_dir, save_image, timestamp_id


class NetInspectorGUI:
    """Segmentation-first GUI with optional live camera."""

    _INLINE_TOKEN_RE = re.compile(
        r"(`[^`]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_|!\[[^\]]*\]\([^)]+\)|\[[^\]]+\]\([^)]+\))"
    )

    def __init__(self) -> None:
        self.config = AppConfig()
        self.segmenter = Segmenter()
        self.glm_client = ChatGLMVisionClient(self.config.glm_vision)
        self.incident_store = IncidentStore(
            output_dir=self.config.outputs_incidents,
            model_version="net-inspector/gui-v1",
        )

        self.root = tk.Tk()
        self.root.title("Net Inspector")
        self.root.minsize(1040, 720)

        self.current_image: Optional[np.ndarray] = None
        self.current_thermal_image: Optional[np.ndarray] = None
        self.segmented_image: Optional[np.ndarray] = None

        self._camera_thread: Optional[threading.Thread] = None
        self._segment_thread: Optional[threading.Thread] = None
        self._camera_running = False
        self._stop_event = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_seg: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_id = 0
        self._seg_frame_id = -1
        self._seg_fps = 0.0
        self._seg_last_time = time.time()
        self._seg_count = 0
        self._fps = 0.0
        self._fps_last_time = time.time()
        self._fps_count = 0
        self._llm_thread: Optional[threading.Thread] = None
        self._llm_busy = False
        self._md_fonts: list[tkfont.Font] = []
        self._incident_ids: list[str] = []
        self._selected_incident_id: Optional[str] = None

        self._apply_style()
        self._build_ui()
        self._start_ui_loop()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_style(self) -> None:
        self.root.configure(bg="#f7f8fa")
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TFrame", background="#f7f8fa")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure(
            "Header.TLabel",
            background="#f7f8fa",
            foreground="#0f172a",
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "SubHeader.TLabel",
            background="#f7f8fa",
            foreground="#64748b",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Badge.TLabel",
            background="#eef2ff",
            foreground="#3730a3",
            font=("Segoe UI", 9, "bold"),
            padding=(8, 2),
        )
        style.configure(
            "TLabel",
            background="#ffffff",
            foreground="#0f172a",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Status.TLabel",
            background="#ffffff",
            foreground="#475569",
            font=("Segoe UI", 9),
        )
        style.configure(
            "HsvValue.TLabel",
            background="#f1f5f9",
            foreground="#0f172a",
            font=("Segoe UI", 9, "bold"),
            padding=(6, 2),
        )
        style.configure(
            "TLabelframe",
            background="#ffffff",
            foreground="#0f172a",
            bordercolor="#e2e8f0",
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "TLabelframe.Label",
            background="#ffffff",
            foreground="#334155",
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "TButton",
            background="#2563eb",
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            padding=(10, 6),
        )
        style.map(
            "TButton",
            background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")],
            foreground=[("disabled", "#f8fafc")],
        )
        style.configure(
            "TCheckbutton",
            background="#ffffff",
            foreground="#0f172a",
            font=("Segoe UI", 9),
            padding=(4, 2),
        )
        style.map("TCheckbutton", foreground=[("disabled", "#94a3b8")])
        style.configure(
            "Hsv.Horizontal.TScale",
            troughcolor="#e2e8f0",
            background="#2563eb",
            sliderlength=14,
        )

    def _build_ui(self) -> None:
        """Build the Tkinter layout.

        @return None
        """
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        self.var_debris = tk.BooleanVar(value=True)
        self.var_tear = tk.BooleanVar(value=True)
        self.var_fire = tk.BooleanVar(value=True)
        self.var_sky = tk.BooleanVar(value=False)
        self.var_net = tk.BooleanVar(value=True)
        self.var_debris_detect = tk.BooleanVar(value=True)
        self.var_fire_detect = tk.BooleanVar(value=True)
        self.var_legend = tk.BooleanVar(value=True)
        self.var_camera = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value=self._segment_status_text())
        self._extra_status = ""
        self.report_var = tk.StringVar(value="Debris: 0.0%")
        self.glm_model_var = tk.StringVar(value=self.config.glm_vision.model)
        self.glm_prompt_var = tk.StringVar(value=self.config.glm_vision.default_prompt)
        self.glm_status_var = tk.StringVar(value=self._glm_status_text())
        self.incident_status_var = tk.StringVar(value="Incidents: 0")
        self.incident_distance_var = tk.StringVar(value="")
        self.incident_position_var = tk.StringVar(value="")
        self.incident_pose_var = tk.StringVar(value="")
        self.incident_conf_var = tk.DoubleVar(value=0.65)

        self.h_min_var = tk.IntVar(value=self.config.heuristic.green_h_min)
        self.h_max_var = tk.IntVar(value=self.config.heuristic.green_h_max)
        self.s_min_var = tk.IntVar(value=self.config.heuristic.green_s_min)
        self.v_min_var = tk.IntVar(value=self.config.heuristic.green_v_min)

        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 12))
        header_left = ttk.Frame(header)
        header_left.pack(side=tk.LEFT, anchor=tk.W)
        title_row = ttk.Frame(header_left)
        title_row.pack(side=tk.TOP, anchor=tk.W)
        ttk.Label(title_row, text="Net Inspector", style="Header.TLabel").pack(
            side=tk.LEFT, anchor=tk.W
        )
        ttk.Label(title_row, text="SEGMENTATION", style="Badge.TLabel").pack(
            side=tk.LEFT, padx=10
        )
        ttk.Label(
            header_left,
            text="Live camera preview with net overlay and fast segmentation refresh.",
            style="SubHeader.TLabel",
        ).pack(side=tk.TOP, anchor=tk.W, pady=(2, 0))

        controls = ttk.Frame(main, style="Card.TFrame", padding=12)
        controls.pack(fill=tk.X)

        options = ttk.LabelFrame(controls, text="Synthetic elements", padding=8)
        options.pack(side=tk.LEFT, padx=(0, 12), pady=4)
        ttk.Checkbutton(options, text="Debris", variable=self.var_debris).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(options, text="Tear", variable=self.var_tear).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(options, text="Fire", variable=self.var_fire).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(options, text="Sky", variable=self.var_sky).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(options, text="Net overlay", variable=self.var_net).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Checkbutton(options, text="Legend", variable=self.var_legend).pack(
            side=tk.LEFT, padx=4
        )

        actions = ttk.LabelFrame(controls, text="Actions", padding=8)
        actions.pack(side=tk.LEFT, padx=(0, 12), pady=4)
        ttk.Button(actions, text="Generate demo image", command=self._on_generate).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(actions, text="Load image", command=self._on_load).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(actions, text="Paste image", command=self._on_paste).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(actions, text="Run Segmentation", command=self._on_segment).pack(
            side=tk.LEFT, padx=4
        )

        camera = ttk.LabelFrame(controls, text="Live camera", padding=8)
        camera.pack(side=tk.LEFT, padx=(0, 12), pady=4)
        ttk.Checkbutton(
            camera, text="Enable", variable=self.var_camera, command=self._toggle_camera
        ).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            camera, text="Debris", variable=self.var_debris_detect
        ).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            camera, text="Fire", variable=self.var_fire_detect
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(camera, text="Stop", command=self._stop_camera).pack(side=tk.LEFT, padx=4)
        ttk.Label(camera, textvariable=self.status_var, style="Status.TLabel").pack(
            side=tk.LEFT, padx=8
        )

        content = ttk.Frame(main, padding=0)
        content.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        sidebar = ttk.Frame(content)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        sliders = ttk.LabelFrame(sidebar, text="Net HSV", padding=8)
        sliders.pack(fill=tk.X, pady=(0, 8))
        self._add_slider(sliders, "H min", self.h_min_var, 0, 179)
        self._add_slider(sliders, "H max", self.h_max_var, 0, 179)
        self._add_slider(sliders, "S min", self.s_min_var, 0, 255)
        self._add_slider(sliders, "V min", self.v_min_var, 0, 255)

        preview = ttk.Frame(sliders)
        preview.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(preview, text="Preview", style="Status.TLabel").pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self._hsv_lower_swatch = tk.Label(
            preview, width=4, height=2, bg="#22c55e", relief="solid", bd=1
        )
        self._hsv_lower_swatch.pack(side=tk.LEFT, padx=(0, 6))
        self._hsv_lower_text = ttk.Label(preview, text="", style="Status.TLabel")
        self._hsv_lower_text.pack(side=tk.LEFT, padx=(0, 8))
        self._hsv_upper_swatch = tk.Label(
            preview, width=4, height=2, bg="#a7f3d0", relief="solid", bd=1
        )
        self._hsv_upper_swatch.pack(side=tk.LEFT, padx=(0, 6))
        self._hsv_upper_text = ttk.Label(preview, text="", style="Status.TLabel")
        self._hsv_upper_text.pack(side=tk.LEFT)
        self._hsv_gradient = tk.Canvas(
            sliders, width=180, height=10, highlightthickness=0, bg="#ffffff"
        )
        self._hsv_gradient.pack(fill=tk.X, pady=(2, 0))

        self.h_min_var.trace_add("write", self._on_hsv_change)
        self.h_max_var.trace_add("write", self._on_hsv_change)
        self.s_min_var.trace_add("write", self._on_hsv_change)
        self.v_min_var.trace_add("write", self._on_hsv_change)
        self._draw_hsv_gradient()
        self._update_hsv_preview()

        legend_box = ttk.LabelFrame(sidebar, text="Segmentation legend", padding=8)
        legend_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(legend_box, text="Show legend", variable=self.var_legend).pack(
            side=tk.LEFT, padx=4
        )

        report_box = ttk.LabelFrame(sidebar, text="Debris report", padding=8)
        report_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(report_box, textvariable=self.report_var, style="Status.TLabel").pack(
            side=tk.LEFT, padx=4
        )

        images = ttk.Frame(content, padding=0)
        images.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        input_frame = ttk.LabelFrame(images, text="Input", padding=8)
        input_frame.pack(side=tk.LEFT, padx=(0, 12), fill=tk.BOTH, expand=True)
        output_frame = ttk.LabelFrame(images, text="Segmentation output", padding=8)
        output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        llm_frame = ttk.LabelFrame(images, text="ChatGLM Vision (Markdown)", padding=8)
        llm_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.input_label = ttk.Label(
            input_frame, text="No image loaded", anchor=tk.CENTER, justify=tk.CENTER
        )
        self.input_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.output_label = ttk.Label(
            output_frame,
            text="Run segmentation to see overlay",
            anchor=tk.CENTER,
            justify=tk.CENTER,
        )
        self.output_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        llm_toolbar = ttk.Frame(llm_frame)
        llm_toolbar.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(
            llm_toolbar, text="Analyze current frame", command=self._on_glm_analyze
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(llm_toolbar, text="Clear", command=self._on_glm_clear).pack(side=tk.LEFT)
        ttk.Label(llm_toolbar, text="Model", style="Status.TLabel").pack(
            side=tk.LEFT, padx=(10, 4)
        )
        self.glm_model_combo = ttk.Combobox(
            llm_toolbar,
            width=18,
            textvariable=self.glm_model_var,
            values=list(self.config.glm_vision.models),
            state="normal",
        )
        self.glm_model_combo.pack(side=tk.LEFT, padx=(0, 6))
        self.glm_model_combo.bind("<<ComboboxSelected>>", self._on_glm_model_change)
        ttk.Label(
            llm_toolbar, textvariable=self.glm_status_var, style="Status.TLabel"
        ).pack(side=tk.RIGHT)

        prompt_row = ttk.Frame(llm_frame)
        prompt_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(prompt_row, text="Prompt", style="Status.TLabel").pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Entry(prompt_row, textvariable=self.glm_prompt_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        incident_row = ttk.LabelFrame(llm_frame, text="Incident capture", padding=6)
        incident_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(
            incident_row, text="Load thermal image", command=self._on_load_thermal
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            incident_row, text="Create incident", command=self._on_create_incident
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            incident_row, text="Export selected", command=self._on_export_incident
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(
            incident_row, textvariable=self.incident_status_var, style="Status.TLabel"
        ).pack(side=tk.RIGHT)

        incident_meta = ttk.Frame(llm_frame)
        incident_meta.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(incident_meta, text="Distance(m)", style="Status.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Entry(
            incident_meta, width=7, textvariable=self.incident_distance_var
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(incident_meta, text="Position", style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Entry(
            incident_meta, width=16, textvariable=self.incident_position_var
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(incident_meta, text="Pose", style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Entry(incident_meta, width=16, textvariable=self.incident_pose_var).pack(
            side=tk.LEFT, padx=(4, 8)
        )
        ttk.Label(incident_meta, text="Conf", style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Scale(
            incident_meta,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.incident_conf_var,
            length=90,
            style="Hsv.Horizontal.TScale",
        ).pack(side=tk.LEFT, padx=(4, 0))

        incident_queue = ttk.LabelFrame(llm_frame, text="Incident queue", padding=6)
        incident_queue.pack(fill=tk.X, pady=(0, 6))
        queue_body = ttk.Frame(incident_queue)
        queue_body.pack(fill=tk.X)
        queue_scroll = ttk.Scrollbar(queue_body, orient=tk.VERTICAL)
        queue_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.incident_listbox = tk.Listbox(
            queue_body,
            height=5,
            yscrollcommand=queue_scroll.set,
            exportselection=False,
        )
        self.incident_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.incident_listbox.bind("<<ListboxSelect>>", self._on_incident_select)
        queue_scroll.configure(command=self.incident_listbox.yview)

        triage_row = ttk.Frame(incident_queue)
        triage_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            triage_row,
            text="Accept",
            command=lambda: self._on_set_disposition("accepted"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            triage_row,
            text="Needs review",
            command=lambda: self._on_set_disposition("needs_review"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            triage_row,
            text="Reject",
            command=lambda: self._on_set_disposition("rejected"),
        ).pack(side=tk.LEFT)

        markdown_container = ttk.Frame(llm_frame)
        markdown_container.pack(fill=tk.BOTH, expand=True)
        markdown_scroll = ttk.Scrollbar(markdown_container, orient=tk.VERTICAL)
        markdown_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.markdown_text = tk.Text(
            markdown_container,
            wrap=tk.WORD,
            yscrollcommand=markdown_scroll.set,
            bg="#ffffff",
            fg="#0f172a",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.markdown_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        markdown_scroll.configure(command=self.markdown_text.yview)
        self._setup_markdown_tags()
        self._set_markdown(
            "# ChatGLM Vision\n\n"
            "Click **Analyze current frame** to send the latest image.\n\n"
            "Use incident controls to create/export grounded reports.\n\n"
            "API key sources:\n"
            "- `CHATGLM_API_KEY`\n"
            f"- `{self.config.glm_vision.api_key_file}`"
        )
        self._refresh_incident_queue()

    def _segment_status_text(self) -> str:
        """Return segmentation status string with FPS.

        @return Status string.
        """
        ready = "Ready" if self.segmenter.available() else "Missing weights"
        return f"Segmentation: {ready} | {self._seg_fps:.1f} FPS"

    def _camera_status_text(self) -> str:
        """Return camera status string with FPS.

        @return Status string.
        """
        cam_state = "On" if self._camera_running else "Off"
        fps_part = f" | {self._fps:.1f} FPS" if self._camera_running else ""
        return f"Camera: {cam_state}{fps_part}"

    def _glm_status_text(self) -> str:
        """Return ChatGLM status string."""
        model = self.glm_model_var.get().strip() if hasattr(self, "glm_model_var") else ""
        if not model:
            model = self.config.glm_vision.model
        if self._llm_busy:
            return f"GLM: Running ({model})..."
        if self.glm_client.available():
            return f"GLM: Ready ({model})"
        return "GLM: API key missing"

    def _on_generate(self) -> None:
        """Generate a synthetic demo image and show it.

        @return None
        """
        self._stop_camera()
        result = generate_demo_image(
            add_debris=self.var_debris.get(),
            add_tear=self.var_tear.get(),
            add_fire=self.var_fire.get(),
            add_sky=self.var_sky.get(),
        )
        self.current_image = result.image
        self.segmented_image = None
        self._set_image(self.input_label, self.current_image)
        self._clear_output()

    def _on_load(self) -> None:
        """Load an image from disk and show it.

        @return None
        """
        self._stop_camera()
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")],
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "Failed to load image")
            return
        self.current_image = image
        self.segmented_image = None
        self._set_image(self.input_label, self.current_image)
        self._clear_output()

    def _on_paste(self) -> None:
        """Paste image from clipboard and show it.

        @return None
        """
        self._stop_camera()
        image = self._load_clipboard_image()
        if image is None:
            messagebox.showwarning(
                "Clipboard",
                "No image found in clipboard. Install xclip (X11) or wl-clipboard (Wayland), "
                "or copy an image file path and try again.",
            )
            return
        self.current_image = image
        self.segmented_image = None
        self._set_image(self.input_label, self.current_image)
        self._clear_output()

    def _on_segment(self) -> None:
        """Run segmentation on the current image.

        @return None
        """
        if self.current_image is None:
            messagebox.showwarning("No image", "Load or generate an image first.")
            return
        if not self.segmenter.available():
            messagebox.showerror(
                "Segmentation unavailable",
                "PyTorch segmentation weights not found. Place models/deeplabv3_resnet50.pth",
            )
            return
        mask = self.segmenter.segment(self.current_image)
        overlay = render_overlay(
            self.current_image,
            mask,
            self.segmenter.labels,
            show_legend=self.var_legend.get(),
        )
        overlay, stats = self._apply_extra_overlays(overlay, self.current_image)
        self._extra_status = stats
        self.segmented_image = overlay
        self._set_image(self.output_label, self.segmented_image)
        self._save_segmentation(self.segmented_image)

    def _on_glm_analyze(self) -> None:
        """Analyze current frame/image with ChatGLM vision."""
        if self._llm_busy:
            return
        image = self._select_glm_image()
        if image is None:
            messagebox.showwarning("No image", "Load image or enable camera before GLM analysis.")
            return
        if not self.glm_client.available():
            self.glm_status_var.set(self._glm_status_text())
            messagebox.showwarning(
                "Missing API key",
                "Set CHATGLM_API_KEY or create secrets/chatglm_api_key.txt first.",
            )
            return

        prompt = self.glm_prompt_var.get().strip()
        if self._selected_incident_id:
            report = self.incident_store.get(self._selected_incident_id)
            if report is not None:
                evidence_ids = [item.evidence_id for item in report.evidence]
                if evidence_ids:
                    prompt = (
                        f"{prompt}\n\n"
                        "Grounding constraint: every sentence must include one evidence tag in "
                        "the format [EVID:<id>]. Use only these IDs: "
                        f"{', '.join(evidence_ids)}."
                    )
        model = self.glm_model_var.get().strip() or self.config.glm_vision.model
        self._llm_busy = True
        self.glm_status_var.set(self._glm_status_text())
        self._set_markdown("## Running ChatGLM request\n\nPlease wait...")
        self._llm_thread = threading.Thread(
            target=self._glm_worker, args=(image, prompt, model), daemon=True
        )
        self._llm_thread.start()

    def _on_glm_model_change(self, _event=None) -> None:
        """Refresh status when model selector changes."""
        self.glm_status_var.set(self._glm_status_text())

    def _on_glm_clear(self) -> None:
        """Clear markdown pane."""
        self._set_markdown("")
        self.glm_status_var.set(self._glm_status_text())

    def _select_glm_image(self) -> Optional[np.ndarray]:
        """Select best available frame for LLM analysis."""
        with self._frame_lock:
            if self._camera_running and self._latest_frame is not None:
                return self._latest_frame.copy()
        if self.current_image is not None:
            return self.current_image.copy()
        if self.segmented_image is not None:
            return self.segmented_image.copy()
        with self._frame_lock:
            if self._latest_seg is not None:
                return self._latest_seg.copy()
        return None

    def _glm_worker(self, image: np.ndarray, prompt: str, model: str) -> None:
        try:
            markdown = self.glm_client.infer_markdown(image, prompt, model=model)
        except Exception as exc:
            self.root.after(0, lambda: self._on_glm_error(str(exc)))
            return
        self.root.after(0, lambda: self._on_glm_result(markdown))

    def _on_glm_result(self, markdown: str) -> None:
        self._llm_busy = False
        self.glm_status_var.set(self._glm_status_text())
        if self._selected_incident_id:
            try:
                errors = self.incident_store.attach_llm_summary(
                    self._selected_incident_id, markdown
                )
            except Exception as exc:
                self.glm_status_var.set("GLM: Attach failed")
                self._set_markdown(f"## Failed to attach summary\n\n```\n{exc}\n```")
                return
            if errors:
                self.glm_status_var.set("GLM: Ungrounded response")
                preview = "\n".join(f"- {err}" for err in errors[:8])
                self._set_markdown(
                    "## ChatGLM response rejected (grounding check)\n\n"
                    "Each sentence must include valid `[EVID:...]` references.\n\n"
                    f"{markdown}\n\nValidation errors:\n{preview}"
                )
                return
            self.glm_status_var.set("GLM: Attached to incident")
            report = self.incident_store.get(self._selected_incident_id)
            if report is not None:
                self._set_markdown(self.incident_store.to_markdown(report))
            return
        self._set_markdown(markdown)

    def _on_glm_error(self, error_text: str) -> None:
        self._llm_busy = False
        self.glm_status_var.set("GLM: Request failed")
        self._set_markdown(f"## ChatGLM request failed\n\n```\n{error_text}\n```")

    def _on_load_thermal(self) -> None:
        """Load a thermal/IR image to pair with new incidents."""
        path = filedialog.askopenfilename(
            title="Select thermal image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")],
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "Failed to load thermal image")
            return
        self.current_thermal_image = image
        self.incident_status_var.set("Thermal image loaded")

    def _on_create_incident(self) -> None:
        """Create a grounded incident from the current capture context."""
        rgb_image = self._select_glm_image()
        if rgb_image is None:
            messagebox.showwarning("No image", "Load image or enable camera first.")
            return
        distance = self._parse_optional_float(self.incident_distance_var.get())
        position_hint = self.incident_position_var.get().strip()
        robot_pose = self.incident_pose_var.get().strip()
        confidence = float(max(0.0, min(1.0, self.incident_conf_var.get())))

        report = self.incident_store.create_incident(
            rgb_frame=rgb_image,
            thermal_frame=self.current_thermal_image.copy()
            if self.current_thermal_image is not None
            else None,
            label="wall_anomaly",
            confidence=confidence,
            severity="medium",
            estimated_distance_m=distance,
            position_hint=position_hint,
            robot_pose=robot_pose,
            metadata={
                "camera_running": self._camera_running,
                "segmentation_fps": round(self._seg_fps, 2),
                "camera_fps": round(self._fps, 2),
            },
        )
        self._refresh_incident_queue(select_incident_id=report.incident_id)
        self._set_markdown(self.incident_store.to_markdown(report))
        self.incident_status_var.set(f"Incident created: {report.incident_id}")

    def _on_incident_select(self, _event=None) -> None:
        selection = self.incident_listbox.curselection()
        if not selection:
            self._selected_incident_id = None
            return
        idx = int(selection[0])
        if idx < 0 or idx >= len(self._incident_ids):
            self._selected_incident_id = None
            return
        incident_id = self._incident_ids[idx]
        self._selected_incident_id = incident_id
        report = self.incident_store.get(incident_id)
        if report is not None:
            self._set_markdown(self.incident_store.to_markdown(report))

    def _on_set_disposition(self, disposition: str) -> None:
        """Set disposition for selected incident."""
        if not self._selected_incident_id:
            messagebox.showinfo("No selection", "Select an incident first.")
            return
        self.incident_store.set_disposition(self._selected_incident_id, disposition)
        report = self.incident_store.get(self._selected_incident_id)
        if report is not None:
            self._set_markdown(self.incident_store.to_markdown(report))
        self._refresh_incident_queue(select_incident_id=self._selected_incident_id)

    def _on_export_incident(self) -> None:
        """Export selected incident as JSON + markdown."""
        if not self._selected_incident_id:
            messagebox.showinfo("No selection", "Select an incident first.")
            return
        try:
            json_path, md_path = self.incident_store.export_incident(
                self._selected_incident_id
            )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.incident_status_var.set(f"Exported: {self._selected_incident_id}")
        messagebox.showinfo(
            "Export complete",
            f"JSON: {json_path}\nMarkdown: {md_path}",
        )

    def _refresh_incident_queue(self, select_incident_id: Optional[str] = None) -> None:
        reports = self.incident_store.all()
        self.incident_listbox.delete(0, tk.END)
        self._incident_ids = []
        for report in reports:
            score = report.risk_score()
            suffix = report.incident_id.replace("INC_", "")[-12:]
            label = (
                f"{suffix} | {report.operator_disposition} | "
                f"risk {score:.2f} | evid {len(report.evidence)}"
            )
            self._incident_ids.append(report.incident_id)
            self.incident_listbox.insert(tk.END, label)

        self.incident_status_var.set(f"Incidents: {len(reports)}")
        target_id = select_incident_id or self._selected_incident_id
        if target_id and target_id in self._incident_ids:
            idx = self._incident_ids.index(target_id)
            self.incident_listbox.selection_clear(0, tk.END)
            self.incident_listbox.selection_set(idx)
            self.incident_listbox.activate(idx)
            self._selected_incident_id = target_id
        elif not reports:
            self._selected_incident_id = None

    def _parse_optional_float(self, value: str) -> Optional[float]:
        raw = value.strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _toggle_camera(self) -> None:
        """Toggle live camera on/off.

        @return None
        """
        if self.var_camera.get():
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self) -> None:
        """Start camera and segmentation threads.

        @return None
        """
        if self._camera_running:
            return
        self._camera_running = True
        self._stop_event.clear()
        self._fps = 0.0
        self._fps_last_time = time.time()
        self._fps_count = 0
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        self._segment_thread = threading.Thread(target=self._segment_loop, daemon=True)
        self._segment_thread.start()

    def _stop_camera(self) -> None:
        """Stop camera and segmentation threads.

        @return None
        """
        self._camera_running = False
        self._stop_event.set()
        self.var_camera.set(False)
        self._fps = 0.0
        self._seg_fps = 0.0
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=1.0)
        if self._segment_thread and self._segment_thread.is_alive():
            self._segment_thread.join(timeout=1.0)
        with self._frame_lock:
            self._latest_frame = None
            self._latest_seg = None
            self._seg_frame_id = -1

    def _camera_loop(self) -> None:
        """Camera capture loop (runs on its own thread).

        @return None
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._camera_running = False
            self.root.after(
                0, lambda: messagebox.showerror("Camera error", "Unable to open camera 0")
            )
            return
        while self._camera_running and not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_id += 1
            self._fps_count += 1
            now = time.time()
            if now - self._fps_last_time >= 1.0:
                self._fps = self._fps_count / max(0.001, now - self._fps_last_time)
                self._fps_last_time = now
                self._fps_count = 0
            time.sleep(0.001)
        cap.release()

    def _segment_loop(self) -> None:
        """Segmentation loop (runs on its own thread).

        @return None
        """
        while self._camera_running and not self._stop_event.is_set():
            if not self.segmenter.available():
                time.sleep(0.1)
                continue
            with self._frame_lock:
                if self._latest_frame is None or self._seg_frame_id == self._frame_id:
                    frame = None
                else:
                    frame = self._latest_frame.copy()
                    self._seg_frame_id = self._frame_id
            if frame is None:
                time.sleep(0.01)
                continue
            mask = self.segmenter.segment(frame)
            overlay = render_overlay(
                frame,
                mask,
                self.segmenter.labels,
                show_legend=self.var_legend.get(),
            )
            overlay, stats = self._apply_extra_overlays(overlay, frame)
            self._extra_status = stats
            with self._frame_lock:
                self._latest_seg = overlay
            self._seg_count += 1
            if self._seg_count % 5 == 0:
                now = time.time()
                elapsed = now - self._seg_last_time
                if elapsed > 0:
                    self._seg_fps = 5.0 / elapsed
                self._seg_last_time = now

    def _start_ui_loop(self) -> None:
        """Periodic UI refresh loop.

        @return None
        """
        base = self._segment_status_text()
        camera = self._camera_status_text()
        extra = f" | {self._extra_status}" if self._extra_status else ""
        self.status_var.set(f"{base} | {camera}{extra}")
        if self._camera_running:
            if self._latest_frame is not None:
                self._set_image(self.input_label, self._latest_frame)
            if self._latest_seg is not None:
                self._set_image(self.output_label, self._latest_seg)
        else:
            if self.current_image is not None:
                self._set_image(self.input_label, self.current_image)
            if self.segmented_image is not None:
                self._set_image(self.output_label, self.segmented_image)
        self.root.after(50, self._start_ui_loop)

    def _set_image(self, label: ttk.Label, image_bgr: np.ndarray) -> None:
        """Render an image into a Tk label.

        @param label Tk label to update.
        @param image_bgr Image in BGR.
        @return None
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image_rgb)
        pil.thumbnail((540, 420))
        tk_image = ImageTk.PhotoImage(pil)
        label.configure(image=tk_image, text="")
        label.image = tk_image

    def _clear_output(self) -> None:
        """Clear the output panel.

        @return None
        """
        self.output_label.configure(image="", text="Run segmentation to see overlay")
        self.output_label.image = None

    def _save_segmentation(self, overlay: np.ndarray) -> None:
        """Save segmentation overlay to outputs/annotated.

        @param overlay Segmentation overlay image.
        @return None
        """
        ensure_dir(self.config.outputs_annotated)
        stamp = timestamp_id()
        out_path = self.config.outputs_annotated / f"seg_{stamp}.jpg"
        save_image(out_path, overlay)

    def _load_clipboard_image(self) -> Optional[np.ndarray]:
        """Load image from clipboard (Linux/macOS).

        @return Image in BGR or None.
        """
        # Try PIL ImageGrab (works on Windows/macOS, sometimes on Linux).
        try:
            from PIL import ImageGrab

            grabbed = ImageGrab.grabclipboard()
            if grabbed is not None:
                if isinstance(grabbed, Image.Image):
                    return cv2.cvtColor(np.array(grabbed), cv2.COLOR_RGB2BGR)
                if isinstance(grabbed, list):
                    for item in grabbed:
                        if isinstance(item, str) and os.path.exists(item):
                            image = cv2.imread(item)
                            if image is not None:
                                return image
        except Exception:
            pass

        # Try Tk clipboard first (may only give text on some platforms).
        try:
            data = self.root.clipboard_get()
            if isinstance(data, str):
                # Clipboard might contain a file path.
                candidate = data.strip().strip("\"'")
                if os.path.exists(candidate):
                    image = cv2.imread(candidate)
                    if image is not None:
                        return image
        except Exception:
            pass

        # Linux: Wayland clipboard via wl-paste
        if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY"):
            if shutil.which("wl-paste"):
                try:
                    proc = subprocess.run(
                        ["bash", "-lc", "wl-paste --type image/png"],
                        capture_output=True,
                        check=True,
                    )
                    if proc.stdout:
                        image = Image.open(io.BytesIO(proc.stdout))
                        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                except Exception:
                    pass

        # Linux: use xclip if available (X11)
        if sys.platform.startswith("linux") and shutil.which("xclip"):
            try:
                proc = subprocess.run(
                    ["bash", "-lc", "xclip -selection clipboard -t image/png -o"],
                    capture_output=True,
                    check=True,
                )
                if proc.stdout:
                    image = Image.open(io.BytesIO(proc.stdout))
                    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception:
                pass

        # Linux: use xsel if available (X11)
        if sys.platform.startswith("linux") and shutil.which("xsel"):
            try:
                proc = subprocess.run(
                    ["bash", "-lc", "xsel --clipboard --output --output"],
                    capture_output=True,
                    check=True,
                )
                if proc.stdout:
                    image = Image.open(io.BytesIO(proc.stdout))
                    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception:
                pass

        # macOS: use pngpaste if available
        if sys.platform == "darwin":
            try:
                proc = subprocess.run(
                    ["bash", "-lc", "pngpaste -"],
                    capture_output=True,
                    check=True,
                )
                if proc.stdout:
                    image = Image.open(io.BytesIO(proc.stdout))
                    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception:
                pass

        return None

    def _apply_extra_overlays(
        self, overlay: np.ndarray, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, str]:
        """Apply net/debris/fire overlays with conservative confidence.

        @param overlay Current overlay image.
        @param image_bgr Source image (BGR).
        @return (overlay image, status text).
        """
        status_parts = []

        if self.var_net.get():
            net_mask = self._compute_net_mask(image_bgr)
            overlay = self._overlay_mask(overlay, net_mask, (0, 255, 0), 0.25)
            net_ratio = float(np.count_nonzero(net_mask)) / float(net_mask.size)
            status_parts.append(f"Net {net_ratio*100:.1f}%")

        if self.var_debris_detect.get():
            net_mask = self._compute_net_mask(image_bgr)
            debris_mask = self._compute_debris_mask(image_bgr, net_mask)
            overlay = self._overlay_mask(overlay, debris_mask, (0, 140, 255), 0.25)
            debris_ratio = float(np.count_nonzero(debris_mask)) / float(debris_mask.size)
            status_parts.append(f"Debris {debris_ratio*100:.1f}% (low)")
            self.report_var.set(f"Debris: {debris_ratio*100:.1f}% (low)")
        else:
            self.report_var.set("Debris: off")

        if self.var_fire_detect.get():
            fire_mask = self._compute_fire_mask(image_bgr)
            overlay = self._overlay_mask(overlay, fire_mask, (0, 0, 255), 0.25)
            fire_ratio = float(np.count_nonzero(fire_mask)) / float(fire_mask.size)
            status_parts.append(f"Fire {fire_ratio*100:.1f}% (low)")

        return overlay, " | ".join(status_parts)

    def _compute_debris_mask(self, image_bgr: np.ndarray, net_mask: np.ndarray) -> np.ndarray:
        """Compute a conservative debris mask within net regions.

        @param image_bgr Input image (BGR).
        @param net_mask Net mask.
        @return Binary debris mask.
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        cfg = self.config.heuristic
        green = cv2.inRange(
            hsv,
            (cfg.green_h_min, cfg.green_s_min, cfg.green_v_min),
            (cfg.green_h_max, 255, 255),
        )
        non_green = cv2.bitwise_not(green)
        debris = cv2.bitwise_and(non_green, net_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        debris = cv2.morphologyEx(debris, cv2.MORPH_OPEN, kernel, iterations=1)
        return debris

    def _compute_fire_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Compute a conservative fire mask (low confidence).

        @param image_bgr Input image (BGR).
        @return Binary fire mask.
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        fire1 = cv2.inRange(hsv, (0, 140, 140), (25, 255, 255))
        fire2 = cv2.inRange(hsv, (160, 140, 140), (179, 255, 255))
        fire = cv2.bitwise_or(fire1, fire2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fire = cv2.morphologyEx(fire, cv2.MORPH_OPEN, kernel, iterations=1)
        return fire

    def _overlay_mask(self, image_bgr: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
        """Overlay a binary mask on an image.

        @param image_bgr Input image (BGR).
        @param mask Binary mask.
        @param color BGR color.
        @param alpha Overlay alpha.
        @return Image with overlay.
        """
        if mask is None or mask.size == 0:
            return image_bgr
        overlay = image_bgr.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)

    def _compute_net_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Compute a simple net mask using green HSV thresholds.

        @param image_bgr Input image (BGR).
        @return Binary net mask.
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower = (self.h_min_var.get(), self.s_min_var.get(), self.v_min_var.get())
        upper = (self.h_max_var.get(), 255, 255)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _add_slider(
        self,
        parent: ttk.LabelFrame,
        label: str,
        var: tk.IntVar,
        minimum: int,
        maximum: int,
    ) -> None:
        """Add a labeled slider row.

        @param parent Parent frame.
        @param label Label text.
        @param var IntVar bound to slider.
        @param minimum Slider min.
        @param maximum Slider max.
        @return None
        """
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, style="Status.TLabel").pack(side=tk.LEFT)
        value_var = tk.StringVar(value=str(var.get()))
        value_label = ttk.Label(row, textvariable=value_var, style="HsvValue.TLabel")
        value_label.pack(side=tk.RIGHT, padx=(6, 0))

        def _sync_value(*_args) -> None:
            value_var.set(str(var.get()))

        def _on_scale(value: str) -> None:
            var.set(int(float(value)))
            value_var.set(str(var.get()))

        scale = ttk.Scale(
            row,
            from_=minimum,
            to=maximum,
            orient=tk.HORIZONTAL,
            variable=var,
            command=_on_scale,
            style="Hsv.Horizontal.TScale",
        )
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=6)
        var.trace_add("write", _sync_value)

    def _on_hsv_change(self, *_args) -> None:
        h_min = self.h_min_var.get()
        h_max = self.h_max_var.get()
        if h_min > h_max:
            self.h_max_var.set(h_min)
            return
        self._update_hsv_preview()

    def _hsv_to_hex(self, h: int, s: int, v: int) -> str:
        h_norm = max(0.0, min(1.0, h / 179.0))
        s_norm = max(0.0, min(1.0, s / 255.0))
        v_norm = max(0.0, min(1.0, v / 255.0))
        r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def _hsv_to_rgb(self, h: int, s: int, v: int) -> tuple[int, int, int]:
        h_norm = max(0.0, min(1.0, h / 179.0))
        s_norm = max(0.0, min(1.0, s / 255.0))
        v_norm = max(0.0, min(1.0, v / 255.0))
        r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
        return int(r * 255), int(g * 255), int(b * 255)

    def _draw_hsv_gradient(self) -> None:
        self._hsv_gradient.delete("all")
        width = int(self._hsv_gradient["width"])
        for x in range(width):
            h = int(179 * x / max(1, width - 1))
            color = self._hsv_to_hex(h, 255, 255)
            self._hsv_gradient.create_line(x, 0, x, 10, fill=color)

    def _update_hsv_preview(self) -> None:
        h_min = self.h_min_var.get()
        h_max = self.h_max_var.get()
        s_min = self.s_min_var.get()
        v_min = self.v_min_var.get()
        lower_color = self._hsv_to_hex(h_min, s_min, v_min)
        upper_color = self._hsv_to_hex(h_max, 255, 255)
        self._hsv_lower_swatch.configure(bg=lower_color)
        self._hsv_upper_swatch.configure(bg=upper_color)
        lower_rgb = self._hsv_to_rgb(h_min, s_min, v_min)
        upper_rgb = self._hsv_to_rgb(h_max, 255, 255)
        self._hsv_lower_text.configure(
            text=(
                f"H:{h_min} S:{s_min} V:{v_min}\n"
                f"{lower_color.upper()}\n"
                f"RGB{lower_rgb}"
            )
        )
        self._hsv_upper_text.configure(
            text=(
                f"H:{h_max} S:255 V:255\n"
                f"{upper_color.upper()}\n"
                f"RGB{upper_rgb}"
            )
        )

        self._hsv_gradient.delete("marker")
        width = int(self._hsv_gradient["width"])
        def _x_from_h(h: int) -> int:
            return int((h / 179.0) * max(1, width - 1))

        x_min = _x_from_h(h_min)
        x_max = _x_from_h(h_max)
        self._hsv_gradient.create_line(x_min, 0, x_min, 10, fill="#0f172a", tags="marker")
        self._hsv_gradient.create_line(x_max, 0, x_max, 10, fill="#0f172a", tags="marker")

    def _setup_markdown_tags(self) -> None:
        """Configure text tags used for markdown rendering."""
        base = tkfont.nametofont("TkDefaultFont")
        body = base.copy()
        code = tkfont.Font(family="Courier", size=max(9, body.cget("size")))
        h1 = body.copy()
        h2 = body.copy()
        h3 = body.copy()
        h1.configure(size=15, weight="bold")
        h2.configure(size=13, weight="bold")
        h3.configure(size=12, weight="bold")
        bold = body.copy()
        bold.configure(weight="bold")
        italic = body.copy()
        italic.configure(slant="italic")
        self._md_fonts = [body, code, h1, h2, h3, bold, italic]

        self.markdown_text.tag_configure("md_body", font=body, spacing3=3)
        self.markdown_text.tag_configure("md_h1", font=h1, spacing1=8, spacing3=4)
        self.markdown_text.tag_configure("md_h2", font=h2, spacing1=6, spacing3=3)
        self.markdown_text.tag_configure("md_h3", font=h3, spacing1=5, spacing3=3)
        self.markdown_text.tag_configure("md_bold", font=bold)
        self.markdown_text.tag_configure("md_italic", font=italic)
        self.markdown_text.tag_configure(
            "md_code_inline",
            font=code,
            background="#f1f5f9",
            foreground="#0f172a",
        )
        self.markdown_text.tag_configure(
            "md_code_block",
            font=code,
            background="#f8fafc",
            foreground="#0f172a",
            lmargin1=12,
            lmargin2=12,
            spacing1=2,
            spacing3=4,
        )
        self.markdown_text.tag_configure(
            "md_quote",
            foreground="#475569",
            lmargin1=12,
            lmargin2=18,
            spacing1=2,
        )

    def _set_markdown(self, markdown: str) -> None:
        """Render markdown text into the LLM pane."""
        self.markdown_text.configure(state=tk.NORMAL)
        self.markdown_text.delete("1.0", tk.END)
        self._render_markdown(markdown)
        self.markdown_text.configure(state=tk.DISABLED)

    def _render_markdown(self, markdown: str) -> None:
        in_code_block = False
        for raw in markdown.splitlines():
            line = raw.rstrip("\r")
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                self.markdown_text.insert(tk.END, f"{line}\n", ("md_code_block",))
                continue

            if not line.strip():
                self.markdown_text.insert(tk.END, "\n", ("md_body",))
                continue

            head = re.match(r"^(#{1,6})\s+(.*)$", line)
            if head:
                level = min(len(head.group(1)), 3)
                self._insert_inline_markdown(head.group(2), (f"md_h{level}",))
                self.markdown_text.insert(tk.END, "\n", ("md_body",))
                continue

            quote = re.match(r"^\s*>\s?(.*)$", line)
            if quote:
                self.markdown_text.insert(tk.END, "│ ", ("md_quote",))
                self._insert_inline_markdown(quote.group(1), ("md_quote", "md_body"))
                self.markdown_text.insert(tk.END, "\n", ("md_quote",))
                continue

            bullet = re.match(r"^\s*[-*+]\s+(.*)$", line)
            if bullet:
                self.markdown_text.insert(tk.END, "• ", ("md_body",))
                self._insert_inline_markdown(bullet.group(1), ("md_body",))
                self.markdown_text.insert(tk.END, "\n", ("md_body",))
                continue

            numbered = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
            if numbered:
                self.markdown_text.insert(tk.END, f"{numbered.group(1)}. ", ("md_body",))
                self._insert_inline_markdown(numbered.group(2), ("md_body",))
                self.markdown_text.insert(tk.END, "\n", ("md_body",))
                continue

            self._insert_inline_markdown(line, ("md_body",))
            self.markdown_text.insert(tk.END, "\n", ("md_body",))

    def _insert_inline_markdown(self, text: str, base_tags: tuple[str, ...]) -> None:
        pos = 0
        for match in self._INLINE_TOKEN_RE.finditer(text):
            if match.start() > pos:
                self.markdown_text.insert(tk.END, text[pos : match.start()], base_tags)
            token = match.group(0)
            rendered, tags = self._render_inline_token(token, base_tags)
            self.markdown_text.insert(tk.END, rendered, tags)
            pos = match.end()
        if pos < len(text):
            self.markdown_text.insert(tk.END, text[pos:], base_tags)

    def _render_inline_token(
        self, token: str, base_tags: tuple[str, ...]
    ) -> tuple[str, tuple[str, ...]]:
        if token.startswith("`") and token.endswith("`") and len(token) >= 2:
            return token[1:-1], base_tags + ("md_code_inline",)
        if token.startswith("**") and token.endswith("**") and len(token) >= 4:
            return token[2:-2], base_tags + ("md_bold",)
        if token.startswith("__") and token.endswith("__") and len(token) >= 4:
            return token[2:-2], base_tags + ("md_bold",)
        if token.startswith("*") and token.endswith("*") and len(token) >= 2:
            return token[1:-1], base_tags + ("md_italic",)
        if token.startswith("_") and token.endswith("_") and len(token) >= 2:
            return token[1:-1], base_tags + ("md_italic",)

        image_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", token)
        if image_match:
            alt = image_match.group(1) or "image"
            return f"[image: {alt}]({image_match.group(2)})", base_tags

        link_match = re.match(r"^\[([^\]]+)\]\(([^)]+)\)$", token)
        if link_match:
            return f"{link_match.group(1)} ({link_match.group(2)})", base_tags

        return token, base_tags

    def _on_close(self) -> None:
        """Handle window close event.

        @return None
        """
        self._stop_camera()
        self.root.quit()
        self.root.destroy()


def launch_gui() -> None:
    """Launch the Tkinter GUI.

    @return None
    """
    app = NetInspectorGUI()
    app.root.mainloop()


if __name__ == "__main__":
    launch_gui()
