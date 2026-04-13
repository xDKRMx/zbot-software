"""Configuration objects and paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"
SECRETS_DIR = ROOT_DIR / "secrets"


@dataclass(frozen=True)
class FireClassifierConfig:
    """Configuration for optional fire classifier TFLite model.

    @field input_size Input width/height.
    @field input_channels Number of channels.
    @field positive_index Index for positive class.
    @field score_threshold Minimum score to trigger.
    @field normalize_0_1 Whether to normalize to [0,1].
    """

    input_size: int = 224
    input_channels: int = 3
    positive_index: int = 1
    score_threshold: float = 0.5
    normalize_0_1: bool = True


@dataclass(frozen=True)
class ObjectDetectorConfig:
    """Configuration for optional generic object detector TFLite model.

    @field score_threshold Minimum detection score.
    @field labels_path Optional labels file path.
    """

    score_threshold: float = 0.5
    labels_path: Path = MODELS_DIR / "object_labels.txt"


@dataclass(frozen=True)
class HeuristicConfig:
    """Thresholds for heuristic detection.

    @field green_h_min HSV lower bound for green.
    @field green_h_max HSV upper bound for green.
    @field green_s_min HSV saturation min.
    @field green_v_min HSV value min.
    @field debris_min_area Minimum debris area.
    @field tear_min_area Minimum tear area.
    @field fire_min_area Minimum fire area.
    @field smoke_s_max Max saturation for smoke.
    @field smoke_v_min Min value for smoke.
    @field sky_h_min HSV lower bound for sky.
    @field sky_h_max HSV upper bound for sky.
    @field sky_s_min HSV saturation min for sky.
    @field sky_v_min HSV value min for sky.
    @field sky_min_area_ratio Min sky area ratio.
    @field unknown_min_area_ratio Min unknown area ratio.
    @field tear_green_ratio_max Max green ratio inside tear bbox.
    """

    green_h_min: int = 35
    green_h_max: int = 85
    green_s_min: int = 60
    green_v_min: int = 40

    debris_min_area: int = 120
    tear_min_area: int = 300
    fire_min_area: int = 80
    tear_green_ratio_max: float = 0.22

    smoke_s_max: int = 40
    smoke_v_min: int = 120

    sky_h_min: int = 85
    sky_h_max: int = 135
    sky_s_min: int = 40
    sky_v_min: int = 120

    sky_min_area_ratio: float = 0.08
    unknown_min_area_ratio: float = 0.06


@dataclass(frozen=True)
class GLMVisionConfig:
    """Configuration for ChatGLM vision API usage.

    @field endpoint Chat completion endpoint.
    @field model ChatGLM vision-capable model name.
    @field models Suggested model list for GUI selector.
    @field api_key_file Text file that stores API key.
    @field timeout_s Request timeout in seconds.
    @field system_prompt System-level guardrails for grounded reporting.
    @field default_prompt Default user prompt for visual analysis.
    """

    endpoint: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    model: str = "glm-4v-flash"
    models: tuple[str, ...] = (
        "glm-4v-flash",
        "glm-4v-plus",
    )
    api_key_file: Path = SECRETS_DIR / "chatglm_api_key.txt"
    timeout_s: float = 45.0
    system_prompt: str = (
        "You are NetInspector-Report, a conservative vision analyst for wall-climbing robot "
        "inspection images. You may receive RGB images or thermal/infrared images, including "
        "low-resolution infrared frames.\n"
        "Rules:\n"
        "1) Ground every claim in visible evidence from the provided frame.\n"
        "2) Never invent entities, identities, causes, or scenarios that are not clearly visible.\n"
        "3) For thermal/IR images without calibrated scale or legend, do not state absolute "
        "temperature values.\n"
        "4) If resolution/noise limits interpretation, state this explicitly.\n"
        "5) Separate observed facts from inferences; keep inferences conservative and conditional.\n"
        "6) Use confidence language strictly: high='detected', medium='possible', "
        "low='unverified indication'.\n"
        "7) If user provides allowed evidence IDs, include at least one [EVID:<id>] tag in each "
        "sentence under Observed/Inferred/Risks and use only those IDs. If no IDs are provided, "
        "use [EVID:frame].\n"
        "8) Do not provide medical/legal conclusions.\n"
        "9) Keep output concise and operational.\n"
        "Output markdown sections exactly:\n"
        "## Summary\n"
        "## Observed (Grounded)\n"
        "## Inferred (Conservative)\n"
        "## Risks (Conditional)\n"
        "## Suggested Next Actions\n"
        "## Unknowns / Not Verifiable"
    )
    default_prompt: str = (
        "Analyze this inspection frame and produce a concise grounded report. "
        "If this appears to be infrared/thermal and low-resolution, explicitly include uncertainty "
        "and avoid unsupported object identification."
    )


@dataclass(frozen=True)
class PanoramaConfig:
    """Configuration for thermal panorama stitching.

    @field enabled Enable panorama mode.
    @field capture_interval_s Seconds between frame captures.
    @field max_frames Maximum frames in buffer.
    @field live_preview Show live preview window.
    @field duration_s Auto-stop after N seconds (0=unlimited).
    @field canvas_padding_factor Canvas size multiplier vs frame size.
    @field akaze_threshold AKAZE detector threshold.
    @field ransac_threshold RANSAC reprojection threshold pixels.
    @field min_inliers Minimum inliers for valid homography.
    @field blend_width Feathering width in pixels.
    @field bundle_adjust_interval Frames between drift correction passes.
    @field thermal_minv Fixed cold threshold (matches run_thermal.py MINV).
    @field thermal_maxv Fixed hot threshold (matches run_thermal.py MAXV).
    @field rgb_thermal_dx Pixel offset X between RGB and thermal cameras.
    @field rgb_thermal_dy Pixel offset Y between RGB and thermal cameras.
    """

    enabled: bool = False
    capture_interval_s: float = 0.5
    max_frames: int = 500
    live_preview: bool = False
    duration_s: float = 0.0
    canvas_padding_factor: int = 3
    akaze_threshold: float = 0.001
    ransac_threshold: float = 5.0
    min_inliers: int = 10
    blend_width: int = 32
    bundle_adjust_interval: int = 20
    thermal_minv: int = 40
    thermal_maxv: int = 160
    rgb_thermal_dx: int = 0
    rgb_thermal_dy: int = 0
    # Hybrid stitching — override + edge-catching
    overlap_margin_px: int = 20
    rotation_threshold_deg: float = 5.0
    max_move_px: float = 150.0
    min_move_px: float = 3.0


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration.

    @field heuristic HeuristicConfig.
    @field fire_classifier FireClassifierConfig.
    @field object_detector ObjectDetectorConfig.
    @field glm_vision GLMVisionConfig.
    @field log_path Default JSONL log path.
    @field events_log_path Realtime JSONL log path.
    @field outputs_annotated Annotated output directory.
    @field outputs_raw Raw output directory.
    @field outputs_incidents Incident report output directory.
    """

    heuristic: HeuristicConfig = HeuristicConfig()
    fire_classifier: FireClassifierConfig = FireClassifierConfig()
    object_detector: ObjectDetectorConfig = ObjectDetectorConfig()
    glm_vision: GLMVisionConfig = field(default_factory=GLMVisionConfig)
    panorama: PanoramaConfig = PanoramaConfig()

    log_path: Path = LOG_DIR / "detections.jsonl"
    events_log_path: Path = LOG_DIR / "events.jsonl"
    outputs_annotated: Path = OUTPUT_DIR / "annotated"
    outputs_raw: Path = OUTPUT_DIR / "raw"
    outputs_incidents: Path = OUTPUT_DIR / "incidents"
