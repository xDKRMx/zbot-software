"""Unified Detection Orchestrator - GLM-powered conversational robot system.

This module integrates:
1. Wall/Net Detection (from run_headless.py)
2. Debris Detection (from run_headless.py)
3. Heat Detection (from Z-bot Heat Detection project)

All detection events are aggregated and sent to ChatGLM Vision API at controlled intervals
to generate conversational responses for the robot's audio/display output.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from net_inspector.config import AppConfig
from net_inspector.llm_glm import ChatGLMVisionClient


@dataclass
class DetectionEvent:
    """Single detection event from any subsystem."""
    
    timestamp: str
    source: str  # "wall_net", "debris", "heat"
    event_type: str  # "NET", "WALL", "DEBRIS", "FIRE", "HOTSPOT"
    confidence: float
    frame_rgb: Optional[np.ndarray] = None
    frame_thermal: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (excluding frames)."""
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "event_type": self.event_type,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class GLMRequest:
    """Aggregated request to send to GLM."""
    
    timestamp: str
    events: list[DetectionEvent]
    primary_frame_rgb: Optional[np.ndarray] = None
    primary_frame_thermal: Optional[np.ndarray] = None
    robot_position: Optional[dict[str, float]] = None
    
    def build_prompt(self) -> str:
        """Build MANDATORY conversational prompt with detection logs."""
        # Build detection log summary (critical statistical data)
        log_lines = ["=== DETECTION LOGS ==="]
        for evt in self.events:
            log_lines.append(
                f"[{evt.timestamp}] {evt.event_type}: confidence={evt.confidence:.2f}, source={evt.source}"
            )
            if evt.metadata:
                for k, v in evt.metadata.items():
                    log_lines.append(f"  └─ {k}: {v}")
        
        if self.robot_position:
            log_lines.append(f"\n[ROBOT] Position: X={self.robot_position.get('x', 0):.2f}m, "
                           f"Y={self.robot_position.get('y', 0):.2f}m, "
                           f"Heading={self.robot_position.get('heading', 0):.1f}°")
        
        log_summary = "\n".join(log_lines)
        
        # Dynamic context based on events present
        has_heat = any(e.event_type.upper() in ["FIRE", "HOTSPOT"] for e in self.events)
        has_debris = any(e.event_type.upper() == "DEBRIS" for e in self.events)
        
        focus_instructions = "注意：请综合汇报安全网的覆盖率和墙面状态。"
        if has_heat:
            focus_instructions = "🚨 紧急注意：日志中包含高温（HOTSPOT）或火情（FIRE）警报！请立即在第一句话中强调温度异常、热点或火势情况，发出严正警告！"
        elif has_debris:
            focus_instructions = "⚠️ 注意：日志中包含异物或碎片（DEBRIS）！请在汇报中明确指出检测到了异物碎片，并提醒需要注意清理！"

        # MANDATORY conversational prompt (CHINESE ONLY for China deployment)
        prompt = f"""你是Z-BOT，一个爬墙检测机器人，正在向中国的评委汇报。

检测日志（用于准确性）：
{log_summary}

图像：分析提供的画面和上述日志。

{focus_instructions}

强制规则：
1. 响应必须严格基于提供的检测日志（如热点、异物、安全网）。不要死板地重复示例！
2. 响应必须是生动的对话式，将通过扬声器大声朗读（最多3-4句话）。
3. 如果日志中有HOTSPOT（高温热点）或DEBRIS（异物碎片），必须优先汇报它们！
4. 必须引用日志中的具体检测数据（如置信度数值、检测到的事物）。
5. 必须听起来专业且充满自信。如果是火情或高温警告，语气必须严肃且紧急。
6. 不要使用markdown格式或特殊符号（纯文本，无星号或加粗，用于TTS发音）。
7. 关键：响应必须完全用流利的中文。

示例 1（常规安全网）：各位评委好！我检测到墙面12%被安全网覆盖，置信度为80%。目前结构看起来很稳定！
示例 2（发现高温热点）：各位评委，紧急报告！我在红外摄像机中检测到了高温热点，置信度高达85%，请立即排查该区域的异常发热情况！
示例 3（发现异物碎片）：各位评委好！我在巡检时发现墙面上存在异物碎片，置信度为75%，建议后续安排清理工作。

现在请根据上述真实的【检测日志】提供你的中文响应："""
        
        return prompt


class EventAggregator:
    """Aggregates detection events and prevents SS spam."""
    
    def __init__(
        self,
        min_interval_s: float = 5.0,
        max_queue_size: int = 100,
    ):
        self.min_interval_s = float(min_interval_s)
        self.max_queue_size = int(max_queue_size)
        self._queue: queue.Queue[DetectionEvent] = queue.Queue(maxsize=max_queue_size)
        self._last_glm_request_ts = 0.0
        self._lock = threading.Lock()
    
    def add_event(self, event: DetectionEvent) -> bool:
        """Add a detection event. Returns True if added, False if queue full."""
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            return False
    
    def should_send_glm_request(self) -> bool:
        """Check if enough time has passed since last GLM request.

        PANORAMA_COMPLETE events bypass the cooldown so GLM analyses the
        final heat map immediately.
        """
        # Check for priority events that bypass cooldown
        try:
            items = list(self._queue.queue)
            if any(getattr(e, "event_type", "") == "PANORAMA_COMPLETE" for e in items):
                return True
        except Exception:
            pass

        with self._lock:
            now = time.time()
            if (now - self._last_glm_request_ts) >= self.min_interval_s:
                return True
            return False
    
    def build_glm_request(self) -> Optional[GLMRequest]:
        """Build GLM request from queued events and mark request timestamp."""
        if not self.should_send_glm_request():
            return None
        
        events: list[DetectionEvent] = []
        primary_rgb = None
        primary_thermal = None
        
        # Drain queue
        while not self._queue.empty():
            try:
                evt = self._queue.get_nowait()
                events.append(evt)
                # Use most recent frame as primary
                if evt.frame_rgb is not None:
                    primary_rgb = evt.frame_rgb
                if evt.frame_thermal is not None:
                    primary_thermal = evt.frame_thermal
            except queue.Empty:
                break
        
        if not events:
            return None
        
        with self._lock:
            self._last_glm_request_ts = time.time()
        
        return GLMRequest(
            timestamp=datetime.now(timezone.utc).isoformat(),
            events=events,
            primary_frame_rgb=primary_rgb,
            primary_frame_thermal=primary_thermal,
        )


class UnifiedOrchestrator:
    """Main orchestrator that coordinates all detection systems and GLM integration."""
    
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        glm_interval_s: float = 5.0,
        output_dir: Optional[Path] = None,
        enable_audio_output: bool = False,
        current_response_file: Optional[Path] = None,
    ):
        self.config = config or AppConfig()
        self.glm_client = ChatGLMVisionClient(self.config.glm_vision)
        self.aggregator = EventAggregator(min_interval_s=glm_interval_s)
        
        if output_dir is None:
            output_dir = Path("outputs/orchestrator")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File for Raspberry Pi sensor to read current GLM response
        if current_response_file is None:
            current_response_file = Path("GLMCurrentResponse.txt")
        self.current_response_file = current_response_file
        
        self.enable_audio_output = bool(enable_audio_output)
        self._tts_lock = threading.Lock()
        
        self._running = False
        self._glm_thread: Optional[threading.Thread] = None
        self._response_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    
    def start(self) -> None:
        """Start the orchestrator background thread."""
        if self._running:
            return
        self._running = True
        self._glm_thread = threading.Thread(target=self._glm_worker, daemon=True)
        self._glm_thread.start()
        print("[ORCHESTRATOR] Started GLM worker thread.")
    
    def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        if self._glm_thread and self._glm_thread.is_alive():
            self._glm_thread.join(timeout=2.0)
        print("[ORCHESTRATOR] Stopped.")
    
    def submit_event(self, event: DetectionEvent) -> None:
        """Submit a detection event from any subsystem."""
        added = self.aggregator.add_event(event)
        if not added:
            print(f"[ORCHESTRATOR] Warning: event queue full, dropping event from {event.source}")
    
    def get_latest_response(self) -> Optional[dict[str, Any]]:
        """Get the latest GLM response (non-blocking)."""
        try:
            return self._response_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _glm_worker(self) -> None:
        """Background worker that sends aggregated events to GLM."""
        while self._running:
            try:
                glm_req = self.aggregator.build_glm_request()
                if glm_req is None:
                    time.sleep(0.5)
                    continue
                
                if not self.glm_client.available():
                    print("[ORCHESTRATOR] GLM API key not available, skipping request.")
                    time.sleep(1.0)
                    continue
                
                shared_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "shared"
                
                frame_rgb = None
                if (shared_dir / "latest_rgb.jpg").exists():
                    frame_rgb = cv2.imread(str(shared_dir / "latest_rgb.jpg"))
                    
                frame_thermal = None
                if (shared_dir / "latest_thermal.jpg").exists() and any(e.event_type == "HOTSPOT" for e in glm_req.events):
                    frame_thermal = cv2.imread(str(shared_dir / "latest_thermal.jpg"))
                
                # Select best frame (prefer thermal if hotspot detected, else RGB)
                frame_to_send = frame_thermal if frame_thermal is not None else frame_rgb
                
                if frame_to_send is None:
                    print("[ORCHESTRATOR] No frame available for GLM request, skipping.")
                    continue
                
                # Convert RGB to BGR if needed (GLM expects BGR from cv2)
                if frame_to_send.shape[2] == 3:
                    # Assume it's already BGR from cv2
                    frame_bgr = frame_to_send
                else:
                    frame_bgr = cv2.cvtColor(frame_to_send, cv2.COLOR_RGB2BGR)
                
                prompt = glm_req.build_prompt()
                
                print(f"[ORCHESTRATOR] Sending GLM request with {len(glm_req.events)} events...")
                
                try:
                    markdown_response = self.glm_client.infer_markdown(
                        frame_bgr,
                        prompt=prompt,
                    )
                except Exception as exc:
                    print(f"[ORCHESTRATOR] GLM request failed: {exc}")
                    continue
                    
                print(f"\n[ORCHESTRATOR] ================= GLM RESPONSE =================")
                print(markdown_response)
                print(f"===============================================================\n")
                
                # Package response
                response_data = {
                    "timestamp": glm_req.timestamp,
                    "events": [evt.to_dict() for evt in glm_req.events],
                    "markdown": markdown_response,
                    "robot_position": glm_req.robot_position,
                }
                
                # Save to disk
                self._save_response(response_data, frame_bgr)
                
                # Update current response file for Raspberry Pi sensor
                self._update_current_response_file(markdown_response)
                
                # Queue for retrieval
                try:
                    self._response_queue.put_nowait(response_data)
                except queue.Full:
                    # Drop oldest response
                    try:
                        self._response_queue.get_nowait()
                        self._response_queue.put_nowait(response_data)
                    except queue.Empty:
                        pass
                
                print(f"[ORCHESTRATOR] GLM response received and saved.")
                
                # Optional: trigger audio output
                if self.enable_audio_output:
                    self._trigger_audio_output(markdown_response)
                
            except Exception as exc:
                print(f"[ORCHESTRATOR] GLM worker error: {exc}")
                time.sleep(1.0)
    
    def _save_response(self, response_data: dict[str, Any], frame_bgr: np.ndarray) -> None:
        """Save GLM response and associated frame to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        response_dir = self.output_dir / f"response_{timestamp}"
        response_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = response_dir / "response.json"
        json_path.write_text(
            json.dumps(response_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        # Save markdown
        md_path = response_dir / "response.md"
        md_path.write_text(response_data["markdown"], encoding="utf-8")
        
        # Save frame
        frame_path = response_dir / "frame.jpg"
        cv2.imwrite(str(frame_path), frame_bgr)
        
        print(f"[ORCHESTRATOR] Saved response to {response_dir}")
    
    def _update_current_response_file(self, markdown: str) -> None:
        """Update GLMCurrentResponse.txt with latest response for Raspberry Pi sensor."""
        try:
            self.current_response_file.write_text(
                markdown,
                encoding="utf-8"
            )
            print(f"[ORCHESTRATOR] Updated {self.current_response_file} for sensor reading.")
        except OSError as exc:
            print(f"[ORCHESTRATOR] Failed to update current response file: {exc}")
    
    def _trigger_audio_output(self, markdown: str) -> None:
        """Trigger audio output for the latest response.
 
         Uses Edge TTS + pygame if installed. Non-blocking: if another utterance
         is already playing, this will skip to avoid overlapping audio.
         """
        if not markdown or not markdown.strip():
            return

        acquired = self._tts_lock.acquire(blocking=False)
        if not acquired:
            print("[ORCHESTRATOR] TTS busy, skipping this response.")
            return

        def _tts_worker(text: str) -> None:
            try:
                try:
                    import edge_tts  # type: ignore
                    import pygame  # type: ignore
                except Exception as exc:
                    print(f"[ORCHESTRATOR] TTS dependencies not available: {exc}")
                    return

                voice = "zh-CN-XiaoxiaoNeural"  # chinese_cute

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    mp3_path = tmp.name

                async def _render() -> None:
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(mp3_path)

                try:
                    try:
                        asyncio.run(_render())
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(_render())
                        finally:
                            loop.close()
                except Exception as exc:
                    print(f"[ORCHESTRATOR] TTS render failed: {exc}")
                    return

                try:
                    pygame.mixer.init()
                    pygame.mixer.music.load(mp3_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    pygame.mixer.music.unload()
                except Exception as exc:
                    print(f"[ORCHESTRATOR] TTS playback failed: {exc}")
                finally:
                    try:
                        Path(mp3_path).unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception as exc:
                print(f"[ORCHESTRATOR] TTS failed: {exc}")
            finally:
                try:
                    self._tts_lock.release()
                except RuntimeError:
                    pass

        threading.Thread(target=_tts_worker, args=(markdown.strip(),), daemon=True).start()
