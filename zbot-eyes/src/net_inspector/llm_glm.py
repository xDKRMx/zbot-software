"""ChatGLM vision client for markdown analysis output."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env from current working directory if present

    try:
        project_root = Path(__file__).resolve().parents[3]
        dotenv_path = project_root / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
    except Exception:
        pass
except ImportError:
    pass  # python-dotenv not installed, skip

from net_inspector.config import GLMVisionConfig


class ChatGLMVisionClient:
    """Minimal ChatGLM vision wrapper using OpenAI-compatible chat API."""

    def __init__(self, config: GLMVisionConfig) -> None:
        self.config = config

    def available(self) -> bool:
        """Return whether an API key is currently available."""
        return bool(self._read_api_key())

    def infer_markdown(
        self,
        image_bgr: np.ndarray,
        prompt: str | None = None,
        model: str | None = None,
    ) -> str:
        """Analyze an image and return markdown content."""
        api_key = self._read_api_key()
        if not api_key:
            raise RuntimeError(
                "Missing ChatGLM API key. Set CHATGLM_API_KEY or create secrets/chatglm_api_key.txt."
            )

        image_data_url = self._to_image_data_url(image_bgr)
        request_prompt = prompt.strip() if prompt and prompt.strip() else self.config.default_prompt
        request_model = model.strip() if model and model.strip() else self.config.model

        payload = {
            "model": request_model,
            "messages": [
                {
                    "role": "system",
                    "content": self.config.system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "stream": False,
        }

        req = Request(
            self.config.endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=self.config.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ChatGLM request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"ChatGLM network error: {exc.reason}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"ChatGLM returned non-JSON response: {raw[:240]}") from exc
        return self._extract_message_text(data)

    def _read_api_key(self) -> str:
        self._ensure_secrets_layout()

        env_key = os.environ.get("CHATGLM_API_KEY", "").strip()
        if env_key:
            return env_key
        key_file = Path(self.config.api_key_file)
        if key_file.exists():
            try:
                key = key_file.read_text(encoding="utf-8").strip()
                if key and key not in {"YOUR_CHATGLM_API_KEY_HERE", "REPLACE_WITH_CHATGLM_API_KEY"}:
                    return key
            except OSError:
                pass
                
        # Fallback to direct .env read
        try:
            # We know the .env is at the root of the project (c:\WallNet Detection\zbot-eyes)
            # The current file is in src/net_inspector/ so parents[2] is zbot-eyes
            env_file = Path(__file__).resolve().parents[2] / ".env"
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    # Strip whitespace and ignore comments
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("CHATGLM_API_KEY"):
                        # Split by first '='
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            val = parts[1].strip().strip("\"'")
                            if val and val not in {"YOUR_CHATGLM_API_KEY_HERE", "REPLACE_WITH_CHATGLM_API_KEY"}:
                                return val
        except Exception as e:
            print(f"[DEBUG] Fallback .env read failed: {e}")
            
        return ""

    def _ensure_secrets_layout(self) -> None:
        """Keep secrets directory scaffold in place."""
        key_file = Path(self.config.api_key_file)
        secrets_dir = key_file.parent
        secrets_dir.mkdir(parents=True, exist_ok=True)
        gitkeep = secrets_dir / ".gitkeep"
        if not gitkeep.exists():
            try:
                gitkeep.touch(exist_ok=True)
            except OSError:
                pass

    def _to_image_data_url(self, image_bgr: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Failed to encode image for ChatGLM vision request.")
        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _extract_message_text(self, data: dict[str, Any]) -> str:
        api_error = data.get("error")
        if isinstance(api_error, dict):
            msg = api_error.get("message")
            if isinstance(msg, str) and msg:
                raise RuntimeError(f"ChatGLM API error: {msg}")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Unexpected ChatGLM response: {data}")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "\n".join(parts)
        raise RuntimeError(f"Unable to parse ChatGLM response content: {content}")
