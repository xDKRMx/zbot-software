"""Logging helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from net_inspector.utils.io import ensure_dir


def iso_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp string.

    @return Timestamp in UTC (YYYY-MM-DDTHH:MM:SSZ).
    """
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def log_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a record to a JSONL file.

    @param path JSONL file path.
    @param record Serializable dict to append.
    @return None
    """
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def detections_to_dicts(detections: Iterable[Any]) -> list[dict]:
    """Convert detections to serializable dictionaries.

    @param detections Iterable of detection objects.
    @return List of dicts with type, score, bbox, meta.
    """
    out = []
    for det in detections:
        out.append(
            {
                "type": det.label,
                "score": float(det.score),
                "bbox": [int(v) for v in det.bbox],
                "meta": det.meta or {},
            }
        )
    return out
