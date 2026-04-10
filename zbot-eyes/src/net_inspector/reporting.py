"""Structured incident reporting with evidence-grounded exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Optional

import numpy as np

from net_inspector.utils.io import ensure_dir, save_image, timestamp_id


Disposition = str


def iso_utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def confidence_phrase(confidence: float) -> str:
    """Map confidence to conservative language."""
    if confidence >= 0.80:
        return "detected"
    if confidence >= 0.55:
        return "possible"
    return "unverified indication"


@dataclass
class EvidenceItem:
    """Immutable evidence reference for one frame/sensor reading."""

    evidence_id: str
    modality: str
    timestamp_utc: str
    frame_ref: str
    image_path: str
    sha256: str
    measurements: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationItem:
    """Grounded observation and its claim."""

    observation_id: str
    label: str
    confidence: float
    severity: str
    claim: str
    evidence_ids: list[str]
    measurements: dict[str, Any] = field(default_factory=dict)


@dataclass
class IncidentReport:
    """Structured incident report record."""

    incident_id: str
    created_at_utc: str
    time_range_start_utc: str
    time_range_end_utc: str
    estimated_distance_m: Optional[float]
    position_hint: str
    robot_pose: str
    model_version: str
    operator_disposition: Disposition = "needs_review"
    operator_notes: str = ""
    llm_summary_markdown: str = ""
    evidence: list[EvidenceItem] = field(default_factory=list)
    observations: list[ObservationItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def risk_score(self) -> float:
        """Compute a compact risk score for queue ordering."""
        severity_weight = {"critical": 1.0, "high": 0.85, "medium": 0.6, "low": 0.35}
        if not self.observations:
            return 0.0
        values = []
        for obs in self.observations:
            w = severity_weight.get(obs.severity, 0.5)
            values.append(float(obs.confidence) * w)
        return max(values)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)


def validate_grounded_markdown(markdown: str, evidence_ids: list[str]) -> list[str]:
    """Validate that claims reference concrete evidence IDs.

    Rule: each non-empty sentence with alphabetic content must include `[EVID:<id>]`
    where `<id>` exists in this incident.
    """
    if not markdown.strip():
        return []

    allowed = set(evidence_ids)
    problems: list[str] = []
    sentence_re = re.compile(r"(?<=[.!?])\s+|\n+")
    tag_re = re.compile(r"\[EVID:([A-Za-z0-9_\-]+)\]")

    for chunk in sentence_re.split(markdown):
        sentence = chunk.strip()
        if not sentence:
            continue
        if sentence.startswith("#") or sentence.startswith("```"):
            continue
        if not re.search(r"[A-Za-z]", sentence):
            continue

        tags = tag_re.findall(sentence)
        if not tags:
            problems.append(f"Missing evidence tag: {sentence[:140]}")
            continue

        missing = [tag for tag in tags if tag not in allowed]
        if missing:
            problems.append(
                f"Unknown evidence tag(s) {missing} in sentence: {sentence[:140]}"
            )

    return problems


class IncidentStore:
    """In-memory incident store backed by deterministic file exports."""

    def __init__(self, output_dir: Path, model_version: str) -> None:
        self.output_dir = output_dir
        self.model_version = model_version
        self._items: list[IncidentReport] = []
        ensure_dir(self.output_dir)

    def all(self) -> list[IncidentReport]:
        """Return all incidents sorted by descending risk."""
        return sorted(self._items, key=lambda item: item.risk_score(), reverse=True)

    def get(self, incident_id: str) -> Optional[IncidentReport]:
        """Find one incident by ID."""
        for report in self._items:
            if report.incident_id == incident_id:
                return report
        return None

    def create_incident(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: Optional[np.ndarray],
        label: str,
        confidence: float,
        severity: str,
        estimated_distance_m: Optional[float],
        position_hint: str,
        robot_pose: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IncidentReport:
        """Create a new incident with RGB (and optional thermal) evidence."""
        ts = iso_utc_now()
        incident_id = f"INC_{timestamp_id()}"
        incident_dir = self.output_dir / incident_id
        ensure_dir(incident_dir)

        evidence: list[EvidenceItem] = []
        rgb_ev = self._save_evidence(
            incident_dir=incident_dir,
            modality="rgb",
            frame=rgb_frame,
            timestamp_utc=ts,
            measurements={
                "estimated_distance_m": estimated_distance_m,
                "position_hint": position_hint,
                "robot_pose": robot_pose,
            },
        )
        evidence.append(rgb_ev)

        if thermal_frame is not None:
            thermal_ev = self._save_evidence(
                incident_dir=incident_dir,
                modality="thermal",
                frame=thermal_frame,
                timestamp_utc=ts,
                measurements={
                    "estimated_distance_m": estimated_distance_m,
                    "position_hint": position_hint,
                    "robot_pose": robot_pose,
                },
            )
            evidence.append(thermal_ev)

        phrase = confidence_phrase(confidence)
        primary_evid = evidence[0].evidence_id if evidence else "missing"
        claim = f"{label}: {phrase} in current capture. [EVID:{primary_evid}]"
        observation = ObservationItem(
            observation_id=f"OBS_{timestamp_id()}",
            label=label,
            confidence=float(max(0.0, min(1.0, confidence))),
            severity=severity,
            claim=claim,
            evidence_ids=[item.evidence_id for item in evidence],
            measurements={
                "estimated_distance_m": estimated_distance_m,
                "position_hint": position_hint,
                "robot_pose": robot_pose,
            },
        )

        report = IncidentReport(
            incident_id=incident_id,
            created_at_utc=ts,
            time_range_start_utc=ts,
            time_range_end_utc=ts,
            estimated_distance_m=estimated_distance_m,
            position_hint=position_hint,
            robot_pose=robot_pose,
            model_version=self.model_version,
            evidence=evidence,
            observations=[observation],
            metadata=metadata or {},
        )
        self._items.append(report)
        return report

    def set_disposition(self, incident_id: str, disposition: Disposition) -> None:
        report = self.get(incident_id)
        if report is None:
            raise KeyError(f"Incident not found: {incident_id}")
        report.operator_disposition = disposition

    def attach_llm_summary(self, incident_id: str, markdown: str) -> list[str]:
        """Attach LLM summary after evidence-grounding checks."""
        report = self.get(incident_id)
        if report is None:
            raise KeyError(f"Incident not found: {incident_id}")
        evidence_ids = [item.evidence_id for item in report.evidence]
        errors = validate_grounded_markdown(markdown, evidence_ids)
        if errors:
            return errors
        report.llm_summary_markdown = markdown
        return []

    def export_incident(self, incident_id: str) -> tuple[Path, Path]:
        """Export one incident to JSON + markdown files."""
        report = self.get(incident_id)
        if report is None:
            raise KeyError(f"Incident not found: {incident_id}")

        validation_errors = self.validate_incident(report)
        if validation_errors:
            text = "; ".join(validation_errors)
            raise ValueError(f"Incident validation failed: {text}")

        incident_dir = self.output_dir / report.incident_id
        ensure_dir(incident_dir)
        json_path = incident_dir / "report.json"
        md_path = incident_dir / "report.md"

        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8"
        )
        md_path.write_text(self.to_markdown(report), encoding="utf-8")
        return json_path, md_path

    def validate_incident(self, report: IncidentReport) -> list[str]:
        """Validate incident grounding constraints before export."""
        errors: list[str] = []
        evidence_ids = {item.evidence_id for item in report.evidence}
        if not evidence_ids:
            errors.append("Incident has no evidence.")

        for obs in report.observations:
            if not obs.evidence_ids:
                errors.append(f"{obs.observation_id} has no linked evidence IDs.")
            unknown = [eid for eid in obs.evidence_ids if eid not in evidence_ids]
            if unknown:
                errors.append(f"{obs.observation_id} references unknown evidence: {unknown}")
            if "[EVID:" not in obs.claim:
                errors.append(f"{obs.observation_id} claim missing evidence tag.")

        if report.llm_summary_markdown.strip():
            md_errors = validate_grounded_markdown(
                report.llm_summary_markdown, list(evidence_ids)
            )
            errors.extend(md_errors)

        return errors

    def to_markdown(self, report: IncidentReport) -> str:
        """Render deterministic markdown report."""
        lines = [
            f"# Incident Report {report.incident_id}",
            "",
            "## Metadata",
            f"- Created: {report.created_at_utc}",
            f"- Time range: {report.time_range_start_utc} to {report.time_range_end_utc}",
            f"- Estimated distance (m): {report.estimated_distance_m}",
            f"- Position hint: {report.position_hint or 'n/a'}",
            f"- Robot pose: {report.robot_pose or 'n/a'}",
            f"- Model version: {report.model_version}",
            f"- Disposition: {report.operator_disposition}",
            "",
            "## Evidence",
        ]

        for item in report.evidence:
            lines.extend(
                [
                    f"- {item.evidence_id} ({item.modality})",
                    f"  - Time: {item.timestamp_utc}",
                    f"  - Frame: {item.frame_ref}",
                    f"  - Image: {item.image_path}",
                    f"  - SHA256: {item.sha256}",
                ]
            )

        lines.append("")
        lines.append("## Observed")
        for obs in report.observations:
            lines.extend(
                [
                    f"- {obs.observation_id}",
                    f"  - Label: {obs.label}",
                    f"  - Confidence: {obs.confidence:.2f}",
                    f"  - Severity: {obs.severity}",
                    f"  - Claim: {obs.claim}",
                    f"  - Evidence IDs: {', '.join(obs.evidence_ids)}",
                ]
            )

        if report.llm_summary_markdown.strip():
            lines.extend(
                [
                    "",
                    "## Inferred (LLM, Evidence-tagged)",
                    report.llm_summary_markdown.strip(),
                ]
            )

        if report.operator_notes.strip():
            lines.extend(["", "## Operator Notes", report.operator_notes.strip()])

        return "\n".join(lines).strip() + "\n"

    def _save_evidence(
        self,
        incident_dir: Path,
        modality: str,
        frame: np.ndarray,
        timestamp_utc: str,
        measurements: dict[str, Any],
    ) -> EvidenceItem:
        evidence_id = f"EVID_{modality}_{timestamp_id()}"
        image_path = incident_dir / f"{evidence_id}.jpg"
        save_image(image_path, frame)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        return EvidenceItem(
            evidence_id=evidence_id,
            modality=modality,
            timestamp_utc=timestamp_utc,
            frame_ref=evidence_id,
            image_path=str(image_path),
            sha256=digest,
            measurements=measurements,
        )
