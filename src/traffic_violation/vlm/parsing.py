"""Strict parsing boundary for untrusted VLM text."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from traffic_violation.events import ViolationKind


class VlmParseError(ValueError):
    """Raised when model text contains no valid report object."""


class VlmReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    violation_confirmed: bool
    violation_type: ViolationKind | None
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1, max_length=1000)
    visible_evidence: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        return min(1.0, max(0.0, float(value)))


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def parse_vlm_output(text: str) -> VlmReport:
    """Extract and validate the first schema-conforming JSON object from model text."""

    if not text.strip():
        raise VlmParseError("VLM output is empty")
    errors: list[str] = []
    for payload in _json_objects(text):
        try:
            return VlmReport.model_validate(payload)
        except ValidationError as exc:
            errors.append(str(exc))
    detail = f"; schema errors: {len(errors)}" if errors else ""
    raise VlmParseError(f"VLM output contains no valid report JSON{detail}")
