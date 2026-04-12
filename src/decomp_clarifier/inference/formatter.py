from __future__ import annotations

import json

from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _fallback_output(text: str) -> ClarifiedFunctionOutput:
    return ClarifiedFunctionOutput(
        summary="Model output was not valid JSON.",
        confidence=0.0,
        renamings={},
        cleaned_c=text.strip(),
    )


def normalize_output_with_status(text: str) -> tuple[ClarifiedFunctionOutput, bool]:
    json_fragment = extract_json_object(text)
    if json_fragment is None:
        return _fallback_output(text), False
    try:
        return ClarifiedFunctionOutput.model_validate(json.loads(json_fragment)), True
    except Exception:  # noqa: BLE001 - tolerate malformed generations during inference
        return _fallback_output(text), False


def normalize_output(text: str) -> ClarifiedFunctionOutput:
    return normalize_output_with_status(text)[0]
