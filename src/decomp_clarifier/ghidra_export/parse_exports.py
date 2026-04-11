from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from decomp_clarifier.schemas.ghidra import GhidraFunctionRow, GhidraProjectManifest

_BINARY_EXTENSIONS = {".exe", ".elf", ".out", ".bin"}


def _strip_binary_extension(name: str) -> str:
    stem, _, ext = name.rpartition(".")
    if stem and f".{ext}".lower() in _BINARY_EXTENSIONS:
        return stem
    return name


@dataclass(frozen=True)
class ParsedGhidraProject:
    manifest: GhidraProjectManifest
    functions: list[GhidraFunctionRow]


def parse_ghidra_export_dir(path: Path) -> ParsedGhidraProject:
    raw = json.loads((path / "project_manifest.json").read_text(encoding="utf-8"))
    raw["project_id"] = _strip_binary_extension(raw.get("project_id", ""))
    raw["binary_name"] = _strip_binary_extension(raw.get("binary_name", ""))
    manifest = GhidraProjectManifest.model_validate(raw)
    functions: list[GhidraFunctionRow] = []
    for line in (path / "functions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["project_id"] = _strip_binary_extension(row.get("project_id", ""))
            row["binary_name"] = _strip_binary_extension(row.get("binary_name", ""))
            functions.append(GhidraFunctionRow.model_validate(row))
    return ParsedGhidraProject(manifest=manifest, functions=functions)
