from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class FilesystemCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for_payload(payload: dict[str, Any], model: str, schema_version: str) -> str:
        canonical = json.dumps(
            {"payload": payload, "model": model, "schema_version": schema_version},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def path_for_key(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.path_for_key(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: dict[str, Any]) -> Path:
        path = self.path_for_key(key)
        with tempfile.NamedTemporaryFile(
            "w", dir=self.root, suffix=".tmp", delete=False, encoding="utf-8"
        ) as f:
            json.dump(value, f, indent=2, sort_keys=True)
            tmp = f.name
        os.replace(tmp, path)
        return path
