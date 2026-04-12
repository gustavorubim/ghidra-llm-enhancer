from __future__ import annotations

from typing import Any


def normalize_optional_flag(value: Any) -> bool:
    if isinstance(value, tuple):
        return bool(value[0])
    return bool(value)


def patch_trl_optional_availability() -> None:
    import trl.import_utils as trl_import_utils  # type: ignore[import-not-found]

    for name in dir(trl_import_utils):
        if not (name.startswith("_") and name.endswith("_available")):
            continue
        value = getattr(trl_import_utils, name)
        if isinstance(value, tuple):
            setattr(trl_import_utils, name, normalize_optional_flag(value))


def ensure_model_warnings_issued(model: Any) -> int:
    seen: set[int] = set()
    stack = [model]
    updated = 0

    while stack:
        current = stack.pop()
        if current is None:
            continue
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)

        try:
            warnings_issued = getattr(current, "warnings_issued")
        except Exception:  # noqa: BLE001 - compatibility shim over third-party wrappers
            warnings_issued = None

        if not isinstance(warnings_issued, dict):
            try:
                setattr(current, "warnings_issued", {})
                updated += 1
            except Exception:  # noqa: BLE001 - best-effort compatibility shim
                pass

        for attr_name in ("base_model", "model"):
            try:
                child = getattr(current, attr_name)
            except Exception:  # noqa: BLE001 - compatibility shim over wrapper proxies
                continue
            if child is not None:
                stack.append(child)

    return updated
