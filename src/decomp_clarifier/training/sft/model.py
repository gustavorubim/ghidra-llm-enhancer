from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import re
import time
from pathlib import Path
from typing import Any

from decomp_clarifier.settings import TrainingConfig

_PUBLIC_DNS_SERVERS = ("1.1.1.1", "8.8.8.8")
_HF_DNS_SUFFIXES = ("huggingface.co", "hf.co")
_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_DNS_FALLBACK_INSTALLED = False
_PREFETCH_RETRY_ATTEMPTS = 6
_PREFETCH_RETRY_DELAY_SECONDS = 2.0
_OPTIONAL_MODEL_FILES = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "README.md",
    ".gitattributes",
)
_LOGGER = logging.getLogger("decomp_clarifier")


def _host_needs_hf_dns_fallback(host: str | bytes | None) -> bool:
    if not isinstance(host, str):
        return False
    normalized = host.strip().lower().rstrip(".")
    return any(
        normalized == suffix or normalized.endswith(f".{suffix}") for suffix in _HF_DNS_SUFFIXES
    )


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _parse_nslookup_addresses(output: str) -> list[str]:
    relevant = output.split("Name:", 1)[1] if "Name:" in output else output
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", relevant)
    return _dedupe_preserve_order(ips)


def _resolve_host_with_public_dns(host: str) -> list[str]:
    for server in _PUBLIC_DNS_SERVERS:
        try:
            completed = subprocess.run(
                ["nslookup", host, server],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
        except OSError:
            continue
        addresses = _parse_nslookup_addresses(completed.stdout + "\n" + completed.stderr)
        if addresses:
            return addresses
    return []


def _install_public_dns_fallback_if_needed() -> bool:
    global _DNS_FALLBACK_INSTALLED
    if _DNS_FALLBACK_INSTALLED:
        return True
    seed_addresses = _resolve_host_with_public_dns("huggingface.co")
    if not seed_addresses:
        return False

    def _fallback_getaddrinfo(
        host: str | bytes | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ):
        try:
            return _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)
        except OSError:
            if not _host_needs_hf_dns_fallback(host):
                raise
            fallback_addresses = _resolve_host_with_public_dns(host)
            if (
                not fallback_addresses
                and isinstance(host, str)
                and host.endswith(".huggingface.co")
            ):
                fallback_addresses = _resolve_host_with_public_dns("huggingface.co")
            if not fallback_addresses:
                raise
            resolved: list[object] = []
            fallback_family = family if family not in (0, socket.AF_UNSPEC) else socket.AF_INET
            for address in fallback_addresses:
                try:
                    resolved.extend(
                        _ORIGINAL_GETADDRINFO(
                            address,
                            port,
                            fallback_family,
                            type,
                            proto,
                            flags,
                        )
                    )
                except OSError:
                    continue
            if resolved:
                return resolved
            raise

    socket.getaddrinfo = _fallback_getaddrinfo
    _DNS_FALLBACK_INSTALLED = True
    return True


def _checkpoint_has_lora_adapters(model_name: str | None) -> bool:
    if not model_name:
        return False
    path = Path(model_name)
    return path.exists() and (path / "adapter_config.json").exists()


def _is_local_model_reference(model_name: str) -> bool:
    return Path(model_name).exists()


def _cached_remote_snapshot_dir(model_name: str) -> Path | None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return Path(snapshot_download(repo_id=model_name, local_files_only=True))
    except Exception:  # noqa: BLE001 - training-only probe against optional dependency
        return None


def _snapshot_dir_has_required_files(snapshot_dir: Path) -> bool:
    if not snapshot_dir.exists():
        return False
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        return False
    index_path = snapshot_dir / "model.safetensors.index.json"
    sharded_safetensors = list(snapshot_dir.glob("model-*-of-*.safetensors"))
    if not index_path.exists():
        if sharded_safetensors:
            return False
        return (snapshot_dir / "model.safetensors").exists() or (
            snapshot_dir / "pytorch_model.bin"
        ).exists()
    try:
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, dict):
        return False
    required_shards = _dedupe_preserve_order(
        [value for value in weight_map.values() if isinstance(value, str)]
    )
    if not required_shards:
        return False
    return all((snapshot_dir / shard_name).exists() for shard_name in required_shards)


def _candidate_remote_model_ids(model_name: str, load_in_4bit: bool) -> list[str]:
    if load_in_4bit and model_name.startswith("unsloth/") and not model_name.endswith(
        "-unsloth-bnb-4bit"
    ):
        return [f"{model_name}-unsloth-bnb-4bit"]
    return [model_name]


def _is_transient_snapshot_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "winerror 10051",
            "winerror 10054",
            "winerror 10060",
            "connection reset",
            "connection aborted",
            "forcibly closed",
            "unreachable network",
            "timed out",
            "temporary failure",
        )
    )


def _prefetch_remote_snapshot_dir(model_name: str, load_in_4bit: bool) -> Path | None:
    if _is_local_model_reference(model_name):
        return Path(model_name)
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError:
        return None

    errors: list[str] = []
    for repo_id in _candidate_remote_model_ids(model_name, load_in_4bit):
        last_error: Exception | None = None
        for attempt in range(1, _PREFETCH_RETRY_ATTEMPTS + 1):
            cached = _cached_remote_snapshot_dir(repo_id)
            if cached is not None and _snapshot_dir_has_required_files(cached):
                _LOGGER.info(
                    "using cached training model snapshot repo=%s path=%s",
                    repo_id,
                    cached,
                )
                os.environ["HF_HUB_OFFLINE"] = "1"
                return cached
            if cached is not None:
                _LOGGER.info(
                    "cached training model snapshot is incomplete repo=%s path=%s; resuming download",
                    repo_id,
                    cached,
                )
            try:
                _LOGGER.info(
                    "prefetching training model snapshot repo=%s max_workers=1 attempt=%s/%s",
                    repo_id,
                    attempt,
                    _PREFETCH_RETRY_ATTEMPTS,
                )
                if repo_id.endswith("-unsloth-bnb-4bit"):
                    snapshot_dir = _download_repo_files_individually(repo_id)
                else:
                    snapshot_dir = Path(snapshot_download(repo_id=repo_id, max_workers=1))
            except Exception as exc:  # noqa: BLE001 - training-only network/download errors
                last_error = exc
                if attempt < _PREFETCH_RETRY_ATTEMPTS and _is_transient_snapshot_error(exc):
                    _LOGGER.warning(
                        "training model prefetch hit a transient network error repo=%s attempt=%s/%s: %s; retrying in %.1fs",
                        repo_id,
                        attempt,
                        _PREFETCH_RETRY_ATTEMPTS,
                        exc,
                        _PREFETCH_RETRY_DELAY_SECONDS,
                    )
                    time.sleep(_PREFETCH_RETRY_DELAY_SECONDS)
                    continue
                break
            if _snapshot_dir_has_required_files(snapshot_dir):
                _LOGGER.info(
                    "prefetched training model snapshot repo=%s path=%s",
                    repo_id,
                    snapshot_dir,
                )
                os.environ["HF_HUB_OFFLINE"] = "1"
                return snapshot_dir
            last_error = RuntimeError(
                f"snapshot download for {repo_id} completed without all required files present"
            )
            if attempt < _PREFETCH_RETRY_ATTEMPTS:
                time.sleep(_PREFETCH_RETRY_DELAY_SECONDS)
        if last_error is not None:
            errors.append(f"{repo_id}: {last_error}")

    if errors:
        raise RuntimeError(
            "Failed to prefetch the configured training model. "
            + " | ".join(errors)
        )
    return None


def _download_repo_file_with_retries(
    repo_id: str,
    filename: str,
    *,
    required: bool = True,
) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
        from huggingface_hub.errors import EntryNotFoundError  # type: ignore[import-not-found]
    except ImportError:
        return None

    last_error: Exception | None = None
    for attempt in range(1, _PREFETCH_RETRY_ATTEMPTS + 1):
        try:
            return Path(hf_hub_download(repo_id=repo_id, filename=filename))
        except EntryNotFoundError:
            if required:
                raise
            return None
        except Exception as exc:  # noqa: BLE001 - training-only network/download errors
            last_error = exc
            if attempt < _PREFETCH_RETRY_ATTEMPTS and _is_transient_snapshot_error(exc):
                _LOGGER.warning(
                    "training model file download hit a transient network error repo=%s file=%s attempt=%s/%s: %s; retrying in %.1fs",
                    repo_id,
                    filename,
                    attempt,
                    _PREFETCH_RETRY_ATTEMPTS,
                    exc,
                    _PREFETCH_RETRY_DELAY_SECONDS,
                )
                time.sleep(_PREFETCH_RETRY_DELAY_SECONDS)
                continue
            if not required:
                _LOGGER.warning(
                    "skipping optional training model file repo=%s file=%s after error: %s",
                    repo_id,
                    filename,
                    exc,
                )
                return None
            raise
    if last_error is not None:
        raise last_error
    return None


def _download_repo_files_individually(repo_id: str) -> Path:
    config_path = _download_repo_file_with_retries(repo_id, "config.json")
    if config_path is None:
        raise RuntimeError(f"Could not download config.json for {repo_id}")
    snapshot_dir = config_path.parent

    for optional_filename in _OPTIONAL_MODEL_FILES:
        _download_repo_file_with_retries(repo_id, optional_filename, required=False)

    index_path = _download_repo_file_with_retries(
        repo_id,
        "model.safetensors.index.json",
        required=False,
    )
    if index_path is not None:
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index_payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError(f"Invalid weight map in {index_path}")
        shard_names = _dedupe_preserve_order(
            [value for value in weight_map.values() if isinstance(value, str)]
        )
        for shard_name in shard_names:
            _download_repo_file_with_retries(repo_id, shard_name)
        return snapshot_dir

    safetensors_path = _download_repo_file_with_retries(
        repo_id,
        "model.safetensors",
        required=False,
    )
    if safetensors_path is not None:
        return safetensors_path.parent
    pytorch_bin_path = _download_repo_file_with_retries(
        repo_id,
        "pytorch_model.bin",
        required=False,
    )
    if pytorch_bin_path is not None:
        return pytorch_bin_path.parent
    raise RuntimeError(
        "Could not locate model weights in the Hugging Face repo after downloading "
        f"required files for {repo_id}"
    )


def _can_resolve_huggingface() -> bool:
    try:
        socket.getaddrinfo("huggingface.co", 443)
    except OSError:
        return False
    return True


def _resolve_model_source(model_name: str | None) -> str:
    if not model_name:
        raise RuntimeError("training config missing model.base_model_id")
    if _is_local_model_reference(model_name):
        return model_name
    snapshot_dir = _cached_remote_snapshot_dir(model_name)
    if snapshot_dir is not None:
        if not _can_resolve_huggingface():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            return str(snapshot_dir)
        return model_name
    _install_public_dns_fallback_if_needed()
    if _can_resolve_huggingface():
        return model_name
    if _install_public_dns_fallback_if_needed():
        return model_name
    raise RuntimeError(
        "Could not resolve huggingface.co while loading "
        f"{model_name}. The model is not available in the local Hugging Face cache. "
        "Restore DNS/internet access to Hugging Face or change model.base_model_id "
        "to a local checkpoint directory."
    )


def load_model_and_tokenizer(config: TrainingConfig) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    _install_public_dns_fallback_if_needed()
    model_source = config.model.base_model_id
    prefetched_snapshot = _prefetch_remote_snapshot_dir(
        model_source,
        load_in_4bit=bool(config.training.load_in_4bit),
    )
    if prefetched_snapshot is not None:
        model_source = str(prefetched_snapshot)
    else:
        model_source = _resolve_model_source(config.model.base_model_id)
    max_seq_length = config.training.max_seq_length or 512
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_source,
            max_seq_length=max_seq_length,
            load_in_4bit=bool(config.training.load_in_4bit),
            device_map="cuda:0",
        )
    except RuntimeError as exc:
        if "No config file found" in str(exc):
            raise RuntimeError(
                "Failed to load the configured training model "
                f"{config.model.base_model_id}. If this host is offline, ensure the "
                "model is already cached locally or point model.base_model_id to a "
                "local checkpoint directory."
            ) from exc
        raise
    if _checkpoint_has_lora_adapters(config.model.base_model_id) or (
        "PeftModel" in type(model).__name__
    ):
        return model, tokenizer
    lora_r = config.training.lora_rank or 16
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer
