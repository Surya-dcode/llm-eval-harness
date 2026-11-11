"""On-disk response cache.

A cached entry is keyed by a SHA-256 hash of the provider, model, base URL,
params, messages and repeat index. The repeat index is part of the key so
that ``repeats: 3`` really makes three calls the first time.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = ".evalh_cache"


def cache_key(
    provider: str,
    model: str,
    params: dict[str, Any],
    messages: list[dict[str, str]],
    repeat: int = 0,
    base_url: str | None = None,
) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "params": params,
        "messages": messages,
        "repeat": repeat,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, directory: str | Path | None = None, enabled: bool = True):
        self.directory = Path(directory or os.environ.get("EVALH_CACHE_DIR") or DEFAULT_CACHE_DIR)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.misses += 1
            return None
        self.hits += 1
        return data

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
