"""HTTP helpers with shared caching."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests
import requests_cache

_CACHE_PATH = Path("data/interim/cache/http_cache")
_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

requests_cache.install_cache(
    str(_CACHE_PATH),
    backend="sqlite",
    expire_after=3600,
)


def get(
    url: str,
    *,
    timeout: int = 30,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
) -> requests.Response:
    """Perform a GET request with default headers and shared cache."""
    base_headers = {"User-Agent": "ctr25/1.0 (+https://example.com)"}
    if headers:
        base_headers.update(headers)
    response = requests.get(url, timeout=timeout, headers=base_headers, params=params)
    response.raise_for_status()
    return response
