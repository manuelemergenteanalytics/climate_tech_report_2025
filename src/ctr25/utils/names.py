from __future__ import annotations

import re
import unicodedata
from typing import Mapping, Any


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    text = _strip_accents(str(value))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_company_name(row: Mapping[str, Any]) -> str:
    """Devuelve un nombre limpio de compañía con correcciones conservadoras."""

    current = row.get("company_name") if isinstance(row, Mapping) else None
    if current is None:
        return ""
    current_str = str(current)
    base_norm = _normalize(current_str)

    context_parts = [
        row.get("text_snippet"),
        row.get("title"),
        row.get("description"),
    ]
    url_val = row.get("url")
    if url_val:
        context_parts.append(url_val)

    context_norm = _normalize(" ".join(str(part) for part in context_parts if part))

    # Caso: Del Pinar mal transcripto cuando el contenido menciona Del Pilar
    if "del pinar" in base_norm and "del pilar" not in base_norm:
        if "del pilar" in context_norm:
            return "Del Pilar Sociedad Anónima"
    if "del pinar/del pilar" in base_norm:
        return "Del Pilar Sociedad Anónima"

    return current_str
