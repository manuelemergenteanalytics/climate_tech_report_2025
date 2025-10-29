from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Tuple

import re
import unicodedata

import yaml


TOKENS_EXT: Dict[str, str] = {
    "energy": "energy_power",
    "utility": "energy_power",
    "power": "energy_power",
    "mining": "mining_metals",
    "metal": "mining_metals",
    "smelter": "mining_metals",
    "quim": "chemicals_materials",
    "chem": "chemicals_materials",
    "manufact": "manufacturing",
    "factory": "manufacturing",
    "construct": "construction_realestate",
    "real estate": "construction_realestate",
    "logist": "transport_logistics",
    "transport": "transport_logistics",
    "shipping": "transport_logistics",
    "agro": "agro_food",
    "agri": "agro_food",
    "food": "agro_food",
    "beverage": "agro_food",
    "retail": "retail_consumer",
    "consumer": "retail_consumer",
    "supermerc": "retail_consumer",
    "mall": "retail_consumer",
    "waste": "water_waste_circularity",
    "water": "water_waste_circularity",
    "recycling": "water_waste_circularity",
    "bank": "finance_insurance",
    "insur": "finance_insurance",
    "capital": "finance_insurance",
    "telecom": "ict_telecom",
    "software": "ict_telecom",
    "data center": "ict_telecom",
    "it ": "ict_telecom",
    "hospital": "healthcare_pharma_biotech",
    "clinic": "healthcare_pharma_biotech",
    "biotech": "healthcare_pharma_biotech",
    "pharma": "healthcare_pharma_biotech",
    "medical": "healthcare_pharma_biotech",
    "consult": "professional_services_consulting",
    "advisory": "professional_services_consulting",
    "law": "professional_services_consulting",
    "hotel": "hospitality_tourism_leisure",
    "tourism": "hospitality_tourism_leisure",
    "travel": "hospitality_tourism_leisure",
    "institut": "public_social_education",
    "fundaci": "public_social_education",
    "universidad": "public_social_education",
    "minister": "public_social_education",
    "gobiern": "public_social_education",
    "municip": "public_social_education",
    "escuela": "public_social_education",
}

SECTOR_HINT_MAP: Dict[str, str] = {
    "food producers": "agro_food",
    "biotechnology": "healthcare_pharma_biotech",
    "retailers": "retail_consumer",
}


def clean_text(s: str) -> str:
    """Limpia espacios múltiples y recorta extremos."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_fragment(value: Any) -> str:
    if value is None:
        return ""
    text = _strip_accents(str(value))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9&/\\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_industry_map(path: str = "config/industry_map.yml") -> dict:
    """Carga y normaliza la configuración de industrias."""

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        return {"aliases": {}, "mappings": []}

    raw_aliases = data.get("aliases", {})
    aliases: Dict[str, list[str]] = {}
    if isinstance(raw_aliases, dict):
        for slug, items in raw_aliases.items():
            if not items:
                continue
            cleaned = [str(item).strip() for item in items if str(item).strip()]
            if cleaned:
                aliases[str(slug)] = cleaned

    # Compatibilidad con formato legado (slug -> lista)
    if not aliases:
        for slug, items in data.items():
            if slug in {"aliases", "mappings"}:
                continue
            if isinstance(items, (list, tuple)):
                cleaned = [str(item).strip() for item in items if str(item).strip()]
                if cleaned:
                    aliases[str(slug)] = cleaned

    raw_mappings = data.get("mappings", [])
    mappings: list[dict[str, Any]] = []
    if isinstance(raw_mappings, list):
        for entry in raw_mappings:
            if not isinstance(entry, dict):
                continue
            pattern = entry.get("pattern")
            target = entry.get("to")
            if not pattern or not target:
                continue
            weight = entry.get("weight", 1.0)
            try:
                weight_val = float(weight)
            except (TypeError, ValueError):
                weight_val = 1.0
            mappings.append({"pattern": str(pattern), "to": str(target), "weight": weight_val})

    return {"aliases": aliases, "mappings": mappings}


def _prepare_input(raw: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(raw, dict):
        context = raw.get("context") or {}
        value = raw.get("value") or raw.get("industry") or ""
        return str(value), context
    if isinstance(raw, (tuple, list)):
        value = raw[0] if raw else ""
        context = raw[1] if len(raw) > 1 and isinstance(raw[1], dict) else {}
        return str(value or ""), context
    return str(raw or ""), {}


def _extract_sector_hint(source_meta: str) -> tuple[str, str | None]:
    normalized_meta = _normalize_fragment(source_meta)
    if not normalized_meta:
        return "", None
    match = re.search(r"sector\s*:\s*([^;|,]+)", source_meta, flags=re.IGNORECASE)
    sector_raw = match.group(1).strip() if match else ""
    for hint, slug in SECTOR_HINT_MAP.items():
        if hint in normalized_meta:
            return slug, sector_raw or hint
    return "", sector_raw or None


def classify_industry(raw: Any, imap: dict) -> tuple[str, dict[str, Any]]:
    """Clasifica la industria devolviendo slug y metadatos."""

    aliases = imap.get("aliases", {}) if isinstance(imap, dict) else {}
    mappings = imap.get("mappings", []) if isinstance(imap, dict) else []

    value, context = _prepare_input(raw)

    fields = {
        "industry": value,
        "company_name": context.get("company_name", ""),
        "display_name": context.get("display_name", ""),
        "source_meta": context.get("source_meta", ""),
        "text_snippet": context.get("text_snippet", ""),
        "title": context.get("title", ""),
        "description": context.get("description", ""),
        "url": context.get("url", ""),
    }

    normalized_fields = {key: _normalize_fragment(val) for key, val in fields.items() if val}

    # 1) Coincidencia por alias
    for slug, alias_list in aliases.items():
        for alias in alias_list:
            alias_norm = _normalize_fragment(alias)
            if not alias_norm:
                continue
            for field_name, field_value in normalized_fields.items():
                if alias_norm and alias_norm in field_value:
                    return slug, {
                        "reason": "alias",
                        "score": None,
                        "matched_alias": alias,
                        "field": field_name,
                        "sector_hint": None,
                    }

    combined_text = " ".join(str(val) for val in fields.values() if val)
    combined_variants = [combined_text, _strip_accents(combined_text)]

    scores: Dict[str, float] = defaultdict(float)
    matches: Dict[str, list[str]] = defaultdict(list)
    for entry in mappings:
        pattern = entry.get("pattern")
        target = entry.get("to")
        weight = entry.get("weight", 1.0)
        if not pattern or not target:
            continue
        try:
            weight_val = float(weight)
        except (TypeError, ValueError):
            weight_val = 1.0
        for variant in combined_variants:
            if not variant:
                continue
            try:
                if re.search(pattern, variant):
                    scores[target] += weight_val
                    matches[target].append(pattern)
                    break
            except re.error:
                continue

    sector_hint_slug = ""
    sector_hint_label = None
    source_meta = fields.get("source_meta", "")
    if source_meta:
        sector_hint_slug, sector_hint_label = _extract_sector_hint(source_meta)
        if sector_hint_slug:
            scores[sector_hint_slug] += 1.5

    if scores:
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_slug, best_score = sorted_scores[0]
        if best_score >= 2.0:
            return best_slug, {
                "reason": "weighted",
                "score": round(best_score, 3),
                "patterns": matches.get(best_slug, []),
                "sector_hint": sector_hint_label if sector_hint_slug == best_slug else sector_hint_label,
                "sector_hint_slug": sector_hint_slug or None,
            }

    combined_normalized = _normalize_fragment(combined_text)
    for token, slug in TOKENS_EXT.items():
        if token in combined_normalized:
            return slug, {
                "reason": "token",
                "score": None,
                "token": token,
                "sector_hint": sector_hint_label,
                "sector_hint_slug": sector_hint_slug or None,
            }

    return "unknown", {
        "reason": "unknown",
        "score": 0.0,
        "sector_hint": sector_hint_label,
        "sector_hint_slug": sector_hint_slug or None,
    }


def normalize_industry(raw: Any, imap: dict) -> str:
    slug, _ = classify_industry(raw, imap)
    return slug
