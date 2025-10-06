"""Keyword expansion helpers using curated LLM-derived synonyms."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Set


# Synonym clusters curated with the assistance of an LLM to cover
# Spanish/Portuguese/English climate-tech terminology. These keywords
# complement the shorter lists defined in config/keywords.yml.
_AUGMENTED_SYNONYMS = {
    "climate": [
        "clima",
        "cambio climático",
        "climate action",
        "climate resilience",
        "resiliencia climática",
        "adaptación climática",
        "net zero",
        "cero neto",
        "neutralidad de carbono",
    ],
    "renewable": [
        "energía renovable",
        "energias renovaveis",
        "renewable energy",
        "energía limpia",
        "clean energy",
        "energía sostenible",
    ],
    "solar": [
        "fotovoltaica",
        "solar pv",
        "paneles solares",
        "energia solar",
        "usina solar",
    ],
    "wind": [
        "energía eólica",
        "energia eolica",
        "wind power",
        "aerogenerador",
        "parque eólico",
    ],
    "hidrogeno": [
        "hidrógeno verde",
        "green hydrogen",
        "hidrogeno verde",
        "hidrógeno renovable",
        "hidrogênio verde",
    ],
    "battery": [
        "almacenamiento de energía",
        "energy storage",
        "baterías de litio",
        "lithium ion",
        "baterias",
        "storage-as-a-service",
    ],
    "efficiency": [
        "eficiencia energética",
        "energy efficiency",
        "gestión energética",
        "retrofit",
        "descarbonización",
        "decarbonization",
    ],
    "emissions": [
        "reducción de emisiones",
        "reducción de CO2",
        "emisiones netas",
        "emissões",
        "carbon footprint",
        "huella de carbono",
    ],
    "carbono": [
        "captura de carbono",
        "carbon capture",
        "CCUS",
        "secuestro de carbono",
        "mercado de carbono",
        "créditos de carbono",
        "carbon credits",
    ],
    "digital": [
        "tecnología digital",
        "digitalización",
        "digital transformation",
        "tecnologia digital",
        "smart grid",
        "sensores IoT",
        "iot",
        "gemelo digital",
    ],
    "blockchain": [
        "web3",
        "ledger distribuido",
        "distributed ledger",
        "tokenización",
        "tokenization",
        "blockchain climática",
    ],
    "token": [
        "token climatico",
        "carbon token",
        "token digital",
        "credit token",
    ],
    "carbono digital": [
        "mrv digital",
        "monitoreo remoto",
        "monitoramento remoto",
        "satélite carbono",
        "digital mrv",
    ],
}


@lru_cache(maxsize=None)
def expand_keywords(keywords: Iterable[str]) -> List[str]:
    """Return a deduplicated list including synonyms for each keyword."""

    seen: Set[str] = set()
    expanded: List[str] = []

    for kw in keywords:
        if not kw:
            continue
        normalized = kw.strip()
        if not normalized:
            continue
        if normalized.lower() not in seen:
            expanded.append(normalized)
            seen.add(normalized.lower())

        augments = _AUGMENTED_SYNONYMS.get(normalized.lower())
        if not augments:
            continue
        for synonym in augments:
            syn_norm = synonym.strip()
            if syn_norm and syn_norm.lower() not in seen:
                expanded.append(syn_norm)
                seen.add(syn_norm.lower())

    return expanded
