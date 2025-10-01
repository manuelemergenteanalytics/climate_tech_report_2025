import re
import yaml


def clean_text(s: str) -> str:
    """Limpia espacios múltiples y recorta extremos."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def load_industry_map(path: str = "config/industry_map.yml") -> dict:
    """Carga el mapa alias->slug canónico de industrias."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_industry(raw: str, imap: dict) -> str:
    """
    Devuelve el slug canónico para una industria dada.
    1) Busca coincidencia exacta por alias en industry_map.yml.
    2) Aplica fallback por tokens comunes si no hay match.
    """
    if not raw:
        return ""
    s = raw.strip().lower()
    # Normaliza caracteres “raros” manteniendo acentos básicos y espacios
    s = re.sub(r"[^a-z0-9áéíóúüñ&/\\ ]+", " ", s)

    # 1) match por alias del YAML
    for slug, aliases in imap.items():
        for a in aliases or []:
            if re.search(rf"\b{re.escape(a.lower())}\b", s):
                return slug

    # 2) fallback por tokens
    tokens = {
        "energy": "energy_power",
        "utility": "energy_power",
        "power": "energy_power",
        "mining": "mining_metals",
        "metal": "mining_metals",
        "quím": "chemicals_materials",
        "chem": "chemicals_materials",
        "manufact": "manufacturing",
        "construct": "construction_realestate",
        "real estate": "construction_realestate",
        "logist": "transport_logistics",
        "transport": "transport_logistics",
        "agro": "agro_food",
        "agri": "agro_food",
        "food": "agro_food",
        "retail": "retail_consumer",
        "consumer": "retail_consumer",
        "waste": "water_waste_circularity",
        "water": "water_waste_circularity",
        "bank": "finance_insurance",
        "insur": "finance_insurance",
        "telecom": "ict_telecom",
        "software": "ict_telecom",
        "it ": "ict_telecom",
    }
    for k, v in tokens.items():
        if k in s:
            return v

    # último recurso
    return "manufacturing"
