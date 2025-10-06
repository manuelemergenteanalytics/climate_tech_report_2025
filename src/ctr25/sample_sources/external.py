"""External datasets to enrich the LATAM universe (e.g., SBTi directory)."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from io import StringIO

SBTI_URL = "https://files.sciencebasedtargets.org/production/files/companies-excel.xlsx"

COUNTRY_NAME_TO_ISO: Dict[str, str] = {
    "argentina": "AR",
    "brazil": "BR",
    "brasil": "BR",
    "chile": "CL",
    "colombia": "CO",
    "mexico": "MX",
    "méxico": "MX",
    "peru": "PE",
    "perú": "PE",
    "uruguay": "UY",
}


SBTI_SECTOR_MAP: Dict[str, str] = {
    "technology hardware and equipment": "ict_telecom",
    "software and services": "ict_telecom",
    "media": "ict_telecom",
    "telecommunication services": "ict_telecom",
    "semiconductors and semiconductors equipment": "ict_telecom",
    "healthcare providers and services, and healthcare technology": "ict_telecom",
    "professional services": "ict_telecom",
    "education services": "ict_telecom",
    "specialized consumer services": "retail_consumer",
    "retailing": "retail_consumer",
    "food and staples retailing": "retail_consumer",
    "hotels, restaurants and leisure, and tourism services": "retail_consumer",
    "trading companies and distributors, and commercial services and supplies": "retail_consumer",
    "consumer durables, household and personal products": "manufacturing",
    "textiles, apparel, footwear and luxury goods": "manufacturing",
    "electrical equipment and machinery": "manufacturing",
    "automobiles and components": "manufacturing",
    "building products": "manufacturing",
    "containers and packaging": "manufacturing",
    "construction materials": "manufacturing",
    "healthcare equipment and supplies": "manufacturing",
    "aerospace and defense": "manufacturing",
    "tires": "manufacturing",
    "food and beverage processing": "agro_food",
    "food production - agricultural production": "agro_food",
    "food production - animal source food production": "agro_food",
    "tobacco": "agro_food",
    "real estate": "construction_realestate",
    "homebuilding": "construction_realestate",
    "construction and engineering": "construction_realestate",
    "air freight transportation and logistics": "transport_logistics",
    "air transportation - airport services": "transport_logistics",
    "air transportation - airlines": "transport_logistics",
    "ground transportation - railroads transportation": "transport_logistics",
    "ground transportation - highways and railtracks": "transport_logistics",
    "ground transportation - trucking transportation": "transport_logistics",
    "water transportation - water transportation": "transport_logistics",
    "water transportation - ports and services": "transport_logistics",
    "electric utilities and independent power producers and energy traders (including fossil, alternative and nuclear energy)": "energy_power",
    "solid waste management utilities": "water_waste_circularity",
    "water utilities": "water_waste_circularity",
    "chemicals": "chemicals_materials",
    "pharmaceuticals, biotechnology and life sciences": "chemicals_materials",
    "banks, diverse financials, insurance": "finance_insurance",
    "specialized financial services, consumer finance, insurance brokerage firms": "finance_insurance",
    "mining - iron, aluminum, other metals": "mining_metals",
    "mining - other (rare minerals, precious metals and gems)": "mining_metals",
    "mining - coal": "mining_metals",
    "forest and paper products - forestry, timber, pulp and paper, rubber": "chemicals_materials",
}


def _normalize_country(value: str | float | None) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    lowered = value.lower()
    for name, iso in COUNTRY_NAME_TO_ISO.items():
        if name in lowered:
            return iso
    return None


def _map_sector_to_industry(sector: str | None) -> str:
    if not isinstance(sector, str):
        return ""
    key = sector.strip().lower()
    if key in SBTI_SECTOR_MAP:
        return SBTI_SECTOR_MAP[key]
    # lightweight fallbacks using keywords
    if "mining" in key:
        return "mining_metals"
    if "transport" in key or "logistic" in key:
        return "transport_logistics"
    if "agric" in key or "food" in key or "beverage" in key:
        return "agro_food"
    if "bank" in key or "financial" in key or "insurance" in key:
        return "finance_insurance"
    if "chemical" in key or "pharma" in key or "biotech" in key:
        return "chemicals_materials"
    if "energy" in key or "power" in key or "utilities" in key:
        return "energy_power"
    if "real estate" in key or "construction" in key or "homebuilding" in key:
        return "construction_realestate"
    if "telecom" in key or "software" in key or "technology" in key or "ict" in key:
        return "ict_telecom"
    if "manufact" in key or "equipment" in key or "products" in key:
        return "manufacturing"
    if "waste" in key or "water" in key:
        return "water_waste_circularity"
    return ""


def _sbti_hash_id(name: str, country: str) -> str:
    digest = hashlib.sha1(f"sbti::{name}::{country}".encode("utf-8")).hexdigest()
    return f"sbti-{digest[:12]}"


def load_sbti_companies(
    *,
    target_countries: Iterable[str],
    out_dir: str | Path = "data/raw/external",
) -> pd.DataFrame:
    """Return SBTi companies filtered by country with mapped industries."""

    target_set = {c.upper() for c in target_countries}
    if not target_set:
        return pd.DataFrame()

    cache_dir = Path(out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "sbti_latam_raw.csv"

    df = pd.read_excel(SBTI_URL, dtype=str)
    df["iso2"] = df["location"].apply(_normalize_country)
    df = df[df["iso2"].isin(target_set)].copy()

    if df.empty:
        cache_path.write_text("", encoding="utf-8")
        return pd.DataFrame()

    df["industry"] = df["sector"].apply(_map_sector_to_industry)
    df = df[df["industry"].astype(bool)]

    if df.empty:
        cache_path.write_text("", encoding="utf-8")
        return pd.DataFrame()

    df = df[df["organization_type"].str.lower() != "sme"]

    df.to_csv(cache_path, index=False)

    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        name = str(row.get("company_name", "")).strip()
        if not name:
            continue
        country = row["iso2"].upper()
        industry = row["industry"]
        qid = _sbti_hash_id(name, country)
        size = "l" if str(row.get("organization_type", "")).lower() == "corporate" else "m"
        description = row.get("full_target_language") or row.get("sector") or ""

        records.append({
            "qid": qid,
            "company_name": name,
            "country": country,
            "industry_raw": row.get("sector", ""),
            "description": description,
            "employees": None,
            "revenue": None,
            "company_domain": "",
            "ticker": "",
            "size_bin": size,
            "source": "sbti",
            "source_rank": 2,
            "industry": industry,
        })

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame.from_records(records)
    return out


WIKIPEDIA_PAGES: Dict[str, str] = {
    "AR": "https://en.wikipedia.org/wiki/List_of_companies_of_Argentina",
    "BR": "https://en.wikipedia.org/wiki/List_of_companies_of_Brazil",
    "CL": "https://en.wikipedia.org/wiki/List_of_companies_of_Chile",
    "CO": "https://en.wikipedia.org/wiki/List_of_companies_of_Colombia",
    "MX": "https://en.wikipedia.org/wiki/List_of_companies_of_Mexico",
    "PE": "https://en.wikipedia.org/wiki/List_of_companies_of_Peru",
    "UY": "https://en.wikipedia.org/wiki/List_of_companies_of_Uruguay",
}


def _guess_industry_from_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("energy", "power", "renewable", "electric", "solar", "wind", "hydro")):
        return "energy_power"
    if any(keyword in lowered for keyword in ("oil", "gas", "petro", "petrol", "refiner", "upstream", "downstream")):
        return "oil_gas"
    if any(keyword in lowered for keyword in ("mining", "minera", "mineral", "metals", "steel", "aluminum", "iron")):
        return "mining_metals"
    if any(keyword in lowered for keyword in ("chemical", "chem", "fertilizer", "plastic", "pharma", "biotech")):
        return "chemicals_materials"
    if any(keyword in lowered for keyword in ("manufactur", "factory", "industrial", "automotive", "auto", "machinery", "equipment")):
        return "manufacturing"
    if any(keyword in lowered for keyword in ("construct", "construction", "real estate", "inmobili", "developer", "infrastructure", "cement")):
        return "construction_realestate"
    if any(keyword in lowered for keyword in ("transport", "logistic", "aero", "airline", "shipping", "rail", "bus", "subway", "metro", "port", "freight")):
        return "transport_logistics"
    if any(keyword in lowered for keyword in ("agro", "agric", "food", "beverage", "brew", "farm", "lácte", "meat", "dairy", "sugar", "coffee")):
        return "agro_food"
    if any(keyword in lowered for keyword in ("retail", "consumer", "store", "supermarket", "e-commerce", "fashion", "restaurant", "hotel", "tourism", "media", "telecom", "tech", "software", "digital", "bank", "finance", "insurance", "holding")):
        # refine subgroups
        if any(keyword in lowered for keyword in ("bank", "finance", "insurance", "fund", "holding")):
            return "finance_insurance"
        if any(keyword in lowered for keyword in ("telecom", "software", "digital", "tech", "media", "internet", "television", "radio", "it", "ict")):
            return "ict_telecom"
        return "retail_consumer"
    if any(keyword in lowered for keyword in ("water", "waste", "recycling", "sanitation", "environment")):
        return "water_waste_circularity"
    return ""


def load_wikipedia_companies(
    *,
    target_countries: Iterable[str],
    out_dir: str | Path = "data/raw/external",
) -> pd.DataFrame:
    target_set = {c.upper() for c in target_countries}
    relevant_pages = {
        iso: url for iso, url in WIKIPEDIA_PAGES.items() if iso in target_set
    }
    if not relevant_pages:
        return pd.DataFrame()

    cache_dir = Path(out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (ctr25)"}
    records: List[Dict[str, object]] = []

    for iso, url in relevant_pages.items():
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
        except Exception:
            continue
        html = response.text
        try:
            tables = pd.read_html(StringIO(html))
        except ValueError:
            continue

        all_rows: List[Dict[str, object]] = []
        for table in tables:
            table = table.copy()
            table.columns = [" ".join(map(str, col if isinstance(col, tuple) else (col,))).strip().lower() for col in table.columns]
            if not any("name" in col or "company" in col for col in table.columns):
                continue
            name_col: Optional[str] = next((c for c in table.columns if "name" in c), None)
            if name_col is None:
                name_col = next((c for c in table.columns if "company" in c), table.columns[0])
            industry_col: Optional[str] = next((c for c in table.columns if "industry" in c or "sector" in c or "type" in c or "field" in c or "products" in c), None)
            desc_col: Optional[str] = next((c for c in table.columns if "notes" in c or "description" in c or "remarks" in c or "activity" in c), None)

            for _, row in table.iterrows():
                name = str(row.get(name_col, "")).strip()
                if not name or name.lower() in {"name", "company", "organisation"}:
                    continue
                industry_raw = str(row.get(industry_col, "")) if industry_col else ""
                description = str(row.get(desc_col, "")) if desc_col else industry_raw
                text_to_classify = f"{industry_raw} {description}"
                industry = _guess_industry_from_text(text_to_classify)
                if not industry:
                    continue
                all_rows.append({
                    "company_name": name,
                    "industry_raw": industry_raw or description,
                    "description": description,
                    "industry": industry,
                })

        if not all_rows:
            continue

        raw_df = pd.DataFrame(all_rows).drop_duplicates(subset=["company_name"])
        cache_path = cache_dir / f"wikipedia_{iso.lower()}.csv"
        raw_df.assign(country=iso).to_csv(cache_path, index=False)

        for _, row in raw_df.iterrows():
            name = row["company_name"]
            industry = row["industry"]
            qid = _sbti_hash_id(name, iso)  # reuse helper for deterministic id
            records.append({
                "qid": f"wiki-{qid.split('-', 1)[-1]}",
                "company_name": name,
                "country": iso,
                "industry_raw": row.get("industry_raw", ""),
                "description": row.get("description", ""),
                "employees": None,
                "revenue": None,
                "company_domain": "",
                "ticker": "",
                "size_bin": "m",
                "source": "wikipedia",
                "source_rank": 3,
                "industry": industry,
            })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)
