#!/usr/bin/env python3
"""Combine membership datasets into events_normalized.csv."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ctr25.signals.news import _load_keywords, _enrich_events
from ctr25.industry import IndustryResolver

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
EVENTS_PATH = PROCESSED_DIR / "events_normalized.csv"
UNIVERSE_PATH = PROCESSED_DIR / "universe_sample.csv"
KEYWORDS_PATH = Path("config/keywords.yml")

EVENT_COLUMNS = [
    "company_id",
    "company_qid",
    "company_name",
    "country",
    "industry",
    "size_bin",
    "source",
    "signal_type",
    "signal_strength",
    "ts",
    "url",
    "title",
    "text_snippet",
    "description",
    "climate_score",
    "sentiment_label",
    "sentiment_score",
]

CONTEXT_COLS = [
    "context_sector",
    "context_type",
    "context_description",
    "context_extra",
]

# Mapping of country names present in source datasets -> ISO2 codes used internally.
_COUNTRY_MAP: Dict[str, str] = {
    "argentina": "AR",
    "bolivia": "BO",
    "bolivia (plurinational state of)": "BO",
    "brazil": "BR",
    "brasil": "BR",
    "chile": "CL",
    "colombia": "CO",
    "costa rica": "CR",
    "dominican republic": "DO",
    "ecuador": "EC",
    "el salvador": "SV",
    "guatemala": "GT",
    "honduras": "HN",
    "mexico": "MX",
    "méxico": "MX",
    "nicaragua": "NI",
    "panama": "PA",
    "panamá": "PA",
    "paraguay": "PY",
    "peru": "PE",
    "perú": "PE",
    "uruguay": "UY",
    "venezuela": "VE",
}
_LATAM_CODES = {
    "AR",
    "BO",
    "BR",
    "CL",
    "CO",
    "CR",
    "DO",
    "EC",
    "GT",
    "HN",
    "MX",
    "NI",
    "PA",
    "PE",
    "PY",
    "SV",
    "UY",
    "VE",
}

_TITLE_MAP = {
    "bcorp": "Certificación B Corp",
    "sbti": "Compromiso SBTi",
    "ungc": "Participación Pacto Global",
}

# Canonicalización industrial extendida para reducir falsos positivos en "manufacturing".
_INDUSTRY_PATTERNS: List[tuple[str, str]] = [
    (r"oil|petro|gas|hidrocarb", "oil_gas"),
    (
        r"energy|energ[ií]a|eléctric|electric|power|solar|fotovoltaic|e[oó]lic|wind|hidro|renew|utilities?",
        "energy_power",
    ),
    (r"mining|miner|minería|minera|metal|sider", "mining_metals"),
    (r"chem|qu[ií]m|material|plast|poly|fertil|forest and paper products|rubber", "chemicals_materials"),
    (
        r"construction materials|real estate|inmobili|infraestruct|building|arquitect|engineering|obra",
        "construction_realestate",
    ),
    (r"transport|log[ií]st|storage|shipping|ferro|a[eé]re|aviaci|metro|rail|trucking|mar[ií]t|portu", "transport_logistics"),
    (
        r"agri|agro|food|alimento|bev|cervec|brew|café|cafe|cacao|ganad|harin|az[uú]car|fish|forestry",
        "agro_food",
    ),
    (
        r"retail|consumer|wholesal|commerce|tienda|supermerc|department|distribu|e-?commerce",
        "retail_consumer",
    ),
    (r"water|sewer|waste|residu|circular|recicl|sanit|hidric|hydric", "water_waste_circularity"),
    (r"finance|bank|banca|insur|segur|investment|capital|bursatil|microfin|fintech", "finance_insurance"),
    (
        r"telecom|ict|telefon|software|internet|technolog|digital|comunicaci|media|televis|notic|radio",
        "ict_telecom",
    ),
    (r"other services", "professional_services"),
    (
        r"professional|consult|advisory|services? support|technical services|legal|account|auditor|marketing",
        "professional_services",
    ),
    (r"educat|school|universit|academ|training|learning", "education_research"),
    (r"health|hospital|cl[ií]nic|medical|pharma|biotech|salud", "healthcare"),
    (r"hotel|hospitality|restaurant|leisure|tourism|alojam|lodg|food service", "hospitality_tourism"),
    (r"media|entertainment|publishing|press|television|news|radio|cultural|art", "media_entertainment"),
    (r"government|public policy|publica|municip|minister|state-owned|sector p[uú]blico", "public_sector"),
    (r"non[- ]?profit|ngo|fundaci[oó]n|impacto social", "social_impact"),
    (r"environmental service|sustainability service|carbon market|carbon credit", "environmental_services"),
    (
        r"manufactured goods|manufactur|industrial|factory|production|automotive|machinery|equipment|textil|apparel|packaging",
        "manufacturing",
    ),
]


# --- text helpers (copy of membership normalization utilities) ---
def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9áéíóúüñ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokenize_name(norm_name: str) -> List[str]:
    tokens: List[str] = []
    for raw in norm_name.split():
        tok = raw.strip()
        if not tok or len(tok) == 1:
            continue
        if tok in {
            "de",
            "del",
            "la",
            "las",
            "los",
            "el",
            "y",
            "the",
            "grupo",
            "group",
            "holding",
            "company",
            "companies",
            "corp",
            "corporation",
            "inc",
            "sa",
            "saa",
            "sae",
            "srl",
            "plc",
            "ltd",
            "ltda",
            "co",
            "cv",
            "s",
            "sac",
            "saic",
            "spa",
            "air",
            "cargo",
        }:
            continue
        tokens.append(tok)
    return tokens


def _slugify(value: str, *, max_len: int = 60) -> str:
    if not value:
        return "unknown"
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
    slug = slug.strip("-") or "unknown"
    return slug[:max_len]


def _clean_text(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _map_country(value: str) -> str:
    if not isinstance(value, str):
        return ""
    key = value.strip().lower()
    return _COUNTRY_MAP.get(key, "")


def _parse_ts(*values: object) -> str:
    for val in values:
        if val is None:
            continue
        if isinstance(val, float) and pd.isna(val):
            continue
        s = str(val).strip()
        if not s:
            continue
        if s.isdigit() and len(s) == 4:
            return f"{s}-01-01T00:00:00Z"
        ts = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        return ts.isoformat()
    # fallback to today UTC to avoid empty timestamps
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT00:00:00Z")


def _size_from_bcorp(raw: str) -> str:
    mapping = {
        "0": "s",
        "1-9": "s",
        "10-49": "s",
        "50-249": "m",
        "250-999": "m",
        "250+": "l",
        "1000+": "l",
    }
    if not isinstance(raw, str):
        return ""
    return mapping.get(raw.strip(), "")


@dataclass
class UniverseRecord:
    company_id: object
    company_qid: str
    company_name: str
    country: str
    industry: str
    size_bin: str
    norm: str
    tokens: List[str]


class UniverseIndex:
    def __init__(self, universe: pd.DataFrame) -> None:
        self.records: List[UniverseRecord] = []
        self.by_norm: Dict[str, UniverseRecord] = {}
        self.by_token: Dict[str, List[UniverseRecord]] = {}
        for _, row in universe.iterrows():
            norm = _norm(row.get("company_name", ""))
            tokens = _tokenize_name(norm)
            record = UniverseRecord(
                company_id=row.get("company_id"),
                company_qid=str(row.get("company_qid", "") or ""),
                company_name=str(row.get("company_name", "") or ""),
                country=str(row.get("country", "") or ""),
                industry=str(row.get("industry", "") or ""),
                size_bin=str(row.get("size_bin", "") or ""),
                norm=norm,
                tokens=tokens,
            )
            self.records.append(record)
            if norm and norm not in self.by_norm:
                self.by_norm[norm] = record
            for token in tokens:
                self.by_token.setdefault(token, []).append(record)

    def match(self, name: str) -> Optional[UniverseRecord]:
        if not name:
            return None
        direct = self.by_norm.get(name)
        if direct is not None:
            return direct
        tokens = _tokenize_name(name)
        if len(tokens) < 2:
            return None
        candidates: Dict[int, UniverseRecord] = {}
        for token in tokens:
            for rec in self.by_token.get(token, []):
                candidates[id(rec)] = rec
        if not candidates:
            return None
        best: Optional[UniverseRecord] = None
        best_score = 0
        query_tokens = set(tokens)
        for rec in candidates.values():
            shared = len(query_tokens.intersection(rec.tokens))
            if shared > best_score:
                best = rec
                best_score = shared
        if best_score >= 2:
            return best
        return None


def _load_universe() -> UniverseIndex:
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(
            "No se encontró universe_sample.csv. Corré `ctr25 sample` antes de reconstruir eventos."
        )
    universe = pd.read_csv(UNIVERSE_PATH)
    return UniverseIndex(universe)


def _guess_industry(raw: str) -> str:
    if not isinstance(raw, str):
        return "other_services"
    s = raw.strip().lower()
    if not s:
        return "other_services"
    for pattern, slug in _INDUSTRY_PATTERNS:
        if re.search(pattern, s):
            return slug
    return "other_services"


def _common_event_fields(row: pd.Series, *, signal_type: str) -> Dict[str, object]:
    title = _TITLE_MAP.get(signal_type, f"Membership: {signal_type}")
    text = _clean_text(str(row.get("text_snippet", "") or row.get("member_name", "")))
    return {
        "source": "memberships",
        "signal_type": signal_type,
        "signal_strength": 1.0,
        "ts": row.get("ts"),
        "url": row.get("url", ""),
        "title": title,
        "text_snippet": text,
        "climate_score": 1.0,
        "sentiment_label": "positive",
        "sentiment_score": 0.6,
    }


def _build_event(row: pd.Series, match: Optional[UniverseRecord], *, signal_type: str) -> Dict[str, object]:
    base = _common_event_fields(row, signal_type=signal_type)
    if match is not None:
        base.update(
            {
                "company_id": match.company_id,
                "company_qid": match.company_qid,
                "company_name": match.company_name,
                "country": match.country or row.get("country", ""),
                "industry": match.industry or row.get("industry", ""),
                "size_bin": match.size_bin or row.get("size_bin", ""),
            }
        )
    else:
        slug = _slugify(str(row.get("member_name", "")))
        country = str(row.get("country", ""))
        if country:
            slug = f"{slug}-{country.lower()}"
        base.update(
            {
                "company_id": f"ext::{signal_type}::{slug}",
                "company_qid": "",
                "company_name": row.get("member_name", ""),
                "country": country,
                "industry": row.get("industry", ""),
                "size_bin": row.get("size_bin", ""),
            }
        )
    return base


def _prepare_members(df: pd.DataFrame, *, signal_type: str) -> pd.DataFrame:
    out = df.copy()
    out["member_name"] = out["member_name"].fillna("").astype(str)
    out = out[out["member_name"].str.strip() != ""]
    out["member_name_norm"] = out["member_name"].apply(_norm)
    out["industry"] = out["industry"].apply(_guess_industry)
    out["ts"] = out["ts"].apply(lambda x: _parse_ts(x))
    out["url"] = out["url"].fillna("")
    out["country"] = out["country"].fillna("")
    out["size_bin"] = out["size_bin"].fillna("")
    out["text_snippet"] = out.get("text_snippet", "").apply(_clean_text)
    for col in CONTEXT_COLS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("")
    out["signal_type"] = signal_type
    return out


def _match_events(
    df: pd.DataFrame,
    index: UniverseIndex,
    *,
    signal_type: str,
    resolver: IndustryResolver,
) -> pd.DataFrame:
    events: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        match = index.match(row.get("member_name_norm", ""))
        event = _build_event(row, match, signal_type=signal_type)
        event = _classify_event(event, row=row, match=match, signal_type=signal_type, resolver=resolver)
        events.append(event)
    return pd.DataFrame(events)


def _build_context_fields(row: pd.Series) -> Dict[str, object]:
    context: Dict[str, object] = {}
    for col in CONTEXT_COLS:
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            context[col] = value.strip()
    # Ensure key aliases for prompt compatibility
    if "context_sector" in context and "sector" not in context:
        context["sector"] = context["context_sector"]
    if "context_type" in context and "type" not in context:
        context["type"] = context["context_type"]
    if "context_description" in context and "description" not in context:
        context["description"] = context["context_description"]
    if "context_extra" in context and "text_snippet" not in context:
        context["text_snippet"] = context["context_extra"]
    return context


def _classify_event(
    event: Dict[str, object],
    *,
    row: pd.Series,
    match: Optional[UniverseRecord],
    signal_type: str,
    resolver: IndustryResolver,
) -> Dict[str, object]:
    hints: List[str] = []
    existing = [
        row.get("industry"),
        event.get("industry"),
        row.get("context_sector"),
    ]
    if match and match.industry:
        existing.append(match.industry)
    hints = [h for h in existing if isinstance(h, str) and h]
    fields = _build_context_fields(row)
    # always include the canonical snippet sent to the event
    fields.setdefault("text_snippet", event.get("text_snippet", ""))
    classification = resolver.classify_event(
        company_name=str(event.get("company_name", "")),
        source=signal_type,
        country=str(event.get("country", "")),
        fields=fields,
        hints=hints,
    )
    if classification.industry_slug:
        event["industry"] = classification.industry_slug
    description = classification.description.strip()
    if not description:
        description = fields.get("description") or fields.get("text_snippet") or ""
    event["description"] = _clean_text(description)[:400]
    event["_industry_confidence"] = classification.confidence
    event["_industry_provider"] = classification.provider
    return event


def _load_bcorp() -> pd.DataFrame:
    path = RAW_DIR / "memberships" / "b_corp" / "b_corp_data.csv"
    if not path.exists():
        return pd.DataFrame(columns=["member_name"])
    df = pd.read_csv(path)
    df["country"] = df["country"].apply(_map_country)
    df = df[df["country"].isin(_LATAM_CODES)].copy()
    df["member_name"] = df["company_name"].fillna("").astype(str)
    df = df[df["member_name"].str.strip() != ""]
    df["ts"] = df.apply(
        lambda row: _parse_ts(
            row.get("date_certified"),
            row.get("date_first_certified"),
            row.get("assessment_year"),
        ),
        axis=1,
    )
    df["url"] = df[["b_corp_profile", "website"]].fillna("").agg(
        lambda s: next((x for x in s if isinstance(x, str) and x.strip()), ""),
        axis=1,
    )
    df["industry"] = df["industry_category"].fillna(df.get("industry", ""))
    df["size_bin"] = df["size"].apply(_size_from_bcorp)
    df["text_snippet"] = df[["description", "products_and_services"]].fillna("").agg(
        lambda s: next((x for x in s if isinstance(x, str) and x.strip()), ""),
        axis=1,
    )
    df["context_sector"] = df["industry_category"].fillna("")
    df["context_description"] = (
        df["description"] if "description" in df.columns else pd.Series(["" for _ in range(len(df))])
    ).fillna("")
    df["context_extra"] = (
        df["products_and_services"] if "products_and_services" in df.columns else pd.Series(["" for _ in range(len(df))])
    ).fillna("")
    df = df[
        [
            "member_name",
            "ts",
            "url",
            "country",
            "industry",
            "size_bin",
            "text_snippet",
            "context_sector",
            "context_description",
            "context_extra",
        ]
    ]
    return _prepare_members(df, signal_type="bcorp")


def _load_ungc() -> pd.DataFrame:
    path = PROCESSED_DIR / "ungc_participants_latam.csv"
    if not path.exists():
        return pd.DataFrame(columns=["member_name"])
    df = pd.read_csv(path)
    df["country"] = df["country"].apply(_map_country)
    df = df[df["country"].isin(_LATAM_CODES)].copy()
    df["member_name"] = df["name"].fillna("").astype(str)
    df = df[df["member_name"].str.strip() != ""]
    df["ts"] = df["joined_on"].apply(lambda x: _parse_ts(x))
    df["industry"] = df["sector"].fillna("")
    size_map = {
        "small or medium-sized enterprise": "m",
        "company": "l",
    }
    df["size_bin"] = df["type"].str.lower().map(size_map).fillna("")
    df["url"] = df["participant_url"].fillna("")
    df["text_snippet"] = df[["status", "ownership"]].fillna("").agg(lambda s: " | ".join([x for x in s if x]), axis=1)
    df["context_sector"] = df["sector"].fillna("")
    df["context_type"] = df["type"].fillna("")
    df["context_description"] = df["status"].fillna("")
    df["context_extra"] = df["ownership"].fillna("")
    df = df[
        [
            "member_name",
            "ts",
            "url",
            "country",
            "industry",
            "size_bin",
            "text_snippet",
            "context_sector",
            "context_type",
            "context_description",
            "context_extra",
        ]
    ]
    return _prepare_members(df, signal_type="ungc")


def _load_sbti() -> pd.DataFrame:
    path_candidates = [
        RAW_DIR / "memberships" / "sbti" / "sbti_data.csv",
        RAW_DIR / "memberships" / "sbti.csv",
    ]
    path = next((p for p in path_candidates if p.exists()), None)
    if path is None:
        return pd.DataFrame(columns=["member_name"])
    df = pd.read_csv(path)
    df["country"] = df["country"].apply(_map_country)
    df = df[df["country"].isin(_LATAM_CODES)].copy()
    df["member_name"] = df["member_name"].fillna("").astype(str)
    df = df[df["member_name"].str.strip() != ""]
    df["ts"] = df["ts"].apply(lambda x: _parse_ts(x))
    df["industry"] = df["sector"].fillna("")
    df["size_bin"] = ""
    df["url"] = df["url"].fillna("")
    df["text_snippet"] = "Compromiso con objetivos basados en ciencia"
    df["context_sector"] = df["sector"].fillna("")
    df["context_type"] = (
        df["type"] if "type" in df.columns else pd.Series(["" for _ in range(len(df))])
    ).fillna("")
    df["context_description"] = (
        df["target_type"] if "target_type" in df.columns else pd.Series(["" for _ in range(len(df))])
    ).fillna("")
    df["context_extra"] = (
        df["source"] if "source" in df.columns else pd.Series(["" for _ in range(len(df))])
    ).fillna("")
    df = df[
        [
            "member_name",
            "ts",
            "url",
            "country",
            "industry",
            "size_bin",
            "text_snippet",
            "context_sector",
            "context_type",
            "context_description",
            "context_extra",
        ]
    ]
    return _prepare_members(df, signal_type="sbti")


def _load_news(keywords_path: Path = KEYWORDS_PATH, resolver: Optional[IndustryResolver] = None) -> pd.DataFrame:
    news_dir = RAW_DIR / "news"
    if not news_dir.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)

    files = sorted(news_dir.rglob("*.csv"))
    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[rebuild-events] no se pudo leer {path}: {exc}")
            continue
        if df.empty:
            continue
        for col in EVENT_COLUMNS:
            if col not in df.columns:
                default: object = ""
                if col in {"signal_strength", "climate_score", "sentiment_score"}:
                    default = 0.0
                df[col] = default
        frames.append(df[EVENT_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    news = pd.concat(frames, ignore_index=True)
    keywords = _load_keywords(str(keywords_path))
    news = _enrich_events(news, keywords)
    news = news[news["signal_strength"] > 0]
    if news.empty:
        return news

    news = news.copy()
    news["source"] = "news"
    news["signal_type"] = "news"
    news["title"] = news["title"].apply(_clean_text)
    news["text_snippet"] = news["text_snippet"].apply(_clean_text)
    news["country"] = news["country"].fillna("").astype(str).str.upper()
    news = news[news["country"].isin(_LATAM_CODES)]
    news["ts"] = pd.to_datetime(news["ts"], errors="coerce", utc=True)
    news = news.dropna(subset=["ts"])
    if news.empty:
        return news

    news = news.sort_values("ts")
    news["ts"] = news["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    news = news.drop_duplicates(subset=["company_id", "url", "ts"], keep="last")
    if resolver is not None and not news.empty:
        enriched_rows: List[pd.Series] = []
        for _, row in news.iterrows():
            fields = {
                "title": row.get("title", ""),
                "text_snippet": row.get("text_snippet", ""),
                "industry": row.get("industry", ""),
            }
            hints = [row.get("industry", "")]
            classification = resolver.classify_event(
                company_name=str(row.get("company_name", "")),
                source="news",
                country=str(row.get("country", "")),
                fields=fields,
                hints=hints,
            )
            if classification.industry_slug:
                row["industry"] = classification.industry_slug
            description = classification.description.strip()
            if not description:
                description = f"News: {row.get('title', '')}".strip()
            row["description"] = _clean_text(description)[:400]
            row["_industry_confidence"] = classification.confidence
            row["_industry_provider"] = classification.provider
            enriched_rows.append(row)
        news = pd.DataFrame(enriched_rows)
    else:
        news["description"] = news["text_snippet"]
    return news


def main() -> None:
    resolver = IndustryResolver()
    universe_index = _load_universe()

    sources = [
        _load_bcorp(),
        _load_ungc(),
        _load_sbti(),
        _load_news(resolver=resolver),
    ]
    sources = [df for df in sources if not df.empty]
    if not sources:
        raise RuntimeError("No se encontraron datasets de memberships para procesar.")

    frames: List[pd.DataFrame] = []
    for df in sources:
        signal_type = df["signal_type"].iloc[0]
        if signal_type == "news":
            events = df[EVENT_COLUMNS].copy()
        else:
            events = _match_events(df, universe_index, signal_type=signal_type, resolver=resolver)
        frames.append(events)

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise RuntimeError("No se generaron eventos normalizados.")

    # Ordenar columnas según esquema esperado
    combined = combined[EVENT_COLUMNS]
    combined = combined.drop_duplicates(subset=["company_id", "signal_type", "url", "ts"], keep="last")
    combined = combined.sort_values("ts").reset_index(drop=True)

    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(EVENTS_PATH, index=False)
    print(f"Eventos normalizados escritos en {EVENTS_PATH} ({len(combined)} filas)")


if __name__ == "__main__":
    main()
