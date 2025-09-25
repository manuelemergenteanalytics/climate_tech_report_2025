from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable, List
import pandas as pd
from datetime import datetime, timezone

# ---------- helpers ----------
def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9áéíóúüñ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _load_universe_sample() -> pd.DataFrame:
    p = Path("data/processed/universe_sample.csv")
    if not p.exists():
        raise FileNotFoundError("Falta data/processed/universe_sample.csv (corré `ctr25 sample`).")
    df = pd.read_csv(p)
    if "company_id" not in df.columns or "company_name" not in df.columns:
        raise ValueError("universe_sample.csv debe tener company_id y company_name.")
    df["company_name_norm"] = df["company_name"].apply(_norm)
    return df

def _load_events() -> pd.DataFrame:
    p = Path("data/processed/events_normalized.csv")
    if p.exists():
        return pd.read_csv(p)
    cols = [
        "company_id","company_name","country","industry","size_bin",
        "source","signal_type","signal_strength","ts","url","title","text_snippet"
    ]
    return pd.DataFrame(columns=cols)

def _write_events(df: pd.DataFrame):
    out = Path("data/processed/events_normalized.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

def _dedupe_events(df: pd.DataFrame) -> pd.DataFrame:
    # Dedupe conservador por (company_id, signal_type, url)
    df = df.drop_duplicates(subset=["company_id","signal_type","url"], keep="last")
    return df

def _match_memberships_to_universe(universe: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """
    members: columns => member_name, url(optional), ts(optional), source, signal_type
    Estrategia de matching simple por nombre normalizado (exacto o containment de tokens).
    """
    members = members.copy()
    if "member_name" not in members.columns:
        raise ValueError("members DataFrame requiere columna member_name.")
    members["member_name_norm"] = members["member_name"].apply(_norm)

    # exact match
    exact = members.merge(
        universe[["company_id","company_name","company_name_norm","country","industry","size_bin"]],
        left_on="member_name_norm", right_on="company_name_norm", how="inner"
    )

    # containment (tokens)
    remaining = members[~members.index.isin(exact.index)].copy()
    rows: List[dict] = []
    for idx, row in remaining.iterrows():
        q = row["member_name_norm"]
        q_tokens = set(q.split())
        if not q_tokens:
            continue
        candidates = universe[universe["company_name_norm"].str.contains(list(q_tokens)[0], regex=False)]
        # quick filter: at least 2 shared tokens
        def shared_tokens(x: str) -> int:
            return len(q_tokens.intersection(set(x.split())))
        candidates = candidates.assign(shared=candidates["company_name_norm"].apply(shared_tokens))
        candidates = candidates[candidates["shared"] >= 2].sort_values("shared", ascending=False)
        if not candidates.empty:
            best = candidates.iloc[0]
            d = {
                "member_name": row["member_name"],
                "member_name_norm": q,
                "url": row.get("url", ""),
                "ts": row.get("ts", _now_iso()),
                "source": row["source"],
                "signal_type": row["signal_type"],
                "company_id": best["company_id"],
                "company_name": best["company_name"],
                "country": best.get("country",""),
                "industry": best.get("industry",""),
                "size_bin": best.get("size_bin",""),
            }
            rows.append(d)
    contain = pd.DataFrame(rows)

    if not exact.empty:
        exact = exact.rename(columns={"member_name":"member_name", "url":"url"})
        exact = exact[[
            "member_name","member_name_norm","url","ts","source","signal_type",
            "company_id","company_name","country","industry","size_bin"
        ]]

    matched = pd.concat([exact, contain], ignore_index=True) if not exact.empty or not contain.empty else contain
    return matched

def _events_from_matches(matched: pd.DataFrame) -> pd.DataFrame:
    if matched.empty:
        return matched
    events = pd.DataFrame({
        "company_id": matched["company_id"],
        "company_name": matched["company_name"],
        "country": matched.get("country",""),
        "industry": matched.get("industry",""),
        "size_bin": matched.get("size_bin",""),
        "source": matched["source"],
        "signal_type": matched["signal_type"],
        "signal_strength": 1.0,
        "ts": matched.get("ts", _now_iso()),
        "url": matched.get("url",""),
        "title": matched["signal_type"].apply(lambda s: f"Membership: {s}"),
        "text_snippet": matched["member_name"],
    })
    return events

# ---------- loaders (input flexibles) ----------
def _load_csv_if_exists(path: str, required_cols: Iterable[str]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=list(required_cols))
    df = pd.read_csv(p)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} le faltan columnas: {missing}")
    return df

# Cada fetch_* primero intenta leer un CSV crudo (para QA local).
# Luego podés reemplazar/expandir con requests+parse cuando decidas scrapear directo.

def fetch_sbti() -> pd.DataFrame:
    raw = _load_csv_if_exists(
        "data/raw/memberships/sbti.csv",
        required_cols=["member_name","url","ts"]
    )
    if raw.empty:
        # Placeholder mínimo (para que corra el pipeline)
        return pd.DataFrame(columns=["member_name","url","ts","source","signal_type"])
    raw["source"] = "memberships"
    raw["signal_type"] = "sbti"
    return raw

def fetch_re100() -> pd.DataFrame:
    raw = _load_csv_if_exists(
        "data/raw/memberships/re100.csv",
        required_cols=["member_name","url","ts"]
    )
    if raw.empty:
        return pd.DataFrame(columns=["member_name","url","ts","source","signal_type"])
    raw["source"] = "memberships"
    raw["signal_type"] = "re100"
    return raw

def fetch_bcorps() -> pd.DataFrame:
    raw = _load_csv_if_exists(
        "data/raw/memberships/bcorps.csv",
        required_cols=["member_name","url","ts"]
    )
    if raw.empty:
        return pd.DataFrame(columns=["member_name","url","ts","source","signal_type"])
    raw["source"] = "memberships"
    raw["signal_type"] = "sistema_b"
    return raw

def fetch_ungc() -> pd.DataFrame:
    raw = _load_csv_if_exists(
        "data/raw/memberships/ungc.csv",
        required_cols=["member_name","url","ts"]
    )
    if raw.empty:
        return pd.DataFrame(columns=["member_name","url","ts","source","signal_type"])
    raw["source"] = "memberships"
    raw["signal_type"] = "pacto_global"
    return raw

# ---------- entrypoint ----------
def collect_memberships() -> pd.DataFrame:
    universe = _load_universe_sample()
    sources = [fetch_sbti(), fetch_re100(), fetch_bcorps(), fetch_ungc()]
    sources = [s for s in sources if not s.empty]
    if not sources:
        print("No hay archivos en data/raw/memberships/*.csv aún. Crealos para probar.")
        return pd.DataFrame()

    members = pd.concat(sources, ignore_index=True)
    matched = _match_memberships_to_universe(universe, members)
    events = _events_from_matches(matched)

    current = _load_events()
    out = pd.concat([current, events], ignore_index=True)
    out = _dedupe_events(out)
    _write_events(out)
    print(f"Members → events_normalized.csv (+{len(events)})")
    return events
