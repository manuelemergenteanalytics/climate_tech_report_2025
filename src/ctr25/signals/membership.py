from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Iterable, List
import pandas as pd
from datetime import datetime, timezone

from ctr25.utils.events import append_events

from ctr25.signals.memberships_live import (
    fetch_bcorps as fetch_bcorps_live,
    fetch_re100 as fetch_re100_live,
    fetch_sbti as fetch_sbti_live,
    fetch_ungc as fetch_ungc_live,
    load_memberships_cfg,
)

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


RAW_DIR = Path("data/raw/memberships")


def _persist_raw_membership(df: pd.DataFrame, label: str) -> Path | None:
    if df.empty:
        return None
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "memberships"
    out_dir = RAW_DIR / safe_label
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{ts_stamp}_{len(df)}.csv"
    df.to_csv(path, index=False)
    return path


def _coerce_company_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _normalize_company_id_output(value):
    cid = _coerce_company_id(value)
    if cid.isdigit():
        try:
            return int(cid)
        except ValueError:
            return cid
    return cid


_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _coerce_ts(value) -> str:
    if pd.isna(value):
        return _now_iso()
    s = str(value).strip()
    if not s:
        return _now_iso()
    if _DATE_ONLY_RE.match(s):
        return f"{s}T00:00:00Z"
    return s


def _ensure_membership_df(
    df: pd.DataFrame,
    *,
    default_signal_type: str,
    default_source: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "member_name" not in df.columns:
        if "company_name" in df.columns:
            df["member_name"] = df["company_name"]
        else:
            df["member_name"] = ""
    df["member_name"] = df["member_name"].fillna("").astype(str)

    if "company_name" in df.columns:
        df["company_name"] = df["company_name"].fillna("").astype(str)

    if "company_id" in df.columns:
        df["company_id"] = df["company_id"].apply(_coerce_company_id)

    if "signal_type" in df.columns:
        df["signal_type"] = df["signal_type"].fillna(default_signal_type)
    else:
        df["signal_type"] = default_signal_type

    if "source" in df.columns:
        df["source"] = df["source"].fillna(default_source)
    else:
        df["source"] = default_source

    if "ts" in df.columns:
        df["ts"] = df["ts"].apply(_coerce_ts)
    else:
        df["ts"] = _now_iso()

    if "url" in df.columns:
        df["url"] = df["url"].fillna("")
    else:
        df["url"] = ""

    for col in ["country", "industry", "size_bin"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df

def _load_universe_sample() -> pd.DataFrame:
    p = Path("data/processed/universe_sample.csv")
    if not p.exists():
        raise FileNotFoundError("Falta data/processed/universe_sample.csv (corré `ctr25 sample`).")
    df = pd.read_csv(p)
    if "company_id" not in df.columns or "company_name" not in df.columns:
        raise ValueError("universe_sample.csv debe tener company_id y company_name.")
    if "company_qid" not in df.columns:
        df["company_qid"] = ""
    df["company_name_norm"] = df["company_name"].apply(_norm)
    return df

def _match_memberships_to_universe(universe: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Enlaza filas de memberships con el universo por company_id o nombre."""

    members = members.copy()
    if members.empty:
        return members

    if "member_name" not in members.columns:
        raise ValueError("members DataFrame requiere columna member_name.")

    members["member_name"] = members["member_name"].fillna("").astype(str)
    members["member_name_norm"] = members["member_name"].apply(_norm)
    members["_orig_idx"] = members.index

    universe = universe.copy()
    if "company_name_norm" not in universe.columns:
        universe["company_name_norm"] = universe["company_name"].apply(_norm)

    universe_lookup: Dict[str, pd.Series] = {}
    for _, row in universe.iterrows():
        cid = _coerce_company_id(row.get("company_id"))
        if cid:
            universe_lookup[cid] = row

    matched_rows: List[dict] = []
    processed_idx: set[int] = set()

    if "company_id" in members.columns:
        for idx, row in members.iterrows():
            cid = _coerce_company_id(row.get("company_id"))
            if not cid:
                continue
            processed_idx.add(row["_orig_idx"])
            best = universe_lookup.get(cid)
            company_id_val = best.get("company_id", cid) if best is not None else cid
            data = {
                "member_name": row["member_name"],
                "member_name_norm": row["member_name_norm"],
                "url": str(row.get("url", "") or ""),
                "ts": _coerce_ts(row.get("ts")),
                "source": row["source"],
                "signal_type": row["signal_type"],
                "company_id": _normalize_company_id_output(company_id_val),
                "company_qid": best.get("company_qid", "") if best is not None else "",
                "company_name": best.get("company_name", row.get("company_name", row["member_name"])) if best is not None else (row.get("company_name") or row["member_name"]),
                "country": best.get("country", row.get("country", "")) if best is not None else row.get("country", ""),
                "industry": best.get("industry", row.get("industry", "")) if best is not None else row.get("industry", ""),
                "size_bin": best.get("size_bin", row.get("size_bin", "")) if best is not None else row.get("size_bin", ""),
            }
            matched_rows.append(data)

    matched_frames: List[pd.DataFrame] = []
    if matched_rows:
        matched_frames.append(pd.DataFrame(matched_rows))

    remaining = members[~members["_orig_idx"].isin(processed_idx)].copy()

    if not remaining.empty:
        base_cols = ["member_name", "member_name_norm", "url", "ts", "source", "signal_type", "_orig_idx"]
        left = remaining[base_cols]
        exact = left.merge(
            universe[["company_id", "company_qid", "company_name", "company_name_norm", "country", "industry", "size_bin"]],
            left_on="member_name_norm",
            right_on="company_name_norm",
            how="inner",
        )
        if not exact.empty:
            processed_idx.update(exact["_orig_idx"].tolist())
            exact = exact[[
                "member_name",
                "member_name_norm",
                "url",
                "ts",
                "source",
                "signal_type",
                "company_id",
                "company_qid",
                "company_name",
                "country",
                "industry",
                "size_bin",
            ]]
            matched_frames.append(exact)

        fuzzy_candidates = remaining[~remaining["_orig_idx"].isin(processed_idx)].copy()
        if not fuzzy_candidates.empty:
            rows: List[dict] = []
            for _, row in fuzzy_candidates.iterrows():
                q = row["member_name_norm"]
                q_tokens = set(q.split())
                if not q_tokens:
                    continue
                first = next(iter(q_tokens))
                candidates = universe[universe["company_name_norm"].str.contains(first, regex=False)]

                def shared_tokens(x: str) -> int:
                    return len(q_tokens.intersection(set(x.split())))

                candidates = candidates.assign(shared=candidates["company_name_norm"].apply(shared_tokens))
                candidates = candidates[candidates["shared"] >= 2].sort_values("shared", ascending=False)
                if candidates.empty:
                    continue
                best = candidates.iloc[0]
                rows.append({
                    "member_name": row["member_name"],
                    "member_name_norm": q,
                    "url": row["url"],
                    "ts": row["ts"],
                    "source": row["source"],
                    "signal_type": row["signal_type"],
                    "company_id": best["company_id"],
                    "company_qid": best.get("company_qid", ""),
                    "company_name": best["company_name"],
                    "country": best.get("country", ""),
                    "industry": best.get("industry", ""),
                    "size_bin": best.get("size_bin", ""),
                })
            if rows:
                matched_frames.append(pd.DataFrame(rows))

    if not matched_frames:
        return pd.DataFrame(columns=[
            "member_name",
            "member_name_norm",
            "url",
            "ts",
            "source",
            "signal_type",
            "company_id",
            "company_qid",
            "company_name",
            "country",
            "industry",
            "size_bin",
        ])

    result = pd.concat(matched_frames, ignore_index=True)
    return result

def _events_from_matches(matched: pd.DataFrame) -> pd.DataFrame:
    if matched.empty:
        return matched
    events = pd.DataFrame({
        "company_id": matched["company_id"],
        "company_qid": matched.get("company_qid", ""),
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
        "climate_score": 1.0,
        "sentiment_label": "positive",
        "sentiment_score": 0.6,
    })
    return events

# ---------- loaders (input flexibles) ----------
def _load_csv_if_exists(
    path: str,
    required_cols: Iterable[str],
    aliases: Dict[str, Iterable[str]] | None = None,
) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=list(required_cols))
    df = pd.read_csv(p)
    missing: List[str] = []
    for col in required_cols:
        if col in df.columns:
            continue
        if aliases and col in aliases:
            for alt in aliases[col]:
                if alt in df.columns:
                    df[col] = df[alt]
                    break
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"{path} le faltan columnas: {missing}")
    return df

# Cada fetch_* primero intenta leer un CSV crudo (para QA local).
# Luego podés reemplazar/expandir con requests+parse cuando decidas scrapear directo.

def fetch_sbti() -> pd.DataFrame:
    raw = _load_csv_if_exists(
        "data/raw/memberships/sbti.csv",
        required_cols=["member_name","url","ts"],
        aliases={"member_name": ["company_name"]},
    )
    return _ensure_membership_df(raw, default_signal_type="sbti", default_source="memberships")

def fetch_re100() -> pd.DataFrame:
    raw = _load_csv_if_exists(
        "data/raw/memberships/re100.csv",
        required_cols=["member_name","url","ts"],
        aliases={"member_name": ["company_name"]},
    )
    return _ensure_membership_df(raw, default_signal_type="re100", default_source="memberships")

def fetch_bcorps() -> pd.DataFrame:
    raw = _load_csv_if_exists(
        "data/raw/memberships/bcorps.csv",
        required_cols=["member_name","url","ts"],
        aliases={"member_name": ["company_name"]},
    )
    return _ensure_membership_df(raw, default_signal_type="bcorp", default_source="memberships")

def fetch_ungc() -> pd.DataFrame:
    raw = _load_csv_if_exists(
        "data/raw/memberships/ungc.csv",
        required_cols=["member_name","url","ts"],
        aliases={"member_name": ["company_name"]},
    )
    return _ensure_membership_df(raw, default_signal_type="ungc", default_source="memberships")

def fetch_linkedin_memberships() -> pd.DataFrame:
    raw = _load_csv_if_exists(
        "data/raw/memberships/memberships_linkedin.csv",
        required_cols=["member_name", "url"],
        aliases={"member_name": ["company_name"]},
    )
    return _ensure_membership_df(
        raw,
        default_signal_type="linkedin_membership",
        default_source="linkedin",
    )


# ---------- entrypoint ----------
def collect_memberships(cfg_path: str = "config/memberships.yml") -> int:
    universe = _load_universe_sample()
    sources = [
        fetch_sbti(),
        fetch_re100(),
        fetch_bcorps(),
        fetch_ungc(),
        fetch_linkedin_memberships(),
    ]
    cfg = load_memberships_cfg(cfg_path)
    live_sources = []
    if cfg:
        mapping = [
            ("sbti_url", fetch_sbti_live, "sbti"),
            ("re100_url", fetch_re100_live, "re100"),
            ("ungc_url", fetch_ungc_live, "ungc"),
            ("bcorps_url", fetch_bcorps_live, "bcorp"),
        ]
        for key, func, kind in mapping:
            url = cfg.get(key)
            if not url:
                continue
            try:
                df_live = func(url)
            except Exception as exc:
                print(f"[collect-memberships] fallo al descargar {kind} ({url}): {exc}")
                continue
            if df_live.empty:
                continue
            raw_path = _persist_raw_membership(df_live, kind)
            if raw_path is not None:
                print(f"[collect-memberships] dump crudo -> {raw_path}")
            df_live["signal_type"] = kind
            df_live = _ensure_membership_df(
                df_live,
                default_signal_type=kind,
                default_source="memberships",
            )
            live_sources.append(df_live)

    sources.extend(live_sources)
    sources = [s for s in sources if not s.empty]
    if not sources:
        print("No se encontraron memberships locales ni remotos. Revisá data/raw/memberships o config/memberships.yml.")
        return 0

    members = pd.concat(sources, ignore_index=True)
    matched = _match_memberships_to_universe(universe, members)
    events = _events_from_matches(matched)
    if events.empty:
        return 0
    return append_events(events, source="memberships", signal_type="memberships")
