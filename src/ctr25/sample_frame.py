from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml

from ctr25.sample_sources.wikidata import (
    query_wikidata,
    map_industries,
    apply_sampling,
)


def _load_project() -> dict:
    """Return project configuration if available."""
    for path in (Path("config/project.yml"), Path("config/proyect.yml")):
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    return {}


def _write_strata_counts(df: pd.DataFrame) -> None:
    qa_dir = Path("data/interim/qa")
    qa_dir.mkdir(parents=True, exist_ok=True)
    counts = (
        df.groupby(["country", "industry"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    counts.to_csv(qa_dir / "strata_counts.csv", index=False)


def build_sample(
    *,
    force: bool = False,
    per_country: int = 500,
    save_raw: str | None = None,
) -> str:
    """
    Construye data/processed/universe_sample.csv desde Wikidata:
      query_wikidata -> map_industries -> apply_sampling
    """
    out_path = Path("data/processed/universe_sample.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        existing = pd.read_csv(out_path)
        print(f"Muestra existente -> {out_path} (n={len(existing)})")
        return str(out_path)

    project_cfg = _load_project()
    sample_cfg = (project_cfg or {}).get("sample", {}) or {}
    countries = sample_cfg.get("countries") or (project_cfg or {}).get("countries")

    # 1) Query
    raw = query_wikidata(
        companies_per_country=per_country,
        countries=countries,
    )
    if raw.empty:
        raise ValueError(
            "Wikidata devolvió 0 filas. Ajustá per_country o revisá la conexión."
        )

    if save_raw:
        raw_path = Path(save_raw)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_path, index=False)
        print(f"Raw Wikidata -> {raw_path} (n={len(raw)})")

    # 2) Map industries (crea 'industry')
    mapped = map_industries(raw)
    if mapped.empty or "industry" not in mapped.columns:
        raise ValueError(
            "No se pudo mapear industrias. Revisá config/industry_map.yml y 'industry_raw'."
        )

    # 3) Sampling estratificado
    final = apply_sampling(mapped, project_cfg)
    if final.empty:
        raise ValueError(
            "No se pudo construir el universo estratificado. Verificá filtros en project.yml."
        )

    final.to_csv(out_path, index=False)
    print(f"Muestra estratificada -> {out_path} (n={len(final)})")

    _write_strata_counts(final)
    return str(out_path)
