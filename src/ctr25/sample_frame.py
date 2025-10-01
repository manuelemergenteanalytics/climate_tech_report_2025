from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

from ctr25.sample_sources.wikidata import (
    apply_sampling,
    map_industries,
    query_wikidata,
)


def _load_project() -> dict:
    """Return project configuration if available."""
    candidates = [Path("config/project.yml"), Path("config/proyect.yml")]
    for path in candidates:
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
    """Construct universe_sample.csv using Wikidata."""
    out_path = Path("data/processed/universe_sample.csv")
    if out_path.exists() and not force:
        existing = pd.read_csv(out_path)
        print(f"Universo existente -> {out_path} (n={len(existing)})")
        return str(out_path)

    project_cfg = _load_project()

    raw_df = query_wikidata(companies_per_country=per_country)
    if raw_df.empty:
        raise ValueError("Wikidata query returned no companies. Ajustá per-country o revisá conexión.")

    if save_raw:
        raw_path = Path(save_raw)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_path, index=False)
        print(f"Raw Wikidata -> {raw_path} (n={len(raw_df)})")

    mapped_df = map_industries(raw_df)
    if mapped_df.empty:
        raise ValueError("No se pudo mapear industrias. Revisá config/industry_map.yml.")

    sample_df = apply_sampling(mapped_df, project_cfg)
    if sample_df.empty:
        raise ValueError("No se pudo construir el universo estratificado. Verificá filtros en project.yml.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(out_path, index=False)
    print(f"Muestra estratificada -> {out_path} (n={len(sample_df)})")

    _write_strata_counts(sample_df)
    return str(out_path)
