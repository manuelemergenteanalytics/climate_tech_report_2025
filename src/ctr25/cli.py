from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from ctr25.utils.names import normalize_company_name
from ctr25.utils.text import classify_industry, load_industry_map


INDUSTRY_SIMPLIFY_MAP = {
    "agro_food_beverage": "agro_food",
    "agro_food": "agro_food",
    "construction_infrastructure_realestate": "construction_realestate",
    "energy_power_utilities": "energy_power",
    "finance_insurance_capital": "finance_insurance",
    "ict_digital_media": "ict_telecom",
    "industrial_manufacturing": "manufacturing",
    "mining_materials": "mining_metals",
    "retail_consumer_services": "retail_consumer",
    "transport_mobility_logistics": "transport_logistics",
    "oil_gas": "oil_gas",
    "water_waste_circularity": "water_waste_circularity",
    "chemicals_materials": "chemicals_materials",
    "healthcare_pharma_biotech": "healthcare_pharma_biotech",
    "professional_services_consulting": "professional_services_consulting",
    "hospitality_tourism_leisure": "hospitality_tourism_leisure",
    "environmental_circular_services": "environmental_circular_services",
    "public_social_education": "public_social_education",
}

app = typer.Typer(help="CTR25 CLI")

_BASE_DIRS = [
    "data/raw/news",
    "data/raw/jobs",
    "data/raw/finance",
    "data/raw/webscan",
    "data/raw/memberships",
    "data/interim/cache",
    "data/interim/logs",
    "data/processed",
    "logs",
]


@app.command("init")
def init_cmd():
    """Crea la estructura mínima de carpetas para correr el pipeline."""
    for d in _BASE_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
    typer.echo("[init] estructura inicial lista")


@app.command("sample")
def sample_cmd(
    force: bool = typer.Option(False, "--force", help="Reconstruye el universo aunque ya exista."),
    per_country: int = typer.Option(500, "--per-country", help="Máximo de empresas por país desde Wikidata."),
    save_raw: Path | None = typer.Option(None, "--save-raw", help="Ruta CSV para guardar el dump crudo de Wikidata."),
    max_workers: int = typer.Option(4, "--max-workers", help="Número de hilos concurrentes por muestreo."),
):
    from ctr25.sample_frame import build_sample

    path = build_sample(
        force=force,
        per_country=per_country,
        save_raw=str(save_raw) if save_raw else None,
        max_workers=max_workers,
    )
    typer.echo(f"[sample] wrote {path}")


@app.command("collect-memberships")
def collect_memberships_cmd():
    from ctr25.signals.membership import collect_memberships

    added = collect_memberships()
    typer.echo(f"[collect-memberships] eventos agregados: {added}")


@app.command("collect-news")
def collect_news(
    universe: str = typer.Option("data/processed/universe_sample.csv", help="Ruta universe_sample.csv"),
    keywords: str = typer.Option("config/keywords.yml", help="Ruta keywords.yml"),
    news_cfg: str = typer.Option("config/news.yml", help="Config de agregador"),
    country: str | None = typer.Option(None, help="Filtra por país"),
    industry: str | None = typer.Option(None, help="Filtra por industria"),
    max_companies: int = typer.Option(0, help="Límite de empresas evaluadas"),
    months: int = typer.Option(12, help="Ventana temporal en meses"),
    since: str | None = typer.Option(None, help="Fecha ISO para reemplazar la ventana"),
):
    from ctr25.signals.news import run_collect_news

    n = run_collect_news(
        universe_path=universe,
        keywords_path=keywords,
        aggregator_path=news_cfg,
        country=country,
        industry=industry,
        max_companies=max_companies,
        months=months,
        since=since,
    )
    typer.echo(f"[collect-news] eventos agregados: {n}")


@app.command("collect-news-discovery")
def collect_news_discovery(
    keywords: str = typer.Option("config/keywords.yml", help="Ruta keywords.yml"),
    industry_map: str = typer.Option("config/industry_map.yml", help="Ruta industry_map.yml"),
    out: str = typer.Option(
        "data/processed/events_normalized_news_discovery.csv",
        help="CSV de salida",
    ),
    since: str = typer.Option("2024-10-01", help="Fecha inicial ISO"),
    until: str = typer.Option("2025-10-13", help="Fecha final ISO"),
    batch_size: int = typer.Option(500, help="Tamaño de lote para procesamiento"),
    gdelt_max: int = typer.Option(250, help="Máximo de artículos GDELT"),
    fetch_content_ratio: float = typer.Option(0.25, help="Fracción de URLs con snippet completo"),
    append_events_csv: bool = typer.Option(
        False,
        "--append-events/--no-append-events",
        help="Agrega los eventos descubiertos a events_normalized.csv",
    ),
    mode: str = typer.Option(
        "general",
        "--mode",
        help="Modo de recolección: general (descubrimiento abierto) o known (empresas del universo)",
    ),
    universe_path: str = typer.Option(
        "data/processed/universe_sample.csv",
        "--universe",
        help="CSV con universe_sample.csv",
    ),
    events_path: str = typer.Option(
        "data/processed/events_normalized.csv",
        "--events",
        help="CSV con events_normalized.csv",
    ),
    known_max_companies: int = typer.Option(
        200,
        "--known-max-companies",
        help="Máximo de compañías conocidas a consultar (0 = todas)",
    ),
):
    from ctr25.signals.news_discovery import run_collect_news_discovery

    mode_value = (mode or "general").lower()
    out_path = out
    if mode_value == "known" and out_path == "data/processed/events_normalized_news_discovery.csv":
        out_path = "data/processed/events_normalized_news_known.csv"

    df, appended = run_collect_news_discovery(
        keywords_path=keywords,
        industry_map_path=industry_map,
        out_csv=out_path,
        since=since,
        until=until,
        batch_size=batch_size,
        gdelt_max=gdelt_max,
        fetch_content_ratio=fetch_content_ratio,
        append_events_csv=append_events_csv,
        mode=mode_value,
        universe_path=universe_path,
        events_path=events_path,
        known_max_companies=known_max_companies,
    )
    typer.echo(f"[collect-news-discovery] filas nuevas: {len(df)}")
    if append_events_csv:
        typer.echo(f"[collect-news-discovery] eventos agregados a normalized: {appended}")


@app.command("collect-jobs")
def collect_jobs(
    universe: str = "data/processed/universe_sample.csv",
    keywords: str = "config/keywords.yml",
    country: str | None = None,
    industry: str | None = None,
    max_companies: int = 0,
    months: int = 12,
    since: str | None = None,
    jobs_cfg: str = "config/jobs.yml",
):
    from ctr25.signals.jobs import run_collect_jobs

    n = run_collect_jobs(
        universe_path=universe,
        keywords_path=keywords,
        country=country,
        industry=industry,
        max_companies=max_companies,
        months=months,
        since=since,
        jobs_cfg_path=jobs_cfg,
    )
    typer.echo(f"[collect-jobs] eventos agregados: {n}")


@app.command("collect-finance")
def collect_finance(
    universe: str = "data/processed/universe_sample.csv",
    country: str | None = None,
    industry: str | None = None,
    max_companies: int = 0,
    months: int = 12,
    since: str | None = None,
):
    from ctr25.signals.finance import run_collect_finance

    n = run_collect_finance(
        universe_path=universe,
        country=country,
        industry=industry,
        max_companies=max_companies,
        months=months,
        since=since,
    )
    typer.echo(f"[collect-finance] eventos agregados: {n}")


@app.command("reclassify-industries")
def reclassify_industries(
    input_csv: Path = typer.Option(
        Path("data/processed/events_normalized.csv"),
        "--in",
        help="CSV de entrada con events_normalized",
    ),
    output_csv: Path = typer.Option(
        Path("data/processed/events_normalized.reclass.csv"),
        "--out",
        help="CSV de salida con industrias reclasificadas",
    ),
    log_csv: Path = typer.Option(
        Path("data/processed/reclass.log.csv"),
        "--log",
        help="CSV de log con los cambios aplicados",
    ),
    industry_map_path: Path = typer.Option(
        Path("config/industry_map.yml"),
        "--industry-map",
        help="Ruta al archivo industry_map.yml",
    ),
    fix_names: bool = typer.Option(
        False,
        "--fix-names/--no-fix-names",
        help="Aplica heurísticas conservadoras para limpiar nombres de compañías",
    ),
):
    imap = load_industry_map(str(industry_map_path))

    if not input_csv.exists():
        raise typer.BadParameter(f"No existe el archivo de entrada: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        typer.echo("[reclassify-industries] CSV sin filas, nada que hacer")
        return

    columns = df.columns.tolist()

    alias_changes = 0
    weighted_changes = 0
    token_hits = 0
    unknown_hits = 0
    log_rows: list[dict[str, object]] = []

    for idx, row in df.iterrows():
        original_industry = row.get("industry", "")
        context = {
            "company_name": row.get("company_name", ""),
            "display_name": row.get("display_name", ""),
            "source_meta": row.get("source_meta", ""),
            "text_snippet": row.get("text_snippet", ""),
            "title": row.get("title", ""),
            "description": row.get("description", ""),
            "url": row.get("url", ""),
        }

        slug, details = classify_industry((original_industry, context), imap)
        target_slug = INDUSTRY_SIMPLIFY_MAP.get(slug, slug)
        reason = details.get("reason") if isinstance(details, dict) else None

        if reason == "token":
            token_hits += 1
        if target_slug == "unknown":
            unknown_hits += 1

        updated_industry = original_industry
        change_reason: str | None = None
        score_value = details.get("score") if isinstance(details, dict) else None
        sector_hint = None
        if isinstance(details, dict):
            sector_hint = details.get("sector_hint_slug") or details.get("sector_hint")

        if reason == "alias" and target_slug and target_slug != original_industry:
            updated_industry = target_slug
            alias_changes += 1
            change_reason = "alias"
        elif reason == "weighted" and target_slug and target_slug != original_industry:
            updated_industry = target_slug
            weighted_changes += 1
            change_reason = "weighted"

        if change_reason:
            df.at[idx, "industry"] = updated_industry
            entity_id = row.get("company_id") or row.get("company_qid") or row.get("company_name")
            log_rows.append(
                {
                    "row_idx": int(idx),
                    "entity_id": entity_id,
                    "old_industry": original_industry,
                    "new_industry": updated_industry,
                    "reason": change_reason,
                    "score": score_value,
                    "sector_hint": sector_hint,
                }
            )

        if fix_names:
            normalized_name = normalize_company_name(row)
            if normalized_name and normalized_name != row.get("company_name", ""):
                df.at[idx, "company_name"] = normalized_name

    # Reordenamos columnas para preservar el orden original (por precaución)
    df = df[columns]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    log_csv.parent.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(log_rows, columns=[
        "row_idx",
        "entity_id",
        "old_industry",
        "new_industry",
        "reason",
        "score",
        "sector_hint",
    ])
    log_df.to_csv(log_csv, index=False)

    typer.echo(
        f"[reclassify-industries] Scanned {len(df)}; changes: alias={alias_changes} "
        f"weighted={weighted_changes} token={token_hits}; unknown={unknown_hits}"
    )


@app.command("collect-webscan")
def collect_webscan(
    universe: str = "data/processed/universe_sample.csv",
    keywords: str = "config/keywords.yml",
    country: str | None = None,
    industry: str | None = None,
    max_companies: int = 0,
):
    from ctr25.signals.webscan import run_collect_webscan

    n = run_collect_webscan(
        universe_path=universe,
        keywords_path=keywords,
        country=country,
        industry=industry,
        max_companies=max_companies,
    )
    typer.echo(f"[collect-webscan] eventos agregados: {n}")


@app.command("qa")
def qa_cmd(
    universe: str = "data/processed/universe_sample.csv",
    events: str = "data/processed/events_normalized.csv",
    out_dir: str = "data/interim/qa",
):
    from ctr25.utils.qa import run_qa

    out = run_qa(universe_path=universe, events_path=events, out_dir=out_dir)
    typer.echo(f"[qa] summary: {out}")


@app.command("compute-iic")
def compute_iic_cmd():
    from ctr25.iic import compute_iic_cli

    compute_iic_cli()


@app.command("compute-ps")
def compute_ps_cmd():
    from ctr25.prospect import compute_ps_cli

    compute_ps_cli()


@app.command("viz")
def viz_cmd():
    from ctr25.visualize import main as do_viz

    do_viz()


@app.command("export")
def export_cmd(out: str = typer.Option(..., "--out", help="Destino del CSV exportado")):
    from ctr25.visualize import export_prospects

    export_prospects(out)


if __name__ == "__main__":
    app()
