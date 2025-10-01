from __future__ import annotations

from pathlib import Path
import typer

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
):
    from ctr25.sample_frame import build_sample

    path = build_sample(force=force, per_country=per_country, save_raw=str(save_raw) if save_raw else None)
    typer.echo(f"[sample] wrote {path}")


@app.command("collect-memberships")
def collect_memberships_cmd():
    from ctr25.signals.membership import collect_memberships

    events = collect_memberships()
    added = events.attrs.get("added", len(events)) if hasattr(events, "attrs") else len(events)
    typer.echo(f"[collect-memberships] eventos agregados: {added}")


@app.command("collect-news")
def collect_news(
    universe: str = typer.Option("data/processed/universe_sample.csv", help="Ruta universe_sample.csv"),
    keywords: str = typer.Option("config/keywords.yml", help="Ruta keywords.yml"),
    news_cfg: str = typer.Option("config/news.yml", help="Config de agregador"),
    country: str | None = typer.Option(None, help="MX,BR,CO,CL,AR,UY"),
    industry: str | None = typer.Option(None, help="slug industria"),
    max_companies: int = typer.Option(0, help="límite debug"),
):
    from ctr25.signals.news import run_collect_news

    n = run_collect_news(
        universe_path=universe,
        keywords_path=keywords,
        aggregator_path=news_cfg,
        country=country,
        industry=industry,
        max_companies=max_companies,
    )
    typer.echo(f"[collect-news] eventos agregados: {n}")


@app.command("collect-jobs")
def collect_jobs(
    universe: str = "data/processed/universe_sample.csv",
    keywords: str = "config/keywords.yml",
    country: str | None = None,
    industry: str | None = None,
    max_companies: int = 0,
):
    from ctr25.signals.jobs import run_collect_jobs

    n = run_collect_jobs(
        universe_path=universe,
        keywords_path=keywords,
        country=country,
        industry=industry,
        max_companies=max_companies,
    )
    typer.echo(f"[collect-jobs] eventos agregados: {n}")


@app.command("collect-finance")
def collect_finance(
    universe: str = "data/processed/universe_sample.csv",
    country: str | None = None,
    industry: str | None = None,
    max_companies: int = 0,
):
    from ctr25.signals.finance import run_collect_finance

    n = run_collect_finance(
        universe_path=universe,
        country=country,
        industry=industry,
        max_companies=max_companies,
    )
    typer.echo(f"[collect-finance] eventos agregados: {n}")


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
