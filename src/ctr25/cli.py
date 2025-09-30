# src/ctr25/cli.py (Typer)
from __future__ import annotations
import typer
from ctr25.sample_frame import build_sample
from ctr25.signals import (
    collect_memberships,
    run_collect_finance,
    run_collect_jobs,
    run_collect_news,
    run_collect_webscan,
)
from ctr25.utils.qa import run_qa
from ctr25.iic import compute_iic_cli
from ctr25.prospect import compute_ps_cli

app = typer.Typer(help="CTR25 CLI")

@app.command("sample")
def sample_cmd():
    path = build_sample()
    typer.echo(f"[sample] wrote {path}")

@app.command("collect-memberships")
def collect_memberships_cmd():
    events = collect_memberships()
    total = len(events) if events is not None else 0
    typer.echo(f"[collect-memberships] eventos agregados: {total}")

@app.command("viz")
def viz_cmd():
    from ctr25.visualize import main as do_viz
    do_viz()

# --- News que te pasé recién (Typer wrapper)
@app.command("collect-news")
def collect_news(
    universe: str = typer.Option("data/processed/universe_sample.csv", help="Ruta universe_sample.csv"),
    keywords: str = typer.Option("config/keywords.yml", help="Ruta keywords.yml"),
    news_cfg: str = typer.Option("config/news.yml", help="Config de agregador"),
    country: str | None = typer.Option(None, help="MX,BR,CO,CL,AR,UY"),
    industry: str | None = typer.Option(None, help="slug industria"),
    max_companies: int = typer.Option(0, help="límite debug"),
):
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
    universe: str = typer.Option("data/processed/universe_sample.csv", help="Ruta universe_sample.csv"),
    keywords: str = typer.Option("config/keywords.yml", help="Ruta keywords.yml"),
    country: str | None = typer.Option(None, help="MX,BR,CO,CL,AR,UY"),
    industry: str | None = typer.Option(None, help="slug industria"),
    max_companies: int = typer.Option(0, help="límite debug"),
):
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
    universe: str = typer.Option("data/processed/universe_sample.csv", help="Ruta universe_sample.csv"),
    country: str | None = typer.Option(None, help="MX,BR,CO,CL,AR,UY"),
    industry: str | None = typer.Option(None, help="slug industria"),
    max_companies: int = typer.Option(0, help="límite debug"),
):
    n = run_collect_finance(
        universe_path=universe,
        country=country,
        industry=industry,
        max_companies=max_companies,
    )
    typer.echo(f"[collect-finance] eventos agregados: {n}")

@app.command("collect-webscan")
def collect_webscan(
    universe: str = typer.Option("data/processed/universe_sample.csv", help="Ruta universe_sample.csv"),
    keywords: str = typer.Option("config/keywords.yml", help="Ruta keywords.yml"),
    country: str | None = typer.Option(None, help="MX,BR,CO,CL,AR,UY"),
    industry: str | None = typer.Option(None, help="slug industria"),
    max_companies: int = typer.Option(0, help="límite debug"),
):
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
    universe: str = typer.Option("data/processed/universe_sample.csv", help="Ruta universe_sample.csv"),
    events: str = typer.Option("data/processed/events_normalized.csv", help="Ruta events_normalized.csv"),
    out_dir: str = typer.Option("data/interim/qa", help="Directorio de salida"),
):
    out = run_qa(universe_path=universe, events_path=events, out_dir=out_dir)
    typer.echo(f"[qa] summary: {out}")

@app.command("compute-iic")
def compute_iic_cmd():
    compute_iic_cli()

@app.command("compute-ps")
def compute_ps_cmd():
    compute_ps_cli()
