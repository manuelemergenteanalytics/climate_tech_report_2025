# src/ctr25/cli.py (Typer)
from __future__ import annotations
import typer
from ctr25.signals.jobs import run_collect_jobs
from ctr25.signals.finance import run_collect_finance

app = typer.Typer(help="CTR25 CLI")

@app.command("sample")
def sample_cmd():
    from ctr25.sample_frame import main as do_sample
    do_sample()

@app.command("collect-memberships")
def collect_memberships_cmd():
    from ctr25.prospect import collect_memberships as do_memberships
    do_memberships()

@app.command("compute-iic")
def compute_iic_cmd():
    from ctr25.iic import compute_iic as do_iic
    do_iic()

@app.command("compute-ps")
def compute_ps_cmd():
    from ctr25.iic import compute_ps as do_ps
    do_ps()

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
    from ctr25.signals.news import run_collect_news
    n = run_collect_news(universe_path=universe, keywords_path=keywords, aggregator_path=news_cfg,
                         country=country, industry=industry, max_companies=max_companies)
    typer.echo(f"[collect-news] eventos agregados: {n}")

@app.command("collect-jobs")
def collect_jobs(
    universe: str = "data/processed/universe_sample.csv",
    keywords: str = "config/keywords.yml",
    country: str | None = None,
    industry: str | None = None,
    max_companies: int = 0,
):
    n = run_collect_jobs(universe_path=universe, keywords_path=keywords,
                         country=country, industry=industry, max_companies=max_companies)
    print(f"[collect-jobs] eventos agregados: {n}")

@app.command("collect-finance")
def collect_finance(
    universe: str = "data/processed/universe_sample.csv",
    country: str | None = None,
    industry: str | None = None,
    max_companies: int = 0,
):
    n = run_collect_finance(universe_path=universe, country=country, industry=industry, max_companies=max_companies)
    print(f"[collect-finance] eventos agregados: {n}")

@app.command("collect-finance")
def collect_finance(
    universe: str = "data/processed/universe_sample.csv",
    country: str | None = None,
    industry: str | None = None,
    max_companies: int = 0,
):
    n = run_collect_finance(universe_path=universe, country=country, industry=industry, max_companies=max_companies)
    print(f"[collect-finance] eventos agregados: {n}")