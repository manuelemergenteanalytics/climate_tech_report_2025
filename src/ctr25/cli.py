from typer import Typer
from . import sample_frame, normalize, iic, prospect, visualize
from pathlib import Path

app = Typer(help="Climate Tech Report 2025 CLI")

@app.command()
def init():
    for p in ["data/raw","data/interim","data/processed","reports/figures","reports/public","reports/collaborators"]:
        Path(p).mkdir(parents=True, exist_ok=True)
    print("Estructura creada.")

@app.command()
def sample():
    """Construye muestra estratificada según config.sample (country×industry)."""
    sample_frame.build_sample()

@app.command()
def collect(demo: bool = True):
    normalize.combine_mock_signals()

@app.command("compute-iic")
def compute_iic():
    iic.compute()

@app.command("compute-ps")
def compute_ps():
    prospect.compute()

@app.command()
def viz():
    visualize.build_all()

@app.command()
def export(out: str = "reports/collaborators/prospects.csv"):
    visualize.export_prospects(out)

if __name__ == "__main__":
    app()
