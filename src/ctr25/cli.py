from typer import Typer, Option
from pathlib import Path
from . import sample_frame, normalize, iic, prospect, visualize


app = Typer(help="Climate Tech Report 2025 CLI")


@app.command()
def init():
"""Crear carpetas base y copiar configs por defecto."""
for p in [
"data/raw", "data/interim", "data/processed",
"reports/figures", "reports/public", "reports/collaborators"
]:
Path(p).mkdir(parents=True, exist_ok=True)
print("Estructura creada.")


@app.command()
def sample(use_mock: bool = Option(True, help="Construir universo/muestra mock")):
sample_frame.build_mock()


@app.command()
def collect(demo: bool = Option(True, help="Recolectar señales demo (mocks)")):
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
def export(out: str = Option("reports/collaborators/prospects.csv")):
visualize.export_prospects(out)


if __name__ == "__main__":
app()