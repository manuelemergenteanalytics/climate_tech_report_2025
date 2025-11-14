"""Render the hover-enabled map HTML and capture a screenshot with all tooltips visible."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

FOCUS_COUNTRIES = ["AR", "BR", "CL", "CO", "MX", "PE", "UY"]
REPO_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = (REPO_ROOT / "docs/market_radar/html/map_intensity_focus_hover.html").resolve()
OUTPUT_PATH = (REPO_ROOT / "assets/map_intensity_focus_hover.png").resolve()
LIB_ARCHIVE_ROOT = REPO_ROOT / "third_party/lib_overrides"
EXTRA_LIB_DIR = LIB_ARCHIVE_ROOT / "usr/lib/x86_64-linux-gnu"
DEB_DEPENDENCIES = [
    REPO_ROOT / "libnspr4_2%3a4.35-1.1build1_amd64.deb",
    REPO_ROOT / "libnss3_2%3a3.98-1build1_amd64.deb",
    REPO_ROOT / "libasound2t64_1.2.11-1ubuntu0.1_amd64.deb",
]


def _ensure_libs_available() -> None:
    """Append local copies of system libs (if present) for headless Chromium."""
    if not EXTRA_LIB_DIR.exists():
        return
    current = os.environ.get("LD_LIBRARY_PATH")
    extra = str(EXTRA_LIB_DIR)
    if current:
        parts = current.split(os.pathsep)
        if extra in parts:
            return
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([extra, current])
    else:
        os.environ["LD_LIBRARY_PATH"] = extra


def _prepare_embedded_libs() -> None:
    """Extract the .deb archives locally when system libs are unavailable."""
    if EXTRA_LIB_DIR.exists():
        return
    available = [path for path in DEB_DEPENDENCIES if path.exists()]
    if not available:
        return
    LIB_ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    for deb_path in available:
        try:
            subprocess.run(
                ["dpkg", "-x", str(deb_path), str(LIB_ARCHIVE_ROOT)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("dpkg no está disponible; omitiendo extracción de librerías locales.")
            break
        except subprocess.CalledProcessError as err:
            print(f"No se pudo extraer {deb_path.name}: {err}")
            break


def main() -> None:
    if not HTML_PATH.exists():
        raise SystemExit(f"No existe el archivo HTML: {HTML_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _prepare_embedded_libs()
    _ensure_libs_available()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 900, "device_scale_factor": 2})
        page.goto(HTML_PATH.as_uri())
        page.wait_for_load_state("networkidle")
        page.wait_for_selector(".plotly-graph-div")
        page.evaluate(
            """
            (focusCodes) => {
                const graphDiv = document.querySelector('.plotly-graph-div');
                if (!graphDiv || !graphDiv.data || !graphDiv.data.length) {
                    return;
                }
                const customCountries = (graphDiv.data[0].customdata || []).map(row => row[0]);
                const hoverPoints = [];
                focusCodes.forEach(code => {
                    const idx = customCountries.indexOf(code);
                    if (idx >= 0) {
                        hoverPoints.push({curveNumber: 0, pointNumber: idx});
                    }
                });
                if (hoverPoints.length) {
                    Plotly.Fx.hover(graphDiv, hoverPoints, 'geo');
                    const hoverLayer = graphDiv && graphDiv.querySelector('.hoverlayer');
                    if (hoverLayer) {
                        hoverLayer.style.pointerEvents = 'none';
                        hoverLayer.style.opacity = '1';
                    }
                }
            }
            """,
            FOCUS_COUNTRIES,
        )
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUTPUT_PATH), full_page=False)
        browser.close()

    print(f"Captura guardada en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
