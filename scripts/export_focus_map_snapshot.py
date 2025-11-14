"""Generate a copy of the intensity map with all hover labels shown at once."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable, List, Dict
from uuid import uuid4

from plotly.utils import PlotlyJSONEncoder

MODULE_PATH = Path(__file__).with_name("generate_market_radar.py")
FOCUS_COUNTRIES = ("AR", "BR", "CL", "CO", "PE", "UY", "MX")
OUTPUT_FILENAME = "map_intensity_focus_hover.html"

FOCUS_COUNTRY_CENTROIDS: Dict[str, Dict[str, float]] = {
    "AR": {"lat": -38.4161, "lon": -63.6167},
    "BR": {"lat": -10.0, "lon": -55.0},
    "CL": {"lat": -35.6751, "lon": -71.5430},
    "CO": {"lat": 4.5709, "lon": -74.2973},
    "MX": {"lat": 23.6345, "lon": -102.5528},
    "PE": {"lat": -9.1899, "lon": -75.0152},
    "UY": {"lat": -32.5228, "lon": -55.7658},
}


def _load_market_radar_module():
    spec = importlib.util.spec_from_file_location("market_radar_hover", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar generate_market_radar.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_geo(companies, radar_module):
    geo = (
        companies.groupby("country", as_index=False)
        .agg(total_score=("demand_score", "sum"), companies=("company_id", "count"))
    )
    geo["country"] = geo["country"].str.upper()
    geo["iso3"] = geo["country"].map(radar_module.ISO2_TO_ISO3)
    geo = geo[geo["iso3"].notna()].reset_index(drop=True)
    return geo


def _build_figure(radar_module, geo):
    fig = radar_module.px.choropleth(
        geo,
        locations="iso3",
        locationmode="ISO-3",
        color="total_score",
        custom_data=geo[["country", "companies", "total_score"]],
        title="Intensidad de señales climáticas/digitales por país",
        color_continuous_scale=[step[1] for step in radar_module.MARKET_RADAR_SEQUENTIAL],
    )
    fig.update_traces(
        hovertemplate=(
            "País=%{customdata[0]}<br>"
            "Empresas=%{customdata[1]:,.0f}<br>"
            "Puntaje total=%{customdata[2]:,.2f}<extra></extra>"
        )
    )
    fig.update_layout(title=None, width=880, height=640, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_coloraxes(
        colorscale=radar_module.MARKET_RADAR_SEQUENTIAL,
        colorbar=radar_module.colorbar_defaults("Puntaje total"),
    )
    fig.update_geos(
        projection_type="mercator",
        center=dict(lat=-18, lon=-70),
        projection_scale=0.95,
        lataxis=dict(range=[-60, 35]),
        lonaxis=dict(range=[-135, -25]),
        showcountries=True,
        countrycolor="#3b3b3b",
        showsubunits=True,
        subunitcolor="#ffffff",
        landcolor="#d4f0e0",
        lakecolor="#f3f9f4",
    )
    return fig


def _safe_range(values, default: tuple[float, float]) -> tuple[float, float]:
    if not values or len(values) != 2:
        return default
    try:
        start = float(values[0])
        end = float(values[1])
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default
    return start, end


def _build_focus_overlay_entries(geo, fig, focus_countries: Iterable[str]) -> List[Dict[str, float]]:
    geo_layout = getattr(fig.layout, "geo", None)
    lat_range = _safe_range(getattr(getattr(geo_layout, "lataxis", None), "range", None), (-60.0, 35.0))
    lon_range = _safe_range(getattr(getattr(geo_layout, "lonaxis", None), "range", None), (-135.0, -25.0))
    lat_span = (lat_range[1] - lat_range[0]) or 1.0
    lon_span = (lon_range[1] - lon_range[0]) or 1.0
    geo_lookup = geo.set_index("country").to_dict(orient="index")
    overlays: List[Dict[str, float]] = []

    for code in (c.upper() for c in focus_countries):
        stats = geo_lookup.get(code)
        coords = FOCUS_COUNTRY_CENTROIDS.get(code)
        if not stats or not coords:
            continue
        top_pct = (lat_range[1] - coords["lat"]) / lat_span * 100.0
        left_pct = (coords["lon"] - lon_range[0]) / lon_span * 100.0
        overlays.append(
            {
                "code": code,
                "country": code,
                "companies": float(stats.get("companies", 0.0)),
                "total_score": float(stats.get("total_score", 0.0)),
                "top_pct": min(max(top_pct, 3.0), 97.0),
                "left_pct": min(max(left_pct, 3.0), 97.0),
            }
        )

    return overlays


def _write_plotly_html_with_hover(radar_module, fig, focus_countries: Iterable[str], overlay_entries: List[Dict[str, float]]):
    cfg = {"responsive": True}
    plotly_json = fig.to_plotly_json()
    div_id = f"plotly-{uuid4().hex}"
    height = fig.layout.height or 600
    width = fig.layout.width or 900

    def _json_to_tabs(data):
        return json.dumps(data, indent=4, cls=PlotlyJSONEncoder)

    data_json = _json_to_tabs(plotly_json.get("data", []))
    layout_json = _json_to_tabs(plotly_json.get("layout", {}))
    config_json = _json_to_tabs(cfg)

    style_lines = [
        line for line in radar_module.MARKET_RADAR_INLINE_STYLE.strip().splitlines() if line
    ]
    style_lines.extend(
        [
            ".plotly-hover-wrapper { position: relative; width: 100%; margin: 0 auto; }",
            ".plotly-hover-wrapper .plotly-graph-div { background-color: #ecf7f0; height: 100%; width: 100%; }",
            ".hover-overlay { position: absolute; inset: 0; pointer-events: none; font-family: 'Josefin Sans', 'Helvetica Neue', Arial, sans-serif; }",
            ".hover-overlay .hover-card { position: absolute; transform: translate(-50%, -105%); background: rgba(255, 255, 255, 0.97); border: 1px solid #2d6c52; border-radius: 6px; padding: 6px 10px; font-size: 13px; line-height: 1.3; color: #0f281f; box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18); min-width: 150px; max-width: 220px; text-align: left; }",
            ".hover-overlay .hover-card::after { content: ''; position: absolute; left: 50%; bottom: -7px; transform: translateX(-50%); border-width: 7px 7px 0 7px; border-style: solid; border-color: #2d6c52 transparent transparent transparent; }",
            ".hover-overlay .hover-card::before { content: ''; position: absolute; left: 50%; bottom: -5px; transform: translateX(-50%); border-width: 7px 7px 0 7px; border-style: solid; border-color: rgba(255, 255, 255, 0.97) transparent transparent transparent; }",
            "@media (max-width: 640px) { .hover-overlay .hover-card { font-size: 12px; padding: 5px 8px; } }",
        ]
    )

    focus_json = json.dumps([code.upper() for code in focus_countries])
    hover_js = (
        "\n\t\t\tfunction attachPersistentHover(graphDiv) {\n"
        "\t\t\t\tif (!graphDiv) { return; }\n"
        f"\t\t\t\tconst focusCodes = {focus_json};\n"
        "\t\t\t\tlet initialized = false;\n"
        "\t\t\t\tconst computeHoverPoints = function() {\n"
        "\t\t\t\t\tconst customSeries = ((graphDiv.data && graphDiv.data[0] && graphDiv.data[0].customdata) || [])\n"
        "\t\t\t\t\t    .map(row => (row && row[0]) ? row[0].toString().toUpperCase() : '');\n"
        "\t\t\t\t\tconst pts = [];\n"
        "\t\t\t\t\tfocusCodes.forEach(code => {\n"
        "\t\t\t\t\t\tconst idx = customSeries.indexOf(code);\n"
        "\t\t\t\t\t\tif (idx >= 0) { pts.push({curveNumber: 0, pointNumber: idx}); }\n"
        "\t\t\t\t\t});\n"
        "\t\t\t\t\treturn pts;\n"
        "\t\t\t\t};\n"
        "\t\t\t\tconst showAllHover = function(delay) {\n"
        "\t\t\t\t\tconst pts = computeHoverPoints();\n"
        "\t\t\t\t\tif (!pts.length) { return; }\n"
        "\t\t\t\t\tsetTimeout(function() { Plotly.Fx.hover(graphDiv, pts, 'geo'); }, delay || 0);\n"
        "\t\t\t\t};\n"
        "\t\t\t\tconst init = function() {\n"
        "\t\t\t\t\tif (initialized) { return; }\n"
        "\t\t\t\t\tinitialized = true;\n"
        "\t\t\t\t\tshowAllHover(0);\n"
        "\t\t\t\t\tgraphDiv.on('plotly_hover', function() { showAllHover(0); });\n"
        "\t\t\t\t\tgraphDiv.on('plotly_unhover', function() { showAllHover(0); });\n"
        "\t\t\t\t\tgraphDiv.on('plotly_relayout', function() { showAllHover(120); });\n"
        "\t\t\t\t\tgraphDiv.on('plotly_restyle', function() { showAllHover(120); });\n"
        "\t\t\t\t\tgraphDiv.on('plotly_afterplot', function() { showAllHover(0); });\n"
        "\t\t\t\t\twindow.addEventListener('resize', function() { showAllHover(150); });\n"
        "\t\t\t\t\tsetInterval(function() { showAllHover(0); }, 2000);\n"
        "\t\t\t\t\tdocument.addEventListener('visibilitychange', function() { if (!document.hidden) { showAllHover(0); } });\n"
        "\t\t\t\t};\n"
        "\t\t\t\tif (graphDiv.data && graphDiv.data.length) {\n"
        "\t\t\t\t\tinit();\n"
        "\t\t\t\t} else {\n"
        "\t\t\t\t\tlet pendingInit = false;\n"
        "\t\t\t\t\tconst afterplotHandler = function() {\n"
        "\t\t\t\t\t\tif (pendingInit) { return; }\n"
        "\t\t\t\t\t\tpendingInit = true;\n"
        "\t\t\t\t\t\tsetTimeout(function() { init(); }, 0);\n"
        "\t\t\t\t\t};\n"
        "\t\t\t\t\tgraphDiv.on('plotly_afterplot', afterplotHandler);\n"
        "\t\t\t\t}\n"
        "\t\t\t}\n"
        "\t\t\tvar graphDiv = document.getElementById('" + div_id + "');\n"
        "\t\t\tvar figPromiseTyped = figPromise && typeof figPromise.then === 'function' ? figPromise : null;\n"
        "\t\t\tif (figPromiseTyped) {\n"
        "\t\t\t\tfigPromiseTyped.then(function() { attachPersistentHover(graphDiv); });\n"
        "\t\t\t} else {\n"
        "\t\t\t\tattachPersistentHover(graphDiv);\n"
        "\t\t\t}\n"
    )

    overlay_lines: List[str] = []
    if overlay_entries:
        overlay_lines.append("\t\t<div class=\"hover-overlay\" aria-hidden=\"true\">")
        for item in overlay_entries:
            text = (
                f"País={item['country']}<br>"
                f"Empresas={item['companies']:,.0f}<br>"
                f"Puntaje total={item['total_score']:,.2f}"
            )
            overlay_lines.append(
                "\t\t\t<div class=\"hover-card\" "
                f"data-country=\"{item['code']}\" "
                f"style=\"top:{item['top_pct']:.2f}%;left:{item['left_pct']:.2f}%;\">"
                f"{text}</div>"
            )
        overlay_lines.append("\t\t</div>")
    else:
        overlay_lines.append("\t\t<div class=\"hover-overlay\" aria-hidden=\"true\"></div>")

    html_lines = [
        "<html>",
        "<head>",
        "\t<meta charset=\"utf-8\" />",
        "\t<script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>",
        f"\t{radar_module.PLOTLYJS_SCRIPT}",
        "\t<style>",
    ]
    html_lines.extend(f"\t{line}" for line in style_lines)
    html_lines.extend(
        [
            "\t</style>",
            "</head>",
            "<body>",
            f"\t<div class=\"plotly-hover-wrapper\" style=\"height:{height}px; width:100%; max-width:{width}px;\">",
            f"\t\t<div id=\"{div_id}\" class=\"plotly-graph-div\" style=\"height:100%; width:100%;\"></div>",
        ]
    )
    html_lines.extend(overlay_lines)
    html_lines.extend(
        [
            "\t</div>",
            "\t<script type=\"text/javascript\">",
            "\t\twindow.PLOTLYENV = window.PLOTLYENV || {};",
            f"\t\tif (document.getElementById('{div_id}')) {{",
            "\t\t\tvar figPromise = Plotly.newPlot(",
            f"\t\t\t\t'{div_id}',",
            "\t\t\t\t" + data_json.replace("\n", "\n\t\t\t\t") + ",",
            "\t\t\t\t" + layout_json.replace("\n", "\n\t\t\t\t") + ",",
            "\t\t\t\t" + config_json.replace("\n", "\n\t\t\t\t"),
            "\t\t\t);",
            hover_js if hover_js else "",
            "\t\t}",
            "\t</script>",
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(html_lines) + "\n"


def main() -> None:
    radar = _load_market_radar_module()
    events = radar._prep_events()
    companies = radar._company_rollup(events)
    geo = _build_geo(companies, radar)
    fig = _build_figure(radar, geo)
    overlay_entries = _build_focus_overlay_entries(geo, fig, FOCUS_COUNTRIES)
    html = _write_plotly_html_with_hover(radar, fig, FOCUS_COUNTRIES, overlay_entries)

    output_path = radar.OUTPUT_DIR / OUTPUT_FILENAME
    output_path.write_text(html, encoding="utf-8")
    print(f"Mapa con hoovers simultáneos guardado en {output_path}")


if __name__ == "__main__":
    main()
