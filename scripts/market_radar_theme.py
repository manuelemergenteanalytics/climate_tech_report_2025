"""Plotly theme, color maps y utilidades de estilo para el Market Radar LATAM 2025."""
from __future__ import annotations

import hashlib
import unicodedata
from typing import Any, Dict, Iterable, List, Sequence

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

MARKET_RADAR_TEMPLATE_NAME = "emergente_market_radar"

PRIMARY_FONT = "Josefin Sans, 'Helvetica Neue', Arial, sans-serif"
PRIMARY_TEXT_COLOR = "#164233"
TITLE_COLOR = "#0d3b29"
BACKGROUND_COLOR = "#f6fbf8"
PLOT_AREA_COLOR = "#ecf7f0"
HOVER_BG_COLOR = "#123b2a"
HOVER_BORDER_COLOR = "#0a251a"
HOVER_TEXT_COLOR = "#f2fbf5"

MARKET_RADAR_COLORWAY: List[str] = [
    "#0b5d1e",
    "#1e7633",
    "#2f9147",
    "#41aa5f",
    "#55c278",
    "#6fce8f",
    "#8bd9a6",
    "#a8e3bd",
    "#c8edd5",
    "#e4f7e9",
]
MARKET_RADAR_SEQUENTIAL: List[List[Any]] = [
    [0.0, "#f2fbf5"],
    [0.2, "#d5f1dd"],
    [0.4, "#a8e0bf"],
    [0.6, "#74cf9f"],
    [0.8, "#3bbd79"],
    [1.0, "#138f4c"],
]

TABLE_HEADER_COLOR = "#b8e3c6"
TABLE_CELL_COLOR = "#e6f6ec"
TABLE_LINE_COLOR = "#cfe8d7"
TABLE_HEADER_FONT_SIZE = 14
TABLE_CELL_FONT_SIZE = 13

MARKER_BORDER_COLOR = "#0a3c28"
BAR_LINE_COLOR = "#0f5134"
COVERAGE_INDICATORS_BAR_COLOR = "#6ccfa0"
SIGNAL_TYPE_BAR_COLOR = "#62c68f"

MARKET_RADAR_INLINE_STYLE = (
    "@font-face { font-family: 'Josefin Sans'; src: url('../../JosefinSans-Medium.ttf') format('truetype'); "
    "font-weight: 500; font-style: normal; font-display: swap; }\n"
    "body { font-family: 'Josefin Sans', 'Helvetica Neue', Arial, sans-serif; background-color: #f6fbf8; color: #164233; }\n"
    ".plotly-graph-div { background-color: #ecf7f0; }\n"
)

COUNTRY_COLOR_OVERRIDES: Dict[str, str] = {
    "AR": "#a4c8f0",  # azul pastel
    "BR": "#81c784",  # verde suave
    "CL": "#ffe0a3",  # amarillo cálido tenue
    "CO": "#9fd6d9",  # turquesa claro
    "MX": "#ffb4a2",  # coral pastel
    "PE": "#b7d8a6",  # verde hoja claro
    "UY": "#cbb7e8",  # lavanda suave
    "VE": "#ffe9b3",  # mostaza clara
}
COUNTRY_CATEGORY_ORDER: List[str] = [
    "AR",
    "BR",
    "CL",
    "CO",
    "MX",
    "PE",
    "UY",
    "VE",
    "BO",
    "EC",
    "CR",
    "PA",
    "PY",
    "SV",
    "HN",
    "NI",
    "DO",
    "GT",
]

INDUSTRY_COLOR_OVERRIDES: Dict[str, str] = {
    "Agro & alimentos": "#1b9aaa",
    "Construcción, Infraestructura & Bienes Raíces": "#f4a259",
    "Finanzas, Seguros & Capital": "#5d5179",
    "TIC, Digital & Medios": "#2ec4b6",
    "Manufactura industrial": "#ff6f59",
    "Petróleo & gas": "#d1495b",
    "Retail & servicios al consumidor": "#9bc53d",
    "Transporte, Movilidad & Logística": "#3b8ea5",
    "Energía & Servicios Públicos": "#00a896",
    "Salud, Pharma & Biotech": "#845ec2",
    "Hospitalidad, Turismo & Ocio": "#f6ae2d",
    "Minería & materiales básicos": "#2a9d8f",
    "Servicios profesionales & técnicos": "#33658a",
    "Sector público, social & educación": "#ff9f1c",
    "Servicios ambientales & circulares": "#76b041",
    "Sin industria": "#8bd9a6",
}
INDUSTRY_CATEGORY_ORDER: List[str] = [
    "Agro & alimentos",
    "Retail & servicios al consumidor",
    "Manufactura industrial",
    "TIC, Digital & Medios",
    "Transporte, Movilidad & Logística",
    "Construcción, Infraestructura & Bienes Raíces",
    "Finanzas, Seguros & Capital",
    "Petróleo & gas",
    "Energía & Servicios Públicos",
    "Salud, Pharma & Biotech",
    "Hospitalidad, Turismo & Ocio",
    "Minería & materiales básicos",
    "Servicios profesionales & técnicos",
    "Sector público, social & educación",
    "Servicios ambientales & circulares",
    "Sin industria",
]

RAW_INDUSTRY_TO_FRIENDLY: Dict[str, str] = {
    "agro_food": "Agro & alimentos",
    "agro_food_beverage": "Agro & alimentos",
    "agro": "Agro & alimentos",
    "retail_consumer": "Retail & servicios al consumidor",
    "retail_consumer_services": "Retail & servicios al consumidor",
    "retail": "Retail & servicios al consumidor",
    "manufacturing": "Manufactura industrial",
    "industrial_manufacturing": "Manufactura industrial",
    "ict_telecom": "TIC, Digital & Medios",
    "ict_digital_media": "TIC, Digital & Medios",
    "digital_media": "TIC, Digital & Medios",
    "transport_logistics": "Transporte, Movilidad & Logística",
    "transport_mobility_logistics": "Transporte, Movilidad & Logística",
    "construction_realestate": "Construcción, Infraestructura & Bienes Raíces",
    "construction_infrastructure_realestate": "Construcción, Infraestructura & Bienes Raíces",
    "finance_insurance": "Finanzas, Seguros & Capital",
    "finance_insurance_capital": "Finanzas, Seguros & Capital",
    "oil_gas": "Petróleo & gas",
    "energy_power": "Energía & Servicios Públicos",
    "energy_power_utilities": "Energía & Servicios Públicos",
    "utilities": "Energía & Servicios Públicos",
    "health_biotech": "Salud, Pharma & Biotech",
    "hospitality_tourism": "Hospitalidad, Turismo & Ocio",
    "hospitality_tourism_leisure": "Hospitalidad, Turismo & Ocio",
    "mining_metals": "Minería & materiales básicos",
    "mining_materials": "Minería & materiales básicos",
    "professional_services": "Servicios profesionales & técnicos",
    "professional_service_consulting": "Servicios profesionales & técnicos",
    "professional_services_consulting": "Servicios profesionales & técnicos",
    "services_environmental": "Servicios ambientales & circulares",
    "environmental_circular_services": "Servicios ambientales & circulares",
    "water_waste_circularity": "Servicios ambientales & circulares",
    "sector_public": "Sector público, social & educación",
    "public_social_education": "Sector público, social & educación",
    "services_professional_technical": "Servicios profesionales & técnicos",
    "chemicals_materials": "Minería & materiales básicos",
    "healtchare_pharma_biotech": "Salud, Pharma & Biotech",
    "healthcare_pharma_biotech": "Salud, Pharma & Biotech",
    "sin industria": "Sin industria",
}


def _normalize_industry_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(c for c in normalized if not unicodedata.combining(c))
    ascii_value = ascii_value.lower()
    for ch in ["&", "/", "-", ",", ".", "·", ":", ";"]:
        ascii_value = ascii_value.replace(ch, " ")
    ascii_value = ascii_value.replace("+", " ")
    ascii_value = "_".join(ascii_value.split())
    return ascii_value


INDUSTRY_FRIENDLY_LABELS: Dict[str, str] = {k: v for k, v in RAW_INDUSTRY_TO_FRIENDLY.items()}
for friendly in INDUSTRY_CATEGORY_ORDER:
    INDUSTRY_FRIENDLY_LABELS[_normalize_industry_key(friendly)] = friendly
INDUSTRY_FRIENDLY_LABELS[_normalize_industry_key("Sin industria")] = "Sin industria"

SIGNAL_TYPE_COLOR_OVERRIDES: Dict[str, str] = {
    "ungc": "#0b6a39",
    "news": "#2f9d8f",
    "sbti": "#48b597",
    "bcorp": "#94c11f",
    "unknown": "#8ddbb0",
    "sin dato": "#c5ecd4",
}
SIGNAL_TYPE_ORDER: List[str] = ["ungc", "news", "sbti", "bcorp", "sin dato", "unknown"]

SANKEY_NODE_BASE_COLORS: List[str] = [
    "#0b6a39",
    "#1b8a6b",
    "#63b995",
    "#94d4b3",
    "#c5ecd4",
    "#55b49d",
    "#349c8b",
    "#6cc7b9",
    "#97decf",
    "#c4efe3",
    "#e9faf3",
]
SANKEY_LINK_BASE_COLORS: List[str] = [
    "rgba(11,106,57,0.55)",
    "rgba(27,138,107,0.4)",
    "rgba(85,180,157,0.45)",
    "rgba(148,212,179,0.50)",
    "rgba(99,196,157,0.45)",
    "rgba(52,156,139,0.45)",
    "rgba(108,199,185,0.50)",
    "rgba(151,222,207,0.45)",
    "rgba(196,239,227,0.50)",
    "rgba(233,250,243,0.55)",
]
SANKEY_NODE_PAD = 18
SANKEY_NODE_THICKNESS = 18
SANKEY_NODE_LINE_COLOR = "#0a3c28"
SANKEY_NODE_LINE_WIDTH = 0.5

COLORBAR_TICK_FONT: Dict[str, Any] = {
    "color": PRIMARY_TEXT_COLOR,
    "family": PRIMARY_FONT,
    "size": 12,
}
COLORBAR_TITLE_FONT: Dict[str, Any] = {
    "color": PRIMARY_TEXT_COLOR,
    "family": PRIMARY_FONT,
    "size": 16,
}


def _deterministic_color(label: str, palette: Sequence[str] | None = None) -> str:
    base = palette or MARKET_RADAR_COLORWAY
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(base)
    return base[idx]


def _build_color_map(values: Iterable[str], overrides: Dict[str, str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {**overrides}
    for raw in values:
        if raw is None:
            continue
        label = str(raw)
        if label not in mapping:
            mapping[label] = _deterministic_color(label)
    return mapping


def ordered_categories(values: Iterable[str], preferred_order: Sequence[str]) -> List[str]:
    seen: List[str] = []
    for raw in values:
        if raw is None:
            continue
        label = str(raw)
        if label not in seen:
            seen.append(label)
    ordered = [item for item in preferred_order if item in seen]
    for item in seen:
        if item not in ordered:
            ordered.append(item)
    return ordered


def build_country_color_map(values: Iterable[str]) -> Dict[str, str]:
    return _build_color_map(values, COUNTRY_COLOR_OVERRIDES)


def build_industry_color_map(values: Iterable[str]) -> Dict[str, str]:
    friendly_values = [friendly_industry_label(val) for val in values if val is not None]
    mapping: Dict[str, str] = {}
    for value in friendly_values:
        if value not in mapping:
            mapping[value] = INDUSTRY_COLOR_OVERRIDES.get(
                value, _deterministic_color(value)
            )
    return mapping


def build_signal_type_color_map(values: Iterable[str]) -> Dict[str, str]:
    return _build_color_map(values, SIGNAL_TYPE_COLOR_OVERRIDES)


def sankey_node_colors(count: int) -> List[str]:
    if count <= len(SANKEY_NODE_BASE_COLORS):
        return SANKEY_NODE_BASE_COLORS[:count]
    colors = SANKEY_NODE_BASE_COLORS[:]
    while len(colors) < count:
        colors.append(_deterministic_color(f"sankey-node-{len(colors)}"))
    return colors


def sankey_link_colors(count: int) -> List[str]:
    if count <= len(SANKEY_LINK_BASE_COLORS):
        return SANKEY_LINK_BASE_COLORS[:count]
    colors = SANKEY_LINK_BASE_COLORS[:]
    fallback = "rgba(27,138,107,0.35)"
    while len(colors) < count:
        colors.append(fallback)
    return colors


def colorbar_defaults(title: str) -> Dict[str, Any]:
    return {
        "title": {"text": title, "font": COLORBAR_TITLE_FONT},
        "outlinewidth": 0,
        "tickfont": COLORBAR_TICK_FONT,
    }


def friendly_industry_label(value: Any) -> str:
    if value is None:
        return "Sin industria"
    label = str(value).strip()
    if not label:
        return "Sin industria"
    key = _normalize_industry_key(label)
    if key in INDUSTRY_FRIENDLY_LABELS:
        return INDUSTRY_FRIENDLY_LABELS[key]
    if label in INDUSTRY_FRIENDLY_LABELS:
        return INDUSTRY_FRIENDLY_LABELS[label]
    return label


def apply_industry_labels(values: Iterable[Any]) -> List[str]:
    return [friendly_industry_label(val) for val in values]


def wrap_label(text: str, width: int = 18) -> str:
    words = str(text).split()
    if not words:
        return text
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


def _build_template() -> go.layout.Template:
    return go.layout.Template(
        data={
            "bar": [
                {
                    "type": "bar",
                    "error_x": {"color": "#2a3f5f"},
                    "error_y": {"color": "#2a3f5f"},
                    "marker": {
                        "line": {"color": "#E5ECF6", "width": 0.5},
                        "pattern": {"fillmode": "overlay", "size": 10, "solidity": 0.2},
                    },
                }
            ],
            "heatmap": [
                {
                    "type": "heatmap",
                    "colorbar": {"outlinewidth": 0, "ticks": ""},
                    "colorscale": MARKET_RADAR_SEQUENTIAL,
                }
            ],
            "histogram2d": [
                {
                    "type": "histogram2d",
                    "colorbar": {"outlinewidth": 0, "ticks": ""},
                    "colorscale": MARKET_RADAR_SEQUENTIAL,
                }
            ],
            "histogram2dcontour": [
                {
                    "type": "histogram2dcontour",
                    "colorbar": {"outlinewidth": 0, "ticks": ""},
                    "colorscale": MARKET_RADAR_SEQUENTIAL,
                }
            ],
            "contour": [
                {
                    "type": "contour",
                    "colorbar": {"outlinewidth": 0, "ticks": ""},
                    "colorscale": MARKET_RADAR_SEQUENTIAL,
                }
            ],
            "surface": [
                {
                    "type": "surface",
                    "colorbar": {"outlinewidth": 0, "ticks": ""},
                    "colorscale": MARKET_RADAR_SEQUENTIAL,
                }
            ],
            "choropleth": [
                {
                    "type": "choropleth",
                    "colorbar": {"outlinewidth": 0, "ticks": ""},
                }
            ],
            "scatter": [
                {
                    "type": "scatter",
                    "fillpattern": {"fillmode": "overlay", "size": 10, "solidity": 0.2},
                }
            ],
            "table": [
                {
                    "type": "table",
                    "header": {
                        "fill": {"color": TABLE_HEADER_COLOR},
                        "line": {"color": TABLE_LINE_COLOR},
                        "font": {
                            "color": PRIMARY_TEXT_COLOR,
                            "family": PRIMARY_FONT,
                            "size": TABLE_HEADER_FONT_SIZE,
                        },
                    },
                    "cells": {
                        "fill": {"color": TABLE_CELL_COLOR},
                        "line": {"color": TABLE_LINE_COLOR},
                        "font": {
                            "color": PRIMARY_TEXT_COLOR,
                            "family": PRIMARY_FONT,
                            "size": TABLE_CELL_FONT_SIZE,
                        },
                    },
                }
            ],
        },
        layout=go.Layout(
            paper_bgcolor=BACKGROUND_COLOR,
            plot_bgcolor=PLOT_AREA_COLOR,
            font={"family": PRIMARY_FONT, "color": PRIMARY_TEXT_COLOR, "size": 14},
            colorway=MARKET_RADAR_COLORWAY,
            hoverlabel={
                "bgcolor": HOVER_BG_COLOR,
                "bordercolor": HOVER_BORDER_COLOR,
                "font": {"color": HOVER_TEXT_COLOR, "family": PRIMARY_FONT, "size": 12},
            },
            legend={"font": {"color": PRIMARY_TEXT_COLOR, "family": PRIMARY_FONT, "size": 14}},
            coloraxis={
                "colorscale": MARKET_RADAR_SEQUENTIAL,
                "colorbar": {"outlinewidth": 0, "tickfont": COLORBAR_TICK_FONT},
            },
            title={"font": {"color": TITLE_COLOR, "family": PRIMARY_FONT, "size": 24}, "x": 0.05},
            xaxis={
                "automargin": True,
                "gridcolor": "white",
                "linecolor": "white",
                "tickfont": {"color": PRIMARY_TEXT_COLOR, "family": PRIMARY_FONT, "size": 12},
                "ticks": "",
                "title": {
                    "font": {"color": TITLE_COLOR, "family": PRIMARY_FONT, "size": 16},
                    "standoff": 15,
                },
                "zerolinecolor": "white",
                "zerolinewidth": 2,
            },
            yaxis={
                "automargin": True,
                "gridcolor": "white",
                "linecolor": "white",
                "tickfont": {"color": PRIMARY_TEXT_COLOR, "family": PRIMARY_FONT, "size": 12},
                "ticks": "",
                "title": {
                    "font": {"color": TITLE_COLOR, "family": PRIMARY_FONT, "size": 16},
                    "standoff": 15,
                },
                "zerolinecolor": "white",
                "zerolinewidth": 2,
            },
            geo={
                "bgcolor": BACKGROUND_COLOR,
                "lakecolor": "#f3f9f4",
                "landcolor": "#d4f0e0",
                "showlakes": True,
                "showland": True,
                "subunitcolor": "#ffffff",
            },
            mapbox={"style": "light"},
            polar={
                "bgcolor": PLOT_AREA_COLOR,
                "angularaxis": {"gridcolor": "#ffffff", "linecolor": "#ffffff", "ticks": ""},
                "radialaxis": {"gridcolor": "#ffffff", "linecolor": "#ffffff", "ticks": ""},
            },
        ),
    )


def register_market_radar_template(name: str = MARKET_RADAR_TEMPLATE_NAME) -> str:
    pio.templates[name] = _build_template()
    return name


def apply_market_radar_theme(set_defaults: bool = True) -> str:
    template_name = register_market_radar_template()
    if set_defaults:
        pio.templates.default = template_name
        px.defaults.template = template_name
        px.defaults.color_discrete_sequence = MARKET_RADAR_COLORWAY
        px.defaults.color_continuous_scale = [step[1] for step in MARKET_RADAR_SEQUENTIAL]
    return template_name
