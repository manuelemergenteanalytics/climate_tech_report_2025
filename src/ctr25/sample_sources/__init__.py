"""Sample sources for universe construction."""
from .wikidata import query_wikidata, map_industries, apply_sampling  # noqa: F401

__all__ = [
    "query_wikidata",
    "map_industries",
    "apply_sampling",
]
