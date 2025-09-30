"""Signal collectors for CTR25."""

from .finance import run_collect_finance  # noqa: F401
from .jobs import run_collect_jobs  # noqa: F401
from .membership import collect_memberships  # noqa: F401
from .news import run_collect_news  # noqa: F401
from .webscan import run_collect_webscan  # noqa: F401

__all__ = [
    "run_collect_finance",
    "run_collect_jobs",
    "collect_memberships",
    "run_collect_news",
    "run_collect_webscan",
]
