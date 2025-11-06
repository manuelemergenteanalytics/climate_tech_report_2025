# Repository Guidelines

## Project Structure & Module Organization
Source lives in `src/ctr25` with the Typer CLI in `cli.py`, signal collectors under `signals/`, scoring logic in `iic.py` and `prospect.py`, and helpers in `utils/`. Configuration YAMLs in `config/` should stay canonical; create per-environment copies when needed. Use `data/raw`, `data/interim`, and `data/processed` consistently and only commit reproducible artifacts. Documentation and visuals belong in `docs/` or `assets/`, while automation lives in `scripts/`.

## Build, Test, and Development Commands
- `python -m venv .venv && .venv/bin/pip install -e .`: set up an editable install (use `.venv\Scripts` on Windows).
- `.venv/bin/ctr25 init`: bootstrap folders and config defaults.
- `.venv/bin/ctr25 collect-news --country MX`: run a collector; swap subcommand for memberships, jobs, finance, or webscan.
- `.venv/bin/ctr25 compute-iic` / `compute-ps`: recompute indices after ingesting data.
- `pytest` or `pytest tests/test_prospect.py -k scenario`: execute the test suite or a focused subset.

## Coding Style & Naming Conventions
Target Python 3.10+, four-space indentation, and PEP 8-aligned snake_case names. Add CLI entry points through the Typer app and keep modules cohesive inside the existing package layout. Document parameters with short docstrings and prefer pure, typed functions so they can be unit tested in isolation. Large assets or notebooks should land in `assets/` with raw data excluded from git.

## Testing Guidelines
Place new tests in `tests/test_<topic>.py`, mirroring the current structure. Use tiny, deterministic fixtures (synthetic or trimmed from `data/processed`) and mock network access inside collectors. Ensure `pytest` passes before raising a PR and include regression coverage when you change scoring rules or schema contracts.

## Commit & Pull Request Guidelines
Keep commits small, present tense, and descriptive, echoing `git log` entries like `graph style changes`. Mention relevant commands or data folders in the body so reviewers can reproduce results quickly. Pull requests should explain intent, list validation steps, and attach snapshots when visuals or exports change.

## Security & Configuration Tips
Load secrets such as `MEDIACLOUD_API_KEY` via environment variables, not committed files. Sensitive dumps belong outside `data/raw` in this repo; store sanitized samples only. If you need overrides, duplicate the YAML into `config/local/` (ignored) and confirm the default CLI path still works.
