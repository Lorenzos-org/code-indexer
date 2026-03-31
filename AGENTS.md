# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/`. The main entrypoint is `__main__.py`, which delegates to `src/cli.py`. Keep orchestration in `src/indexer.py`, configuration loading in `src/config.py`, and focused subsystems in their own modules such as `src/discovery.py`, `src/chunker.py`, `src/embedder.py`, `src/vectorstore.py`, and `src/sessions.py`.

Repository-level support files include `config.yaml` for defaults, `requirements.txt` for Python dependencies, and `code-indexer.service` for service integration. Do not commit `.venv/` or `src/__pycache__/`.

## Build, Test, and Development Commands
Create a local environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the CLI locally:

```bash
python -m code_indexer index . --full
python -m code_indexer query "configuration loading" -n 5
python -m code_indexer stats
python -m code_indexer session --list --limit 20
```

Use `python -m code_indexer daemon <paths> --interval 300` for background polling. Pass `-c path/to/config.yaml` to test alternate configs.

Lint the repository with:

```bash
make lint
make lint-fix
make hooks-install
make hooks-run
```

These targets run Ruff from the project virtualenv, install Git hooks, and let you execute the hook set manually.

## Coding Style & Naming Conventions
Follow Python 3 style with 4-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Prefer small, single-purpose modules and keep CLI handlers thin by moving logic into `CodeIndexer` or subsystem classes. Type hints are already used in parts of the codebase and should be preserved or expanded when touching existing code.

Ruff is configured in `pyproject.toml` and should be used for import sorting and baseline lint checks before opening changes.
Git hook automation is configured in `.pre-commit-config.yaml`. The main stages are `pre-commit`, `pre-push`, and `pre-merge-commit`, and the same checks can be invoked manually through `make hooks-run`.

## Testing Guidelines
There is currently no automated test suite. Before opening a change, run manual smoke tests against the CLI commands above and verify behavior against a local Ollama instance and Chroma persistence directory. When adding tests later, place them under `tests/` and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
Commit messages follow a short conventional style such as `feat: add config.py module` or `chore: scaffold code-indexer project`. Keep subjects imperative and scoped to one change.

Pull requests should include:
- a short summary of the behavior change
- any config or dependency changes
- manual verification steps and sample commands
- linked issues or follow-up work, if applicable

## Security & Configuration Tips
Do not hardcode local paths, credentials, or service URLs beyond documented defaults. Keep `config.yaml` values portable, and prefer user-overridable settings for Ollama and Chroma locations.

For local development, the default Chroma store is `~/.local/share/code-indexer/chroma`. In this environment it already contains persisted data, with a populated `code` collection and an empty `sessions` collection. Treat that store as developer-local state, not repository data, and do not commit generated Chroma files.
