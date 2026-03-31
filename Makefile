.PHONY: lint lint-fix hooks-install hooks-run

lint:
	.venv/bin/ruff check .

lint-fix:
	.venv/bin/ruff check . --fix

hooks-install:
	.venv/bin/pre-commit install

hooks-run:
	.venv/bin/pre-commit run --all-files --hook-stage manual
