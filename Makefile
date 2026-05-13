# Convenience targets. Real entry point is the `applyops` CLI.

.PHONY: install fmt lint type test eval run clean

install:
	uv pip install -e ".[dev,obs,submit]" || pip install -e ".[dev,obs,submit]"

fmt:
	ruff format src tests
	ruff check --fix src tests

lint:
	ruff check src tests
	ruff format --check src tests

type:
	mypy src/applyops

test:
	pytest -m "not eval and not live"

eval:
	pytest -m eval

run:
	applyops run

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
