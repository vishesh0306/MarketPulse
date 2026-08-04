.PHONY: install test lint typecheck pipeline clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src

lint:
	ruff check src/ tests/

typecheck:
	mypy src/

pipeline:
	bash scripts/run_pipeline.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
