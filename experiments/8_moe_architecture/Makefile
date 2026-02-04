.PHONY: install test lint typecheck clean run validate

install:
	pip install -r requirements.txt
	pip install -e ".[dev,viz]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

typecheck:
	mypy src/

format:
	ruff format src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage

run-all:
	bash scripts/run_pipeline.sh

validate:
	bash scripts/validate_outputs.sh

setup-dirs:
	mkdir -p data outputs/manifests outputs/validation_reports outputs/coresets logs
