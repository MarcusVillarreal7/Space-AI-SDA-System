.PHONY: dev test docker-build docker-up docker-down lint

dev:
	python scripts/run_dashboard.py --dev

test:
	PYTHONPATH=. venv/bin/pytest tests/ -v

lint:
	venv/bin/flake8 src/ --max-line-length 120

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down
