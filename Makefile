.PHONY: install test lint run format clean docker-build docker-run

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t nmt-service:latest .

docker-run:
	docker run -p 8000:8000 nmt-service:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
