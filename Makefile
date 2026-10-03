.DEFAULT_GOAL := up
.PHONY: up down logs seed test dev db clean

# One command: build and start app + mongo + redis, then wait until healthy
up: .env
	docker compose up -d --build --wait
	@echo "Chatbot API: http://localhost:8000  (docs: /docs, health: /health)"

down:
	docker compose down

logs:
	docker compose logs -f app

# Create demo plans + customers (prints their API keys)
seed:
	docker compose exec app python -c "import asyncio, tests.test_api as t; asyncio.run(t.setup_test_data())"

# Run the tests inside the container
test:
	docker compose run --rm --no-deps -e MONGODB_URL=mongodb://mongo:27017 app sh -c "pip install -q pytest pytest-asyncio httpx && python -m pytest tests/test_agent_tools.py tests/test_endpoints.py -q"

# Only the databases in docker, app from a local venv with hot reload
db:
	docker compose up -d --wait mongo redis

dev: db
	venv/bin/uvicorn main:app --reload --port 8000

# Remove containers AND data volumes
clean:
	docker compose down -v

.env:
	cp .env.example .env
	@echo "Created .env from .env.example - add your GROQ_API_KEY, then re-run make"
	@exit 1
