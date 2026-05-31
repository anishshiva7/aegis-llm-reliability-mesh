# Aegis — one-command developer workflow.
#
# Quick start (three terminals, or run `make demo` for guided steps):
#   make install     # create venv + install backend deps, install dashboard deps
#   make backend     # terminal 1: start the API on :8000
#   make dashboard   # terminal 2: start the Next.js dashboard on :3000
#   make seed        # terminal 3 (once backend is up): ingest sample docs
#   make test        # run the full backend test suite
#
# The backend pins the HuggingFace model cache to backend/.hf_cache (see run.sh).

PY := backend/venv/bin/python
PIP := backend/venv/bin/pip
AEGIS_URL ?= http://127.0.0.1:8000

.PHONY: help install install-backend install-dashboard backend dashboard seed test test-backend build clean demo

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: install-backend install-dashboard ## Install backend + dashboard dependencies

install-backend: ## Create the backend venv and install dependencies
	cd backend && python3 -m venv venv && \
		./venv/bin/pip install --upgrade pip && \
		./venv/bin/pip install -r requirements.txt

install-dashboard: ## Install dashboard (Next.js) dependencies
	cd frontend && npm install

backend: ## Run the FastAPI backend on :8000 (mock provider by default)
	cd backend && bash scripts/run.sh

dashboard: ## Run the Next.js dashboard on :3000
	cd frontend && npm run dev

seed: ## Ingest the sample documents in data/ (backend must be running)
	AEGIS_URL=$(AEGIS_URL) bash backend/scripts/seed.sh

test: test-backend ## Run all tests (alias for test-backend)

test-backend: ## Run every backend test suite with the offline mock provider
	cd backend && for t in tests/test_*.py; do \
		echo "=== $$t ==="; \
		HF_HOME=$$PWD/.hf_cache ./venv/bin/python "$$t" || exit 1; \
	done

build: ## Production build of the dashboard (verifies the frontend compiles)
	cd frontend && npm run build

clean: ## Remove caches and the generated metrics DB
	rm -f backend/aegis_metrics.db
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf backend/.pytest_cache

demo: ## Print the guided demo flow
	@echo "Aegis demo flow:"
	@echo "  1. make install"
	@echo "  2. make backend      # terminal 1"
	@echo "  3. make dashboard    # terminal 2"
	@echo "  4. make seed         # terminal 3, once the backend is up"
	@echo "  5. open http://localhost:3000 and ask:"
	@echo "       - 'How does Aegis route a query?'        (grounded RAG)"
	@echo "       - 'What is Acme Cloud\\'s refund policy?'  (grounded RAG)"
	@echo "       - 'What is 2 + 2?'                        (direct, no retrieval)"
