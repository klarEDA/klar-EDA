.PHONY: help install install-dev test test-unit test-integration test-coverage lint format dev build clean docs-build docs-serve

# Default target
help: ## Show this help message
	@echo "klar-EDA v2.0 Development Commands"
	@echo "=================================="
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development Setup
install: ## Install production dependencies
	poetry install --only=main

install-dev: ## Install all dependencies including dev tools
	poetry install --with=dev,test
	poetry run pre-commit install
	@echo "✅ Development environment setup complete!"

##@ Code Quality
lint: ## Run all linting tools
	poetry run ruff check .
	poetry run mypy .
	poetry run black --check .
	@echo "✅ All linting checks passed!"

format: ## Format code with black and ruff
	poetry run black .
	poetry run ruff check --fix .
	@echo "✅ Code formatting complete!"

##@ Testing
test: ## Run all tests
	poetry run pytest tests/ -v --cov=klar_eda --cov-report=term-missing

test-unit: ## Run unit tests only
	poetry run pytest tests/unit/ -v -m "not slow"

test-integration: ## Run integration tests only
	poetry run pytest tests/integration/ -v

test-coverage: ## Generate coverage report
	poetry run pytest tests/ --cov=klar_eda --cov-report=html --cov-report=term
	@echo "📊 Coverage report generated in htmlcov/"

##@ Development
dev: ## Start development environment
	@echo "🚀 Starting development servers..."
	@echo "📊 Development servers started!"
	@echo "   - API: http://localhost:8000"
	@echo "   - Docs: http://localhost:8000/docs"

##@ Build and Deploy
build: ## Build the package
	poetry build
	@echo "📦 Package built successfully!"

clean: ## Clean temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	@echo "🧹 Cleanup complete!"

##@ Documentation
docs-build: ## Build documentation
	cd docsource && make html
	@echo "📚 Documentation built in docs/"

docs-serve: ## Serve documentation locally
	cd docsource && make html && cd .. && python -m http.server 8080 -d docs
	@echo "📖 Documentation available at http://localhost:8080"

##@ Utilities
check: ## Run all checks (lint + test)
	make lint
	make test
	@echo "✅ All checks passed!"

setup-hooks: ## Setup git hooks
	poetry run pre-commit install
	@echo "🔗 Git hooks installed!"

##@ Release
release-patch: ## Create a patch release
	poetry version patch
	@echo "📈 Version bumped to patch release"

release-minor: ## Create a minor release
	poetry version minor
	@echo "📈 Version bumped to minor release"

release-major: ## Create a major release
	poetry version major
	@echo "📈 Version bumped to major release"

##@ Examples
example-basic: ## Run basic usage example
	poetry run python examples/basic_usage.py

##@ Quick Commands
quick-test: ## Quick test run (unit tests only, no coverage)
	poetry run pytest tests/unit/ -x -v

quick-lint: ## Quick lint check (ruff only)
	poetry run ruff check .

quick-format: ## Quick format (black only)
	poetry run black .

##@ Project Management
stats: ## Show project statistics
	@echo "📊 Project Statistics"
	@echo "===================="
	@echo "Lines of code:"
	@find . -name "*.py" -not -path "./.*" | xargs wc -l | tail -1
	@echo "Number of Python files:"
	@find . -name "*.py" -not -path "./.*" | wc -l
