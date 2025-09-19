# Makefile for NeuroScope - Neural Network Framework
# Provides common development tasks and automation

.PHONY: help install dev-install test lint format type-check security clean build docs serve-docs pre-commit all-checks publish-test publish

# Default target
help: ## Show this help message
	@echo "NeuroScope Development Makefile"
	@echo "==============================="
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; OFS = "\t"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# Installation targets
install: ## Install the package for production use
	@echo "Installing NeuroScope..."
	poetry install --only main

dev-install: ## Install the package with development dependencies
	@echo "Installing NeuroScope with development dependencies..."
	poetry install --with dev
	@echo "Installing pre-commit hooks..."
	poetry run pre-commit install

# Testing targets
test: ## Run the full test suite
	@echo "Running test suite..."
	poetry run pytest -v --cov=neuroscope --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage for faster execution
	@echo "Running fast tests..."
	poetry run pytest -v -x

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "Running tests in watch mode..."
	poetry run ptw --runner "pytest -v"

test-unit: ## Run only unit tests
	@echo "Running unit tests..."
	poetry run pytest tests/ -v -m "not integration and not slow"

test-integration: ## Run only integration tests
	@echo "Running integration tests..."
	poetry run pytest tests/ -v -m integration

test-benchmark: ## Run performance benchmark tests
	@echo "Running benchmark tests..."
	poetry run pytest tests/ -v -m benchmark --benchmark-json=benchmark.json

# Code quality targets
lint: ## Run all linting checks
	@echo "Running linting checks..."
	@echo "  ├── Black (formatting check)..."
	@poetry run black --check .
	@echo "  ├── isort (import sorting check)..."
	@poetry run isort --check-only .
	@echo "  ├── flake8 (style and complexity)..."
	@poetry run flake8 .
	@echo "  └── All linting checks passed!"

format: ## Format code with black and isort
	@echo "Formatting code..."
	@echo "  ├── Running black..."
	@poetry run black .
	@echo "  ├── Running isort..."
	@poetry run isort .
	@echo "  └── Code formatting completed!"

type-check: ## Run type checking with mypy
	@echo "Running type checking..."
	poetry run mypy src/neuroscope
	@echo "Type checking completed!"

security: ## Run security checks
	@echo "Running security checks..."
	@echo "  ├── Bandit (code security)..."
	@poetry run bandit -r src/neuroscope
	@echo "  ├── Safety (dependency security)..."
	@poetry run safety check
	@echo "  └── Security checks completed!"

# Code quality automation
pre-commit: ## Run pre-commit hooks on all files
	@echo "Running pre-commit hooks..."
	poetry run pre-commit run --all-files

all-checks: lint type-check security test ## Run all code quality checks and tests
	@echo "All quality checks completed successfully!"

# Cleanup targets
clean: ## Clean up temporary files and caches
	@echo "Cleaning up..."
	@echo "  ├── Removing Python cache files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "  ├── Removing test artifacts..."
	@rm -rf .coverage htmlcov/ .pytest_cache/
	@echo "  ├── Removing build artifacts..."
	@rm -rf build/ dist/
	@echo "  ├── Removing mypy cache..."
	@rm -rf .mypy_cache/
	@echo "  └── Cleanup completed!"

clean-docs: ## Clean documentation build files
	@echo "Cleaning documentation..."
	@rm -rf docs/_build/
	@echo "Documentation cleanup completed!"

# Build targets
build: clean ## Build the package for distribution
	@echo "Building package..."
	poetry build
	@echo "Package built successfully!"
	@echo "Distribution files:"
	@ls -la dist/

build-check: build ## Build and verify the package
	@echo "Verifying package..."
	@poetry run twine check dist/*
	@echo "Package verification completed!"

# Documentation targets
docs: ## Build documentation
	@echo "Building documentation..."
	@cd docs && poetry run sphinx-build -b html . _build/html
	@echo "Documentation built successfully!"
	@echo "Open docs/_build/html/index.html to view"

docs-live: ## Build and serve documentation with auto-reload
	@echo "Starting live documentation server..."
	@cd docs && poetry run sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000

serve-docs: docs ## Build and serve documentation locally
	@echo "Serving documentation..."
	@cd docs/_build/html && python -m http.server 8080
	@echo "Documentation available at http://localhost:8080"

# Development workflow targets
dev: dev-install pre-commit ## Set up complete development environment
	@echo "Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  • Run 'make test' to run the test suite"
	@echo "  • Run 'make lint' to check code quality"
	@echo "  • Run 'make docs' to build documentation"

check: format lint type-check test ## Complete development check workflow
	@echo "All development checks passed!"

# CI/CD simulation
ci: ## Simulate CI/CD pipeline locally
	@echo "Simulating CI/CD pipeline..."
	@echo "Step 1: Installing dependencies..."
	@make dev-install
	@echo ""
	@echo "Step 2: Running linting..."
	@make lint
	@echo ""
	@echo "Step 3: Running type checking..."
	@make type-check
	@echo ""
	@echo "Step 4: Running security checks..."
	@make security
	@echo ""
	@echo "Step 5: Running tests..."
	@make test
	@echo ""
	@echo "Step 6: Building package..."
	@make build-check
	@echo ""
	@echo "CI/CD simulation completed successfully!"

# Publishing targets
publish-test: build-check ## Publish to Test PyPI
	@echo "Publishing to Test PyPI..."
	poetry config repositories.testpypi https://test.pypi.org/legacy/
	poetry publish -r testpypi
	@echo "Published to Test PyPI!"
	@echo "View at: https://test.pypi.org/project/neuroscope/"

publish: build-check ## Publish to PyPI (production)
	@echo "Publishing to PyPI..."
	@read -p "Are you sure you want to publish to PyPI? (y/N): " confirm && [ "$$confirm" = "y" ]
	poetry publish
	@echo "Published to PyPI!"
	@echo "View at: https://pypi.org/project/neuroscope/"

# Utility targets
version: ## Show current version
	@echo "NeuroScope version: $$(poetry version -s)"

info: ## Show project information
	@echo "Project Information"
	@echo "=================="
	@echo "Name: $$(poetry version | cut -d' ' -f1)"
	@echo "Version: $$(poetry version -s)"
	@echo "Python: $$(poetry run python --version)"
	@echo "Poetry: $$(poetry --version)"
	@echo ""
	@echo "Dependencies:"
	@poetry show --tree --only main
	@echo ""
	@echo "Development Dependencies:"
	@poetry show --tree --only dev

env: ## Show environment information
	@echo "Environment Information"
	@echo "======================"
	@echo "Virtual Environment: $$(poetry env info --path)"
	@echo "Python Executable: $$(poetry run which python)"
	@echo "Pip Version: $$(poetry run pip --version)"
	@echo ""
	@echo "Installed Packages:"
	@poetry run pip list

# Performance and profiling
profile: ## Run performance profiling on tests
	@echo "Running performance profiling..."
	poetry run pytest tests/ --profile --profile-svg
	@echo "Profile results saved to prof/"

benchmark-compare: ## Compare benchmark results
	@echo "Comparing benchmark results..."
	@if [ -f benchmark.json ]; then \
		poetry run pytest tests/ -m benchmark --benchmark-compare=benchmark.json; \
	else \
		echo "No previous benchmark results found. Run 'make test-benchmark' first."; \
	fi

# Git workflow helpers
tag-version: ## Create and push a version tag
	@echo "Creating version tag..."
	@VERSION=$$(poetry version -s) && \
	git tag -a "v$$VERSION" -m "Release v$$VERSION" && \
	git push origin "v$$VERSION"
	@echo "Tag created and pushed!"

release-notes: ## Generate release notes from git log
	@echo "Generating release notes..."
	@VERSION=$$(poetry version -s) && \
	echo "# Release Notes for v$$VERSION" > RELEASE_NOTES.md && \
	echo "" >> RELEASE_NOTES.md && \
	git log --pretty=format:"- %s (%h)" $$(git describe --tags --abbrev=0 2>/dev/null || echo "HEAD")..HEAD >> RELEASE_NOTES.md
	@echo "Release notes generated in RELEASE_NOTES.md"

# Quick shortcuts
t: test ## Shortcut for test
l: lint ## Shortcut for lint
f: format ## Shortcut for format
c: check ## Shortcut for check
b: build ## Shortcut for build
d: docs ## Shortcut for docs