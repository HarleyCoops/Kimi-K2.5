# Makefile for Kimi K2.5 API Tools
# Usage: make <command>

.PHONY: help install setup test quick example explore validate clean lint format check

# Default target
help:
	@echo "Kimi K2.5 API Tools - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install    - Install dependencies from requirements.txt"
	@echo "  make setup      - Run interactive API key setup"
	@echo ""
	@echo "Test Commands:"
	@echo "  make quick      - Run quick test to verify API connectivity"
	@echo "  make test       - Run pytest test suite"
	@echo "  make validate   - Validate tool calling implementation"
	@echo "  make explore    - Explore advanced tool capabilities"
	@echo ""
	@echo "Example Commands:"
	@echo "  make example    - Run comprehensive API examples"
	@echo "  make multimodal - Run multimodal (image) examples"
	@echo "  make swarm      - Run agent swarm example"
	@echo ""
	@echo "Development Commands:"
	@echo "  make lint       - Run linting (flake8, pylint)"
	@echo "  make format     - Format code with black"
	@echo "  make check      - Run all checks (lint + test)"
	@echo "  make clean      - Remove cache files and build artifacts"

# Setup Commands
install:
	pip install -r requirements.txt

setup:
	python setup_api_key.py

# Test Commands
quick:
	python quick_test.py

test:
	python -m pytest tests/ -v

validate:
	python validate_tool_calling.py

explore:
	python explore_tools.py

# Example Commands
example:
	python kimi_k2_api_example.py

multimodal:
	python -m multimodal.image_understanding

swarm:
	python -m swarm.orchestrator

# Development Commands
lint:
	@echo "Running flake8..."
	-flake8 *.py --max-line-length=100 --ignore=E501,W503
	@echo "Running pylint..."
	-pylint *.py --disable=C0103,C0111,R0903 --max-line-length=100

format:
	black *.py --line-length=100
	black tests/ --line-length=100
	black multimodal/ --line-length=100
	black swarm/ --line-length=100

check: lint test

clean:
	@echo "Cleaning cache files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ 2>/dev/null || true
	@echo "Clean complete!"
