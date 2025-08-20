# EconoNet Makefile
# Production-ready shortcuts for development and deployment

.PHONY: help install test lint format run docker-build docker-run clean deploy

# Default target
help:
	@echo "EconoNet - Economic Intelligence Platform"
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  run         - Run Streamlit app"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run Docker container"
	@echo "  clean       - Clean temporary files"
	@echo "  deploy      - Deploy to production"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "Tests completed!"

# Run linting
lint:
	@echo "Running linting..."
	flake8 src/ --max-line-length=100 --ignore=E203,W503
	@echo "Linting completed!"

# Format code
format:
	@echo "Formatting code..."
	black src/ --line-length=100
	black pages/ --line-length=100
	black app.py --line-length=100
	@echo "Code formatting completed!"

# Run Streamlit application
run:
	@echo "Starting EconoNet application..."
	streamlit run app.py --server.port=8501

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t econet:latest .
	@echo "Docker image built successfully!"

# Run Docker container
docker-run:
	@echo "Running Docker container..."
	docker run -p 8501:8501 econet:latest

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type f -name ".coverage" -delete
	@echo "Cleanup completed!"

# Deploy to production (placeholder)
deploy:
	@echo "Deploying to production..."
	@echo "1. Run tests..."
	make test
	@echo "2. Build Docker image..."
	make docker-build
	@echo "3. Tag for production..."
	docker tag econet:latest econet:prod
	@echo "Deployment ready! Push to your container registry."

# Development setup
dev-setup: install
	@echo "Setting up development environment..."
	pre-commit install
	@echo "Development environment ready!"

# Check code quality
quality-check: lint test
	@echo "Code quality check completed!"

# Quick start for new developers
quick-start:
	@echo "Quick start for EconoNet development:"
	@echo "1. Installing dependencies..."
	make install
	@echo "2. Running tests..."
	make test
	@echo "3. Starting application..."
	make run
