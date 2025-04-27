#!/bin/bash

# Script to run Python linters and formatters on the project
# All configuration files are in linter/config

set -e

echo "Running Python linters and formatters..."

# Format Python files
echo "Formatting Python files with black..."
pipenv run black --config linter/config/pyproject.toml src/datapipeline
if [ -d "src/api" ]; then
    pipenv run black --config linter/config/pyproject.toml src/api
fi
if [ -d "tests" ]; then
    pipenv run black --config linter/config/pyproject.toml tests
fi

# Sort imports in Python files
echo "Sorting imports with isort..."
pipenv run isort --settings-path linter/config/pyproject.toml src/datapipeline
if [ -d "src/api" ]; then
    pipenv run isort --settings-path linter/config/pyproject.toml src/api
fi
if [ -d "tests" ]; then
    pipenv run isort --settings-path linter/config/pyproject.toml tests
fi

# Lint Python files
echo "Linting Python files with flake8..."
pipenv run flake8 --config linter/config/.flake8 src/datapipeline
if [ -d "src/api" ]; then
    pipenv run flake8 --config linter/config/.flake8 src/api
fi
# Skip linting test files with flake8
echo "Skipping flake8 linting for test files..."

# Type checking Python files
echo "Type checking Python files with mypy..."
pipenv run mypy --config-file linter/config/mypy.ini src/datapipeline
if [ -d "src/api" ]; then
    pipenv run mypy --config-file linter/config/mypy.ini src/api
fi
# Skip type checking test files
echo "Skipping mypy type checking for test files..."

echo "All Python linting and formatting complete!"
