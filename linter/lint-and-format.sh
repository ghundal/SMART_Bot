#!/bin/bash

# Script to run linters and formatters on the project
# All configuration files are in linter/config

set -e

echo "Running linters and formatters..."

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
if [ -d "tests" ]; then
    pipenv run flake8 --config linter/config/.flake8 tests
fi

# Type checking Python files
echo "Type checking Python files with mypy..."
pipenv run mypy --config-file linter/config/mypy.ini src/datapipeline
if [ -d "src/api" ]; then
    pipenv run mypy --config-file linter/config/mypy.ini src/api
fi
if [ -d "tests" ]; then
    pipenv run mypy --config-file linter/config/mypy.ini tests
fi

# Format and lint JavaScript/TypeScript files if frontend directory exists
if [ -d "src/frontend" ]; then
    echo "Formatting JavaScript/TypeScript files with prettier..."
    npx prettier --config linter/config/.prettierrc --write "src/frontend/**/*.{js,jsx,ts,tsx,json,css,scss,md}"

    echo "Linting JavaScript/TypeScript files with eslint..."
    npx eslint --config linter/config/.eslintrc.js "src/frontend/**/*.{js,jsx,ts,tsx}"
fi

# Format SQL files if sql directory exists
if [ -d "sql" ]; then
    echo "Formatting SQL files..."
    find sql -name "*.sql" -exec npx sql-formatter -o {} {} \;
fi

echo "All linting and formatting complete!"
