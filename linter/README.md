# Python Linting and Formatting Guide

This guide covers how to set up and use linters and formatters to maintain Python code quality in this project.

## Setup

Run the setup script to install all linters and formatters:

```bash
chmod +x linter/setup.sh
./linter/setup.sh
```

## Tools Included

### Python

- **Black**: Code formatter (line length: 100)
- **isort**: Import statement formatter
- **Flake8**: Style guide enforcer
- **MyPy**: Type checker

### Docker and YAML

- **hadolint**: Dockerfile linter
- **yamllint**: YAML linter

## Running Linters and Formatters

You can run all linters and formatters with a single command:

```bash
./linter/lint-and-format.sh
```

For Docker files specifically:

```bash
./linter/docker-lint.sh
```

## Running Individual Tools

### Python (with Pipenv)

```bash
# Format code
pipenv run black --config linter/config/pyproject.toml src/datapipeline

# Sort imports
pipenv run isort --settings-path linter/config/pyproject.toml src/datapipeline

# Lint
pipenv run flake8 --config linter/config/.flake8 src/datapipeline

# Type check
pipenv run mypy --config-file linter/config/mypy.ini src/datapipeline
```

## Customizing Configuration

You can customize the linter settings by editing the configuration files in `linter/config/`.
