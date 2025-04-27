#!/bin/bash
# Script to set up linters and formatters for Python code only
# All configuration files will be contained within the linter directory

set -e

echo "Setting up Python linters and formatters for your project..."

# Create config directory if it doesn't exist
mkdir -p linter/config

# Install Python linters and formatters using pipenv
# Make sure pipenv is installed
if ! command -v pipenv &> /dev/null; then
    echo "pipenv not found. Installing pipenv..."
    pip install pipenv
fi

# Install linting dependencies with pipenv
echo "Installing Python linting tools with pipenv..."
pipenv install --dev black flake8 isort pylint mypy

# Create configuration files
echo "Creating configuration files..."

# Python - Black configuration
cat > linter/config/pyproject.toml << 'EOF'
[tool.black]
line-length = 100
target-version = ['py38']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
EOF

# Python - Flake8 configuration
cat > linter/config/.flake8 << 'EOF'
[flake8]
max-line-length = 100
extend-ignore = E203
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist,.venv,tests/,src/frontend/
EOF

# Python - MyPy configuration
cat > linter/config/mypy.ini << 'EOF'
[mypy]
mypy_path = src
python_version = 3.12
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
disallow_incomplete_defs = False
ignore_missing_imports = True
namespace_packages = True
explicit_package_bases = True
exclude = ('tests|src/frontend')

[mypy.plugins.numpy.*]
follow_imports = skip

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-google.cloud.*]
ignore_missing_imports = True
EOF

# Create pre-commit config for Python only
cat > linter/config/.pre-commit-config.yaml << 'EOF'
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black
        args: ["--config", "linter/config/pyproject.toml"]

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort
        args: ["--settings-path", "linter/config/pyproject.toml"]

-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        args: ["--config", "linter/config/.flake8"]
EOF

# YAML linting config
cat > linter/config/.yamllint << 'EOF'
extends: default

rules:
  line-length:
    max: 120
    level: warning
  document-start: disable
EOF

# Create a helper script to run linters and formatters (Python only)
cat > linter/lint-and-format.sh << 'EOF'
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
EOF

# Make the script executable
chmod +x linter/lint-and-format.sh

# Create Docker linting script
cat > linter/docker-lint.sh << 'EOF'
#!/bin/bash
# Script to set up and run Docker linting

set -e

echo "Setting up Docker linting tools..."

# Install hadolint for Dockerfile linting if not already installed
if ! command -v hadolint &> /dev/null; then
    if command -v brew &> /dev/null; then
        brew install hadolint
    elif command -v apt-get &> /dev/null; then
        wget -O /tmp/hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
        chmod +x /tmp/hadolint
        sudo mv /tmp/hadolint /usr/local/bin/
    else
        echo "Please install hadolint manually: https://github.com/hadolint/hadolint"
    fi
fi

# Install yamllint for YAML linting
pipenv install --dev yamllint

# Function to lint Docker files
lint_docker_files() {
    echo "Linting Dockerfiles..."

    # Find all Dockerfiles
    dockerfiles=$(find . -name "Dockerfile*")

    if [ -z "$dockerfiles" ]; then
        echo "No Dockerfiles found."
    else
        for file in $dockerfiles; do
            echo "Linting $file"
            hadolint "$file"
        done
    fi

    echo "Linting docker-compose.yml files..."

    # Find all docker-compose.yml files
    compose_files=$(find . -name "docker-compose*.yml")

    if [ -z "$compose_files" ]; then
        echo "No docker-compose files found."
    else
        for file in $compose_files; do
            echo "Linting $file"
            pipenv run yamllint -c linter/config/.yamllint "$file"
        done
    fi
}

# Run the linting
lint_docker_files

echo "Docker linting complete!"
EOF

# Make the Docker linting script executable
chmod +x linter/docker-lint.sh

# Install pre-commit but configure it to use the config in the linter directory
pipenv install --dev pre-commit
mkdir -p .git/hooks
cat > .git/hooks/pre-commit << EOF
#!/bin/sh
# Redirect to our pre-commit config
exec pre-commit run --config \$(pwd)/linter/config/.pre-commit-config.yaml --hook-stage pre-commit
EOF
chmod +x .git/hooks/pre-commit

# Create README
cat > linter/README.md << 'EOF'
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

## Pre-commit Hooks

Pre-commit hooks are installed to run linters and formatters automatically before each commit.

```bash
# Run pre-commit manually on all files
pre-commit run --all-files --config linter/config/.pre-commit-config.yaml
```

## Customizing Configuration

You can customize the linter settings by editing the configuration files in `linter/config/`.
EOF

echo "Setup complete! You can now run ./linter/lint-and-format.sh to lint and format your Python code."
echo "All configuration files are contained within the linter/config directory."
